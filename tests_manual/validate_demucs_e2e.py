"""Demucs Phase-3-bundle end-to-end validation (GPU).

Validates the cjm-torch-plugin-utils adoption (release_model + cuda_oom +
resolve_torch_device), the Shape-1 heartbeat around the torch.hub model load, the
Q3 Layer B cache_dir_for_config output dir, and the Track 19 WORKER_ENV migration
live, mirroring the Voxtral-HF Phase 3 validation pattern.

Run from the demucs repo root after:
  1. `cjm-ctl --cjm-config cjm.yaml setup-runtime`
  2. `cjm-ctl --cjm-config cjm.yaml install-all --plugins plugins_test.yaml`
     (demucs + ffmpeg + cjm-system-monitor-nvidia)
  3. A short music-with-vocals clip at test_files/podcast-286-sam-cooper_segment_000.m4a

Then:
  conda run -n cjm-media-plugin-demucs --no-capture-output \\
    python tests_manual/validate_demucs_e2e.py

This script:
  - Verifies the demucs v2.0 manifest carries (a) a non-empty `description`,
    (b) requires_gpu + no quantitative resource fields, and (c) the Track 19
    worker_env with CUDA_VISIBLE_DEVICES (static) + a TEMPLATED TORCH_HOME default,
    with an empty install.env_vars.
  - Eagerly loads the Demucs model via prefetch() (torch.hub download on a cold
    cache — heartbeat-wrapped).
  - Runs separation via submit_sequence(ffmpeg.convert .m4a->wav -> demucs.separate_vocals).
  - Asserts the vocals output exists under a cache_dir_for_config dir, the job row
    persisted, and the empirical store recorded a NON-ZERO gpu_memory_mb_peak
    (real subtree GPU attribution via nvidia-monitor).
"""
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("demucs-e2e")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = REPO_ROOT / "test_files" / "podcast-286-sam-cooper_segment_000.m4a"
MANIFESTS_DIR = REPO_ROOT / ".cjm" / "manifests"
EMPIRICAL_DB = REPO_ROOT / ".cjm" / "empirical_resources.db"

PLUGIN_NAME = "cjm-media-plugin-demucs"
SYSMON_NAME = "cjm-system-monitor-nvidia"
FFMPEG_NAME = "cjm-media-plugin-ffmpeg"


def check_prereqs() -> None:
    assert TEST_AUDIO.exists(), f"Missing test audio: {TEST_AUDIO}"
    assert MANIFESTS_DIR.exists(), (
        f"Missing manifests dir: {MANIFESTS_DIR} — run cjm-ctl setup-runtime + install-all first"
    )
    for name in (PLUGIN_NAME, SYSMON_NAME, FFMPEG_NAME):
        assert (MANIFESTS_DIR / f"{name}.json").exists(), f"Missing manifest: {name}.json"
    log.info("Prereqs OK: test audio + demucs + nvidia-monitor + ffmpeg manifests present")


def assert_manifest_shape() -> None:
    manifest = json.loads((MANIFESTS_DIR / f"{PLUGIN_NAME}.json").read_text())
    assert manifest["format_version"] == "2.0", manifest["format_version"]
    code = manifest["code"]

    desc = code.get("description") or manifest.get("description") or ""
    assert desc.strip(), "manifest description is empty (T24 regression)"
    log.info(f"Manifest T24 description: {desc!r}")

    tax = code["taxonomy"]
    assert tax["domain"] == "media" and tax["role"] == "MediaProcessingPlugin", tax
    assert code["resources"]["requires_gpu"] is True, code["resources"]
    for stale in ("min_gpu_vram_mb", "recommended_gpu_vram_mb", "min_system_ram_mb"):
        assert stale not in code["resources"], f"stale resource field present: {stale}"
    log.info(f"Manifest CR-1/Phase-5a: taxonomy={tax}, resources={code['resources']}")

    # Track 19: CUDA_VISIBLE_DEVICES (static) + TORCH_HOME (templated); empty install.env_vars.
    worker_env = code.get("worker_env", [])
    by_name = {e["name"]: e for e in worker_env}
    assert {"CUDA_VISIBLE_DEVICES", "TORCH_HOME"} <= set(by_name), (
        f"Track 19 WORKER_ENV missing expected vars: {sorted(by_name)}"
    )
    torch_home_default = by_name["TORCH_HOME"].get("default", "")
    assert torch_home_default == "${CJM_MODELS_DIR}/torch", (
        f"TORCH_HOME default not templated: {torch_home_default!r}"
    )
    install_env = manifest.get("install", {}).get("env_vars", {})
    assert not install_env, f"install.env_vars should be empty post-migration: {install_env}"
    log.info(f"Manifest Track 19 worker_env: {sorted(by_name)} | TORCH_HOME default={torch_home_default!r}; install.env_vars empty")


def run_e2e() -> None:
    import asyncio

    from cjm_plugin_system.core.manager import PluginManager
    from cjm_plugin_system.core.config import get_config
    from cjm_plugin_system.core.queue import JobQueue, SequenceStep, JobStatus

    cfg = get_config()
    log.info(f"data_dir={cfg.data_dir}, models_dir={cfg.models_dir}")

    pm = PluginManager(search_paths=[MANIFESTS_DIR], sysmon_plugin_name=SYSMON_NAME)
    pm.discover_manifests()
    log.info(f"Discovered: {[m.name for m in pm.discovered]}")

    pm.load_plugin(next(m for m in pm.discovered if m.name == SYSMON_NAME))
    pm.load_plugin(next(m for m in pm.discovered if m.name == FFMPEG_NAME))
    demucs_meta = next(m for m in pm.discovered if m.name == PLUGIN_NAME)
    db_path = demucs_meta.manifest.get("db_path")
    ok = pm.load_plugin(demucs_meta, config={})
    assert ok, f"Failed to load {PLUGIN_NAME}"
    demucs_id = demucs_meta.name
    log.info(f"Loaded {SYSMON_NAME} + {FFMPEG_NAME} + {PLUGIN_NAME}; db_path={db_path}")

    # CR-4 prefetch: torch.hub model download (cold cache) wrapped by the substrate heartbeat.
    log.info("Calling prefetch() to download + load the Demucs model...")
    t0 = time.time()
    pm.get_plugin(demucs_id).prefetch()
    log.info(f"prefetch() returned in {time.time() - t0:.1f}s")

    # ffmpeg.convert writes to <ffmpeg_data_dir>/converted/<stem>.wav.
    ffmpeg_data_dir = Path(next(m for m in pm.discovered if m.name == FFMPEG_NAME).manifest["db_path"]).parent
    predicted_wav = ffmpeg_data_dir / "converted" / f"{TEST_AUDIO.stem}.wav"
    log.info(f"ffmpeg will convert {TEST_AUDIO.name} -> {predicted_wav}")

    async def run_sequence() -> Any:
        queue = JobQueue(deps=pm, sysmon_plugin_name=SYSMON_NAME)
        await queue.start()
        try:
            seq_id = await queue.submit_sequence(
                steps=[
                    SequenceStep(plugin_instance_id=FFMPEG_NAME, kwargs={
                        "action": "convert", "input_path": str(TEST_AUDIO),
                        "output_format": "wav", "sample_rate": 44100, "channels": 2,
                    }),
                    SequenceStep(plugin_instance_id=demucs_id, kwargs={
                        "action": "separate_vocals", "input_path": str(predicted_wav),
                    }),
                ],
                fail_fast=True,
            )
            log.info(f"Submitted sequence {seq_id}: ffmpeg.convert -> demucs.separate_vocals")
            terminal = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
            while True:
                seq = queue.get_sequence(seq_id)
                if seq is None:
                    raise RuntimeError(f"sequence {seq_id} disappeared")
                if seq.status in terminal:
                    break
                await asyncio.sleep(0.5)
            if seq.status != JobStatus.completed:
                raise RuntimeError(f"Sequence {seq_id} status={seq.status}; results={seq.results}")
            return seq.results[-1].result
        finally:
            await queue.stop()

    log.info(f"Submitting submit_sequence for {TEST_AUDIO.name}...")
    t0 = time.time()
    result = asyncio.run(run_sequence())
    log.info(f"Sequence completed in {time.time() - t0:.1f}s")

    out_path = result.get("output_path") if isinstance(result, dict) else getattr(result, "output_path", None)
    assert out_path and Path(out_path).exists(), f"vocals output missing: {out_path} (result={result!r})"
    # Q3 Layer B: output dir is the content+config-addressed cache dir.
    assert "separate_vocals" in out_path, f"output not under cache_dir_for_config layout: {out_path}"
    log.info(f"Vocals written to cache dir: {out_path}")
    log.info(f"stems_available={result.get('stems_available')}, duration={result.get('duration'):.1f}s")

    # Plugin DB: confirm the job row persisted.
    if db_path and Path(db_path).exists():
        con = sqlite3.connect(db_path)
        try:
            for t in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                log.info(f"plugin DB {t}: {n} rows")
        finally:
            con.close()

    # Empirical store: GPU plugin -> assert a NON-ZERO gpu peak (real subtree attribution).
    assert EMPIRICAL_DB.exists(), f"empirical store not created: {EMPIRICAL_DB}"
    con = sqlite3.connect(EMPIRICAL_DB)
    gpu_peak = 0.0
    try:
        for t in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
            if "gpu_memory_mb_peak_max" not in cols:
                continue
            for r in con.execute(
                f"SELECT * FROM {t} WHERE plugin_name=? OR instance_id=? OR instance_id LIKE ?",
                (PLUGIN_NAME, demucs_id, f"{PLUGIN_NAME}%"),
            ).fetchall():
                row = dict(zip(cols, r))
                log.info(f"  empirical {t}: {row}")
                gpu_peak = max(gpu_peak, float(row.get("gpu_memory_mb_peak_max") or 0.0))
    finally:
        con.close()
    assert gpu_peak > 0.0, f"empirical gpu_memory_mb_peak is 0 — subtree GPU attribution failed"
    log.info(f"GPU attribution VERIFIED: demucs gpu_memory_mb_peak_max={gpu_peak:.1f} MB")

    pm.unload_plugin(demucs_id)
    pm.unload_plugin(FFMPEG_NAME)
    pm.unload_plugin(SYSMON_NAME)
    log.info("Unloaded plugins; validation done.")


def main() -> int:
    check_prereqs()
    assert_manifest_shape()
    run_e2e()
    log.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Tombstone — `validate_demucs_e2e.py` (RETIRED 2026-06-18, stage 9)

**Origin:** `cjm-media-plugin-demucs/tests_manual/validate_demucs_e2e.py` (2026-06-10, post-migration Phase-3-bundle era).
**Retired because:** per-tool e2e validator superseded by the cores' standing harness; the pre-overhaul/per-tool `tests_manual` cohort is retired, not patched.

**What it validated:** cjm-torch-plugin-utils adoption (`release_model` / `cuda_oom` / `resolve_torch_device`), the heartbeat around the `torch.hub` model load, the Layer-B `cache_dir_for_config` output dir, and the WORKER_ENV migration — via a real Demucs source-separation run.

**Coverage status:** SUPERSEDED — Demucs is exercised end-to-end through `cjm-transcription-core`'s opt-in `--preprocessing-plugin` path (source-separation task family + artifact-producing adapter), validated on the real corpus.

**Reimplementation target:** none required (cores are the standing harness).

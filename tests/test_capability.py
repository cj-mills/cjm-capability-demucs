"""Hermetic tests for cjm_capability_demucs.capability (c25780e8 flip).

The notebook carried no test cells; these cover the pure-compute surface that
runs without the Demucs model: config round-trip/schema, capability shape,
declarative reload triggers, and lazy-load state. Real separation is exercised
by the transcription pipeline's live runs."""
from cjm_capability_demucs.capability import (DemucsCapabilityConfig,
                                              DemucsProcessingCapability)
from cjm_substrate.utils.validation import config_to_dict, dict_to_config


def test_config_defaults_and_round_trip():
    cfg = DemucsCapabilityConfig()
    assert cfg.model == "htdemucs" and cfg.device == "auto"
    assert cfg.shifts == 1 and cfg.overlap == 0.25 and cfg.segment is None
    d = config_to_dict(cfg)
    assert dict_to_config(DemucsCapabilityConfig, d) == cfg


def test_capability_shape_and_lazy_state():
    cap = DemucsProcessingCapability()
    assert cap.config is None and cap._separator is None  # lazy until initialize
    assert cap.name  # identity derived from the installed distribution
    env_names = {e.name for e in cap.WORKER_ENV}
    assert env_names == {"CUDA_VISIBLE_DEVICES", "TORCH_HOME"}


def test_apply_config_and_schema():
    cap = DemucsProcessingCapability()
    cap._apply_config({"model": "mdx_extra", "shifts": 2})
    assert cap.config.model == "mdx_extra" and cap.config.shifts == 2
    assert cap.config.overlap == 0.25  # untouched fields keep defaults
    schema = cap.get_config_schema()
    assert "model" in schema.get("properties", {})
    assert cap.get_current_config()["model"] == "mdx_extra"


def test_reload_trigger_metadata():
    from dataclasses import fields
    from cjm_substrate.core.capability import RELOAD_TRIGGER
    triggers = {f.name: f.metadata.get(RELOAD_TRIGGER) for f in fields(DemucsCapabilityConfig)}
    # model + device changes must fire the model release (CR-4 declarative path)
    assert triggers["model"] == "model" and triggers["device"] == "model"


def test_zero_sample_decode_is_an_input_error_naming_the_file(tmp_path):
    # Sighted 2026-09-10: a stream-copied Ogg/Opus cut with negative granule positions
    # decoded to ZERO samples and demucs died on a bare `assert offset < total_length`
    # — a message-less AssertionError that reached the host as CapabilityFatalError('').
    # The capability names the input instead: the file's fault, typed as user_input.
    import pytest
    from cjm_substrate.core.errors import CapabilityInputError

    class _Sep:
        samplerate = 44100

        def separate_audio_file(self, p):
            assert 0 < 0   # the shape of demucs.apply.TensorChunk's guard on an empty mix

    cap = DemucsProcessingCapability()
    cap._apply_config({})
    cap._separator = _Sep()
    src = tmp_path / "cut.ogg"
    src.write_bytes(b"OggS")
    with pytest.raises(CapabilityInputError) as ei:
        cap.separate_vocals(src, str(tmp_path / "out"))
    assert "cut.ogg" in str(ei.value) and "AssertionError" in str(ei.value)
    assert ei.value.fields_invalid == ["audio"]

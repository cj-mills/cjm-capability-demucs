"""CR-4 reconfigure-lifecycle validation for the Demucs plugin.

Contract-level (no real model load — the model is large/GPU). Exercises the
substrate's reconfigure delta path in-process with a fake separator object:

  1. reconfigure(device flip) -> RELEASE the separator (RELOAD_TRIGGER ->
     _release_model) + RE-APPLY config (_apply_config)
  2. on_disable releases (CR-2)

Run from the repo root in the plugin's env:

    conda run -n cjm-media-plugin-demucs --no-capture-output python tests_manual/test_reconfigure.py
"""
import sys


def main() -> int:
    from cjm_media_plugin_demucs.plugin import DemucsProcessingPlugin

    p = DemucsProcessingPlugin()
    p._apply_config({"device": "cpu"})
    assert p.config.device == "cpu"

    # 1) device trigger: release + re-apply
    p._separator = object()
    p.reconfigure({"device": "cpu"}, {"device": "auto"})
    assert p._separator is None, "device RELOAD_TRIGGER must fire _release_model"
    assert p.config.device == "auto", "reconfigure must re-apply config (CR-4)"
    print("[1] reconfigure device cpu->auto: separator released + applied  OK")

    # 2) on_disable releases (CR-2)
    p._separator = object()
    p.on_disable()
    assert p._separator is None, "on_disable must release the separator"
    print("[2] on_disable: separator released  OK")

    print("RECONFIGURE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

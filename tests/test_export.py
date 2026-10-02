import json

from wakeword.export import build_manifest, export


def test_manifest_fields(cfg):
    cfg.manifest.author = "Someone"
    cfg.manifest.probability_cutoff = 0.9
    manifest = build_manifest(cfg, "demo.tflite")
    assert manifest["type"] == "micro"
    assert manifest["version"] == 2
    assert manifest["wake_word"] == "hey demo"
    assert manifest["author"] == "Someone"
    assert manifest["model"] == "demo.tflite"
    assert manifest["micro"]["probability_cutoff"] == 0.9


def test_export_copies_model_and_writes_manifest(cfg, paths):
    ww = paths.wakeword(cfg.name)
    ww.tflite.parent.mkdir(parents=True)
    ww.tflite.write_bytes(b"model")

    manifest_path = export(cfg, paths)

    assert (ww.output / "demo.tflite").read_bytes() == b"model"
    assert json.loads(manifest_path.read_text())["model"] == "demo.tflite"

import yaml

from wakeword import train_config


def by_dir(rendered):
    return {f["features_dir"].split("/")[-1]: f for f in rendered["features"]}


def test_without_real_recordings(cfg, paths):
    rendered = train_config.render(cfg, paths)
    feats = by_dir(rendered)
    assert list(feats) == [
        "positives",
        "speech",
        "dinner_party",
        "no_speech",
        "dinner_party_eval",
    ]
    assert feats["positives"]["truth"] is True
    assert feats["dinner_party_eval"]["truncation_strategy"] == "split"
    assert rendered["training_steps"] == [10000]
    assert rendered["negative_class_weight"] == [20]


def test_with_real_recordings(cfg, paths):
    ww = paths.wakeword(cfg.name)
    ww.real_positive_features.mkdir(parents=True)
    ww.real_negative_features.mkdir(parents=True)
    cfg.real.positive_weight = 4.5

    feats = by_dir(train_config.render(cfg, paths))
    assert feats["real_positives"]["sampling_weight"] == 4.5
    assert feats["real_positives"]["truth"] is True
    assert feats["real_negatives"]["truth"] is False
    assert feats["real_negatives"]["truncation_strategy"] == "random"


def test_write_round_trips(cfg, paths):
    out = train_config.write(cfg, paths)
    assert yaml.safe_load(out.read_text()) == train_config.render(cfg, paths)

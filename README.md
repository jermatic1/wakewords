# wakewords

Trains custom [microWakeWord](https://github.com/OHF-Voice/micro-wake-word)
models for ESPHome's `micro_wake_word` component. Samples are synthesized
with the Piper LibriTTS-R generator and can be mixed with your own
recordings. CPU only.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev).

```sh
task setup
```

## Usage

```sh
cp wakewords/example.yaml wakewords/hey_dolphin.yaml   # set name and phrase
task train -- wakewords/hey_dolphin.yaml
```

The first run downloads about 30 GB of shared training data into `data/`.
`--skip-background` skips the AudioSet and FMA noise sets.

Stages can be run individually with `uv run wakeword <stage> wakewords/<name>.yaml`:

| Stage      | Output                                              |
| ---------- | --------------------------------------------------- |
| `generate` | Synthetic clips in `data/<name>/positives/`          |
| `features` | Spectrogram sets in `data/<name>/features/`          |
| `train`    | Trained model in `data/<name>/trained_model/`        |
| `export`   | `output/<name>/<name>.tflite` and `<name>.json`      |

## Your own recordings

Optional, picked up by `features`. Any sample rate or channel count.

- `data/<name>/real_positives/*.wav`: recordings of the phrase, one per file.
- `data/<name>/real_negatives/*.wav`: background audio from where the device
  will be used. The best way to reduce false triggers.

`real.positive_weight` and `real.negative_weight` in the YAML set how often
these are sampled relative to the synthetic data.

## Deploy

Copy `output/<name>/<name>.json` and `<name>.tflite` into the same folder in
your Home Assistant config and reference the manifest:

```yaml
micro_wake_word:
  models:
    - model: /config/models/hey_dolphin.json
```

## Tuning

- False triggers: raise `manifest.probability_cutoff`, add real negatives,
  or raise `train.negative_class_weight`.
- Missed detections: lower `probability_cutoff`, add real positives, or
  generate more samples. Phonetic spellings of the phrase can help.
- Model fails to load on the device: raise `manifest.tensor_arena_size`.

## Docker

The published image runs the same CLI with `data/`, `output/`, and
`wakewords/` mounted from the current directory:

```sh
task docker -- wakewords/hey_dolphin.yaml
```

## Development

```sh
task check   # tests, lint, type checks
task fmt     # format
task lock    # refresh uv.lock after editing pyproject.toml
task image   # build the Docker image locally instead of pulling it
```

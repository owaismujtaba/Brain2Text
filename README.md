# NJEM — Neural-to-Speech Decoding

Turn recorded brain activity into text. The pipeline learns to predict what a
person was trying to say directly from their neural signals, then reads the
prediction out as words using OpenAI's Whisper speech model.

## The idea in one picture

```
neural activity ──► ConvBiGRU model ──► Whisper embeddings ──► Whisper decoder ──► text
   (what we                (what we           (a compact             (frozen,
    record)                 train)             "sound fingerprint")   off-the-shelf)
```

We never generate audio at prediction time. Instead the model learns to output
the *embeddings* that Whisper's audio encoder would have produced for the spoken
sentence. Whisper's decoder then turns those embeddings into words. Because
Whisper is frozen and already excellent at going from embeddings to text, the
only thing we have to train is the small model that maps neural signals to
embeddings.

## How the training targets are made

We don't have real audio for every trial, so we build it:

1. **Generate audio** — take each trial's ground-truth transcription and
   synthesise speech for it with a text-to-speech model (StyleTTS2).
2. **Extract features** — run that audio through Whisper's frozen encoder and
   save the resulting embedding. This embedding is the target the model learns
   to reproduce from neural data.

## Multiple Whisper sizes

You can target different Whisper encoders (`tiny.en` = 384-dim embeddings,
`base.en` = 512, `small.en` = 768). One `extract_features` run produces a
separate feature set for every model listed in the config, each in its own
folder:

```
data/features/<model>/<split>/<session>/whisper_features_<split>.hdf5
checkpoints/<model>/best_wer.pt
```

Training picks **one** model per run via the `model` field; its embedding size,
feature folder, and checkpoint folder are all derived from that one choice, so
the sizes never collide or get mismatched.

## Pipeline stages

| Stage | Command | What it does |
|-------|---------|--------------|
| 1. Generate audio | `python main.py generate_audio` | Transcription → speech audio (16 kHz) |
| 2. Extract features | `python main.py extract_features` | Audio → Whisper encoder embeddings, one set per model |
| 3. Train | `python main.py train` | Learn neural → embeddings for the selected `model` |
| 4. Decode | `python main.py decode` | Decode a split to text, score WER |

Run everything in order with `python main.py workflow`.

## Configuration

Every setting lives in **`config.yaml`** — there are no command-line flags to
remember. Each stage has its own section. To change a run, edit the file (or copy
it) and pass it in:

```bash
python main.py train              # uses config.yaml
python main.py train my_run.yaml  # uses a different config
```

The model-independent dimensions (`n_ctx`, `neural_dim`, `frame_samples`) are
defined once at the top and reused by the train and decode sections via a YAML
anchor. The embedding size (`emb_dim`) is **not** set by hand — it is derived
automatically from the chosen Whisper `model`.

## Code map (`src/`)

Modules are grouped by pipeline role:

| Folder / file | Responsibility |
|---------------|----------------|
| **`pipeline/`** | Offline data preparation stages |
| `pipeline/audio.py` | Stage 1 — synthesise speech from transcriptions (StyleTTS2, multiprocess) |
| `pipeline/features.py` | Stage 2 — audio → frozen-Whisper encoder embeddings (multiprocess) |
| **`data/`** | Training-time data handling |
| `data/dataset.py` | Pair each neural recording with its embedding target; batch/pad (`collate`) |
| `data/preprocessing.py` | Per-session z-score normalization + train-only augmentation |
| **`model/`** | The network and its losses |
| `model/model.py` | `ConvBiGRU`: neural → embeddings (conv → Bi-GRU → cross-attention → MLP) |
| `model/losses.py` | Embedding regression loss + optional decoder-in-the-loop loss |
| **`training/`** | Optimisation and evaluation |
| `training/train.py` | The training loop, checkpointing, and periodic WER checks |
| `training/decode.py` | Load a checkpoint, decode a split with Whisper, write a per-trial CSV |
| `training/metrics.py` | Word error rate |
| **`utils/`** | Cross-cutting helpers |
| `utils/config.py` | Load `config.yaml`; hand each stage its settings; resolve `device: auto`; map a Whisper model to its `emb_dim` |
| `utils/logging_utils.py` | Per-module logging to `logs/<name>.log` and the console |

## The model (`ConvBiGRU`)

Four stages, in order:

1. **Conv front-end** — two stride-2 convolutions smooth and downsample the
   neural signal ~4× in time.
2. **Bi-GRU** — models temporal dynamics in both directions. Padding is handled
   with packed sequences so it never affects the outputs.
3. **Cross-attention** — a fixed bank of learnable "query" frames pulls out
   exactly the number of embedding frames we need.
4. **MLP head** — regresses the final embedding for each frame.

## Two training losses

* **Embedding loss** (always on) — SmoothL1 + `(1 − cosine)` between predicted
  and target embeddings, measured only over frames that contain real content.
* **Decoder-in-the-loop loss** (optional, `dec_loss_weight > 0`) — runs the
  predicted embeddings through Whisper's frozen *decoder* and measures how well
  it recovers the true transcription. This optimises decodability directly,
  which is what actually lowers WER.

## Checkpoints

Written to `checkpoints/<model>/` during training:

* `best.pt` — lowest validation loss so far
* `best_wer.pt` — lowest word error rate so far (from the periodic WER check)
* `last.pt` — the latest epoch (training auto-resumes from here)

Each checkpoint stores the full training config, so decoding rebuilds the exact
model without you having to repeat any settings.

## Requirements

Python 3 with `torch`, `h5py`, `numpy`, `openai-whisper`, `librosa`,
`styletts2`, `pyyaml`, and `tqdm`. A GPU is used automatically when available
(`device: auto`).

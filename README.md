# Brain2Text — Neural Joint Embedding Model (NJEM)

Decode text from intracortical neural activity by mapping neural signals into
the embedding space of OpenAI **Whisper** and letting Whisper's frozen decoder
turn those embeddings into words.

Instead of predicting characters/phonemes directly, the model learns to
regress the **Whisper encoder embeddings** that a spoken version of each
sentence would produce. Decoding then reuses Whisper's pretrained decoder,
so the language knowledge in Whisper is leveraged for free.

```
neural activity ──► ConvBiGRU ──► Whisper-encoder embeddings ──► [frozen Whisper decoder] ──► text
   (B, T, 512)                        (B, 1500, 384)                                             + WER
```

---

## How it works

The pipeline has four stages, orchestrated by `main.py` and driven by
`config.yaml`:

| Stage | Module | What it does |
|-------|--------|--------------|
| 1. `generate_audio` | `src/audio_generator.py` | Synthesize speech audio for each trial's transcription with **StyleTTS2**. |
| 2. `extract_features` | `src/whisper_features.py` | Run the synthesized audio through **Whisper** and cache the encoder embeddings (the regression targets). |
| 3. `train` | `src/train_neural2emb.py` | Train the `ConvBiGRU` to map neural features → Whisper encoder embeddings. |
| 4. `decode` | `src/decode.py` | Feed predicted embeddings to the frozen Whisper decoder and measure **WER**. |

Stages 1–2 produce the training targets and only need to be run once; stages
3–4 are the model training and evaluation.

### Dimensions

- **Neural input:** `(T, 512)` — 512 channels per 20 ms bin.
- **Target / prediction:** `(≤1500, 384)` — Whisper `tiny.en` encoder frames × hidden size.

### The model (`src/model.py`, `ConvBiGRU`)

1. **Conv front-end** — stride-2 conv blocks subsample the neural sequence in time.
2. **Recurrent core** — a (bi)GRU/LSTM models temporal dynamics over packed sequences.
3. **Length aligner** — cross-attention with learnable queries (or interpolation) maps the variable-length sequence to the target frame count.
4. **Regression head** — an MLP predicts the 384-dim embedding per frame.

Every component is configurable, so the architecture can be ablated (see below).

### Training objective (`src/train_neural2emb.py`)

The loss combines several terms, each individually weightable (set a weight to 0
to disable it):

- **Embedding regression** — masked SmoothL1 + `(1 − cosine similarity)` over the valid (content) frames (`l1_weight`, `cos_weight`).
- **Decoder-in-the-loop loss** (`src/decoder_loss.py`, `dec_loss_weight`) — runs predictions through the *frozen* Whisper decoder, teacher-forced on the ground-truth transcription. This directly optimizes *decodability* → WER, and itself blends:
  - **hard cross-entropy** against the ground-truth tokens, and
  - **soft KL distillation** (`dec_distill_weight`, `dec_temperature`) — the same decoder is also run on the *target* (real-audio) embeddings to get a teacher distribution, and the student is pulled towards it. This trains the model so the decoder *behaves* as it does on real audio rather than forcing exact embedding values.
- **Decoder-weight ramp** (`dec_ramp_epochs`) — the decoder-loss weight is linearly ramped 0 → `dec_loss_weight` over the first N epochs, so embedding regression warms the model up before the decoder objective takes over. (Validation always uses the full weight, so the early-stopping metric stays comparable across epochs.)

Additional training machinery:

- **Per-session normalization** — neural input is z-scored per channel using statistics computed **only from each session's training split** (no val leakage). Cached to `stats_dir`.
- **Augmentation** (train only) — Gaussian noise, channel dropout, amplitude jitter, and temporal jitter on the neural input.
- **Early stopping** — training stops if validation loss does not improve for `early_stop_patience` epochs (default 10).
- **Checkpointing** — `best.pt` (best val loss), `best_wer.pt` (best WER), and `last.pt` (latest, for resume) are written to `ckpt_dir`. Runs auto-resume from `last.pt`/`best.pt` if present.

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `openai-whisper`, `styletts2`, `h5py`, `librosa`,
`numpy`, `PyYAML`, `tqdm`. A CUDA-capable GPU is strongly recommended for
training (`device: auto` falls back to CPU).

## Data layout

The raw dataset is per-session HDF5 files (participant `t15`, one directory per
recording day):

```
<raw_dir>/<session>/data_{train,val,test}.hdf5     # neural input_features + transcriptions
<features_dir>/<split>/<session>/whisper_features_<split>.hdf5   # cached Whisper encoder embeddings
```

Paths are set in `config.yaml`. The raw data and features are treated as
**read-only inputs** (they may live in a shared location); everything the
project writes — the normalization cache (`stats_dir`), checkpoints, and
results — stays local.

---

## Quickstart

Run the full pipeline, or any single stage:

```bash
python main.py workflow            # run the stages listed under workflow.steps
python main.py generate_audio      # stage 1
python main.py extract_features    # stage 2
python main.py train               # stage 3
python main.py decode              # stage 4
```

Override any config value on the command line (a flag exists for every key):

```bash
python main.py train --epochs 200 --batch_size 64 --device cuda
python main.py --config my_config.yaml train
python main.py decode --ckpt checkpoints/best_wer.pt --split val --beam_size 5
```

---

## Configuration (`config.yaml`)

Grouped by stage. Notable `train` keys:

| Key | Meaning |
|-----|---------|
| `raw_dir`, `features_dir` | read-only data inputs |
| `stats_dir` | local dir for the per-session normalization cache |
| `epochs`, `batch_size`, `lr`, `weight_decay` | optimization |
| `early_stop_patience` | stop after N epochs without val-loss improvement (0 = off) |
| `conv_channels`, `conv_layers`, `conv_kernel` | conv front-end |
| `rnn_type` (`gru`/`lstm`/`none`), `hidden`, `rnn_layers`, `bidirectional` | recurrent core |
| `aligner` (`attn`/`interp`), `attn_heads`, `head_layers` | length aligner + head |
| `dropout`, `normalize`, `augment`, `aug_*` | regularization / augmentation |
| `l1_weight`, `cos_weight`, `dec_loss_weight` | loss-term weights |
| `dec_distill_weight`, `dec_temperature` | soft-KL vs hard-CE blend + temperature inside the decoder loss |
| `dec_ramp_epochs` | ramp the decoder-loss weight up over the first N epochs |
| `dec_model` | frozen Whisper model for the decoder loss / decoding (`tiny.en`) |
| `wer_every`, `wer_trials`, `beam_size` | in-training WER evaluation |
| `wer_out_dir` | where per-epoch WER CSVs are written |
| `seed`, `device` | reproducibility / device |

---

## Outputs

Training and decoding write:

- **Checkpoints** → `ckpt_dir/` (`best.pt`, `best_wer.pt`, `last.pt`).
- **Per-epoch WER CSVs** (during training, every `wer_every` epochs) → `wer_out_dir/`:
  - `val_predictions_epoch*.csv` — `session, trial, actual, predicted, wer`
  - `val_wer_per_session_epoch*.csv` — `session, wer, n_trials`
- **Decode CSVs** → `decode.out` and `<out>_per_session.csv`.
- **Logs** → `logs/<name>.log` (redirectable via `B2T_LOG_DIR` / `B2T_LOG_FILE`).

---

## Ablation studies

`tools/run_ablations.py` launches a grid of training runs that each change **one
thing** from the `config.yaml` baseline, so effects are attributable. Results,
checkpoints, and logs for each run are isolated under `results/ablations/`:

```
results/ablations/
  logs/<run>.log
  <run>/checkpoints/{best,best_wer,last}.pt
  <run>/wer/val_predictions_epoch*.csv
  <run>/wer/val_wer_per_session_epoch*.csv
```

The grid spans these axes (`--group`): `augment`, `temporal`, `aligner`,
`conv`, `capacity`, `reg`, `loss`, `optim`, `combo`, plus the `baseline`.

```bash
# print the full "changed from baseline" config table without running
python tools/run_ablations.py --dry-run

# run one axis, or specific runs
python tools/run_ablations.py --group loss
python tools/run_ablations.py --only baseline no_aug rnn_lstm

# quick CPU smoke test (tiny; proves the grid runs)
python tools/run_ablations.py --limit 8 --batch_size 4 --epochs 1 \
    --wer_every 1 --wer_trials 8 --device cpu
```

Common flags: `--python` (interpreter to use), `--epochs`, `--batch_size`,
`--num_workers`, `--limit`, `--device`, `--wer_every`, `--wer_trials`, `--seed`.

---

## Project layout

```
main.py                     # unified CLI entry point (stages + workflow)
config.yaml                 # all stage configuration
src/
  audio_generator.py        # stage 1: StyleTTS2 audio synthesis
  whisper_features.py       # stage 2: Whisper encoder-embedding extraction
  dataset.py                # neural↔embedding pairing, normalization, augmentation
  model.py                  # ConvBiGRU + build_model + losses
  decoder_loss.py           # decoder-in-the-loop CE loss
  train_neural2emb.py       # training loop, WER eval, early stopping
  decode.py                 # beam-search decoding + WER reports
  utils.py                  # config loading + logging
tools/
  run_ablations.py          # ablation-study runner
checkpoints/                # model checkpoints (gitignored)
results/                    # predictions, WER CSVs, ablation outputs (gitignored)
logs/                       # run logs (gitignored)
```

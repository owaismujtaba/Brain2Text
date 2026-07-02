import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_CTX = 1500
EMB_DIM = 384
NEURAL_DIM = 512


class ConvBiGRU(nn.Module):
    """Maps neural activity (B, T, 512) to Whisper encoder embeddings (B, N, 384).

    The architecture is intentionally modular so its parts can be ablated:

      * conv front-end   -- `conv_layers` stride-2 blocks subsample time by 2^L
      * temporal model   -- `rnn_type` in {"gru", "lstm", "none"}, uni/bidirectional
      * length aligner   -- `aligner` in {"attn", "interp"} maps the variable-length
                            temporal features to the target frame count
      * regression head  -- `head_layers` MLP that regresses the 384-dim embedding

    Every knob is a constructor argument; use `build_model(cfg)` to construct
    from a config dict / argparse Namespace so checkpoints round-trip exactly.
    """

    def __init__(self, in_dim=NEURAL_DIM, conv_channels=256, conv_layers=2,
                 conv_kernel=5, rnn_type="gru", hidden=256, rnn_layers=2,
                 bidirectional=True, aligner="attn", attn_heads=4, head_layers=2,
                 emb_dim=EMB_DIM, n_ctx=N_CTX, dropout=0.3):
        super().__init__()
        self.n_ctx = n_ctx
        self.conv_layers = conv_layers
        self.rnn_type = rnn_type.lower()
        self.aligner = aligner.lower()

        # ── conv front-end: `conv_layers` stride-2 downsampling blocks ──
        pad = conv_kernel // 2
        conv = []
        c_in = in_dim
        for _ in range(conv_layers):
            conv += [
                nn.Conv1d(c_in, conv_channels, kernel_size=conv_kernel, stride=2, padding=pad),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            c_in = conv_channels
        self.conv = nn.Sequential(*conv)

        # ── temporal model ──
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(conv_channels, hidden, num_layers=rnn_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout if rnn_layers > 1 else 0.0)
            feat_dim = hidden * (2 if bidirectional else 1)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(conv_channels, hidden, num_layers=rnn_layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=dropout if rnn_layers > 1 else 0.0)
            feat_dim = hidden * (2 if bidirectional else 1)
        elif self.rnn_type == "none":
            self.rnn = None
            feat_dim = conv_channels
        else:
            raise ValueError(f"unknown rnn_type '{rnn_type}' (use gru/lstm/none)")
        self.feat_dim = feat_dim

        # ── length aligner ──
        if self.aligner == "attn":
            self.query = nn.Parameter(torch.randn(1, n_ctx, feat_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                feat_dim, num_heads=attn_heads, dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(feat_dim)
        elif self.aligner == "interp":
            pass  # parameter-free; interpolation happens in forward
        else:
            raise ValueError(f"unknown aligner '{aligner}' (use attn/interp)")

        # ── regression head ──
        layers = []
        for _ in range(max(head_layers - 1, 0)):
            layers += [nn.Linear(feat_dim, feat_dim), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(feat_dim, emb_dim)]
        self.head = nn.Sequential(*layers)

    def _conv_out_length(self, length):
        """Sequence length after the stride-2 conv layers."""
        for _ in range(self.conv_layers):
            length = (length - 1) // 2 + 1
        return length

    def _interp_align(self, h, out_size, lengths, out_lengths):
        """Length-aware linear resample of the temporal features to out_size.

        Each sample's valid conv frames are interpolated to its own output
        length (from out_lengths, else out_size); trailing frames stay zero and
        are masked out by the loss, matching the attention path.
        """
        B, T, C = h.shape
        if lengths is not None:
            cl = self._conv_out_length(lengths).clamp(min=1)
        else:
            cl = torch.full((B,), T, dtype=torch.long)
        if out_lengths is not None:
            ol = out_lengths
        else:
            ol = torch.full((B,), out_size, dtype=torch.long)

        result = h.new_zeros(B, out_size, C)
        for i in range(B):
            ci = int(cl[i])
            oi = min(int(ol[i]), out_size)
            if oi <= 0:
                continue
            seg = h[i, :ci].transpose(0, 1).unsqueeze(0)          # (1, C, ci)
            seg = F.interpolate(seg, size=oi, mode="linear", align_corners=False)
            result[i, :oi] = seg.squeeze(0).transpose(0, 1)
        return result

    def forward(self, x, lengths=None, n_out=None, out_lengths=None):
        # x: (B, T, 512)
        B = x.size(0)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T', C)

        packed = lengths is not None and self.rnn is not None
        if packed:
            conv_lengths = self._conv_out_length(lengths).clamp(min=1).cpu()
            h = pack_padded_sequence(h, conv_lengths, batch_first=True,
                                     enforce_sorted=False)

        if self.rnn is not None:
            h, _ = self.rnn(h)

        if packed:
            h, _ = pad_packed_sequence(h, batch_first=True)

        out_size = n_out if n_out is not None else self.n_ctx

        if self.aligner == "attn":
            queries = self.query[:, :out_size].expand(B, -1, -1)
            key_mask = None
            if lengths is not None:
                max_klen = h.size(1)
                cl = self._conv_out_length(lengths).clamp(min=1).to(h.device)
                key_mask = torch.arange(max_klen, device=h.device).unsqueeze(0) >= cl.unsqueeze(1)
            h, _ = self.cross_attn(queries, h, h, key_padding_mask=key_mask)
            h = self.attn_norm(h)
        else:  # interp
            h = self._interp_align(h, out_size, lengths, out_lengths)

        return self.head(h)


def build_model(cfg):
    """Construct a ConvBiGRU from a config dict or argparse Namespace.

    Reads every architecture knob with a default so old checkpoints (which only
    stored a subset of keys) still load. `gru_layers` is accepted as an alias for
    `rnn_layers` for backward compatibility.
    """
    def get(key, default):
        if isinstance(cfg, dict):
            val = cfg.get(key, default)
        else:
            val = getattr(cfg, key, default)
        return default if val is None else val

    return ConvBiGRU(
        in_dim=get("neural_dim", NEURAL_DIM),
        conv_channels=get("conv_channels", 256),
        conv_layers=get("conv_layers", 2),
        conv_kernel=get("conv_kernel", 5),
        rnn_type=get("rnn_type", "gru"),
        hidden=get("hidden", 256),
        rnn_layers=get("rnn_layers", get("gru_layers", 2)),
        bidirectional=bool(get("bidirectional", True)),
        aligner=get("aligner", "attn"),
        attn_heads=get("attn_heads", 4),
        head_layers=get("head_layers", 2),
        emb_dim=get("emb_dim", EMB_DIM),
        n_ctx=get("n_ctx", N_CTX),
        dropout=get("dropout", 0.3),
    )


def load_checkpoint_weights(model, state):
    """Load a state_dict, remapping legacy ``gru.*`` keys to ``rnn.*``.

    Checkpoints trained before the RNN module was renamed store parameters under
    ``gru.*``; the model now names it ``rnn`` (it may be a GRU or LSTM). This
    remaps those keys so old checkpoints still load.
    """
    if any(k.startswith("gru.") for k in state) and not any(k.startswith("rnn.") for k in state):
        state = {("rnn." + k[len("gru."):] if k.startswith("gru.") else k): v
                 for k, v in state.items()}
    model.load_state_dict(state)
    return model


def masked_embedding_loss(pred, target, mask, l1_weight=1.0, cos_weight=1.0):
    """SmoothL1 + (1 - cosine) computed only over valid (content) frames.

    pred, target: (B, 1500, 384); mask: (B, 1500) bool. Set l1_weight or
    cos_weight to 0 to ablate that term.
    """
    m = mask.unsqueeze(-1).float()
    denom = m.sum().clamp(min=1.0)

    l1 = (F.smooth_l1_loss(pred, target, reduction="none") * m).sum() / (denom * pred.shape[-1])

    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1)  # (B, 1500)
    fmask = mask.float()
    cos = (cos * fmask).sum() / fmask.sum().clamp(min=1.0)

    loss = l1_weight * l1 + cos_weight * cos
    return loss, l1.detach(), cos.detach()

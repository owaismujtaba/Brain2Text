import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

N_CTX = 1500
EMB_DIM = 384
NEURAL_DIM = 512


class ConvBiGRU(nn.Module):
    """Maps neural activity (B, T, 512) to Whisper encoder embeddings (B, N, 384).

    Conv front-end smooths/subsamples the neural sequence, a bidirectional GRU
    (with packed sequences) models temporal dynamics, cross-attention aligns
    the GRU output to the target frame count via learnable queries, and an MLP
    head regresses the 384-dim embedding per frame.
    """

    def __init__(self, in_dim=NEURAL_DIM, conv_channels=256, hidden=256,
                 gru_layers=2, emb_dim=EMB_DIM, n_ctx=N_CTX, dropout=0.3,
                 attn_heads=4):
        super().__init__()
        self.n_ctx = n_ctx
        gru_out = 2 * hidden

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )  # downsamples time by ~4x

        self.gru = nn.GRU(
            conv_channels, hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.query = nn.Parameter(torch.randn(1, n_ctx, gru_out) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            gru_out, num_heads=attn_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(gru_out)

        self.head = nn.Sequential(
            nn.Linear(gru_out, gru_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out, emb_dim),
        )

    def _conv_out_length(self, length):
        """Compute sequence length after the two stride-2 conv layers."""
        for _ in range(2):
            length = (length - 1) // 2 + 1
        return length

    def forward(self, x, lengths=None, n_out=None):
        # x: (B, T, 512)
        B = x.size(0)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T', C)

        if lengths is not None:
            conv_lengths = self._conv_out_length(lengths).clamp(min=1).cpu()
            h = pack_padded_sequence(h, conv_lengths, batch_first=True,
                                     enforce_sorted=False)

        h, _ = self.gru(h)

        if lengths is not None:
            h, _ = pad_packed_sequence(h, batch_first=True)

        out_size = n_out if n_out is not None else self.n_ctx
        queries = self.query[:, :out_size].expand(B, -1, -1)

        # Build key-padding mask so attention ignores GRU padding frames
        key_mask = None
        if lengths is not None:
            max_klen = h.size(1)
            cl = self._conv_out_length(lengths).clamp(min=1).to(h.device)
            key_mask = torch.arange(max_klen, device=h.device).unsqueeze(0) >= cl.unsqueeze(1)

        h, _ = self.cross_attn(queries, h, h, key_padding_mask=key_mask)
        h = self.attn_norm(h)
        return self.head(h)


def masked_embedding_loss(pred, target, mask, l1_weight=1.0, cos_weight=1.0):
    """SmoothL1 + (1 - cosine) computed only over valid (content) frames.

    pred, target: (B, 1500, 384); mask: (B, 1500) bool.
    """
    m = mask.unsqueeze(-1).float()
    denom = m.sum().clamp(min=1.0)

    l1 = (F.smooth_l1_loss(pred, target, reduction="none") * m).sum() / (denom * pred.shape[-1])

    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1)  # (B, 1500)
    fmask = mask.float()
    cos = (cos * fmask).sum() / fmask.sum().clamp(min=1.0)

    loss = l1_weight * l1 + cos_weight * cos
    return loss, l1.detach(), cos.detach()

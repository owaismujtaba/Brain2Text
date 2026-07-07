"""Training losses.

* ``embedding_loss``     – regression loss between the predicted and target
  Whisper embeddings (SmoothL1 + cosine), measured over content frames only.
* ``WhisperDecoderLoss`` – runs the predicted embeddings through the FROZEN
  Whisper decoder and measures how well the true transcription is predicted.
  This optimises *decodability* directly, which is what actually lowers WER.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_loss(pred, target, mask, l1_weight=1.0, cos_weight=1.0):
    """SmoothL1 + ``(1 - cosine)`` over valid frames only.

    Args:
        pred, target : (B, T, emb_dim)
        mask         : (B, T) bool – True for real (content) frames
    Returns:
        (total_loss, l1_detached, cos_detached)
    """
    frame_mask = mask.unsqueeze(-1).float()             # (B, T, 1)

    l1 = (F.smooth_l1_loss(pred, target, reduction="none") * frame_mask).sum()
    l1 = l1 / (frame_mask.sum().clamp(min=1.0) * pred.shape[-1])

    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1)   # (B, T)
    cos = (cos * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    total = l1_weight * l1 + cos_weight * cos
    return total, l1.detach(), cos.detach()


class WhisperDecoderLoss(nn.Module):
    """Cross-entropy of the true transcription under the frozen Whisper decoder."""

    def __init__(self, model_name="tiny.en", device="cpu", n_ctx=1500):
        super().__init__()
        import whisper
        self.whisper = whisper.load_model(model_name, device=device)
        self.whisper.eval()
        for p in self.whisper.parameters():
            p.requires_grad_(False)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.whisper.is_multilingual, language="en", task="transcribe")
        self.sot = list(self.tokenizer.sot_sequence_including_notimestamps)
        self.eot = self.tokenizer.eot
        self.device = device
        self.n_ctx = int(n_ctx)

    def _teacher_forcing_tokens(self, texts):
        """Build ``(decoder_input, shifted_target)``; -100 marks ignored positions."""
        seqs = [self.sot + self.tokenizer.encode(" " + t.strip()) + [self.eot]
                for t in texts]
        length = max(len(s) for s in seqs)
        n_prompt = len(self.sot)

        inp = torch.full((len(seqs), length), self.eot, dtype=torch.long)
        tgt = torch.full((len(seqs), length), -100, dtype=torch.long)
        for i, seq in enumerate(seqs):
            seq = torch.tensor(seq, dtype=torch.long)
            inp[i, : len(seq)] = seq
            tgt[i, : len(seq) - 1] = seq[1:]      # predict the next token
        tgt[:, : n_prompt - 1] = -100             # ignore positions inside the prompt
        return inp.to(self.device), tgt.to(self.device)

    def forward(self, pred, mask, texts):
        """Args: ``pred`` (B, T, emb_dim), ``mask`` (B, T) bool, ``texts`` list[str]."""
        # Zero the padding frames, then pad up to the full n_ctx frame layout the
        # decoder sees at inference time. Whisper's decoder has no attention mask
        # over the audio axis, so the frame count must match between training and
        # decoding, otherwise trailing zeros would shift the attention softmax.
        feats = (pred * mask.unsqueeze(-1).to(pred.dtype)).float()
        if feats.size(1) < self.n_ctx:
            feats = F.pad(feats, (0, 0, 0, self.n_ctx - feats.size(1)))

        inp, tgt = self._teacher_forcing_tokens(texts)
        logits = self.whisper.decoder(inp, feats)             # (B, L, vocab)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)

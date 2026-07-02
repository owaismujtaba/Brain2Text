"""Decoder-in-the-loop loss.

Runs predicted embeddings through the FROZEN Whisper decoder (teacher-forced
with the ground-truth transcription) and returns cross-entropy over the target
tokens. This optimises *decodability* directly, closing the gap between a low
embedding-regression loss and a low WER.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CTX = 1500   # Whisper encoder frame count (must match decode-time feature length)


class WhisperDecoderLoss(nn.Module):
    def __init__(self, model_name="tiny.en", device="cpu"):
        super().__init__()
        import whisper
        self.model = whisper.load_model(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual, language="en", task="transcribe")
        self.sot = list(self.tok.sot_sequence_including_notimestamps)
        self.eot = self.tok.eot
        self.device = device

    def _build_tokens(self, texts):
        """Teacher-forcing tensors: decoder input + shifted target (-100 = ignore)."""
        seqs = [self.sot + self.tok.encode(" " + t.strip()) + [self.eot] for t in texts]
        L = max(len(s) for s in seqs)
        B = len(seqs)
        n_prompt = len(self.sot)
        inp = torch.full((B, L), self.eot, dtype=torch.long)
        tgt = torch.full((B, L), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            st = torch.tensor(s, dtype=torch.long)
            inp[i, : len(s)] = st
            tgt[i, : len(s) - 1] = st[1:]        # predict next token
        tgt[:, : n_prompt - 1] = -100            # ignore predictions inside the prompt
        return inp.to(self.device), tgt.to(self.device)

    def forward(self, pred, mask_vf, texts):
        """
        pred:    (B, T, 384) predicted encoder embeddings (T == max valid frames in batch)
        mask_vf: (B, T) bool  -- True for valid content frames
        texts:   list[str] ground-truth transcriptions
        """
        # zero out invalid frames so the decoder ignores padding (matches eval-time masking)
        feats = (pred * mask_vf.unsqueeze(-1).to(pred.dtype)).float()
        # Pad to the full N_CTX=1500 frame layout used at decode time. Whisper's
        # decoder has no cross-attention mask over the audio axis, so it must see
        # the same number of frames in training and inference (else the trailing
        # zero frames shift the attention softmax at decode time only).
        T = feats.size(1)
        if T < N_CTX:
            feats = F.pad(feats, (0, 0, 0, N_CTX - T))
        inp, tgt = self._build_tokens(texts)
        logits = self.model.decoder(inp, feats)          # (B, L, vocab)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)

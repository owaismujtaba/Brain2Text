"""Decoder-in-the-loop loss (hard cross-entropy + optional soft distillation).

Runs embeddings through the FROZEN Whisper decoder (teacher-forced with the
ground-truth transcription) to optimise *decodability* directly, closing the gap
between a low embedding-regression loss and a low WER.

Two signals are combined:

* **hard CE** -- cross-entropy of the decoder's tokens (fed the *predicted*
  embeddings) against the ground-truth transcription.
* **soft distillation (KL)** -- the same decoder is also run on the *target*
  (real-audio Whisper) embeddings to get a teacher distribution; the student
  (predicted-embedding) distribution is pulled towards it with a temperature.
  This trains the model so the decoder *behaves* as it does on real audio,
  rather than forcing exact embedding values, and transfers the teacher's
  soft/uncertain targets. Blended as (1-w)*CE + w*KL via ``distill_weight``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CTX = 1500   # Whisper encoder frame count (must match decode-time feature length)


class WhisperDecoderLoss(nn.Module):
    def __init__(self, model_name="tiny.en", device="cpu",
                 distill_weight=0.0, temperature=2.0):
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
        self.distill_weight = float(distill_weight)   # 0 = pure CE, 1 = pure KL
        self.temperature = float(temperature)

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

    def _prep_feats(self, emb, mask_vf):
        """Mask padding frames and pad to the full N_CTX layout used at decode time.

        Whisper's decoder has no cross-attention mask over the audio axis, so it
        must see the same number of frames in training and inference (else the
        trailing zero frames shift the attention softmax at decode time only).
        """
        feats = (emb * mask_vf.unsqueeze(-1).to(emb.dtype)).float()
        T = feats.size(1)
        if T < N_CTX:
            feats = F.pad(feats, (0, 0, 0, N_CTX - T))
        return feats

    def forward(self, pred, mask_vf, texts, target=None):
        """
        pred:    (B, T, 384) predicted encoder embeddings (T == max valid frames)
        mask_vf: (B, T) bool  -- True for valid content frames
        texts:   list[str] ground-truth transcriptions
        target:  (B, T, 384) real-audio Whisper embeddings (teacher). Required
                 for distillation; if None, only the hard CE is returned.
        """
        inp, tgt = self._build_tokens(texts)
        student_logits = self.model.decoder(inp, self._prep_feats(pred, mask_vf))  # (B, L, V)
        V = student_logits.size(-1)
        ce = F.cross_entropy(student_logits.reshape(-1, V), tgt.reshape(-1),
                             ignore_index=-100)

        if self.distill_weight <= 0.0 or target is None:
            return ce

        with torch.no_grad():
            teacher_logits = self.model.decoder(inp, self._prep_feats(target, mask_vf))

        valid = (tgt != -100).reshape(-1)                      # only real target tokens
        tau = self.temperature
        s_logp = F.log_softmax(student_logits.reshape(-1, V)[valid] / tau, dim=-1)
        t_prob = F.softmax(teacher_logits.reshape(-1, V)[valid] / tau, dim=-1)
        # KL(teacher || student), scaled by tau^2 so its gradient magnitude
        # matches the hard-CE term (standard distillation scaling).
        kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (tau * tau)

        w = self.distill_weight
        return (1.0 - w) * ce + w * kl

import torch, numpy as np
from src.dataset import NeuralToEmbeddingDataset, N_CTX, EMB_DIM
from src.model import ConvBiGRU

device = "cpu"
ck = torch.load("/tmp/diag_best.pt", map_location=device, weights_only=False)
a = ck["args"]
print("ckpt epoch:", ck.get("epoch"), "| gru_layers:", a.get("gru_layers"), "| best_val:", ck.get("best_val"))

model = ConvBiGRU(conv_channels=a.get("conv_channels",256), hidden=a["hidden"],
                  gru_layers=a["gru_layers"], dropout=a["dropout"]).to(device)
model.load_state_dict(ck["model"]); model.eval()

ds = NeuralToEmbeddingDataset(a["raw_dir"], a["features_dir"], "val")
N = 12
preds, targs = [], []
with torch.no_grad():
    for i in range(N):
        neural, emb, txt = ds[i]
        vf = emb.shape[0]
        p = model(neural.unsqueeze(0))[0][:vf]      # (vf,384)
        preds.append(p); targs.append(emb)

# 1) per-frame cosine sim pred vs its own target
def fcos(p, t):
    return torch.cosine_similarity(p, t, dim=-1).mean().item()
own = np.mean([fcos(p, t) for p, t in zip(preds, targs)])

# 2) norm scale
pn = np.mean([p.norm(dim=-1).mean().item() for p in preds])
tn = np.mean([t.norm(dim=-1).mean().item() for t in targs])

# 3) COLLAPSE test: how similar are predictions ACROSS different trials?
#    compare each trial's mean-pooled pred embedding to every other trial's.
pooled_p = torch.stack([p.mean(0) for p in preds])   # (N,384)
pooled_t = torch.stack([t.mean(0) for t in targs])
def cross_sim(x):
    xn = torch.nn.functional.normalize(x, dim=-1)
    S = xn @ xn.T
    off = S[~torch.eye(len(x), dtype=bool)]
    return off.mean().item(), off.std().item()
cp_m, cp_s = cross_sim(pooled_p)
ct_m, ct_s = cross_sim(pooled_t)

# 4) constant-mean baseline: cosine of each target vs the AVERAGE target
mean_t = torch.cat(targs, 0).mean(0)
base = np.mean([torch.cosine_similarity(t, mean_t.expand_as(t), dim=-1).mean().item()
                for t in targs])

print(f"\nper-frame cos(pred, own target)      : {own:.3f}   (decoder needs ~0.95+)")
print(f"embedding norm  pred={pn:.2f}  target={tn:.2f}  (ratio {pn/tn:.2f})")
print(f"\nCOLLAPSE check (cross-trial similarity of pooled embeddings):")
print(f"  predictions : {cp_m:.3f} +/- {cp_s:.3f}")
print(f"  targets     : {ct_m:.3f} +/- {ct_s:.3f}")
print(f"  -> if predictions are MUCH more self-similar than targets, model is collapsing")
print(f"\nconstant mean-embedding baseline cos  : {base:.3f}")
print(f"model beats mean baseline by          : {own-base:+.3f}")

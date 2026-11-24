"""Real-to-sim sensor model, Part 4: the per-pixel network and its training. Data and geometry are in real2sim.py.

For every CAD pixel the network reads the inputs (from the CAD and camera pose only) and outputs
  p      probability that the sensor drops the pixel                    (trained with BCE against y_drop)
  mu     typical depth error Z_real - Z_CAD (mm)                         (trained with Gaussian NLL against e_z,
  sigma  spread of the depth error around mu (mm)                         kept pixels only)
The network outputs log sigma^2 instead of sigma, which keeps training stable.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN, LOGVAR_MAX = -6.0, 6.0      # sigma between 0.05 and 20 mm


class SensorNet(nn.Module):
    """Small MLP applied to each pixel on its own. Inputs are standardised inside the model (x_mean, x_std are saved
    with it), so callers pass raw input values."""

    def __init__(self, n_in, hidden=(128, 128, 128)):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 3)
        self.hidden = tuple(hidden)
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))

    def forward(self, x):
        out = self.head(self.body((x - self.x_mean) / self.x_std))
        return out[:, 0], out[:, 1], out[:, 2].clamp(LOGVAR_MIN, LOGVAR_MAX)   # logit p, mu, log sigma^2


def losses(out, y_drop, e_z):
    """(BCE over all pixels, Gaussian NLL over kept pixels), both averaged per pixel."""
    logit, mu, logvar = out
    bce = F.binary_cross_entropy_with_logits(logit, y_drop)
    kept = ~torch.isnan(e_z)
    if not kept.any():
        return bce, torch.zeros((), device=bce.device)
    lv, d = logvar[kept], e_z[kept] - mu[kept]
    nll = 0.5 * (lv + d * d / lv.exp() + math.log(2 * math.pi)).mean()
    return bce, nll


def split_views(df, test_workpiece, val_fraction=0.3, seed=42):
    """The agreed split (2026-11-10): the test workpiece is never used for training; of every other workpiece a
    random val_fraction of its VIEWS (all pixels of a view stay together) is validation, the rest is training.
    Returns dict of boolean masks over df rows: train, val, test."""
    rng = np.random.default_rng(seed)
    is_test = (df.workpiece == test_workpiece).to_numpy()
    is_val = np.zeros(len(df), bool)
    for wp in sorted(df.workpiece[~is_test].unique()):
        views = sorted(df.view[df.workpiece == wp].unique())
        val_views = rng.choice(views, int(round(val_fraction * len(views))), replace=False)
        is_val |= ((df.workpiece == wp) & df.view.isin(val_views)).to_numpy()
    return dict(train=~is_test & ~is_val, val=is_val, test=is_test)


def to_tensors(df, inputs, device):
    """DataFrame rows -> (X, y_drop, e_z) float32 tensors on device; e_z is NaN on dropped pixels."""
    X = torch.as_tensor(df[inputs].to_numpy(np.float32), device=device)
    y = torch.as_tensor(df["y_drop"].to_numpy(np.float32), device=device)
    e = torch.as_tensor(df["e_z"].to_numpy(np.float32), device=device)
    return X, y, e


@torch.no_grad()
def evaluate(model, data, batch=1 << 18):
    """Mean BCE and NLL over a whole split (pixel-weighted, like the training loss)."""
    model.eval()
    X, y, e = data
    s_bce = s_nll = 0.0
    n_kept = 0
    for i in range(0, len(X), batch):
        out = model(X[i:i + batch])
        bce, nll = losses(out, y[i:i + batch], e[i:i + batch])
        k = int((~torch.isnan(e[i:i + batch])).sum())
        s_bce += float(bce) * len(out[0])
        s_nll += float(nll) * k
        n_kept += k
    return s_bce / len(X), s_nll / max(n_kept, 1)


def train(model, train_data, val_data, max_epochs=300, patience=15, lr=1e-3, lr_patience=5, lr_factor=0.5,
          min_delta=1e-4, batch=16384, seed=42, log=print, log_every=5):
    """Adam, trained until the validation loss (BCE + NLL) stops improving.

    - An epoch counts as an improvement if the validation loss drops by more than min_delta below the best so far.
    - After lr_patience epochs without improvement the learning rate is multiplied by lr_factor (ReduceLROnPlateau).
    - After patience epochs without improvement training stops (early stopping), or at max_epochs.
    The weights of the best epoch are restored. The train loss in the history is the mean over the epoch's batches.
    Returns the history as a list of dicts (epoch, train_bce, train_nll, val_bce, val_nll, lr)."""
    X, y, e = train_data
    model.x_mean.copy_(X.mean(0))
    model.x_std.copy_(X.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=lr_factor, patience=lr_patience,
                                                       threshold=min_delta, threshold_mode="abs")
    gen = torch.Generator(device=X.device).manual_seed(seed)
    history, best, best_epoch, best_state = [], math.inf, 0, None
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X), device=X.device, generator=gen)
        s_bce = s_nll = 0.0
        n_batches = 0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            bce, nll = losses(model(X[idx]), y[idx], e[idx])
            opt.zero_grad(set_to_none=True)
            (bce + nll).backward()
            opt.step()
            s_bce += float(bce.detach())
            s_nll += float(nll.detach())
            n_batches += 1
        va_bce, va_nll = evaluate(model, val_data)
        current_lr = opt.param_groups[0]["lr"]
        history.append(dict(epoch=ep, train_bce=s_bce / n_batches, train_nll=s_nll / n_batches,
                            val_bce=va_bce, val_nll=va_nll, lr=current_lr))
        improved = va_bce + va_nll < best - min_delta
        if improved:
            best, best_epoch = va_bce + va_nll, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sched.step(va_bce + va_nll)
        if log and (ep == 1 or ep % log_every == 0 or ep - best_epoch >= patience):
            log(f"epoch {ep:3d}  train BCE {s_bce / n_batches:.4f} NLL {s_nll / n_batches:.4f} | "
                f"val BCE {va_bce:.4f} NLL {va_nll:.4f} | lr {current_lr:.1e} | best epoch {best_epoch}")
        if ep - best_epoch >= patience:
            if log:
                log(f"stopped: no improvement for {patience} epochs; best epoch {best_epoch} restored")
            break
    else:
        if log:
            log(f"reached max_epochs = {max_epochs}; best epoch {best_epoch} restored")
    model.load_state_dict(best_state)
    return history


def train_fixed(model, train_data, lr_schedule, batch=16384, seed=42, log=print):
    """Trains for exactly len(lr_schedule) epochs with the given learning rate per epoch, without validation, and
    keeps the final weights. Used to retrain on all training workpieces once a run with early stopping (train) has
    found how long to train: pass that run's history lr values up to its best epoch."""
    X, y, e = train_data
    model.x_mean.copy_(X.mean(0))
    model.x_std.copy_(X.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=lr_schedule[0])
    gen = torch.Generator(device=X.device).manual_seed(seed)
    history = []
    for ep, lr in enumerate(lr_schedule, start=1):
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        perm = torch.randperm(len(X), device=X.device, generator=gen)
        s_bce = s_nll = 0.0
        n_batches = 0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            bce, nll = losses(model(X[idx]), y[idx], e[idx])
            opt.zero_grad(set_to_none=True)
            (bce + nll).backward()
            opt.step()
            s_bce += float(bce.detach())
            s_nll += float(nll.detach())
            n_batches += 1
        history.append(dict(epoch=ep, train_bce=s_bce / n_batches, train_nll=s_nll / n_batches, lr=lr))
        if log:
            log(f"epoch {ep:3d}  train BCE {s_bce / n_batches:.4f} NLL {s_nll / n_batches:.4f} | lr {lr:.1e}")
    return history


@torch.no_grad()
def predict(model, X, batch=1 << 18):
    """Raw inputs (N, n_in) numpy or tensor -> (p, mu, sigma) numpy arrays."""
    model.eval()
    dev = next(model.parameters()).device
    X = torch.as_tensor(np.asarray(X, np.float32) if not torch.is_tensor(X) else X, device=dev)
    p, mu, sd = [], [], []
    for i in range(0, len(X), batch):
        logit, m, lv = model(X[i:i + batch])
        p.append(torch.sigmoid(logit).cpu())
        mu.append(m.cpu())
        sd.append((0.5 * lv).exp().cpu())
    return torch.cat(p).numpy(), torch.cat(mu).numpy(), torch.cat(sd).numpy()


def scores(y_drop, e_z, p, mu, sigma):
    """Test scores of one prediction (numpy arrays over the same pixels; e_z NaN on dropped pixels).
      dropout_bce    BCE of p against y_drop (lower is better)
      ez_mae         mean |e_z - mu| on kept pixels (mm)
      within_1sigma  fraction of kept pixels with |e_z - mu| < sigma (ideal 0.683)
      within_2sigma  same with 2 sigma (ideal 0.954)
      pred_drop, real_drop  mean predicted p and real dropout rate"""
    y = np.asarray(y_drop, float)
    pc = np.clip(p, 1e-3, 1 - 1e-3)
    kept = ~np.isnan(e_z)
    z = np.abs(e_z[kept] - mu[kept]) / sigma[kept]
    return dict(dropout_bce=float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
                ez_mae=float(np.abs(e_z[kept] - mu[kept]).mean()),
                within_1sigma=float(np.mean(z < 1)), within_2sigma=float(np.mean(z < 2)),
                pred_drop=float(p.mean()), real_drop=float(y.mean()))


def save_model(path, model, inputs, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), inputs=list(inputs), hidden=model.hidden, meta=meta or {}), path)
    return path


def load_model(path, device="cpu"):
    """-> (model, input names, meta)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SensorNet(len(ck["inputs"]), ck["hidden"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["inputs"], ck["meta"]

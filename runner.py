import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from conv_model import DigitConv
from synth_digits import make_dataset


def _shuffle_split(X, y, share):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    cut = int((1.0 - share) * len(X))
    left, right = order[:cut], order[cut:]
    return (
        TensorDataset(torch.from_numpy(X[left]), torch.from_numpy(y[left])),
        TensorDataset(torch.from_numpy(X[right]), torch.from_numpy(y[right])),
    )


def _loop(net, loader, criterion, device, opt=None):
    net.train(opt is not None)
    seen = 0
    hits = 0
    loss_acc = 0.0
    scope = torch.enable_grad() if opt is not None else torch.no_grad()
    with scope:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if opt is not None:
                opt.zero_grad()
            logits = net(xb)
            batch_loss = criterion(logits, yb)
            if opt is not None:
                batch_loss.backward()
                opt.step()
            loss_acc += batch_loss.item() * xb.size(0)
            hits += (logits.argmax(1) == yb).sum().item()
            seen += xb.size(0)
    return loss_acc / seen, hits / seen


def train(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    X, y = make_dataset(config.SAMPLES_PER_DIGIT)
    print("X:", X.shape, "y:", y.shape)

    train_ds, val_ds = _shuffle_split(X, y, config.VALIDATION_SHARE)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    net = DigitConv(config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

    best = 0.0
    no_improve = 0

    for ep in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = _loop(net, train_dl, criterion, device, opt)
        sched.step()
        _, va_acc = _loop(net, val_dl, criterion, device)

        print(
            f"ep {ep:02d}/{config.EPOCHS}  loss={tr_loss:.4f}  "
            f"train={tr_acc * 100:.2f}%  val={va_acc * 100:.2f}%"
        )

        if va_acc > best:
            best, no_improve = va_acc, 0
            torch.save(
                {
                    "state": net.state_dict(),
                    "best_val": best,
                    "epoch": ep,
                    "config": {
                        "epochs": config.EPOCHS,
                        "batch": config.BATCH_SIZE,
                        "lr": config.LEARNING_RATE,
                    },
                },
                config.WEIGHTS_PATH,
            )
            print("  saved:", config.WEIGHTS_PATH)
        else:
            no_improve += 1
            if no_improve >= config.PATIENCE:
                print("  early stop")
                break

    print(f"best val: {best * 100:.2f}%  -> {config.WEIGHTS_PATH.resolve()}")
    return best


def load_trained(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config.WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"{config.WEIGHTS_PATH} not found; run `python app.py train`")
    payload = torch.load(config.WEIGHTS_PATH, map_location=device)
    net = DigitConv(config.NUM_CLASSES).to(device)
    net.load_state_dict(payload["state"])
    net.eval()
    return net, device, payload
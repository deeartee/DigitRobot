import argparse

import numpy as np
import torch
from PIL import Image

import config
from runner import load_trained, train
from synth_digits import make_dataset


def _prep(img):
    gray = img.convert("L").resize((config.IMAGE_SIDE, config.IMAGE_SIDE), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr)[None, None, :, :]


def _forward(net, batch, device):
    with torch.no_grad():
        return torch.softmax(net(batch.to(device)), dim=1).cpu().numpy()


def _demo(net, device):
    X, y = make_dataset(per_digit=5, seed=7)
    probs = _forward(net, torch.from_numpy(X), device)
    preds = probs.argmax(1)
    print(f"demo: {(preds == y).sum()}/{len(y)} correct")
    for truth, pred, conf in zip(y.tolist(), preds.tolist(), probs.max(1)):
        tag = "ok" if truth == pred else "!!"
        print(f"  {tag} true={truth} pred={pred} p={conf:.3f}")


def _predict(net, device, paths):
    for path in paths:
        probs = _forward(net, _prep(Image.open(path)), device)[0]
        print(f"{path}: {int(probs.argmax())}  (p={probs.max():.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("images", nargs="*")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        train()
        return

    net, device, meta = load_trained()
    print(f"loaded best_val={meta.get('best_val', 0) * 100:.2f}%  epoch={meta.get('epoch')}")
    if args.demo:
        _demo(net, device)
    elif args.images:
        _predict(net, device, args.images)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
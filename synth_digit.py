import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import IMAGE_SIDE, SEED

WORK = 56

SEGMENTS = {
    0: [(14, 8, 40, 8), (40, 8, 40, 46), (40, 46, 14, 46), (14, 46, 14, 8)],
    1: [(28, 8, 28, 46)],
    2: [(14, 10, 40, 10), (40, 10, 40, 26), (40, 26, 14, 26),
        (14, 26, 14, 44), (14, 44, 40, 44)],
    3: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44), (20, 27, 40, 27)],
    4: [(14, 10, 14, 30), (14, 30, 40, 30), (40, 10, 40, 46)],
    5: [(40, 10, 14, 10), (14, 10, 14, 27), (14, 27, 40, 27),
        (40, 27, 40, 44), (40, 44, 14, 44)],
    6: [(40, 10, 16, 10), (16, 10, 16, 44), (16, 44, 40, 44),
        (40, 44, 40, 27), (40, 27, 16, 27)],
    7: [(14, 10, 40, 10), (40, 10, 20, 46)],
    8: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44),
        (14, 44, 14, 10), (14, 27, 40, 27)],
    9: [(40, 44, 40, 12), (40, 12, 16, 12), (16, 12, 16, 27),
        (16, 27, 40, 27), (16, 12, 16, 10)],
}


def _paint(digit, rng):
    layer = Image.new("L", (WORK, WORK), 0)
    pencil = ImageDraw.Draw(layer)
    for seg in SEGMENTS[digit]:
        pencil.line(seg, fill=255, width=int(rng.integers(2, 5)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.0))))
    layer = layer.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR, fillcolor=0)
    dx, dy = (int(v) for v in rng.integers(-3, 4, size=2))
    return Image.fromarray(np.roll(np.array(layer), (dy, dx), axis=(0, 1)))


def _fit(layer, rng):
    arr = np.asarray(layer, dtype=np.float32) / 255.0
    rows, cols = np.nonzero(arr > 0.1)
    if rows.size == 0:
        return np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float32)
    boxed = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    shrunk = Image.fromarray((boxed * 255).astype(np.uint8))
    shrunk.thumbnail((20, 20), Image.BILINEAR)
    board = Image.new("L", (IMAGE_SIDE, IMAGE_SIDE), 0)
    board.paste(shrunk, ((IMAGE_SIDE - shrunk.width) // 2, (IMAGE_SIDE - shrunk.height) // 2))
    base = np.asarray(board, dtype=np.float32) / 255.0
    return np.clip(base + rng.normal(0, 0.03, base.shape).astype(np.float32), 0.0, 1.0)


def make_dataset(per_digit=300, seed=SEED):
    rng = np.random.default_rng(seed)
    pool_x, pool_y = [], []
    for d in range(10):
        for _ in range(per_digit):
            pool_x.append(_fit(_paint(d, rng), rng))
            pool_y.append(d)
    X = np.stack(pool_x).astype(np.float32)[:, None, :, :]
    y = np.asarray(pool_y, dtype=np.int64)
    return X, y
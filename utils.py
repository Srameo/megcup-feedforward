def pixel_unshuffle(x):
    assert len(x.shape) == 4
    n, c, h, w = x.shape
    x = x.reshape((n, c, h // 2, 2, w // 2, 2)) \
         .transpose((0, 1, 3, 5, 2, 4)) \
         .reshape((n, c * 2 * 2, h // 2, w // 2))
    return x


def pixel_shuffle(x):
    assert len(x.shape) == 4
    n, c, h, w = x.shape
    x = x.reshape((n, c // (2 * 2), 2, 2, h, w)) \
         .transpose((0, 1, 4, 2, 5, 3)) \
         .reshape((n, c // (2 * 2), h * 2, w * 2))
    return x

import numpy as np

delta_stats = [
    "mean iou",
    "pix acc",
    "abs err",
    "rel err",
    "mean",
    "median",
    "<11.25",
    "<22.5",
    "<30",
]
base = np.array(
    [0.3830, 0.6376, 0.6754, 0.2780, 25.01, 19.21, 0.3014, 0.5720, 0.6915]
)  # base results from CAGrad
sign = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
kk = np.ones(9) * -1


def delta_fn(a):
    return (kk**sign * (a - base) / base).mean() * 100.0  # * 100 for percentage

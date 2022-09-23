import numpy as np
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter


def step(x, a, s):
    return .5 + .5 * np.tanh((x - a) / (s + 1e-8))


def rect(x, c, w, s):
    return step(x, c - w / 2, s) - step(x, c + w / 2, s)


def poly_background(w):
    W = len(w)
    out = 2 * (w - W / 6) * (w - W / 1) * (w - W / 1.98)
    out += np.abs(np.min(out)) + 0.01
    out /= np.max(out)
    return out


def sinusoidal_background(w):
    slope = -3 / len(w)
    bg = slope * w + np.sin(w / 10)
    bg -= np.min(bg)
    return bg / np.max(bg)


def stationary_GP_background(w, kappa, ls):
    K = kappa * kappa * np.exp(-.5 * (w[:, None] - w[None, :]) ** 2 / (ls * ls))
    slope = 1 / len(w)
    mean = slope * w + 0.2
    sample = np.random.multivariate_normal(mean, cov=K)
    sample += np.abs(np.min(sample)) + 0.01
    return sample / np.max(sample)


def gibbs_gp_background(w, kappa):
    def gibbs_kernel(x, l, kappa):
        W = x.shape[0]
        lij = np.outer(l, l)
        ll = l * l
        l_sums_1 = ll.repeat(W, axis=0).reshape(W, W)
        l_sums = l_sums_1 + l_sums_1.T

        x_rep = x.repeat(W, axis=0).reshape(W, W)
        ximj = x_rep - x_rep.T
        ximjsq = ximj * ximj

        prefactor = np.sqrt(2 * lij / l_sums)
        exponential = np.exp(-ximjsq / l_sums)

        K = kappa * kappa * prefactor * exponential
        return K + 1e-7 * np.eye(W)

    length_scale = 10 * degenerate_background(w, [len(w) / 2, len(w) * 3 / 2], [len(w) / 2, len(w) / 3], 8) + 3
    K = gibbs_kernel(w, length_scale, kappa)
    sample = np.random.multivariate_normal(mean=1 / len(w) * w, cov=K)
    sample -= np.min(sample)
    return sample / np.max(sample)


def degenerate_background(w, c, gamma, s):
    bg = np.zeros(*w.shape)
    for center, width in zip(c, gamma):
        bg += np.tanh((w - (center - width / 2)) / s) - np.tanh((w - (center + width / 2)) / s)
    return bg / np.max(bg)


def AR_process_background(w, c, phi, smooth=False):
    eta = np.random.randn(len(w), 1)
    B = np.zeros_like(eta)
    B[0] = np.random.rand()
    for w in range(1, len(w)):
        B[w] = c + phi * B[w - 1] + eta[w]
    if smooth:

        B = savgol_filter(B.ravel(), window_length=149, polyorder=2)

    if len(B) > 1:
        B += np.random.rand()
        B -= np.min(B)
        B /= np.max(B)

    B += np.random.rand()
    return B.ravel()


def B_spline_background(w, knots, coefs, degree=3):
    b_spline = BSpline(knots, coefs, degree)
    bg = b_spline(w)
    bg -= np.min(bg)
    bg /= np.max(bg)
    bg += np.random.rand() 
    return bg


def nsgp_background(w, ls, fun_scale, cs, gammas, var_min=0.0000001, variance_scale=0.000001, width_factor=3):
    W = len(w)
    variance = np.ones(W) * var_min
    cws = zip(cs, gammas)
    dw = np.ones(W) / W

    for center, width in cws:
        variance = np.maximum(variance_scale * rect(w, center, width_factor * width, 0), variance)
        dw = np.minimum(1 / W * (1 - rect(w, center, width_factor * width, 0)), dw)
    warped_w = np.cumsum(dw)
    kernel = fun_scale * np.exp(-(warped_w[None, :] - warped_w[:, None]) ** 2 / ls) + np.diag(variance)

    mu = np.zeros(*w.shape)

    L = np.linalg.cholesky(kernel + 1e-6 * np.eye(W))
    bg = L @ np.random.randn(W) + mu

    bg -= bg.min()
    bg /= bg.max()
    bg += np.random.rand()

    return bg

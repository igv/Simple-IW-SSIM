import sys
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from concurrent.futures import ThreadPoolExecutor

WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)

CHROMA_WEIGHTS = np.array([0.0, 0.0, 0.35, 0.20, 0.10])
CHROMA_WEIGHTS = CHROMA_WEIGHTS / np.sum(CHROMA_WEIGHTS)

def get_structure_tensor(H, Parent, sd, t):
    gy, gx = np.gradient(H)
    s_xx = gaussian_filter(gx * gx, sd, truncate=t)
    s_yy = gaussian_filter(gy * gy, sd, truncate=t)
    s_xy = gaussian_filter(gx * gy, sd, truncate=t)
    if Parent is not None:
        py, px = np.gradient(Parent)
        s_xx = (s_xx + gaussian_filter(px * px, sd, truncate=t)) / 2.0
        s_yy = (s_yy + gaussian_filter(py * py, sd, truncate=t)) / 2.0
        s_xy = (s_xy + gaussian_filter(px * py, sd, truncate=t)) / 2.0
    trace = s_xx + s_yy
    det = s_xx * s_yy - s_xy ** 2
    delta = np.sqrt(np.maximum((trace / 2.0) ** 2 - det, 0))
    l1, l2 = trace / 2.0 + delta, trace / 2.0 - delta

    coherence = (l1 - l2) / (l1 + l2 + 1e-3)
    weight = coherence ** 2

    return l1 * weight, l2 * weight

def linearize(img):
    return np.where(img > 0.04045, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def rgb_to_lab(img_linear, luma_only=False):
    r, g, b = img_linear[:,:,0], img_linear[:,:,1], img_linear[:,:,2]
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

    fy = f(Y)
    L = np.maximum(0.0, 116.0 * fy - 16.0)
    if luma_only:
        return L

    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    fx, fz = f(X / 0.95047), f(Z / 1.08883)
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b

def gaussian_pyramid(image_linear, levels=6, luma_only=False):
    if luma_only:
        Y = 0.2126729 * image_linear[:,:,0] + 0.7151522 * image_linear[:,:,1] + 0.0721750 * image_linear[:,:,2]
        def f(t):
            return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
        gpyr_L = [np.maximum(0.0, 116.0 * f(Y) - 16.0)]
        current = Y
        for _ in range(levels - 1):
            current = gaussian_filter(current, sigma=1.08, truncate=1.5)[::2, ::2]
            gpyr_L.append(np.maximum(0.0, 116.0 * f(current) - 16.0))
        return gpyr_L, None, None

    res = rgb_to_lab(image_linear, luma_only=False)
    gpyr_L, gpyr_a, gpyr_b = [res[0]], [res[1]], [res[2]]

    current = image_linear
    for _ in range(levels - 1):
        current = np.stack([
            gaussian_filter(current[:,:,c], sigma=1.08, truncate=1.5)[::2, ::2]
            for c in range(3)
        ], axis=-1)
        res = rgb_to_lab(current, luma_only=False)
        gpyr_L.append(res[0])
        gpyr_a.append(res[1])
        gpyr_b.append(res[2])

    return gpyr_L, gpyr_a, gpyr_b

def laplacian_pyramid(G_pyr, start_level=0, levels=5):
    L_pyr = []
    for s in range(start_level, levels):
        l, l2 = G_pyr[s], G_pyr[s+1]
        h, w = l.shape
        h2, w2 = l2.shape
        exp = np.zeros((h, w), dtype=l.dtype)
        exp[0:min(h, h2*2):2, 0:min(w, w2*2):2] = l2[0:(h+1)//2, 0:(w+1)//2]
        upsampled = gaussian_filter(exp, sigma=1.08, truncate=1.5) * 4.0
        L_pyr.append(l - upsampled)
    return L_pyr

def compute_ssim_maps(lpyr1, lpyr2, gpyr1, gpyr2, sd=1.5, t=2.5, dyn_range=100.0):
    cs_maps = []
    C1 = (0.01 * dyn_range) ** 2
    C2 = (0.03 * dyn_range) ** 2
    for H1, H2 in zip(lpyr1, lpyr2):
        mu1 = gaussian_filter(H1, sd, truncate=t)
        mu2 = gaussian_filter(H2, sd, truncate=t)
        sigma1_sq = np.maximum(0, gaussian_filter(H1 * H1, sd, truncate=t) - mu1 ** 2)
        sigma2_sq = np.maximum(0, gaussian_filter(H2 * H2, sd, truncate=t) - mu2 ** 2)
        sigma12   = gaussian_filter(H1 * H2, sd, truncate=t) - mu1 * mu2

        cs_maps.append((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))

    base1, base2 = gpyr1[4], gpyr2[4]
    mu_base1 = gaussian_filter(base1, sd, truncate=t)
    mu_base2 = gaussian_filter(base2, sd, truncate=t)
    l_map = (2.0 * mu_base1 * mu_base2 + C1) / (mu_base1 ** 2 + mu_base2 ** 2 + C1)
    return cs_maps, l_map

def compute_iw_maps(lpyr1, lpyr2, sd=1.2, t=2.0):
    sigma_nsq = 0.05
    eps = 1e-6
    iw_maps = []

    for scale in range(1, 6):
        if scale == 5:
            iw_maps.append(None)
            continue

        H1, H2 = lpyr1[scale-1], lpyr2[scale-1]
        parent = lpyr1[scale]
        zoom_factors = (H1.shape[0] / parent.shape[0], H1.shape[1] / parent.shape[1])
        P = zoom(parent, zoom_factors, order=2)

        lam1, lam2 = get_structure_tensor(H1, P, sd, t)

        mu1 = gaussian_filter(H1, sd, truncate=t)
        mu2 = gaussian_filter(H2, sd, truncate=t)
        H1_sq = gaussian_filter(H1 * H1, sd, truncate=t)
        H2_sq = gaussian_filter(H2 * H2, sd, truncate=t)
        H1_H2 = gaussian_filter(H1 * H2, sd, truncate=t)

        sigma1_sq = np.maximum(H1_sq - mu1 ** 2, eps)
        sigma2_sq = np.maximum(H2_sq - mu2 ** 2, eps)
        sigma12 = np.maximum(H1_H2 - mu1 * mu2, eps)

        g = sigma12 / sigma1_sq
        g[sigma1_sq < sigma_nsq] = 1
        # g[g>1] = 1

        sv_sq = np.maximum(sigma2_sq - g * sigma12, 0)
        # sv_sq[sigma1_sq<sigma_nsq] *= eps

        info_dist = np.log2(1 + ((sv_sq + (1 + g ** 2) * sigma_nsq) * lam1 + sv_sq * sigma_nsq) / (sigma_nsq ** 2)) + \
                    np.log2(1 + ((sv_sq + (1 + g ** 2) * sigma_nsq) * lam2 + sv_sq * sigma_nsq) / (sigma_nsq ** 2))
        info_dist[info_dist < 1e-10] = 0
        iw_maps.append(info_dist)

    return iw_maps

def compute_channel_iwssim(gpyr1, gpyr2, lpyr1, lpyr2, iw_maps, dyn_range=100.0, is_chroma=False):
    cs_maps, l_map = compute_ssim_maps(lpyr1, lpyr2, gpyr1, gpyr2, dyn_range=dyn_range)
    weights = CHROMA_WEIGHTS if is_chroma else WEIGHTS
    start_level = int(np.flatnonzero(weights)[0])
    wmcs = []

    for i, cs in enumerate(cs_maps):
        scale_idx = start_level + i
        if scale_idx == 4:
            cs *= l_map
            iw = np.ones_like(cs)
        else:
            iw = iw_maps[scale_idx]

        crop = 1
        cs_crop = cs[crop:-crop, crop:-crop]
        iw_crop = iw[crop:-crop, crop:-crop]

        val = np.sum(cs_crop * iw_crop) / np.sum(iw_crop)
        wmcs.append(np.clip(val, 0.0, 1.0) ** weights[scale_idx])

    return np.prod(wmcs)

def iwssim(file1, file2, luma_only=False):
    img1 = np.array(Image.open(file1).convert('RGB'), dtype=np.float32) / 255.0
    img2 = np.array(Image.open(file2).convert('RGB'), dtype=np.float32) / 255.0

    lin1 = linearize(img1)
    lin2 = linearize(img2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(gaussian_pyramid, lin1, 6, luma_only)
        f2 = executor.submit(gaussian_pyramid, lin2, 6, luma_only)
        (gpyr_L1, gpyr_a1, gpyr_b1), (gpyr_L2, gpyr_a2, gpyr_b2) = f1.result(), f2.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(laplacian_pyramid, gpyr_L1)
        f2 = executor.submit(laplacian_pyramid, gpyr_L2)
        lpyr_L1, lpyr_L2 = f1.result(), f2.result()

    iw_maps_L = compute_iw_maps(lpyr_L1, lpyr_L2)
    score_L = compute_channel_iwssim(gpyr_L1, gpyr_L2, lpyr_L1, lpyr_L2, iw_maps_L, dyn_range=100.0, is_chroma=False)

    if luma_only:
        return score_L

    chroma_start = int(np.flatnonzero(CHROMA_WEIGHTS)[0])
    lpyr_a1 = laplacian_pyramid(gpyr_a1, start_level=chroma_start)
    lpyr_a2 = laplacian_pyramid(gpyr_a2, start_level=chroma_start)
    score_a = compute_channel_iwssim(gpyr_a1, gpyr_a2, lpyr_a1, lpyr_a2, iw_maps_L, dyn_range=100.0, is_chroma=True)

    lpyr_b1 = laplacian_pyramid(gpyr_b1, start_level=chroma_start)
    lpyr_b2 = laplacian_pyramid(gpyr_b2, start_level=chroma_start)
    score_b = compute_channel_iwssim(gpyr_b1, gpyr_b2, lpyr_b1, lpyr_b2, iw_maps_L, dyn_range=100.0, is_chroma=True)

    return 0.9 * score_L + 0.05 * score_a + 0.05 * score_b

def main():
    args = sys.argv[1:]
    luma_only = False
    filtered_args = []
    for arg in args:
        if arg in ('--luma', '--luma-only', '-l'):
            luma_only = True
        else:
            filtered_args.append(arg)
    args = filtered_args

    if len(args) < 2:
        print("Usage: python iwssim.py [--luma] <ref> <dist1> [dist2...]")
        return

    ref = args[0]
    for dist in args[1:]:
        score = iwssim(ref, dist, luma_only=luma_only)
        print(f"{score:.6f}\t{dist}")

if __name__ == '__main__':
    main()

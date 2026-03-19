import sys
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from concurrent.futures import ThreadPoolExecutor

WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

def get_structure_tensor_evals(H, Parent, sd, t):
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
    delta = np.sqrt(np.maximum((trace / 2) ** 2 - det, 0))

    l1, l2 = trace / 2 + delta, trace / 2 - delta
    coherence = (l1 - l2) / (l1 + l2 + 1e-6)

    # angle = 0.5 * np.arctan2(2 * s_xy, s_xx - s_yy)
    # factor = 0.4142 * np.abs(np.sin(2 * angle))
    # norm = 1.0 + factor * coherence

    return l1 * (coherence + 0.1), l2 * (coherence + 0.1)

def linearize(img):
    return np.where(img > 0.04045, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def to_Luma(img):
    return 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]

def to_L(Y):
    return np.where(Y > 0.008856, np.power(Y, 1./3.) * 116 - 16, Y * 903.3)

def gaussian_pyramid(image, levels=6):
    pyramid = [to_L(image)]
    current = image
    for _ in range(levels - 1):
        current = gaussian_filter(current, sigma=1.08, truncate=1.5)[::2, ::2]
        pyramid.append(to_L(current))
    return pyramid

def laplacian_pyramid(G_pyr, levels=5):
    L_pyr = []
    for s in range(levels):
        l, l2 = G_pyr[s], G_pyr[s+1]
        h, w = l.shape
        h2, w2 = l2.shape
        exp = np.zeros((h, w), dtype=l.dtype)
        h, w = min(h, h2 * 2), min(w, w2 * 2)
        exp[0:h:2, 0:w:2] = l2[0:(h+1)//2, 0:(w+1)//2]
        upsampled = gaussian_filter(exp, sigma=1.08, truncate=1.5) * 4.0
        H = l - upsampled
        L_pyr.append(H)
    return L_pyr

def compute_ssim_maps(lpyr1, lpyr2, sd=1.5, t=2.5):
    cs_maps = []
    l_map = None
    C1 = (0.01 * 100) ** 2
    C2 = (0.03 * 100) ** 2

    for scale in range(1, 6):
        H1, H2 = lpyr1[scale-1], lpyr2[scale-1]
        mu1 = gaussian_filter(H1, sd, truncate=t)
        mu2 = gaussian_filter(H2, sd, truncate=t)

        sigma1_sq = gaussian_filter(H1 * H1, sd, truncate=t) - mu1 ** 2
        sigma2_sq = gaussian_filter(H2 * H2, sd, truncate=t) - mu2 ** 2
        sigma12 = gaussian_filter(H1 * H2, sd, truncate=t) - mu1 * mu2

        cs_maps.append((2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))

        if scale == 5:
            l_map = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)

    return cs_maps, l_map

def compute_iw_maps(lpyr1, lpyr2, pyr1, sd=1.2, t=2.0):
    iw_maps = []
    sigma_nsq = 0.05
    eps = 1e-6

    for scale in range(1, 6):
        if scale == 5:
            iw_maps.append(None) 
            continue

        H1, H2 = lpyr1[scale-1], lpyr2[scale-1]

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

        parent = lpyr1[scale]
        zoom_factors = (H1.shape[0] / parent.shape[0], H1.shape[1] / parent.shape[1])
        P = zoom(parent, zoom_factors, order=2)

        lam1, lam2 = get_structure_tensor_evals(H1, P, sd, t)

        info_dist = np.log2(1 + ((sv_sq + (1 + g ** 2) * sigma_nsq) * lam1 + sv_sq * sigma_nsq) / \
                                (sigma_nsq ** 2)) + \
                    np.log2(1 + ((sv_sq + (1 + g ** 2) * sigma_nsq) * lam2 + sv_sq * sigma_nsq) / \
                                (sigma_nsq ** 2))

        # info_dist = np.log2(1 + (lam1 / sigma_nsq)) + np.log2(1 + (lam2 / sigma_nsq))

        info_dist[info_dist < 1e-10] = 0
        iw_maps.append(info_dist)

    return iw_maps

def iwssim(file1, file2):
    img1 = Image.open(file1).convert('RGB')
    img2 = Image.open(file2).convert('RGB')

    width, height = img1.size
    img1 = np.frombuffer(img1.tobytes(), dtype=np.uint8).reshape(height, width, 3)
    img2 = np.frombuffer(img2.tobytes(), dtype=np.uint8).reshape(height, width, 3)

    Y1 = to_Luma(linearize((img1 / 255)))
    Y2 = to_Luma(linearize((img2 / 255)))

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(gaussian_pyramid, Y1)
        f2 = executor.submit(gaussian_pyramid, Y2)
        pyr1, pyr2 = f1.result(), f2.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(laplacian_pyramid, pyr1)
        f2 = executor.submit(laplacian_pyramid, pyr2)
        lpyr1, lpyr2 = f1.result(), f2.result()

    cs_maps, l_map = compute_ssim_maps(lpyr1, lpyr2)

    iw_maps = compute_iw_maps(lpyr1, lpyr2, pyr1)

    wmcs = []

    for scale in range(1, 6):
        cs = cs_maps[scale-1]
        if scale == 5:
            cs *= l_map
            iw = np.ones_like(cs)
        else:
            iw = iw_maps[scale-1]

        cs = cs[1:-1, 1:-1]
        iw = iw[1:-1, 1:-1]

        wmcs.append(np.sum(cs * iw) / np.sum(iw))

    score = np.prod(np.array(wmcs) ** np.array(WEIGHTS / np.sum(WEIGHTS)))
    return score

def main():
    if len(sys.argv) < 3:
        print("Usage: python iwssim.py <ref> <dist1> [dist2...]")
        return

    for arg in sys.argv[2:]:
        score = iwssim(sys.argv[1], arg)
        print(f"{score:.6f}\t{arg}")

if __name__ == '__main__':
    main()

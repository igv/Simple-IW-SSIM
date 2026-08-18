# Simple IW-SSIM
A lightweight Python implementation of Information-Weighted Structural Similarity (IW-SSIM) that balances perceptual accuracy and speed.

This metric enhances traditional MS-SSIM by using an information-theoretic weighting strategy, prioritizing areas of the image that contain more visual information where the human eye is more sensitive to distortions.

## Key Differences from the Original IW-SSIM
While based on the original research by Wang and Li, simple-iw-ssim introduces several practical modifications:
* **Simplified Information Map:** Instead of a full GSM-based statistical model, this version uses a streamlined structure tensor approach to calculate the information distribution.
* **Linear-Light CIELAB Workflow:** Downsampling is performed strictly in physically linear RGB before per-scale conversion to CIELAB, preventing gamma shifts and chromatic distortion artifacts.
* **Chroma CSF Adaptation:** Bypasses high-frequency chromatic subpixel noise (Scales 1 & 2) for the a and b channels, matching the human visual system's lower spatial acuity for color.
* **Coherence-Adjusted Weighting:** Structural tensor eigenvalues are scaled by coherence to prioritize coherent structural edges over stochastic noise.

## No Heavy Dependencies
Built only on **NumPy**, **SciPy**, and **Pillow**. No need for OpenCV or PyTorch.

## Usage

### Command Line
```bash
# Full perceptual color evaluation (L*a*b*)
python iwssim.py reference.png distorted.png [distorted2.png ...]

# Luma-only evaluation (L*)
python iwssim.py --luma reference.png distorted.png
```

### Python API
```python
from iwssim import iwssim

# Full color score
score = iwssim("reference.png", "distorted.png")
print(f"IW-SSIM: {score:.6f}")

# Luma-only score
score_luma = iwssim("reference.png", "distorted.png", luma_only=True)
print(f"Luma IW-SSIM: {score_luma:.6f}")
```

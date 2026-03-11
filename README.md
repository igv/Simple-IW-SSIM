# Simple IW-SSIM
A lightweight Python implementation of IW-SSIM that balances accuracy and speed.

This metric enhances the traditional MS-SSIM by using an information-theoretic weighting strategy, prioritizing areas of the image that contain more visual information where the human eye is more sensitive to distortions.

## How It Works
Traditional SSIM treats all pixels equally. IW-SSIM, however, applies Information Content (IC) weighting. It calculates a "weight map" based on a Gaussian Scale Mixture (GSM) model, ensuring that complex structural regions have a higher impact on the final score than flat, uninformative areas.

## This specific implementation utilizes:
* **Laplacian Pyramids** for multi-scale analysis.
* **Structure Tensors** to evaluate local image statistics.
* **Perceptual Color Space:** The code linearizes RGB values and converts them to the perceptual luminance ($L^*$) of the CIELAB space for better alignment with human vision.

## Key Differences from the Original IW-SSIM
While based on the original research by Wang and Li, simple-iw-ssim introduces several practical modifications:
* **Simplified Information Map:** Instead of a full GSM-based statistical model, this version uses a streamlined structure tensor approach to calculate the information distribution.
* **Linearized Workflow:** Unlike many basic implementations that work directly on gamma-corrected sRGB, this tool performs linearization and uses perceptual luminance ($L^*$).

## No Heavy Dependencies
Built only on NumPy, SciPy, and Pillow. No need for massive frameworks like OpenCV or PyTorch.

## Usage
```python iwssim.py reference.png distorted.png ```

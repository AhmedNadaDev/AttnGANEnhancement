"""
Post-processing enhancement pipeline for AttnGAN-generated images.

Enhancement chain (applied in order)
--------------------------------------
1. Upscale   256 → 512 px via Lanczos resampling
2. Denoise   mild Gaussian blur to suppress GAN checkerboard artefacts
3. Sharpen   UnsharpMask to restore / enhance fine structural detail
4. Contrast  moderate global contrast boost (×1.15)
5. Saturation  mild colour saturation boost (×1.08)

All five steps use pure PIL — no additional model download required,
guaranteeing the enhancement always runs on Kaggle.

Real-ESRGAN path (optional)
----------------------------
If the `realesrgan` and `basicsr` packages are installed and a GPU is
available, the class will attempt to load the RealESRGAN_x4plus model
automatically.  The PIL chain is then applied after ESRGAN upscaling
to apply the quality corrections.  If the ESRGAN model download fails
(e.g. offline Kaggle), the pipeline falls back silently to pure PIL.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter


class ImageEnhancer:
    """
    Deterministic post-processing enhancer for GAN-generated images.

    Parameters
    ----------
    output_size       : Target (width, height) after upscaling.
    denoise_radius    : Gaussian blur σ for artefact suppression.
    sharpen_radius    : UnsharpMask radius (pixels).
    sharpen_percent   : UnsharpMask strength (percent ≥ 0).
    sharpen_threshold : UnsharpMask threshold (pixel difference).
    contrast_factor   : ImageEnhance.Contrast multiplier (1.0 = no change).
    color_factor      : ImageEnhance.Color multiplier (1.0 = no change).
    use_esrgan        : If True, attempt to load Real-ESRGAN (optional).

    Example
    -------
    >>> enhancer = ImageEnhancer()
    >>> result   = enhancer.enhance(pil_256)  # → PIL Image 512×512
    """

    def __init__(
        self,
        output_size:        Tuple[int, int] = (512, 512),
        denoise_radius:     float           = 0.6,
        sharpen_radius:     float           = 1.5,
        sharpen_percent:    int             = 130,
        sharpen_threshold:  int             = 3,
        contrast_factor:    float           = 1.15,
        color_factor:       float           = 1.08,
        use_esrgan:         bool            = True,
    ) -> None:
        self.output_size       = output_size
        self.denoise_radius    = denoise_radius
        self.sharpen_radius    = sharpen_radius
        self.sharpen_percent   = sharpen_percent
        self.sharpen_threshold = sharpen_threshold
        self.contrast_factor   = contrast_factor
        self.color_factor      = color_factor

        self._esrgan: Optional[object] = (
            self._try_load_esrgan() if use_esrgan else None
        )

    # ── ESRGAN loader (optional) ──────────────────────────────────────────────

    def _try_load_esrgan(self) -> Optional[object]:
        """
        Attempt to load Real-ESRGAN x4plus for learned super-resolution.
        Returns the upsampler object on success, None on any failure.
        """
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from realesrgan import RealESRGANer              # type: ignore

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            upsampler = RealESRGANer(
                scale=4,
                model_path=(
                    "https://github.com/xinntao/Real-ESRGAN/releases/"
                    "download/v0.1.0/RealESRGAN_x4plus.pth"
                ),
                model=model,
                half=torch.cuda.is_available(),
            )
            print("[Enhancer] Real-ESRGAN x4plus loaded (4× super-resolution).")
            return upsampler
        except Exception as exc:
            print(f"[Enhancer] Real-ESRGAN unavailable ({exc}); using PIL upscaling.")
            return None

    # ── public API ────────────────────────────────────────────────────────────

    def enhance(self, image: Image.Image) -> Image.Image:
        """
        Apply the full enhancement chain to *image*.

        Selects Real-ESRGAN path when available, otherwise falls back
        to pure-PIL Lanczos upscaling.

        Returns a PIL Image at self.output_size (default 512×512).
        """
        if self._esrgan is not None:
            return self._enhance_esrgan(image)
        return self._enhance_pil(image)

    # ── enhancement paths ─────────────────────────────────────────────────────

    def _enhance_esrgan(self, image: Image.Image) -> Image.Image:
        """Real-ESRGAN 4× upscale then resize to output_size + quality chain."""
        try:
            import cv2  # type: ignore
            import numpy as np

            bgr    = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            output, _ = self._esrgan.enhance(bgr, outscale=4)
            rgb    = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            img    = Image.fromarray(rgb).resize(self.output_size, Image.LANCZOS)
            return self._quality_chain(img)
        except Exception as exc:
            print(f"[Enhancer] ESRGAN inference failed ({exc}); falling back to PIL.")
            return self._enhance_pil(image)

    def _enhance_pil(self, image: Image.Image) -> Image.Image:
        """Pure-PIL Lanczos upscale + quality chain (always available)."""
        img = image.resize(self.output_size, Image.LANCZOS)
        return self._quality_chain(img)

    def _quality_chain(self, img: Image.Image) -> Image.Image:
        """
        Apply denoise → sharpen → contrast → colour saturation.

        Steps 2–5 of the enhancement chain; applied after any upscaling.
        """
        # Step 2 — Denoise: light Gaussian blur removes GAN ringing/checkerboard
        img = img.filter(ImageFilter.GaussianBlur(radius=self.denoise_radius))

        # Step 3 — Sharpen: UnsharpMask restores edge detail lost in denoising
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=self.sharpen_radius,
                percent=self.sharpen_percent,
                threshold=self.sharpen_threshold,
            )
        )

        # Step 4 — Contrast boost: improves perceived depth and object definition
        img = ImageEnhance.Contrast(img).enhance(self.contrast_factor)

        # Step 5 — Colour saturation: makes plumage colours more vivid
        img = ImageEnhance.Color(img).enhance(self.color_factor)

        return img

    # ── meta ──────────────────────────────────────────────────────────────────

    @property
    def using_esrgan(self) -> bool:
        """True when Real-ESRGAN is active for super-resolution."""
        return self._esrgan is not None

    def __repr__(self) -> str:
        backend = "Real-ESRGAN" if self.using_esrgan else "PIL-Lanczos"
        return (
            f"ImageEnhancer("
            f"backend={backend!r}, "
            f"output={self.output_size[0]}×{self.output_size[1]}, "
            f"contrast={self.contrast_factor}, "
            f"color={self.color_factor})"
        )

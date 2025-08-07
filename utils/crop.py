from PIL import Image

class CropBorders:
    """
    Crop a fixed percentage off the *top* and off *both* left+right sides.

    top_ratio   – fraction to remove from the top   (e.g.  1/3  →  0.333…)
    side_ratio  – fraction to remove from *each* side (e.g. 1/4 → 0.25)
    """
    def __init__(self, top_ratio=1/3, side_ratio=1/4):
        self.top_ratio  = top_ratio
        self.side_ratio = side_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        left   = int(w * self.side_ratio)
        right  = int(w * (1 - self.side_ratio))
        top    = int(h * self.top_ratio)
        bottom = h                                # keep full bottom edge
        return img.crop((left, top, right, bottom))
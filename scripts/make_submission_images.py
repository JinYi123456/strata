"""Generate the submission images Devpost actually accepts.

Devpost's gallery renders at 3:2 and its project thumbnail at 16:9. The
screenshots in ``docs/screenshots/`` are 1600x1000 (1.60) because that is what
the console's own verification script captures, and the platform will crop them
wherever it likes - which is how a screenshot of a dense dashboard loses the
approval gate at the bottom, or the header at the top, in a way nobody chose.

So the crops are chosen here, deliberately, and written to ``submission/`` where
they can be looked at before they are uploaded.

    python scripts/make_submission_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "screenshots"
OUT = ROOT / "submission" / "gallery"

GALLERY_RATIO = 3 / 2          # 1500x1000 from a 1600x1000 source
THUMBNAIL_SIZE = (1280, 720)   # 16:9, Devpost's project thumbnail

# The screenshots are dense dashboards with the change-set panel against the
# right edge and the approval gate at the bottom, so they are padded to 3:2
# rather than cropped: 100 columns off each side would take the confidence
# numbers with them. The pad colour is the console's own ground, so the padding
# is invisible against the interface.
GROUND = "#080B0F"
SHOTS = ("01-sites.png", "02-scan-result.png", "03-compare.png",
         "04-changes.png", "05-coverage.png", "06-gate.png")


def fit(image: Image.Image, size: tuple[int, int], background: str) -> Image.Image:
    """Scale to fit inside ``size`` and pad, rather than crop a dashboard."""
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (int(round(image.width * scale)), int(round(image.height * scale))),
        Image.LANCZOS,
    )
    canvas = Image.new("RGB", size, background)
    canvas.paste(resized, ((size[0] - resized.width) // 2,
                           (size[1] - resized.height) // 2))
    return canvas


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name in SHOTS:
        source = SRC / name
        if not source.is_file():
            raise SystemExit(f"missing screenshot: {source}")
        image = Image.open(source).convert("RGB")
        target_h = int(round(image.width / GALLERY_RATIO))
        canvas = Image.new("RGB", (image.width, target_h), GROUND)
        canvas.paste(image, (0, (target_h - image.height) // 2))
        out = OUT / name
        canvas.save(out, optimize=True)
        written.append(out)

    # The architecture diagram belongs in the gallery too: the rules ask for a
    # diagram showing the OpenCV 5 and AWS components, and a judge should not
    # have to open the repository to see it. The diagram is 1.59, so it is
    # padded to 1.50 rather than cropped - every label in it is load-bearing.
    diagram = Image.open(ROOT / "docs" / "architecture.png").convert("RGB")
    canvas_h = int(round(diagram.width / GALLERY_RATIO))
    canvas = Image.new("RGB", (diagram.width, canvas_h), "#F7F9FB")
    canvas.paste(diagram, (0, (canvas_h - diagram.height) // 2))
    out = OUT / "07-architecture.png"
    canvas.save(out, optimize=True)
    written.append(out)

    # Thumbnail: the film's own title card is 16:9 already, so it is fitted
    # rather than cropped. Falls back to the changes screenshot if the film has
    # not been built in this checkout.
    title = ROOT / "build" / "video" / "cards" / "01-title.png"
    if title.is_file():
        thumb = fit(Image.open(title).convert("RGB"), THUMBNAIL_SIZE, "#080B0F")
    else:
        thumb = fit(Image.open(SRC / "04-changes.png").convert("RGB"),
                    THUMBNAIL_SIZE, "#080B0F")
        draw = ImageDraw.Draw(thumb)
        draw.rectangle((0, THUMBNAIL_SIZE[1] - 4, THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1]),
                       fill="#4FE3C1")
    thumbnail = ROOT / "submission" / "thumbnail.png"
    thumb.save(thumbnail, optimize=True)
    written.append(thumbnail)
    return written


def main() -> int:
    for path in build():
        image = Image.open(path)
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(ROOT).as_posix():<40} "
              f"{image.width}x{image.height} ({image.width / image.height:.2f}) "
              f"{size_kb:,.0f} KB")
    print("\nUnder Devpost's 5 MB per image, at the aspect ratio it renders at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

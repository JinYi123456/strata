"""Generate the submission images Devpost actually accepts.

Devpost renders both its project gallery and its project thumbnail at 3:2
(the form says so: "For best results, use a 3:2 ratio"), so nothing here is
cropped to fit a ratio the platform will not show. The
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

# One source for the palette and the letter-spaced text helper. Duplicating them
# across the film and the stills is how a brand colour ends up meaning two
# different things in two files.
from build_video import (  # noqa: E402
    ACCENT,
    ACCENT_BRIGHT,
    EDGE,
    GROUND,
    INK,
    INK_FAINT,
    INK_MUTED,
    RAISED,
    _rule,
    _text,
)
SRC = ROOT / "docs" / "screenshots"
OUT = ROOT / "submission" / "gallery"

GALLERY_RATIO = 3 / 2          # 1500x1000 from a 1600x1000 source
THUMBNAIL_SIZE = (1200, 800)   # 3:2 — what Devpost's thumbnail field asks for

# The screenshots are dense dashboards with the change-set panel against the
# right edge and the approval gate at the bottom, so they are padded to 3:2
# rather than cropped: 100 columns off each side would take the confidence
# numbers with them. The pad colour is the console's own ground, so the padding
# is invisible against the interface.
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


def youtube_thumbnail(out: Path) -> Path:
    """A thumbnail for YouTube, which wants 1280x720 and nothing else.

    The Devpost thumbnail is 3:2; YouTube crops a 3:2 image to 16:9 and eats the
    top and bottom of the type. So this is composed for the platform's own frame
    rather than converted into it. It has to survive being 210 pixels wide in a
    sidebar, which means one wordmark, one sentence, and one real screenshot -
    not the whole product description.
    """
    size = (1280, 720)
    img = Image.new("RGB", size, GROUND)
    d = ImageDraw.Draw(img)
    for x in range(0, size[0], 32):
        d.line([(x, 0), (x, size[1])], fill=RAISED, width=1)
    for y in range(0, size[1], 32):
        d.line([(0, y), (size[0], y)], fill=RAISED, width=1)

    shot = Image.open(SRC / "04-changes.png").convert("RGB")
    scale = 470 / shot.width
    shot = shot.resize((470, int(round(shot.height * scale))), Image.LANCZOS)
    box = (size[0] - shot.width - 64, 64)
    d.rectangle((box[0] - 3, box[1] - 3, box[0] + shot.width + 3, box[1] + shot.height + 3),
                outline=EDGE, width=2)
    img.paste(shot, box)

    _text(d, (64, 150), "S T R A T A", size=76, fill=INK, role="sans_bold", tracking=9)
    # Span the wordmark rather than a third of it, or the rule reads
    # as an underline under one letter.
    _rule(d, (66, 250), 500, fill=ACCENT)
    _text(d, (64, 290), "Version control for", size=40, fill=INK_MUTED)
    _text(d, (64, 340), "the physical world", size=40, fill=INK_MUTED)
    _text(d, (64, 452), "OpenCV 5  +  AWS Graviton", size=28, role="mono_bold",
          fill=ACCENT_BRIGHT)
    _text(d, (64, 500), "Typed changes, measured, human-gated", size=24, fill=INK_FAINT)
    _text(d, (64, 600), "OpenCV AI Competition 2026", size=22, role="mono",
          fill=INK_FAINT)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)
    return out


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
    written.append(youtube_thumbnail(ROOT / "submission" / "youtube-thumbnail.png"))

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

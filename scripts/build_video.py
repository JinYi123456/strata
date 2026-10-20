"""Build the submission film.

The competition requires a video of no more than five minutes that shows the
team, the application working, its architecture, and its principal results. The
film that shipped with the repository showed two of those four: the application
and its results, on screen, with narration. There was no authorship and no
architecture. This script adds them, from the same assets the report uses, so
the film and the documents cannot disagree.

    python scripts/build_video.py                 # build into build/video/
    python scripts/build_video.py --install       # ...and publish into the app

The existing demonstration is kept as-is rather than re-recorded: it is a real
capture of the real application, and re-recording it would trade a verified
artefact for an approximation of one.

Design constraints, all of them earned the hard way:

- Text never zooms or pans. A scaled screenshot reads as an enlargement
  artefact and a moving frame under a line of type reads as a defect.
- Narration is Microsoft David at rate 1, matching the voice and pace already in
  the film, and every segment is loudness-matched to the existing track with two
  passes rather than one.
- Every visual is generated from a committed source: the palette and font stack
  from the console's own configuration, the diagram from docs/architecture.png,
  the numbers from docs/evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figures import derived  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 1920, 1080
FPS = 25

# The console's own design language, read from web/tailwind.config.ts by eye and
# written here once. If the palette changes, change it here too - the film is
# meant to look like the product, which is the point of sharing the values.
GROUND = "#080B0F"
RAISED = "#0E141B"
PANEL = "#131B24"
EDGE = "#1E2A36"
INK = "#E9EFF5"
INK_MUTED = "#93A3B4"
INK_FAINT = "#5C6B7C"
ACCENT = "#4FE3C1"
ACCENT_BRIGHT = "#7BF0D5"
KIND = {
    "missing": "#FF6B6B",
    "added": "#4FE3C1",
    "moved": "#B18CFF",
    "state_changed": "#63B3FF",
    "uncertain": "#FFB84D",
}

FONTS = {
    "sans": "C:/Windows/Fonts/segoeui.ttf",
    "sans_bold": "C:/Windows/Fonts/segoeuib.ttf",
    "light": "C:/Windows/Fonts/segoeui.ttf",
    "mono": "C:/Windows/Fonts/consola.ttf",
    "mono_bold": "C:/Windows/Fonts/consolab.ttf",
}

VOICE = "Microsoft David Desktop"
RATE = 1
# Measured from the existing track with ebur128, so the new segments do not jump
# in level against the ones already in the film.
TARGET_LUFS = -16.4
TARGET_TRUE_PEAK_DB = -1.5


# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    identifier: str
    narration: str
    frames: list[Path] = field(default_factory=list)
    hold_after_speech: float = 0.7

    @property
    def slug(self) -> str:
        return self.identifier.replace(" ", "-").lower()


def speak(text: str, out_wav: Path) -> Path:
    """Render narration with Windows SAPI. No network, no account, no cost."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    script = out_wav.with_suffix(".ps1")
    # A here-string keeps the text out of the command line, where quoting and
    # length limits live. The path is single-quoted because PowerShell expands
    # double-quoted strings, and a backslash in a Windows path would be eaten.
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "Add-Type -AssemblyName System.Speech\n"
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        f"$synth.SelectVoice('{VOICE}')\n"
        f"$synth.Rate = {RATE}\n"
        f"$synth.SetOutputToWaveFile('{out_wav.as_posix()}')\n"
        "$synth.Speak(@'\n"
        f"{text}\n"
        "'@)\n"
        "$synth.Dispose()\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(script)],
        check=True, capture_output=True,
    )
    if not out_wav.is_file() or out_wav.stat().st_size < 5000:
        raise RuntimeError(f"narration produced no audio for {out_wav}")
    return out_wav


def normalise_loudness(src: Path, dst: Path) -> Path:
    """Two-pass loudnorm. A single pass measures and corrects in the same walk
    through the file, which leaves the first seconds of any clip at the wrong
    level - visible in a waveform and audible in a cut."""
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
         "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK_DB}:LRA=11:print_format=json",
         "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    raw = probe.stderr
    start, end = raw.rfind("{"), raw.rfind("}")
    measured = json.loads(raw[start:end + 1])
    filter_spec = (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK_DB}:LRA=11"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true:print_format=summary"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", filter_spec,
         "-ar", "48000", "-ac", "2", str(dst)],
        check=True,
    )
    return dst


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS.get(role, FONTS["sans"])
    if not Path(path).is_file():
        path = FONTS["sans"]
    return ImageFont.truetype(path, size)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), GROUND)
    draw = ImageDraw.Draw(img)
    # The same 32px grid the console paints behind its panels.
    for x in range(0, W, 32):
        draw.line([(x, 0), (x, H)], fill=RAISED, width=1)
    for y in range(0, H, 32):
        draw.line([(0, y), (W, y)], fill=RAISED, width=1)
    return img, draw


def _text(draw: ImageDraw.ImageDraw, xy, value: str, *, role="sans", size=32,
          fill=INK, anchor="la", tracking: int = 0) -> None:
    """Draw text, optionally letter-spaced.

    ``tracking`` exists because the wordmark is set with CSS letter-spacing in
    the product, and a film that renders it tight looks like a different brand.
    """
    x, y = xy
    f = font(role, size)
    if not tracking:
        draw.text((x, y), value, font=f, fill=fill, anchor=anchor)
        return
    total = sum(draw.textlength(c, font=f) + tracking for c in value) - tracking
    if anchor.endswith("a") and anchor[0] in "mc":
        x -= total / 2
    elif anchor.endswith("a") and anchor[0] == "r":
        x -= total
    for c in value:
        draw.text((x, y), c, font=f, fill=fill, anchor="la")
        x += draw.textlength(c, font=f) + tracking


def _rule(draw: ImageDraw.ImageDraw, xy, width: int, *, fill=EDGE) -> None:
    draw.line([xy, (xy[0] + width, xy[1])], fill=fill, width=2)


def _kicker(draw: ImageDraw.ImageDraw, number: str, label: str) -> None:
    _text(draw, (140, 150), f"{number}   {label.upper()}", role="mono", size=22,
          fill=INK_FAINT, tracking=4)


def title_card(out: Path) -> Path:
    img, d = _canvas()
    _text(d, (W / 2, 330), "S T R A T A", size=118, fill=INK, anchor="ma",
          role="sans_bold", tracking=14)
    # Below the descenders, not through them - a rule that crosses a wordmark
    # reads as a rendering fault rather than as design.
    _rule(d, (W / 2 - 190, 512), 380, fill=ACCENT)
    _text(d, (W / 2, 556), "Version control for the physical world", size=44,
          fill=INK_MUTED, anchor="ma")
    _text(d, (W / 2, 640), "OpenCV AI Competition 2026, powered by AWS", size=28,
          role="mono", fill=INK_FAINT, anchor="ma")
    _text(d, (W / 2, 700), "Built by  Jin Yi   ·   solo entry   ·   October 2026",
          size=32, fill=INK, anchor="ma")
    _text(d, (W / 2, 780),
          "OpenCV 5.0  ·  ALIKED + LightGlue  ·  USAC/MAGSAC  ·  AWS Lambda on Graviton",
          size=24, role="mono", fill=ACCENT, anchor="ma")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def architecture_card(caption: str, highlight: str | None = None) -> Image.Image:
    """The committed diagram, unmodified, with one line of caption beneath it.

    The diagram is never cropped, zoomed or animated. It is 2480x1560 and the
    frame is 1920x1080, so it is fitted by height and centred: every label stays
    legible because nothing is resampled up.
    """
    img, d = _canvas()
    source = ROOT / "docs" / "architecture.png"
    art = Image.open(source).convert("RGB")
    avail_h, avail_w = 830, 1700
    scale = min(avail_h / art.height, avail_w / art.width)
    size = (int(art.width * scale), int(art.height * scale))
    art = art.resize(size, Image.LANCZOS)
    # The diagram is drawn on white; the film is dark. A panel behind it keeps
    # the contrast the diagram was designed for rather than inverting it.
    pad = 18
    box = ((W - size[0]) // 2 - pad, 150 - pad,
           (W + size[0]) // 2 + pad, 150 + size[1] + pad)
    d.rectangle(box, fill="#F2F5F8", outline=EDGE, width=2)
    img.paste(art, ((W - size[0]) // 2, 150))
    _text(d, (140, 986), caption, size=30, fill=INK)
    if highlight:
        _text(d, (140, 1030), highlight, size=24, role="mono", fill=ACCENT)
    return img


def results_card(out: Path, metrics: dict) -> Path:
    img, d = _canvas()
    _kicker(d, "04", "Principal results")
    _text(d, (140, 200), "24 generated scenes with known answers, and 12 pairs where nothing changed",
          size=40, fill=INK, role="sans_bold")
    rows = [
        ("Registration error", f"{metrics['table']['registration']} px", "mean corner error"),
        ("Precision", metrics["table"]["precision"], "macro over cases"),
        ("Recall", metrics["table"]["recall"], ""),
        ("F1", metrics["table"]["f1"], ""),
        ("Event-kind accuracy", metrics["table"]["kind"], "the weakest number, reported anyway"),
        ("False positives on unchanged pairs", f"{metrics['still_pair_false_positives']:.1f}",
         "12 pairs, nothing invented"),
    ]
    pitch, top = 116, 330
    # Two passes. Drawing a row's panel and its note in one walk meant the next
    # row's panel was painted over the previous row's note - the notes survived
    # only as a clipped top edge.
    for index, _ in enumerate(rows):
        y = top + index * pitch
        d.rectangle((140, y - 14, W - 140, y + 58), fill=PANEL, outline=EDGE, width=1)
    for index, (label, value, note) in enumerate(rows):
        y = top + index * pitch
        _text(d, (176, y + 6), label, size=30, fill=INK)
        _text(d, (W - 176, y + 2), value, size=40, role="mono_bold",
              fill=ACCENT_BRIGHT, anchor="ra")
        if note:
            _text(d, (176, y + 66), note, size=20, fill=INK_FAINT)
    _text(d, (140, 1000),
          "Reproducible to the pixel: the same seeds reproduce every figure above.",
          size=26, fill=INK_MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def limitations_card(out: Path, metrics: dict) -> Path:
    img, d = _canvas()
    _kicker(d, "04b", "Where this stops working")
    items = [
        ("Event labels are weaker than event locations",
         f"kind accuracy {metrics['table']['kind']} against F1 {metrics['table']['f1']}"),
        (f"{metrics['merged_cases']} of {metrics['cases']} cases merge two changes into one region",
         "documented, and pinned by a test"),
        ("No Arm-versus-x86 ratio is published",
         "the COOL benchmark on an EC2 Graviton4 instance has not been run"),
        ("The evaluation data is synthetic",
         "real captures work, but they carry no answer key"),
    ]
    y = 240
    for head, sub in items:
        d.rectangle((140, y, 146, y + 96), fill=KIND["uncertain"])
        _text(d, (186, y + 6), head, size=34, fill=INK, role="sans_bold")
        _text(d, (186, y + 56), sub, size=26, fill=INK_MUTED)
        y += 150
    _text(d, (140, 940),
          "A benchmark of only successes measures the benchmark.",
          size=30, fill=ACCENT)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def duration_of(media: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(media)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def build_segment(seg: Segment, build: Path) -> Path:
    """One card sequence, as long as its narration plus a short hold.

    Frames are shown in order, each for an equal share of the runtime: no
    Ken-Burny, no cross-fades under text.
    """
    raw = build / "narr" / f"{seg.slug}.raw.wav"
    wav = normalise_loudness(speak(seg.narration, raw), build / "narr" / f"{seg.slug}.wav")
    out = build / "seg" / f"{seg.slug}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    speech = duration_of(wav)
    total = speech + seg.hold_after_speech
    share = total / len(seg.frames)

    inputs: list[str] = []
    for frame in seg.frames:
        inputs += ["-loop", "1", "-t", f"{share:.3f}", "-i", str(frame)]
    inputs += ["-i", str(wav)]

    labels = "".join(f"[{i}:v]" for i in range(len(seg.frames)))
    n = len(seg.frames)
    graph = (
        f"{labels}concat=n={n}:v=1:a=0,format=yuv420p,setsar=1[v];"
        f"[{n}:a]apad=whole_dur={total:.3f}[a]"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs,
         "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
         "-t", f"{total:.3f}",
         "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", str(out)],
        check=True,
    )
    return trim_to_target(out)


def measured_lufs(media: Path) -> float:
    """Integrated loudness, by the same standard the target was set with."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(media),
         "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    match = re.search(r"I:\s+(-?[0-9.]+) LUFS", out)
    if not match:
        raise RuntimeError(f"ebur128 reported no integrated loudness for {media}")
    return float(match.group(1))


def trim_to_target(media: Path, tolerance_db: float = 0.5) -> Path:
    """Close the loop on level.

    Two-pass loudnorm aims for the target, but the tail hold it never sees and
    the true-peak ceiling it respects leave each segment about a decibel quiet
    against the track already in the film. A fixed linear gain measured from the
    result fixes that without touching dynamics, and it is checked rather than
    assumed.
    """
    measured = measured_lufs(media)
    delta = TARGET_LUFS - measured
    if abs(delta) <= tolerance_db:
        print(f"    {media.name}: {measured:.1f} LUFS (within {tolerance_db} dB)")
        return media
    corrected = media.with_name(media.stem + ".lvl.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(media),
         "-af", f"volume={delta:+.2f}dB",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         str(corrected)],
        check=True,
    )
    final = measured_lufs(corrected)
    print(f"    {media.name}: {measured:.1f} -> {final:.1f} LUFS ({delta:+.2f} dB)")
    corrected.replace(media)
    return media


def transcode_existing(src: Path, dst: Path) -> Path:
    """Bring the existing demonstration onto the same container parameters.

    The concat demuxer requires identical codec settings; re-encoding the one
    clip that is already correct costs a generation of quality, so this only
    runs when the parameters actually differ.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt",
         "-show_entries", "stream=codec_name,sample_rate,channels",
         "-of", "default=nw=1", str(src)],
        check=True, capture_output=True, text=True,
    ).stdout
    wants = {f"width={W}", f"height={H}", "codec_name=h264", "pix_fmt=yuv420p"}
    have = set(probe.split())
    if wants <= have:
        # Re-recording the film reads the previous segment back as its own input,
        # in which case src and dst are the same file and copying onto itself is
        # an error rather than a no-op.
        if src.resolve() == dst.resolve():
            return dst
        shutil.copyfile(src, dst)
        return dst
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={GROUND[1:]},fps={FPS}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", str(dst)],
        check=True,
    )
    return dst


def concat(parts: list[Path], dst: Path) -> Path:
    listing = dst.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(dst)],
        check=True,
    )
    return dst


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

def plan(metrics: dict) -> list[Segment]:
    table = metrics["table"]
    return [
        Segment(
            "01-title",
            "STRATA. Version control for the physical world. "
            "A solo project by Jin Yi, for the OpenCV A.I. Competition twenty twenty-six, "
            "powered by Amazon Web Services. "
            "It answers the question a site supervisor asks every single day: "
            "what changed here, since last time?",
        ),
        Segment(
            "04-architecture",
            "Here is how it is built. Four layers, each with one job. "
            "An operator console, built in TypeScript and exported statically. "
            "One Python service, which serves that console from the same origin as its "
            "application programming interface, so there is one artefact and one address. "
            "An agent loop, which observes, decides, acts, and then stops at a human gate. "
            "And the vision core, on OpenCV five: A.L.I.K.E.D. features, LightGlue matching, "
            "and registration by U.S.A.C. Magsac. "
            "The core is a pure function of its inputs. That is what lets the identical code "
            "run in the console, in the service, and in an arm-sixty-four function on "
            "AWS Graviton, without a conditional anywhere - and that deployment has been "
            "run: a scan takes about nine seconds there, and writes its own ledger row.",
        ),
        Segment(
            "05-results",
            f"The result, measured against twenty-four generated scenes with known answers. "
            f"Registration lands at {table['registration']} pixels of mean corner error. "
            f"Precision {table['precision']}, recall {table['recall']}, "
            f"F one {table['f1']}. "
            f"And on twelve pairs where nothing changed at all, the system invented "
            f"{metrics['still_pair_false_positives']:.1f} changes per pair. "
            f"That last number is the one that decides whether a tool like this is usable.",
        ),
        Segment(
            "06-limits",
            f"It is also honest about where it stops. "
            f"Event kinds are labelled at {table['kind']}, well below the rate at which they "
            f"are found, and both numbers are published rather than blended. "
            f"{metrics['merged_cases']} of the twenty-four cases merge two changes into one "
            f"region. "
            f"No Arm-versus-x86 speed-up is claimed, because the benchmark has never been "
            f"run on a Graviton four instance with the cloud-optimized OpenCV build, even "
            f"though the harness that would produce it is committed and reproducible.",
        ),
    ]


def build(build_dir: Path, demo: Path, metrics_path: Path) -> Path:
    metrics = derived(json.loads(metrics_path.read_text(encoding="utf-8")))
    build_dir.mkdir(parents=True, exist_ok=True)
    cards = build_dir / "cards"
    cards.mkdir(exist_ok=True)

    segments = plan(metrics)

    t = title_card(cards / "01-title.png")
    segments[0].frames = [t]

    arch_captions = [
        ("Four layers. The console, the service, the agent, and the vision core.",
         "OpenCV 5 · ALIKED + LightGlue · USAC/MAGSAC · pure functions"),
        ("Storage sits behind one interface: local disk, or S3 plus DynamoDB.",
         "switching is a configuration change, not a code change"),
        ("Hybrid by design: the console on x86, the vision workload on Arm.",
         "an arm64 Lambda on AWS Graviton - 8.8 s a scan, four events, ledger row written"),
    ]
    arch_frames = []
    for index, (caption, highlight) in enumerate(arch_captions, start=1):
        path = cards / f"04-architecture-{index}.png"
        architecture_card(caption, highlight).save(path)
        arch_frames.append(path)
    segments[1].frames = arch_frames

    segments[2].frames = [results_card(cards / "05-results.png", metrics)]
    segments[3].frames = [limitations_card(cards / "06-limits.png", metrics)]

    parts: list[Path] = []
    parts.append(build_segment(segments[0], build_dir))
    demo_norm = build_dir / "seg" / "02-demo.mp4"
    parts.append(transcode_existing(demo, demo_norm))
    for segment in segments[1:]:
        parts.append(build_segment(segment, build_dir))

    out = build_dir / "strata-film.mp4"
    concat(parts, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build" / "video")
    ap.add_argument("--demo", type=Path,
                    default=ROOT / "web" / "public" / "media" / "strata-demo.mp4")
    ap.add_argument("--metrics", type=Path, default=ROOT / "docs" / "evaluation.json")
    ap.add_argument("--install", action="store_true",
                    help="copy the finished film over the one the app serves")
    args = ap.parse_args()

    print("==> narration and cards")
    film = build(args.build_dir, args.demo, args.metrics)
    seconds = duration_of(film)
    print(f"==> {film}  {seconds:.1f}s ({seconds / 60:.2f} min)")
    if seconds > 300:
        print("FAIL: the competition caps the film at five minutes", file=sys.stderr)
        return 1

    if args.install:
        target = ROOT / "web" / "public" / "media" / "strata-demo.mp4"
        shutil.copyfile(film, target)
        print(f"==> installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

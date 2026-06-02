#!/usr/bin/env python3
"""
audiosync - pair separate WAV recordings to video clips and find the sync offset.

Made for the run-it-once workflow: point it at a folder of footage (video files
plus the WAV files from your recorder), and it figures out which WAV belongs to
which clip and how far off they are, frame-accurate.

How it works
------------
1. Scans the folder (and any subfolders) for video + WAV files.
2. Reads each file's recording time - real timecode (SMPTE/BWF) if present,
   otherwise the embedded creation time, otherwise the file's modified time.
3. Pairs a WAV to a video when their recording times are close.
4. For each pair, pulls the video's scratch (camera-mic) audio and cross-
   correlates it against the WAV to find the exact offset - this is what gets
   you frame accuracy, not the timestamps.
5. Prints a table, and optionally writes a CSV and/or copies matched pairs into
   tidy per-clip folders so they line up in your NLE.

It NEVER deletes or overwrites your originals. --apply only COPIES into a new
'synced/' folder.

Usage
-----
    python3 audiosync.py /path/to/footage
    python3 audiosync.py /path/to/footage --report sync.csv
    python3 audiosync.py /path/to/footage --apply
    python3 audiosync.py /path/to/footage --fps 25 --window 300

Do you even need this?
----------------------
Probably not. This script is the heavy-duty option, only worth installing for:
  - .mxf files, or
  - 4K clips too big for a browser tab to handle.
For everything normal (iPhone .mov/.mp4 + WAV), use the browser app instead -
no Python, no ffmpeg, no install: just open audiosync.html and drop files in.

Don't want to install Python at all? Upload the footage to Claude and ask it to
run this for you. Fine for the odd awkward file; big video uploads are slow, so
it's not for routine card dumps. Only install the stack below if this becomes a
regular, every-shoot job.

Requires: Python 3.9+, numpy, and ffmpeg/ffprobe on your PATH.
  macOS:  brew install ffmpeg python && pip3 install numpy
"""

import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".m4v", ".avi", ".mts", ".m2ts", ".mpg", ".mkv"}
AUDIO_EXTS = {".wav"}

# Sample rate we resample everything down to before correlating. Low enough to
# be fast, high enough to keep a clean alignment peak.
ANALYSIS_SR = 4000


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    path: str
    kind: str  # "video" or "audio"
    duration: float = 0.0
    start_time: float | None = None  # POSIX seconds, best estimate
    time_source: str = "unknown"  # "timecode" | "creation" | "mtime"
    timecode: str | None = None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class Match:
    video: Clip
    audio: Clip
    offset_s: float  # +ve => WAV was rolling first; slide the WAV LATER by this much to align
    confidence: float  # 0..1, normalised correlation at best lag
    time_gap: float | None = None  # seconds between the two recording starts


# --------------------------------------------------------------------------- #
# ffprobe / ffmpeg helpers
# --------------------------------------------------------------------------- #
def _check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(
                f"error: '{tool}' not found on PATH.\n"
                f"Install ffmpeg first (macOS: brew install ffmpeg)."
            )


def probe_video(path: str) -> Clip:
    """Read duration, creation time and any embedded timecode from a video."""
    clip = Clip(path=path, kind="video")
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, text=True, check=True,
        ).stdout
        meta = json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        meta = {}

    fmt = meta.get("format", {})
    try:
        clip.duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        clip.duration = 0.0

    tags = {**fmt.get("tags", {})}
    for stream in meta.get("streams", []):
        tags.update(stream.get("tags", {}))
        if stream.get("timecode"):
            clip.timecode = stream["timecode"]
    if not clip.timecode and tags.get("timecode"):
        clip.timecode = tags["timecode"]

    creation = tags.get("creation_time")
    if creation:
        ts = _parse_iso(creation)
        if ts is not None:
            clip.start_time, clip.time_source = ts, "creation"

    if clip.start_time is None:
        clip.start_time, clip.time_source = os.path.getmtime(path), "mtime"
    return clip


def load_audio(path: str) -> np.ndarray:
    """
    Decode any media file's audio to a mono float array at ANALYSIS_SR.
    Routed through ffmpeg so it handles every WAV flavour (16/24-bit PCM AND
    32-bit float, e.g. DJI Mic internal recordings) plus the scratch track of
    video files. Float values may exceed +/-1 with 32-bit float; that's fine,
    the envelope/correlation normalise it away.
    """
    proc = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", path,
         "-vn", "-ac", "1", "-ar", str(ANALYSIS_SR),
         "-f", "f32le", "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# --------------------------------------------------------------------------- #
# WAV / BWF helpers
# --------------------------------------------------------------------------- #
def _parse_iso(s: str) -> float | None:
    s = s.strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def read_bext(path: str) -> tuple[str | None, float | None, int | None]:
    """
    Parse the broadcast-wave 'bext' chunk for origination date/time and the
    sample-accurate TimeReference. Pure stdlib byte reading, no deps.
    Returns (origination_iso, posix_start, time_reference_samples).
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"RIFF":
                return None, None, None
            f.read(4)
            if f.read(4) != b"WAVE":
                return None, None, None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid, size = struct.unpack("<4sI", hdr)
                if cid == b"bext":
                    data = f.read(size)
                    # description[256] originator[32] originatorRef[32]
                    # origDate[10] origTime[8] timeRef(low u32, high u32)
                    date = data[320:330].decode("ascii", "ignore").strip("\x00 ")
                    time = data[330:338].decode("ascii", "ignore").strip("\x00 ")
                    tref_lo, tref_hi = struct.unpack_from("<II", data, 338)
                    tref = (tref_hi << 32) | tref_lo
                    iso = posix = None
                    if date and time:
                        iso = f"{date.replace(':', '-')}T{time}"
                        posix = _parse_iso(iso)
                    return iso, posix, tref
                f.seek(size + (size & 1), os.SEEK_CUR)  # chunks are word-aligned
    except OSError:
        pass
    return None, None, None


def probe_audio(path: str) -> Clip:
    clip = Clip(path=path, kind="audio")
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", path],
            capture_output=True, text=True, check=True,
        ).stdout
        clip.duration = float(json.loads(out).get("format", {}).get("duration", 0.0))
    except (subprocess.CalledProcessError, json.JSONDecodeError, TypeError, ValueError):
        clip.duration = 0.0

    iso, posix, tref = read_bext(path)
    if posix is not None:
        clip.start_time, clip.time_source = posix, "creation"
        clip.timecode = iso
    if clip.start_time is None:
        clip.start_time, clip.time_source = os.path.getmtime(path), "mtime"
    return clip


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if x.size == 0 or sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, x.size - 1, n_out)
    return np.interp(idx, np.arange(x.size), x).astype(np.float32)


# --------------------------------------------------------------------------- #
# Sync maths
# --------------------------------------------------------------------------- #
def _envelope(x: np.ndarray) -> np.ndarray:
    """Loudness envelope: correlate on energy, robust to differing mics/levels."""
    if x.size == 0:
        return x
    e = np.abs(x)
    win = max(1, ANALYSIS_SR // 100)  # ~10 ms smoothing
    e = np.convolve(e, np.ones(win) / win, mode="same")
    e -= e.mean()
    norm = np.linalg.norm(e)
    return e / norm if norm else e


def find_offset(video_audio: np.ndarray, wav_audio: np.ndarray) -> tuple[float, float]:
    """
    Cross-correlate two clips. Returns (offset_seconds, confidence 0..1).
    offset > 0  =>  the WAV recorder started rolling BEFORE the camera, so its
    content sits earlier; slide the WAV later by `offset` to line them up.
    """
    a, b = _envelope(video_audio), _envelope(wav_audio)
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0

    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    corr = np.concatenate((corr[-(b.size - 1):], corr[:a.size]))
    lags = np.arange(-(b.size - 1), a.size)

    peak = int(np.argmax(corr))
    best = corr[peak]
    confidence = float(np.clip(best / (np.std(corr) * 4 + 1e-9), 0, 1))
    offset_samples = lags[peak]
    return offset_samples / ANALYSIS_SR, confidence


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def pair_clips(videos: list[Clip], audios: list[Clip], window: float,
               min_conf: float) -> tuple[list[Match], list[Clip], list[Clip]]:
    matches: list[Match] = []
    used_audio: set[str] = set()

    # Cache the analysis audio so we only decode each file once.
    vcache: dict[str, np.ndarray] = {}
    acache: dict[str, np.ndarray] = {}

    for v in videos:
        # Candidate WAVs: those whose recording time is within `window` seconds.
        cands = []
        for a in audios:
            if a.path in used_audio:
                continue
            gap = None
            if v.start_time is not None and a.start_time is not None:
                gap = abs(v.start_time - a.start_time)
                if gap > window:
                    continue
            cands.append((a, gap))
        if not cands:
            continue

        if v.path not in vcache:
            vcache[v.path] = load_audio(v.path)
        va = vcache[v.path]

        best: Match | None = None
        for a, gap in cands:
            if a.path not in acache:
                acache[a.path] = load_audio(a.path)
            offset, conf = find_offset(va, acache[a.path])
            if best is None or conf > best.confidence:
                best = Match(v, a, offset, conf, gap)
        if best and best.confidence >= min_conf:
            matches.append(best)
            used_audio.add(best.audio.path)

    matched_a = {m.audio.path for m in matches}
    matched_v = {m.video.path for m in matches}
    orphan_v = [v for v in videos if v.path not in matched_v]
    orphan_a = [a for a in audios if a.path not in matched_a]
    return matches, orphan_v, orphan_a


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def fmt_tc(seconds: float, fps: float) -> str:
    sign = "-" if seconds < 0 else "+"
    s = abs(seconds)
    f = int(round((s - int(s)) * fps))
    s = int(s)
    return f"{sign}{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}:{f:02d}"


def print_table(matches: list[Match], orphan_v, orphan_a, fps: float) -> None:
    if matches:
        print(f"\n  Matched {len(matches)} pair(s):\n")
        vw = max(len(m.video.name) for m in matches)
        aw = max(len(m.audio.name) for m in matches)
        print(f"  {'VIDEO'.ljust(vw)}   {'WAV'.ljust(aw)}   "
              f"{'OFFSET (' + str(int(fps)) + 'fps)':>14}   CONF")
        print("  " + "-" * (vw + aw + 28))
        for m in sorted(matches, key=lambda x: -x.confidence):
            bar = "high" if m.confidence > 0.6 else \
                  "ok" if m.confidence > 0.35 else "LOW - check"
            print(f"  {m.video.name.ljust(vw)}   {m.audio.name.ljust(aw)}   "
                  f"{fmt_tc(m.offset_s, fps):>14}   {m.confidence:.2f} {bar}")
    else:
        print("\n  No confident matches found.")

    if orphan_v:
        print(f"\n  Video with no matched WAV ({len(orphan_v)}):")
        for c in orphan_v:
            print(f"    - {c.name}")
    if orphan_a:
        print(f"\n  WAV with no matched video ({len(orphan_a)}):")
        for c in orphan_a:
            print(f"    - {c.name}")
    print()


def write_csv(matches: list[Match], path: str, fps: float) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "wav", "offset_seconds", "offset_timecode",
                    "offset_frames", "confidence", "time_gap_seconds"])
        for m in matches:
            w.writerow([m.video.name, m.audio.name, f"{m.offset_s:.4f}",
                        fmt_tc(m.offset_s, fps), int(round(m.offset_s * fps)),
                        f"{m.confidence:.3f}",
                        "" if m.time_gap is None else f"{m.time_gap:.1f}"])
    print(f"  Wrote report: {path}")


def apply_layout(matches: list[Match], root: str) -> None:
    """Copy each matched pair into synced/<clip>/ with matching names. Non-destructive."""
    out = os.path.join(root, "synced")
    os.makedirs(out, exist_ok=True)
    for m in matches:
        stem = os.path.splitext(m.video.name)[0]
        folder = os.path.join(out, stem)
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(m.video.path, os.path.join(folder, m.video.name))
        wav_ext = os.path.splitext(m.audio.name)[1]
        shutil.copy2(m.audio.path, os.path.join(folder, stem + wav_ext))
    print(f"  Copied {len(matches)} pair(s) into: {out}  (originals untouched)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def scan(folder: str) -> tuple[list[Clip], list[Clip]]:
    videos, audios = [], []
    paths = []
    for root, dirs, files in os.walk(folder):
        # skip hidden dirs and our own output folder
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() != "synced"]
        for name in files:
            if not name.startswith("."):
                paths.append(os.path.join(root, name))
    for path in sorted(paths, key=lambda p: os.path.basename(p)):
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTS:
            videos.append(probe_video(path))
        elif ext in AUDIO_EXTS:
            audios.append(probe_audio(path))
    return videos, audios


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pair WAV audio to video clips and find the sync offset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "do you even need this?\n"
            "  Probably not. This is the heavy-duty option - only worth it for .mxf files\n"
            "  or 4K clips too big for a browser tab. For normal jobs (iPhone .mov/.mp4 +\n"
            "  WAV) use the browser app (audiosync.html): no Python, no install.\n"
            "  No install at all? Upload the footage to Claude and ask it to run this for\n"
            "  you - fine for the odd file, slow for whole cards.\n"
        ),
    )
    p.add_argument("folder", help="folder containing the video and WAV files")
    p.add_argument("--fps", type=float, default=25.0, help="frame rate for offset display (default 25)")
    p.add_argument("--window", type=float, default=600.0,
                   help="max seconds between recording times to consider a pair (default 600)")
    p.add_argument("--min-conf", type=float, default=0.25,
                   help="minimum confidence to accept a match (0-1, default 0.25)")
    p.add_argument("--report", metavar="CSV", help="write a CSV report to this path")
    p.add_argument("--apply", action="store_true",
                   help="copy matched pairs into synced/<clip>/ folders (never deletes originals)")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"error: not a folder: {args.folder}")
    _check_tools()

    videos, audios = scan(args.folder)
    print(f"\n  Found {len(videos)} video file(s) and {len(audios)} WAV file(s).")
    if not videos or not audios:
        sys.exit("  Need at least one of each to do anything useful.")

    print("  Analysing audio and matching...")
    matches, orphan_v, orphan_a = pair_clips(videos, audios, args.window, args.min_conf)
    print_table(matches, orphan_v, orphan_a, args.fps)

    if args.report:
        write_csv(matches, args.report, args.fps)
    if args.apply and matches:
        apply_layout(matches, args.folder)


if __name__ == "__main__":
    main()

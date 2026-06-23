"""Scene-change frame extraction helpers for video captioning."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_PTS_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def scene_threshold() -> float:
    return float(os.getenv("VIDEO_FRAME_SCENE_THRESHOLD", "0.35"))


def max_frames_per_video() -> int:
    return int(os.getenv("VIDEO_FRAME_MAX_PER_VIDEO", "60"))


def _select_evenly_spaced_indices(total: int, keep: int) -> list[int]:
    if keep <= 0 or total <= 0:
        return []
    if keep >= total:
        return list(range(total))
    if keep == 1:
        return [0]
    last = total - 1
    indices = []
    for i in range(keep):
        idx = round(i * last / (keep - 1))
        indices.append(min(last, max(0, idx)))
    return sorted(set(indices))


def cap_frames_evenly(frames: list[dict], max_count: int) -> list[dict]:
    if max_count <= 0:
        return []
    if len(frames) <= max_count:
        return frames
    idx = _select_evenly_spaced_indices(len(frames), max_count)
    return [frames[i] for i in idx]


def extract_scene_frames(
    video_path: Path,
    frame_dir: Path,
    *,
    threshold: float | None = None,
) -> list[dict]:
    """Extract scene-change frames and return [{'path': Path, 'start_seconds': float}]."""
    threshold = scene_threshold() if threshold is None else threshold
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frame_dir / "raw_%06d.png"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-vsync",
            "vfr",
            str(output_pattern),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    pts_times = [float(match.group(1)) for match in _PTS_TIME_RE.finditer(proc.stderr)]
    files = sorted(frame_dir.glob("raw_*.png"))
    paired = []
    for idx, frame_file in enumerate(files):
        ts = pts_times[idx] if idx < len(pts_times) else 0.0
        pretty_name = f"{ts:08.3f}.png"
        final_path = frame_dir / pretty_name
        if frame_file != final_path:
            frame_file.replace(final_path)
        paired.append({"path": final_path, "start_seconds": ts})
    return paired

"""Frame-by-frame driver that stitches GPU renders into a GIF.

The animator is intentionally tiny: callers pass a ``render_frame`` callback
that returns an ``(H, W, 3) uint8`` image for a given frame index, plus a
frame count and an output path. Each frame is saved as a numbered PNG into
a staging directory and then mimsaved into a GIF (or any other container
imageio supports).

Why save frames to disk first instead of holding them all in memory?
800 × 600 × 3 bytes per frame is ~1.4 MiB, which is fine in RAM for ~120
frames, but checkpointing PNGs lets a long render survive a crash and the
final GIF can be re-stitched without re-rendering. Pass ``keep_frames=True``
to preserve the staging directory, or ``False`` (default) to clean it up.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image


FrameRenderFn = Callable[[int, int], NDArray[np.uint8]]


class Animator:
    """Frame loop with progress reporting and GIF assembly.

    Parameters
    ----------
    render_frame:
        Callable ``(frame_idx, total_frames) -> (H, W, 3) uint8`` invoked
        once per frame. The callable owns scene setup and renderer reuse;
        the animator only sequences calls and writes files.
    frame_count:
        Number of frames to render.
    output_path:
        Final animation path. Extension picks the container — ``.gif`` for a
        GIF (default), ``.mp4`` if ``imageio-ffmpeg`` is installed.
    fps:
        Frames per second written into the container metadata.
    staging_dir:
        Directory to dump per-frame PNGs into. Default is a sibling of
        ``output_path`` named ``<output_stem>_frames``.
    keep_frames:
        If ``True``, leave the staging directory in place after stitching.
        Default ``False`` deletes it on success.
    resume:
        If ``True``, frames whose PNG already exists in ``staging_dir`` are
        loaded from disk instead of re-rendered. Lets a long animation be
        run in multiple sessions. Default ``True``.
    max_frames_per_run:
        Optional cap on how many frames to render in this call (already-
        existing frames don't count). Useful when each session has a hard
        wall-clock budget. ``None`` means render to completion.
    label:
        Optional callable ``(frame_idx, total_frames) -> str`` whose return
        value is overlaid in the top-left corner of each frame.
    """

    def __init__(
        self,
        render_frame: FrameRenderFn,
        frame_count: int,
        output_path: str | Path,
        *,
        fps: int = 24,
        staging_dir: str | Path | None = None,
        keep_frames: bool = False,
        resume: bool = True,
        max_frames_per_run: int | None = None,
        label: Callable[[int, int], str] | None = None,
    ) -> None:
        if frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count}.")
        self.render_frame: FrameRenderFn = render_frame
        self.frame_count: int = int(frame_count)
        self.output_path: Path = Path(output_path)
        self.fps: int = int(fps)
        self.staging_dir: Path = (
            Path(staging_dir)
            if staging_dir is not None
            else self.output_path.with_name(self.output_path.stem + "_frames")
        )
        self.keep_frames: bool = bool(keep_frames)
        self.resume: bool = bool(resume)
        self.max_frames_per_run: int | None = max_frames_per_run
        self.label: Callable[[int, int], str] | None = label

    def _save_frame(self, idx: int, frame: NDArray[np.uint8]) -> Path:
        path = self.staging_dir / f"frame_{idx:05d}.png"
        img = Image.fromarray(frame, mode="RGB")
        if self.label is not None:
            from PIL import ImageDraw, ImageFont

            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except OSError:
                font = ImageFont.load_default()
            text = self.label(idx, self.frame_count)
            draw = ImageDraw.Draw(img)
            draw.text((10, 8), text, fill=(230, 230, 230), font=font)
        img.save(path)
        return path

    def _stitch(self, paths: list[Path]) -> None:
        """Read the staged PNGs back and write the final container."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.output_path.suffix.lower()
        frames = [imageio.imread(p) for p in paths]
        if suffix == ".gif":
            # ``duration`` is in seconds per frame for the v2 GIF writer.
            imageio.mimsave(
                self.output_path, frames, duration=1.0 / self.fps, loop=0
            )
        else:
            imageio.mimsave(self.output_path, frames, fps=self.fps)
        size_mb = self.output_path.stat().st_size / (1024 * 1024)
        print(
            f"Wrote {self.output_path} ({len(frames)} frames, {size_mb:.1f} MiB)"
        )

    def _frame_path(self, idx: int) -> Path:
        return self.staging_dir / f"frame_{idx:05d}.png"

    def run(self) -> Path:
        """Render missing frames, save them as PNGs, stitch when complete.

        If ``resume`` is True, frames whose PNG already exists are skipped.
        If ``max_frames_per_run`` is set, the call returns early after
        rendering that many fresh frames and the GIF is *not* stitched —
        run the animator again to fill in the rest.
        """
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.name for p in self.staging_dir.glob("frame_*.png")} if self.resume else set()
        if existing:
            print(
                f"Animator: resuming with {len(existing)} of {self.frame_count} "
                f"frames already on disk in {self.staging_dir}"
            )
        else:
            print(
                f"Animator: {self.frame_count} frames -> {self.output_path} "
                f"@ {self.fps} fps via {self.staging_dir}"
            )

        paths: list[Path] = []
        rendered_this_run: int = 0
        budget = self.max_frames_per_run
        wall_start = time.perf_counter()
        for i in range(self.frame_count):
            path = self._frame_path(i)
            if path.name in existing:
                paths.append(path)
                continue
            if budget is not None and rendered_this_run >= budget:
                print(
                    f"Animator: hit max_frames_per_run={budget}; "
                    f"{self.frame_count - i} frames remaining. Re-run to continue."
                )
                return self.staging_dir

            frame_start = time.perf_counter()
            frame = self.render_frame(i, self.frame_count)
            saved = self._save_frame(i, frame)
            paths.append(saved)
            rendered_this_run += 1
            frame_elapsed = time.perf_counter() - frame_start
            elapsed = time.perf_counter() - wall_start
            avg = elapsed / rendered_this_run
            remaining = (
                self.frame_count - i - 1
                if budget is None
                else min(self.frame_count - i - 1, budget - rendered_this_run)
            )
            eta = avg * remaining
            print(
                f"  frame {i + 1:>4}/{self.frame_count}  "
                f"{frame_elapsed:5.1f}s  total {elapsed:6.1f}s  ETA {eta:6.1f}s",
                flush=True,
            )

        # All frames present — stitch the final container.
        self._stitch(paths)
        if not self.keep_frames:
            shutil.rmtree(self.staging_dir, ignore_errors=True)
        return self.output_path

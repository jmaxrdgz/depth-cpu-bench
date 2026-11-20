#!/usr/bin/env python3
"""Run a ZipDepth ncnn bundle on an OpenCV camera, video file, or stream."""
import argparse
from collections import deque
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'export/models/zipdepth')
    parser.add_argument('--source', default='0', help='Camera index, file path, or stream URL')
    parser.add_argument('--threads', type=positive, default=2)
    parser.add_argument('--cores', help='Linux CPU affinity, e.g. 2,3 on RPi5')
    parser.add_argument('--resize', choices=['letterbox', 'stretch'], default='letterbox')
    parser.add_argument('--headless', action='store_true', help='Disable the display window')
    parser.add_argument('--output', type=Path, help='Save the side-by-side visualization as MP4')
    parser.add_argument('--frames', type=positive, help='Stop after this many frames')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if args.cores:
        if not hasattr(os, 'sched_setaffinity'):
            parser.error('--cores requires Linux; omit it on macOS')
        try:
            cores = {int(v) for v in args.cores.split(',')}
            if not cores or not cores <= os.sched_getaffinity(0):
                raise ValueError('cores must be available to this process')
            os.sched_setaffinity(0, cores)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

    # Import native libraries after affinity is set so worker threads inherit it.
    import cv2
    import numpy as np
    from inference.zipdepth_ncnn import ZipDepthNCNN, colorize

    cv2.setNumThreads(1)
    model = ZipDepthNCNN(args.model, threads=args.threads)
    source = int(args.source) if args.source.isdecimal() else args.source
    live = isinstance(source, int) or '://' in args.source
    capture = cv2.VideoCapture(source)
    writer = None
    count = 0
    times = deque(maxlen=30)
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video source: {args.source}')
        if live:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; backend dependent.
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30
        started = last_report = time.perf_counter()
        output_size = None
        while args.frames is None or count < args.frames:
            ok, frame = capture.read()
            if not ok:
                if live:
                    raise RuntimeError('Video feed stopped delivering frames')
                break
            depth, milliseconds = model.predict(frame, args.resize)
            times.append(milliseconds)
            view = np.concatenate([frame, colorize(depth)], axis=1)
            model_fps = 1000 / (sum(times) / len(times))
            cv2.putText(view, f'Relative inverse depth | model {model_fps:.1f} FPS | near = bright',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            # Video codecs require even dimensions.
            view = cv2.copyMakeBorder(view, 0, view.shape[0] % 2, 0, view.shape[1] % 2,
                                      cv2.BORDER_REPLICATE)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    output_size = (view.shape[1], view.shape[0])
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, output_size)
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot create video: {args.output}')
                if (view.shape[1], view.shape[0]) != output_size:
                    raise RuntimeError('Source dimensions changed while recording')
                writer.write(view)
            if not args.headless:
                cv2.imshow('ZipDepth: RGB / relative inverse depth', view)
                if cv2.waitKey(1) & 0xff in (27, ord('q')):
                    count += 1
                    break
            count += 1
            now = time.perf_counter()
            if now - last_report >= 1:
                print(f'{count} frames | model {model_fps:.1f} FPS | '
                      f'pipeline {count / (now - started):.1f} FPS', flush=True)
                last_report = now
        if count == 0:
            raise RuntimeError('Source contained no decodable frames')
        print(f'Processed {count} frames in {time.perf_counter() - started:.2f}s', flush=True)
    except KeyboardInterrupt:
        print(f'Stopped after {count} frames', flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

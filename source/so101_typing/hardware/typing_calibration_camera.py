"""Record the robot arm MJPEG feed beside a teleoperation recording.

Frame timestamps use the host monotonic clock at the MJPEG reader. They can be
compared with teleop sample monotonic timestamps, but are not camera exposure
or hardware synchronization timestamps.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
import time


CAMERAS = {
    'typing_calibration': ('http://127.0.0.1:8766/arm.mjpg', 'USB2.0 CAM1 arm view'),
}
FPS = 15.


class DualCameraCapture:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.stop = Event()
        self.workers = []
        self.errors = []

    def __enter__(self):
        import cv2
        try:
            for name, (url, label) in CAMERAS.items():
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f'{name}: cannot open {url}')
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    raise RuntimeError(f'{name}: cannot read first frame')
                height, width = frame.shape[:2]
                path = self.directory/f'{name}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                                         (width, height))
                if not writer.isOpened():
                    cap.release()
                    raise RuntimeError(f'{name}: cannot create {path}')
                stamp_path = self.directory/f'{name}_timestamps.jsonl'
                try:
                    stamps = stamp_path.open('x', buffering=1)
                except BaseException:
                    writer.release()
                    cap.release()
                    raise
                first_stamp = (time.monotonic_ns(), time.time_ns())
                worker = {'name': name, 'label': label, 'url': url, 'cap': cap,
                          'writer': writer, 'stamps': stamps, 'path': path,
                          'stamp_path': stamp_path, 'first_frame': frame,
                          'first_stamp': first_stamp, 'width': width, 'height': height,
                          'count': 0, 'thread': None}
                self.workers.append(worker)
            for worker in self.workers:
                thread = Thread(target=self._capture, args=(worker,), daemon=True,
                                name=f"record-{worker['name']}")
                worker['thread'] = thread
                thread.start()
            self.check()
            for worker in self.workers:
                print(f"{worker['name']}_recording={worker['path']}", flush=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _capture(self, worker):
        cap, writer, stamps = worker['cap'], worker['writer'], worker['stamps']
        frame, stamp = worker['first_frame'], worker['first_stamp']
        try:
            while not self.stop.is_set():
                if frame.shape[:2] != (worker['height'], worker['width']):
                    raise RuntimeError(f"{worker['name']}: frame size changed")
                writer.write(frame)
                stamps.write(json.dumps({'frame': worker['count'],
                                         'received_monotonic_ns': stamp[0],
                                         'received_unix_ns': stamp[1]})+'\n')
                worker['count'] += 1
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{worker['name']}: stream ended")
                stamp = (time.monotonic_ns(), time.time_ns())
        except BaseException as error:
            if not self.stop.is_set():
                self.errors.append(f"{worker['name']}: {type(error).__name__}: {error}")
        finally:
            writer.release()
            cap.release()
            stamps.close()
            (self.directory/f"{worker['name']}_metadata.json").write_text(json.dumps({
                'source': worker['label'], 'transport': worker['url'],
                'timestamp_semantics': 'host MJPEG frame receipt, not camera exposure time',
                'frame_count': worker['count'], 'width': worker['width'],
                'height': worker['height'], 'video_fps': FPS,
                'video': worker['path'].name,
                'frame_timestamps': worker['stamp_path'].name,
                'ended_monotonic_ns': time.monotonic_ns(),
            }, indent=2)+'\n')

    def check(self):
        if self.errors:
            raise RuntimeError('camera recording failed: '+'; '.join(self.errors))
        if any(w['thread'] and not w['thread'].is_alive() for w in self.workers):
            raise RuntimeError('camera recording thread stopped unexpectedly')

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        for worker in self.workers:
            thread = worker['thread']
            if thread:
                thread.join(timeout=3)
                if thread.is_alive():
                    worker['cap'].release()
                    thread.join(timeout=2)
            else:
                worker['writer'].release()
                worker['cap'].release()
                worker['stamps'].close()
        for worker in self.workers:
            print(f"{worker['name']}_saved={worker['path']} frames={worker['count']}", flush=True)
        if exc_type is None and self.errors:
            raise RuntimeError('camera recording failed: '+'; '.join(self.errors))
        return False

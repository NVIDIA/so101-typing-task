# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist-camera observations and deterministic CPU actor for physical typing."""

from __future__ import annotations

from dataclasses import dataclass, field

from .fixed_cartesian_policy import FixedCartesianTypingState
from so101_typing.vision_contract import supports_vision_task

import hashlib
import json
from pathlib import Path
from threading import Event, Lock, Thread
import time

import numpy as np


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PHRASE_CHARACTERS = ALPHABET + "0123456789 ."


@dataclass
class VisionTypingState(FixedCartesianTypingState):
    goal_characters: str = ALPHABET
    target_heights: dict = field(default_factory=dict)
    timeouts_s: tuple = (6., 2., 2.)

    @classmethod
    def from_environment(cls, target, profile, environment, characters):
        command = environment['commands']['typing']
        labels = command['slot_labels']
        mapped = ''.join(' ' if labels[i] == 'Space' else labels[i] for i in command['letter_slots'])
        if mapped != characters or characters not in (ALPHABET, PHRASE_CHARACTERS):
            raise ValueError('Environment key order differs from actor goal vocabulary')
        target = target.upper()
        fixed = command.get('fixed_target_slots')
        expected = ''.join(' ' if labels[i] == 'Space' else labels[i] for i in fixed) if fixed else None
        if (expected is not None and target != expected) or not target or any(c not in characters for c in target):
            raise ValueError('Target does not match the trained phrase or vocabulary')
        if len(target) != command['letter_length'][0] or len(target) != command['letter_length'][1]:
            raise ValueError('Target length differs from training')
        points = np.asarray(command['letter_xyz_b_m'], dtype=float)
        if points.shape != (len(characters), 3) or not np.isfinite(points).all():
            raise ValueError('Invalid trained target geometry')
        if not np.allclose(points[:26], profile.az_xyz_b_m, atol=1e-10, rtol=0):
            raise ValueError('Training key geometry differs from established physical mapping')
        # The established mapping uses world axes. Z is unchanged by this planar root rotation.
        root = environment['scene']['robot']['init_state']
        if abs(root['pos'][2]) > 1e-12 or abs(root['rot'][0]) > 1e-12 or abs(root['rot'][1]) > 1e-12:
            raise ValueError('Robot root changes the hardware clearance Z convention')
        clearance = float(command['clearance_m'])
        ticks = int(command['clearance_control_ticks'])
        timeouts = tuple(float(v) for v in command['phase_timeouts_s'])
        if not np.isfinite(clearance) or clearance <= 0 or ticks < 1 or len(timeouts) != 3 or not all(np.isfinite(t) and t > 0 for t in timeouts):
            raise ValueError('Invalid trained phase timing or clearance')
        return cls(target=target, profile=profile, goal_characters=characters,
                   target_heights=dict(zip(characters, points[:, 2])),
                   clearance_m=clearance, clearance_control_ticks=ticks, timeouts_s=timeouts)

    def active_target_z(self):
        return float(self.target_heights[self.active_letter()])

    def phase_timed_out(self, now_s):
        return self.phase_elapsed_s(now_s) > self.timeouts_s[self.phase]


def vision_observation(q_rad, velocity_rad_s, command_rad, state, now_s) -> np.ndarray:
    """Match vision_proprio in the Newton task, with physical keyboard phase."""
    q = np.asarray(q_rad, dtype=np.float32)
    velocity = np.asarray(velocity_rad_s, dtype=np.float32)
    command = np.asarray(command_rad, dtype=np.float32)
    reset = np.asarray(state.profile.q_reset_rad, dtype=np.float32)
    if any(x.shape != (5,) or not np.isfinite(x).all() for x in (q, velocity, command, reset)):
        raise ValueError("vision observation requires finite five-joint inputs")
    letter = state.active_letter()
    characters = getattr(state, "goal_characters", ALPHABET)
    if len(letter) != 1 or letter not in characters:
        raise ValueError(f"invalid active letter {letter!r}")
    goal = np.zeros(len(characters), dtype=np.float32)
    goal[characters.index(letter)] = 1.0
    phase = np.zeros(3, dtype=np.float32)
    phase[state.phase] = 1.0
    result = np.concatenate((q - reset, velocity, command - reset, goal,
                             phase, np.asarray([state.phase_elapsed_s(now_s)], dtype=np.float32)))
    if result.shape != (19 + len(characters),) or not np.isfinite(result).all():
        raise ValueError("invalid vision observation")
    return result


class VisionActor:
    def __init__(self, actor_path: Path, metadata_path: Path, environment_path: Path):
        import torch

        actor_path, metadata_path, environment_path = map(Path, (actor_path, metadata_path, environment_path))
        meta = json.loads(metadata_path.read_text())
        env = json.loads(environment_path.read_text())
        if meta.get("kind") != "vision_cpu_export" or meta.get("actor_sha256") != hashlib.sha256(actor_path.read_bytes()).hexdigest():
            raise ValueError("vision actor metadata or SHA-256 mismatch")
        characters = meta.get("goal_characters", ALPHABET)
        if characters not in (ALPHABET, PHRASE_CHARACTERS):
            raise ValueError("Unsupported actor goal vocabulary")
        if meta.get("proprio_dim") != 19 + len(characters) or meta.get("rgb_shape") != [6, 120, 160] or meta.get("action_dim") != 5 or meta.get("control_hz") != 25:
            raise ValueError("unsupported vision actor input or control contract")
        if meta.get("rgb_order") != "RGB float32 [0,1]; previous frame followed by current frame":
            raise ValueError("unsupported wrist RGB history order")
        if not supports_vision_task(env.get("task_contract", ""), len(characters)) or env.get("actuator_profile") != "gtc_berlin":
            raise ValueError("environment does not match the GTC Berlin vision task")
        if actor_path.name != meta.get("actor_file"):
            raise ValueError("vision actor filename mismatch")
        checkpoint_recorded = Path(meta["source_checkpoint"])
        checkpoint = environment_path.parent / checkpoint_recorded.name
        if checkpoint_recorded.parent.name != environment_path.parent.name or not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != meta.get("source_checkpoint_sha256"):
            raise ValueError("source vision checkpoint is missing or has changed")
        torch.set_num_threads(4)
        self.model = torch.jit.load(str(actor_path), map_location="cpu").eval()
        self.goal_characters = characters
        self.metadata = meta
        self.environment = env
        self.actor_path = actor_path

    def action(self, proprio: np.ndarray, rgb_history: np.ndarray) -> np.ndarray:
        import torch

        if proprio.shape != (19 + len(self.goal_characters),) or rgb_history.shape != (6, 120, 160):
            raise ValueError("vision actor input shape mismatch")
        if not np.isfinite(proprio).all() or not np.isfinite(rgb_history).all() or rgb_history.min() < 0 or rgb_history.max() > 1:
            raise ValueError("nonfinite or out-of-range vision input")
        with torch.inference_mode():
            result = self.model(torch.from_numpy(np.ascontiguousarray(proprio[None], dtype=np.float32)),
                                torch.from_numpy(np.ascontiguousarray(rgb_history[None], dtype=np.float32)))
        action = result.detach().cpu().numpy().reshape(-1)
        if action.shape != (5,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError("invalid bounded vision action")
        return action.astype(np.float64)


class CameraRecorder:
    """Write both MP4s with monotonic timestamps; share wrist frames with actor."""

    def __init__(self, directory: Path, wrist_device: str = "/dev/video0",
                 external_url: str = "http://127.0.0.1:8765/stream.mjpg"):
        self.directory = Path(directory)
        self.sources = {"wrist": wrist_device}
        if external_url:
            self.sources["external"] = external_url
        self.stop = Event()
        self.lock = Lock()
        self.workers = []
        self.errors = []
        self.latest_wrist = None

    def __enter__(self):
        import cv2

        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            for name, source in self.sources.items():
                cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if source.startswith("/dev/") else cv2.VideoCapture(source)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f"{name}: camera cannot open {source}")
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    raise RuntimeError(f"{name}: no initial frame")
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if np.isfinite(fps) and 5 <= fps <= 60 else 15.0
                video = self.directory / f"{name}.mp4"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    cap.release()
                    raise RuntimeError(f"{name}: MP4 writer failed")
                stamps = (self.directory / f"{name}_timestamps.jsonl").open("x", buffering=1)
                worker = {"name": name, "source": source, "cap": cap, "writer": writer, "stamps": stamps,
                          "fps": fps, "size": (height, width), "first": frame, "count": 0, "thread": None}
                self.workers.append(worker)
            for worker in self.workers:
                thread = Thread(target=self._capture, args=(worker,), daemon=True, name=f"vision-{worker['name']}")
                worker["thread"] = thread
                thread.start()
            deadline = time.monotonic() + 2
            while self.latest_wrist is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.check()
            if self.latest_wrist is None:
                raise RuntimeError("wrist camera did not produce a policy frame")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _capture(self, worker):
        import cv2

        name, cap, writer, stamps = (worker[k] for k in ("name", "cap", "writer", "stamps"))
        frame = worker["first"]
        try:
            while not self.stop.is_set():
                if frame.shape[:2] != worker["size"]:
                    raise RuntimeError("camera size changed")
                monotonic_ns, unix_ns = time.monotonic_ns(), time.time_ns()
                if name == "wrist":
                    resized = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    with self.lock:
                        self.latest_wrist = (worker["count"], monotonic_ns, rgb)
                writer.write(frame)
                stamps.write(json.dumps({"frame": worker["count"], "received_monotonic_ns": monotonic_ns,
                                         "received_unix_ns": unix_ns}) + "\n")
                worker["count"] += 1
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("camera stream ended")
        except BaseException as exc:
            if not self.stop.is_set():
                self.errors.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            writer.release()
            cap.release()
            stamps.close()
            (self.directory / f"{name}_metadata.json").write_text(json.dumps({
                "source": worker["source"], "fps": worker["fps"], "frames": worker["count"],
                "timestamp_semantics": "host frame receipt, not exposure time"}, indent=2) + "\n")

    def check(self):
        if self.errors:
            raise RuntimeError("camera recording failed: " + "; ".join(self.errors))
        if any(w["thread"] and not w["thread"].is_alive() for w in self.workers):
            raise RuntimeError("camera recording stopped")

    def wrist_rgb_history(self, previous_rgb):
        self.check()
        with self.lock:
            latest = self.latest_wrist
        if latest is None or time.monotonic_ns() - latest[1] > 250_000_000:
            raise RuntimeError("wrist policy frame is stale")
        index, stamp, current = latest
        if previous_rgb is None:
            previous_rgb = current
        history = np.concatenate((previous_rgb.transpose(2, 0, 1), current.transpose(2, 0, 1))).astype(np.float32) / 255
        return history, current.copy(), {"frame": index, "received_monotonic_ns": stamp}

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        for worker in self.workers:
            thread = worker["thread"]
            if thread:
                thread.join(timeout=2)
                if thread.is_alive():
                    worker["cap"].release()
                    thread.join(timeout=2)
            else:
                worker["writer"].release()
                worker["cap"].release()
                worker["stamps"].close()
        if exc_type is None and self.errors:
            raise RuntimeError("camera recording failed: " + "; ".join(self.errors))


def retime_cameras(directory: Path, *, fps: float = 25.0, names=("wrist", "external")) -> dict[str, dict]:
    """Create real-time CFR copies from capture timestamps after motor cleanup.

    OpenCV's MP4 writer assigns a constant frame interval even when a USB or
    MJPEG source delivers fewer frames. Retain raw videos and timestamp sidecars;
    duplicate/drop frames for viewer copies so their timeline matches the log.
    """
    import cv2

    directory = Path(directory)
    reports = {}
    for name in names:
        source = directory / f"{name}.mp4"
        stamp_path = directory / f"{name}_timestamps.jsonl"
        if not source.is_file() or not stamp_path.is_file():
            raise FileNotFoundError(f"missing {name} video or frame timestamps")
        stamps = [json.loads(line)["received_monotonic_ns"] for line in stamp_path.read_text().splitlines()]
        if not stamps or any(b < a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"{name} timestamps are empty or out of order")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"cannot decode {source}")
        raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if raw_count != len(stamps):
            cap.release()
            raise ValueError(f"{name} video/timestamp count mismatch: {raw_count} vs {len(stamps)}")
        ok, current = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"{name} has no decodable first frame")
        height, width = current.shape[:2]
        destination = directory / f"{name}_realtime.mp4"
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot write {destination}")
        next_index = 1
        next_ok, next_frame = cap.read()
        duration_s = (stamps[-1] - stamps[0]) / 1e9
        output_count = int(round(duration_s * fps)) + 1
        try:
            for output_index in range(output_count):
                output_ns = stamps[0] + round(output_index * 1e9 / fps)
                while next_ok and next_index < len(stamps) and stamps[next_index] <= output_ns:
                    current = next_frame
                    next_index += 1
                    next_ok, next_frame = cap.read()
                writer.write(current)
        finally:
            writer.release()
            cap.release()
        report = {"raw_video": source.name, "realtime_video": destination.name,
                  "raw_frames": len(stamps), "realtime_frames": output_count,
                  "fps": fps, "monotonic_start_ns": stamps[0],
                  "monotonic_end_ns": stamps[-1], "duration_s": duration_s,
                  "source_timestamps": stamp_path.name}
        (directory / f"{name}_realtime_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        reports[name] = report
    return reports

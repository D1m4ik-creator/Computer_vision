import logging
import threading
import time
from queue import Empty, Full, Queue

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from ultralytics import YOLO

from config import (
    CLEAR_CANVAS_FRAMES,
    CURSOR_SMOOTHING_ALPHA,
    DRAW_DISABLE_FRAMES,
    DRAW_ENABLE_FRAMES,
    FINGER_EXTENDED_ANGLE,
    FINGER_FOLDED_ANGLE,
    FRAME_QUEUE_SIZE,
    HAND_LANDMARKER_PATH,
    HAND_LOST_FRAME_TOLERANCE,
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_PRESENCE_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE,
    HAND_NUM_HANDS,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    YOLO_ONNX_PATH,
    YOLO_PT_PATH,
)

logger = logging.getLogger("CV_Core")


def put_latest(frame_queue, frame):
    frame_copy = frame.copy()
    try:
        frame_queue.put_nowait(frame_copy)
        return
    except Full:
        pass

    try:
        frame_queue.get_nowait()
    except Empty:
        pass

    try:
        frame_queue.put_nowait(frame_copy)
    except Full:
        pass


class YOLODetector:
    def __init__(self):
        if not YOLO_ONNX_PATH.exists():
            logger.info("Конвертация в ONNX...")
            temp_model = YOLO(str(YOLO_PT_PATH))
            temp_model.export(format="onnx", imgsz=YOLO_IMAGE_SIZE)

        self.model = YOLO(str(YOLO_ONNX_PATH), task="detect")
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.latest_boxes = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("YOLO Thread запущен.")

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                results = self.model(
                    frame,
                    verbose=False,
                    conf=YOLO_CONFIDENCE,
                    imgsz=YOLO_IMAGE_SIZE,
                )

                boxes_data = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_name = self.model.names[int(box.cls[0])]
                        boxes_data.append((x1, y1, x2, y2, conf, cls_name))

                with self.lock:
                    self.latest_boxes = boxes_data
            except Exception:
                logger.exception("Ошибка YOLO inference")

    def update_frame(self, frame):
        put_latest(self.frame_queue, frame)

    def get_boxes(self):
        with self.lock:
            return list(self.latest_boxes)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


class GestureCalculator:
    def __init__(self):
        if not HAND_LANDMARKER_PATH.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {HAND_LANDMARKER_PATH}")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=HAND_NUM_HANDS,
            min_hand_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.cursor_pos = None
        self.smoothed_cursor = None
        self.lost_frames = 0
        self.draw_enable_count = 0
        self.draw_disable_count = 0
        self.clear_count = 0
        self.is_drawing = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("MediaPipe Thread запущен.")

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                self._process_frame(frame)
            except Exception:
                logger.exception("Ошибка MediaPipe inference")

    def _process_frame(self, frame):
        with self.lock:
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        timestamp_ms = self._next_timestamp_ms()
        results = self.hands.detect_for_video(mp_image, timestamp_ms)

        if results.hand_landmarks:
            self._handle_landmarks(frame, results.hand_landmarks[0])
        else:
            self._handle_missing_hand()

    def _handle_landmarks(self, frame, landmarks):
        height, width, _ = frame.shape
        raw_cursor = (int(landmarks[8].x * width), int(landmarks[8].y * height))

        index_extended, index_folded = self._finger_state(landmarks, 5, 6, 7, 8)
        _, middle_folded = self._finger_state(landmarks, 9, 10, 11, 12)
        _, ring_folded = self._finger_state(landmarks, 13, 14, 15, 16)
        _, pinky_folded = self._finger_state(landmarks, 17, 18, 19, 20)

        drawing_candidate = index_extended and middle_folded and ring_folded and pinky_folded
        fist_candidate = index_folded and middle_folded and ring_folded and pinky_folded

        with self.lock:
            self.lost_frames = 0
            cursor = self._smooth_cursor(raw_cursor)
            self._update_drawing_state(drawing_candidate)
            self._update_clear_state(frame, fist_candidate)

            if self.is_drawing and not fist_candidate:
                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = cursor
                cv2.line(
                    self.canvas,
                    (self.prev_x, self.prev_y),
                    cursor,
                    (255, 255, 255),
                    10,
                )
                self.prev_x, self.prev_y = cursor
            else:
                if not self.is_drawing:
                    self.prev_x, self.prev_y = 0, 0

            self.cursor_pos = cursor

    def _handle_missing_hand(self):
        with self.lock:
            self.lost_frames += 1
            if self.lost_frames <= HAND_LOST_FRAME_TOLERANCE:
                return

            self.cursor_pos = None
            self.smoothed_cursor = None
            self.prev_x, self.prev_y = 0, 0
            self.draw_enable_count = 0
            self.draw_disable_count = 0
            self.clear_count = 0
            self.is_drawing = False

    def _smooth_cursor(self, raw_cursor):
            import math
            if self.smoothed_cursor is None:
                self.smoothed_cursor = (float(raw_cursor[0]), float(raw_cursor[1]))
                return int(raw_cursor[0]), int(raw_cursor[1])

            prev_x, prev_y = self.smoothed_cursor
            raw_x, raw_y = raw_cursor

            # Считаем "скорость" движения пальца (расстояние в пикселях между кадрами)
            distance = math.hypot(raw_x - prev_x, raw_y - prev_y)

            # Умный коэффициент альфа:
            # Медленно двигаем (<10px) = 0.2 (очень плавная линия)
            # Резко двигаем (>100px) = стремится к 0.9 (моментальный отклик без отставания)
            dynamic_alpha = min(0.9, max(0.2, distance / 100.0))

            self.smoothed_cursor = (
                dynamic_alpha * raw_x + (1 - dynamic_alpha) * prev_x,
                dynamic_alpha * raw_y + (1 - dynamic_alpha) * prev_y,
            )

            return int(self.smoothed_cursor[0]), int(self.smoothed_cursor[1])

    def _update_drawing_state(self, drawing_candidate):
        if drawing_candidate:
            self.draw_enable_count += 1
            self.draw_disable_count = 0
        else:
            self.draw_disable_count += 1
            self.draw_enable_count = 0

        if self.draw_enable_count >= DRAW_ENABLE_FRAMES:
            self.is_drawing = True
        elif self.draw_disable_count >= DRAW_DISABLE_FRAMES:
            self.is_drawing = False

    def _update_clear_state(self, frame, fist_candidate):
        if not fist_candidate:
            self.clear_count = 0
            return

        self.clear_count += 1
        self.is_drawing = False
        self.prev_x, self.prev_y = 0, 0

        if self.clear_count >= CLEAR_CANVAS_FRAMES:
            self.canvas = np.zeros_like(frame)

    @classmethod
    def _finger_state(cls, landmarks, mcp_id, pip_id, dip_id, tip_id):
        pip_angle = cls._angle(landmarks[mcp_id], landmarks[pip_id], landmarks[dip_id])
        dip_angle = cls._angle(landmarks[pip_id], landmarks[dip_id], landmarks[tip_id])

        extended = pip_angle >= FINGER_EXTENDED_ANGLE and dip_angle >= FINGER_EXTENDED_ANGLE - 10
        folded = pip_angle <= FINGER_FOLDED_ANGLE or dip_angle <= FINGER_FOLDED_ANGLE - 5
        return bool(extended), bool(folded)

    @staticmethod
    def _angle(a, b, c):
        vector_ab = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        vector_cb = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        norm_ab = np.linalg.norm(vector_ab)
        norm_cb = np.linalg.norm(vector_cb)

        if norm_ab == 0 or norm_cb == 0:
            return 0

        cosine = np.dot(vector_ab, vector_cb) / (norm_ab * norm_cb)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _next_timestamp_ms(self):
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def update_frame(self, frame):
        put_latest(self.frame_queue, frame)

    def get_render_data(self):
        with self.lock:
            canvas = self.canvas.copy() if self.canvas is not None else None
            return canvas, self.cursor_pos

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.hands.close()

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
    FRAME_QUEUE_SIZE,
    HAND_LANDMARKER_PATH,
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

        cursor = None
        if results.hand_landmarks:
            for landmarks in results.hand_landmarks:
                cursor = self._handle_landmarks(frame, landmarks)

        with self.lock:
            self.cursor_pos = cursor

    def _handle_landmarks(self, frame, landmarks):
        height, width, _ = frame.shape
        wrist = landmarks[0]

        index_up = self._get_dist(landmarks[8], wrist) > self._get_dist(landmarks[6], wrist)
        middle_down = self._get_dist(landmarks[12], wrist) < self._get_dist(landmarks[10], wrist)
        ring_down = self._get_dist(landmarks[16], wrist) < self._get_dist(landmarks[14], wrist)
        pinky_down = self._get_dist(landmarks[20], wrist) < self._get_dist(landmarks[18], wrist)

        is_drawing = index_up and middle_down and ring_down and pinky_down
        is_fist = not index_up and middle_down and ring_down and pinky_down

        x, y = int(landmarks[8].x * width), int(landmarks[8].y * height)

        with self.lock:
            if is_drawing:
                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = x, y
                cv2.line(
                    self.canvas,
                    (self.prev_x, self.prev_y),
                    (x, y),
                    (255, 255, 255),
                    10,
                )
                self.prev_x, self.prev_y = x, y
            else:
                self.prev_x, self.prev_y = 0, 0

            if is_fist:
                self.canvas = np.zeros_like(frame)

        return x, y

    def _next_timestamp_ms(self):
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    @staticmethod
    def _get_dist(p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

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

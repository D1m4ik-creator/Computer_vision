import os
import cv2
import numpy as np
import mediapipe as mp
import threading
import logging
import time
from pathlib import Path
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from ultralytics import YOLO

logger = logging.getLogger("CV_Core")

class YOLODetector:
    def __init__(self):
        model_name = 'yolo11s' 
        pt_path = f"{model_name}.pt"
        onnx_path = f"{model_name}.onnx"

        if not os.path.exists(onnx_path):
            logger.info("Конвертация в ONNX...")
            temp_model = YOLO(pt_path)
            temp_model.export(format='onnx', imgsz=640)

        self.model = YOLO(onnx_path, task='detect') 
        
        # Многопоточные переменные для YOLO
        self.latest_boxes = []
        self.frame_to_process = None
        self.lock = threading.Lock()
        self.stopped = False
        
        # Запуск фонового потока инференса
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()
        logger.info("YOLO Thread запущен.")

    def _worker(self):
        while not self.stopped:
            frame = None
            # Безопасно забираем кадр для обработки
            with self.lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process.copy()
                    self.frame_to_process = None 
            
            if frame is not None:
                # stream=False лучше для одиночных кадров в фоне
                results = self.model(frame, verbose=False, conf=0.5, imgsz=640)
                
                boxes_data = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_name = self.model.names[int(box.cls[0])]
                        boxes_data.append((x1, y1, x2, y2, conf, cls_name))
                
                # Безопасно обновляем список рамок
                with self.lock:
                    self.latest_boxes = boxes_data

    def update_frame(self, frame):
        # Отдаем кадр рабочему потоку
        with self.lock:
            self.frame_to_process = frame
            
    def get_boxes(self):
        # Получаем последние вычисленные рамки
        with self.lock:
            return self.latest_boxes
            
    def stop(self):
        self.stopped = True
        self.thread.join()

class GestureCalculator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        
        # Многопоточные переменные
        self.frame_to_process = None
        self.cursor_pos = None  
        self.lock = threading.Lock()
        self.stopped = False
        
        # Запускаем MediaPipe в фоне!
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def _worker(self):
        import math
        
        def get_dist(p1, p2):
            # Вычисляет расстояние между двумя точками
            return math.hypot(p1.x - p2.x, p1.y - p2.y)

        while not self.stopped:
            frame = None
            with self.lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process.copy()
                    self.frame_to_process = None
            
            if frame is not None:
                if self.canvas is None:
                    self.canvas = np.zeros_like(frame)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                cursor = None
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        lm = hand_landmarks.landmark
                        h, w, c = frame.shape
                        
                        # Новая логика: независима от наклона руки
                        # Палец выпрямлен, если кончик дальше от запястья (0), чем сустав (PIP)
                        wrist = lm[0]
                        index_up = get_dist(lm[8], wrist) > get_dist(lm[6], wrist)
                        middle_down = get_dist(lm[12], wrist) < get_dist(lm[10], wrist)
                        ring_down = get_dist(lm[16], wrist) < get_dist(lm[14], wrist)
                        pinky_down = get_dist(lm[20], wrist) < get_dist(lm[18], wrist)
                        
                        is_drawing = index_up and middle_down and ring_down and pinky_down
                        is_fist = not index_up and middle_down and ring_down and pinky_down
                        
                        x, y = int(lm[8].x * w), int(lm[8].y * h)
                        cursor = (x, y)

                        # Обновляем холст безопасно
                        with self.lock:
                            if is_drawing:
                                if self.prev_x == 0 and self.prev_y == 0:
                                    self.prev_x, self.prev_y = x, y
                                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (255, 255, 255), 10)
                                self.prev_x, self.prev_y = x, y
                            else:
                                self.prev_x, self.prev_y = 0, 0
                                
                            if is_fist:
                                self.canvas = np.zeros_like(frame)
                                
                with self.lock:
                    self.cursor_pos = cursor

    def update_frame(self, frame):
        # Передаем кадр рабочему потоку
        with self.lock:
            self.frame_to_process = frame

    def get_render_data(self):
        # Отдаем холст и курсор для отрисовки в главном потоке
        with self.lock:
            return (self.canvas.copy() if self.canvas is not None else None), self.cursor_pos

    def stop(self):
        self.stopped = True
        self.thread.join()
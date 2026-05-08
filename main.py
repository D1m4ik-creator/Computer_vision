import cv2
import logging
import time

# Импорты из наших модулей
from camera import ThreadedCamera
from config import CAMERA_HEIGHT, CAMERA_SRC, CAMERA_WIDTH
from cv_models import YOLODetector, GestureCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("cv_system.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("CV_Core")

class CVSystem:
    def __init__(self):
        self.mode = 1 
        logger.info("Инициализация компонентов...")
        
        # Подключаем модули
        self.camera = ThreadedCamera(src=CAMERA_SRC, width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        self.yolo = YOLODetector()
        self.calculator = GestureCalculator()
        
        self.prev_time = time.time()

    def run(self):
        logger.info("Главный цикл запущен.")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    continue

                frame = cv2.flip(frame, 1)

                if self.mode == 1:
                    self.yolo.update_frame(frame)
                    boxes = self.yolo.get_boxes()
                    self.draw_yolo_boxes(frame, boxes)
                else:
                    # Отдаем кадр в фон
                    self.calculator.update_frame(frame)

                    # Забираем слои для отрисовки
                    canvas, cursor = self.calculator.get_render_data()

                    if canvas is not None:
                        # Супербыстрое наложение холста (O(1) вместо сложной маскировки)
                        frame = cv2.add(frame, canvas)

                    if cursor is not None:
                        # Рисуем кружок-курсор на кончике пальца
                        cv2.circle(frame, cursor, 8, (0, 255, 255), cv2.FILLED)

                # Подсчет FPS
                curr_time = time.time()
                fps = 1 / (curr_time - self.prev_time)
                self.prev_time = curr_time

                self.draw_ui(frame, fps)
                cv2.imshow("CV System", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 9:  # Tab
                    self.mode = 2 if self.mode == 1 else 1
        finally:
            self.cleanup()

    def draw_yolo_boxes(self, frame, boxes):
        for (x1, y1, x2, y2, conf, cls_name) in boxes:
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def draw_ui(self, frame, fps):
        mode_text = "Mode: YOLO Detection" if self.mode == 1 else "Mode: Gesture Calculator"
        color = (0, 255, 0) if self.mode == 1 else (0, 255, 255)
        
        cv2.putText(frame, f"{mode_text} | FPS: {int(fps)} | 'Tab' switch | 'q' quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def cleanup(self):
        logger.info("Завершение работы...")
        self.camera.stop()
        self.yolo.stop()
        self.calculator.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CVSystem()
    app.run()

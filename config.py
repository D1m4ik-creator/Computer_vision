from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CAMERA_SRC = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

YOLO_MODEL_NAME = "yolo11s"
YOLO_PT_PATH = BASE_DIR / f"{YOLO_MODEL_NAME}.pt"
YOLO_ONNX_PATH = BASE_DIR / f"{YOLO_MODEL_NAME}.onnx"
YOLO_CONFIDENCE = 0.5
YOLO_IMAGE_SIZE = 640

HAND_LANDMARKER_PATH = BASE_DIR / "hand_landmarker.task"
HAND_NUM_HANDS = 1
HAND_MIN_DETECTION_CONFIDENCE = 0.4
HAND_MIN_PRESENCE_CONFIDENCE = 0.4
HAND_MIN_TRACKING_CONFIDENCE = 0.15

CURSOR_SMOOTHING_ALPHA = 0.6     # Делаем курсор резче (0.35 давал сильную задержку за кистью)
HAND_LOST_FRAME_TOLERANCE = 15   # Ждем ~0.5 сек перед сбросом, если камера потеряла руку из-за быстрого взмаха
DRAW_ENABLE_FRAMES = 2           # Включаем режим рисования быстро
DRAW_DISABLE_FRAMES = 10          # Выключаем медленно (защита от разрывов, если палец случайно дрогнул)
CLEAR_CANVAS_FRAMES = 15         # Защита от ложных срабатываний. Теперь кулак нужно осознанно держать полсекунды
FINGER_EXTENDED_ANGLE = 135      # Палец считается прямым, даже если фаланга немного согнута (155 - это слишком строгий идеал)
FINGER_FOLDED_ANGLE = 160        # Остальные пальцы считаются сжатыми, даже если они просто расслаблены полукругом

FRAME_QUEUE_SIZE = 1

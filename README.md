# Computer Vision System

Проект демонстрирует простую real-time систему компьютерного зрения на Python:

- детекция объектов через YOLO;
- обработка видеопотока с веб-камеры;
- режим жестов руки для рисования на кадре;
- фоновая обработка кадров в отдельных потоках.

## Структура проекта

```text
.
├── main.py              # Точка входа, главный цикл приложения и UI
├── camera.py            # Поточное чтение кадров с камеры
├── cv_models.py         # YOLO-детектор и обработчик жестов
├── .gitignore           # Исключения для Git
├── yolo11s.pt           # Локальная YOLO-модель, не хранится в Git
├── yolo11s.onnx         # ONNX-версия модели, генерируется локально
├── yolov8n.pt           # Дополнительная локальная модель
├── hand_landmarker.task # Локальная модель MediaPipe
└── cv_system.log        # Лог выполнения, не хранится в Git
```

## Требования

Нужен Python и доступная веб-камера. Основные Python-зависимости:

```text
opencv-python
numpy
ultralytics
mediapipe
```

Если используется виртуальное окружение:

```powershell
.\.venv\Scripts\activate
pip install opencv-python numpy ultralytics mediapipe
```

## Локальные модели

Модели не добавляются в Git, потому что это большие бинарные файлы и локальные артефакты.

Ожидаемые файлы в корне проекта:

- `yolo11s.pt` - исходная YOLO-модель;
- `yolo11s.onnx` - ONNX-модель для инференса;
- `hand_landmarker.task` - модель MediaPipe для распознавания руки.

Если `yolo11s.onnx` отсутствует, приложение попробует создать его из `yolo11s.pt`.

## Запуск

Из корня проекта:

```powershell
.\.venv\Scripts\python.exe main.py
```

Или, если нужный Python уже активирован:

```powershell
python main.py
```

## Управление

В окне приложения:

- `Tab` - переключить режим;
- `q` - закрыть приложение.

Режимы:

- `YOLO Detection` - отображает найденные объекты рамками;
- `Gesture Calculator` - отслеживает указательный палец и рисует по кадру.

В режиме жестов:

- поднят только указательный палец - рисование;
- кулак - очистка холста.

## Логи и Git

Файл `cv_system.log` создается при запуске приложения и исключен из Git.

Также исключены:

- виртуальные окружения: `.venv/`, `venv/`, `env/`;
- Python-кэш: `__pycache__/`, `*.pyc`;
- модели: `*.pt`, `*.onnx`, `*.task`, `*.engine`, `*.tflite`;
- runtime-вывод: `.cache/`, `runs/`, `outputs/`, `results/`;
- локальные файлы IDE: `.vscode/`, `.idea/`.

## Возможные проблемы

### Камера не доступна

Проверьте, что камера не занята другим приложением. По умолчанию используется устройство `src=0` в `main.py`.

### MediaPipe: `module 'mediapipe' has no attribute 'solutions'`

В новых версиях MediaPipe старый API `mp.solutions` может быть недоступен. Для таких версий нужно использовать MediaPipe Tasks API и локальный файл `hand_landmarker.task`.

### Git: `detected dubious ownership`

Если Git сообщает о `dubious ownership`, репозиторий открыт от другого системного пользователя. Для локальной работы можно добавить проект в safe directory:

```powershell
git config --global --add safe.directory C:/VS/python/ML/Computer_vision
```

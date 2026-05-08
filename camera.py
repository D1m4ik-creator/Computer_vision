import cv2
import threading

class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError("Камера не доступна")
            
        # Читаем первый кадр
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        
        # Запускаем поток
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Поток умрет вместе с закрытием программы
        self.thread.start()

    def update(self):
        # Бесконечный цикл чтения кадров в фоне
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        # Отдаем самый свежий кадр
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
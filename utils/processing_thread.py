from PyQt6.QtCore import QThread, pyqtSignal


class ProcessingThread(QThread):
    progress_update = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        def callback(msg):
            pct = None
            try:
                if isinstance(msg, str):
                    s = msg.strip()
                    if s.endswith("%"):
                        s = s[:-1].strip()
                    pct = int(float(s))
                elif isinstance(msg, (int, float)):
                    pct = int(msg)
            except Exception:
                pct = None

            if pct is not None:
                pct = max(0, min(100, pct))
                self.progress_update.emit(pct)

        try:
            result = self.func(
                *self.args, progress_callback=callback, **self.kwargs)
            if result is None:
                result = []
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))

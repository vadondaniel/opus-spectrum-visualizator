from PyQt6.QtWidgets import QWidget, QStylePainter, QStyleOptionSlider
from PyQt6.QtCore import Qt, QRect, pyqtSignal


class QRangeSlider(QWidget):
    rangeChanged = pyqtSignal(float, float)

    def __init__(self, minimum=0.0, maximum=100.0, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._lower = minimum
        self._upper = maximum
        self._handle_size = 10
        self.setMinimumHeight(30)
        self.setMouseTracking(True)
        self._active_handle = None

    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = maximum
        self._lower = minimum
        self._upper = maximum
        self.update()

    def lowerValue(self):
        return self._lower

    def upperValue(self):
        return self._upper

    def setLowerValue(self, value):
        self._lower = max(self._min, min(value, self._upper))
        self.rangeChanged.emit(self._lower, self._upper)
        self.update()

    def setUpperValue(self, value):
        self._upper = min(self._max, max(value, self._lower))
        self.rangeChanged.emit(self._lower, self._upper)
        self.update()

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionSlider()
        opt.initFrom(self)

        # Draw groove
        groove_rect = QRect(10, self.height() // 2 - 4,
                            self.width() - 20, 8)
        painter.setPen(Qt.GlobalColor.gray)
        painter.setBrush(Qt.GlobalColor.lightGray)
        painter.drawRect(groove_rect)

        # Draw selected range
        min_pos = self._valueToPos(self._lower)
        max_pos = self._valueToPos(self._upper)
        sel_rect = QRect(min_pos, self.height() // 2 - 4,
                         max_pos - min_pos, 8)
        painter.setBrush(Qt.GlobalColor.blue)
        painter.drawRect(sel_rect)

        # Draw handles
        painter.setBrush(Qt.GlobalColor.white)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawEllipse(min_pos - self._handle_size // 2,
                            self.height() // 2 - self._handle_size // 2,
                            self._handle_size, self._handle_size)
        painter.drawEllipse(max_pos - self._handle_size // 2,
                            self.height() // 2 - self._handle_size // 2,
                            self._handle_size, self._handle_size)

    def mousePressEvent(self, event):
        x = event.position().x()
        min_pos = self._valueToPos(self._lower)
        max_pos = self._valueToPos(self._upper)
        if abs(x - min_pos) < 10:
            self._active_handle = "lower"
        elif abs(x - max_pos) < 10:
            self._active_handle = "upper"

    def mouseMoveEvent(self, event):
        if self._active_handle:
            val = self._posToValue(event.position().x())
            if self._active_handle == "lower":
                self.setLowerValue(val)
            elif self._active_handle == "upper":
                self.setUpperValue(val)

    def mouseReleaseEvent(self, event):
        self._active_handle = None

    def _valueToPos(self, value):
        return int((value - self._min) / (self._max - self._min) * (self.width() - 20)) + 10

    def _posToValue(self, pos):
        return (pos - 10) / (self.width() - 20) * (self._max - self._min) + self._min

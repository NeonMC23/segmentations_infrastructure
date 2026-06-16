from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import numpy as np
import cv2


class ImageWidget(QWidget):
    """Simple widget that displays a numpy RGB image and keeps aspect ratio on resize."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self._pixmap = None

    def set_image(self, img_np):
        if img_np is None:
            self.label.clear()
            self._pixmap = None
            return
        # Ensure uint8 RGB
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            # drop alpha
            img_np = img_np[:, :, :3]
        elif img_np.shape[2] == 3:
            pass
        else:
            raise ValueError('Unsupported image shape: %s' % (img_np.shape,))

        # Make contiguous
        if not img_np.flags['C_CONTIGUOUS']:
            img_np = np.ascontiguousarray(img_np)

        h, w = img_np.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._pixmap = pixmap
        self._update_label_pixmap()

    def _update_label_pixmap(self):
        if self._pixmap is None:
            self.label.clear()
            return
        scaled = self._pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_label_pixmap()

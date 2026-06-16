"""Minimal PyQt6-based CETI Segmentation Explorer V1.

Usage (from repository root):
  pip install -r requirements.txt
  python -m explorer.main

The UI is intentionally simple:
- Open HDF5 of segmentations
- (Optionally) open associated video
- Display one frame with segmentations rendered via `get_video_frame(..., show_masks=True)`
- Previous / Next frame buttons and frame index label

All data access is isolated in `explorer.services.segmentation_service.SegmentationService`.
"""

import os
import sys
import traceback

# Ensure repository root is on sys.path so imports for Segmentations work.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt

from services.segmentation_service import SegmentationService
from widgets.image_widget import ImageWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CETI Segmentation Explorer V1')

        self.service = SegmentationService()
        self.h5_path = None
        self.video_path = None
        self.current_frame = 0
        self.total_frames = 0

        central = QWidget()
        vbox = QVBoxLayout()

        # Top bar: file open buttons
        top_h = QHBoxLayout()
        self.btn_open_h5 = QPushButton('Open HDF5')
        self.btn_open_video = QPushButton('Open Video')
        self.lbl_file = QLabel('No file')
        top_h.addWidget(self.btn_open_h5)
        top_h.addWidget(self.btn_open_video)
        top_h.addWidget(self.lbl_file)
        top_h.addStretch()
        vbox.addLayout(top_h)

        # Image display
        self.image_widget = ImageWidget()
        vbox.addWidget(self.image_widget, 1)

        # Bottom: frame index and navigation
        bottom_h = QHBoxLayout()
        self.lbl_frame = QLabel('Frame: -')
        bottom_h.addWidget(self.lbl_frame)
        bottom_h.addStretch()
        self.btn_prev = QPushButton('Précédente')
        self.btn_next = QPushButton('Suivante')
        bottom_h.addWidget(self.btn_prev)
        bottom_h.addWidget(self.btn_next)
        vbox.addLayout(bottom_h)

        central.setLayout(vbox)
        self.setCentralWidget(central)

        # Connections
        self.btn_open_h5.clicked.connect(self.on_open_h5)
        self.btn_open_video.clicked.connect(self.on_open_video)
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)

    def on_open_h5(self):
        start_dir = os.path.join(os.getcwd(), 'data', 'segmentations')
        path, _ = QFileDialog.getOpenFileName(self, 'Open Segmentations HDF5', start_dir, 'HDF5 Files (*.h5 *.hdf5);;All files (*)')
        if not path:
            return
        try:
            # Attempt to open with any previously selected video
            self.service.open(path, self.video_path)
            self.h5_path = path
            self.total_frames = self.service.get_num_frames_total()
            self.current_frame = 0
            self.lbl_file.setText(os.path.basename(path))
            self.update_frame()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to open HDF5:\n{e}')

    def on_open_video(self):
        start_dir = os.path.join(os.getcwd(), 'data', 'videos')
        path, _ = QFileDialog.getOpenFileName(self, 'Open Video', start_dir, 'Video Files (*.mp4 *.avi *.mov *.mkv);;All files (*)')
        if not path:
            return
        self.video_path = path
        # If an HDF5 is already open, reopen service to associate the video.
        if self.h5_path:
            try:
                self.service.open(self.h5_path, self.video_path)
                self.update_frame()
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to associate video:\n{e}')
        else:
            QMessageBox.information(self, 'Video loaded', 'Video selected. Open an HDF5 file to use it with segmentations.')

    def update_frame(self):
        try:
            img = self.service.get_frame_image(self.current_frame)
            if img is None:
                self.image_widget.set_image(None)
                self.lbl_frame.setText('Frame: -')
                return
            # The service returns an RGB numpy array.
            self.image_widget.set_image(img)
            total = self.total_frames if self.total_frames and self.total_frames > 0 else '?'
            self.lbl_frame.setText(f'Frame: {self.current_frame} / {total}')
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, 'Render error', f'Error rendering frame {self.current_frame}:\n{e}\n\n{tb}')

    def on_prev(self):
        self.current_frame = max(0, self.current_frame - 1)
        self.update_frame()

    def on_next(self):
        if self.total_frames and self.total_frames > 0:
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        else:
            self.current_frame = self.current_frame + 1
        self.update_frame()

    def closeEvent(self, event):
        try:
            self.service.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

"""SegmentationService: isolate data access to `Segmentations`.

This service exposes a small API used by the UI:
- `open(h5_path, video_path=None)`
- `get_num_frames_total()`
- `get_frame_shape()`
- `get_frame_image(frame_index)` -> numpy RGB image
- `close()`

The implementation tries to import `Segmentations` using the repository layout, with a fallback
that loads `Segmentations.py` by path so the UI is resilient to module name differences.
"""

import os
import sys
import importlib.util

# Attempt to import Segmentations from common locations, fall back to loading by path.
try:
    from segmentation_infrastructure.Segmentations import Segmentations
except Exception:
    try:
        from Segmentations import Segmentations
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        seg_path = os.path.join(repo_root, 'Segmentations.py')
        spec = importlib.util.spec_from_file_location('Segmentations', seg_path)
        seg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(seg_mod)
        Segmentations = seg_mod.Segmentations


class SegmentationService:
    def __init__(self):
        self.seg = None
        self.h5_path = None
        self.video_path = None
        self.video_key = 'original'

    def open(self, h5_path, video_path=None):
        """Open an HDF5 of segmentations and optionally associate a video.
        Raises FileNotFoundError if the HDF5 does not exist.
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f'HDF5 file not found: {h5_path}')
        video_filepaths = {}
        if video_path and os.path.exists(video_path):
            video_filepaths[self.video_key] = video_path
            self.video_path = video_path
        else:
            self.video_path = None

        # Close previous if any.
        if self.seg is not None:
            try:
                self.seg.close()
            except Exception:
                pass
            self.seg = None

        # Instantiate Segmentations (read-only)
        self.seg = Segmentations(h5_filepath=h5_path, writable=False, video_filepaths=video_filepaths)
        self.h5_path = h5_path

    def get_num_frames_total(self):
        if self.seg is None:
            return 0
        n = self.seg.get_num_frames_total()
        return int(n) if n is not None else 0

    def get_frame_shape(self):
        if self.seg is None:
            return None
        return tuple(self.seg.get_frame_shape()) if self.seg.get_frame_shape() is not None else None

    def get_frame_image(self, frame_index):
        """Return an RGB numpy image for the requested frame index.
        Uses `get_video_frame(..., show_masks=True)` when a video is available (primary path).
        Falls back to `visualize_segmentations_on_image` if video access fails or is not present.
        """
        if self.seg is None:
            raise RuntimeError('No Segmentations file opened')

        # Clamp the requested index to available frames if known.
        num_frames = self.get_num_frames_total()
        if num_frames > 0:
            frame_index = max(0, min(int(frame_index), num_frames - 1))

        # Primary rendering path: ask Segmentations to return the video frame with masks.
        if self.video_path is not None:
            try:
                img = self.seg.get_video_frame(self.video_key, int(frame_index), resize_to_mask_shape=True, show_masks=True)
                return img
            except Exception:
                # fall through to image-only rendering
                pass

        # Fallback: create an image with segmentations drawn (black background if no video)
        try:
            img = self.seg.visualize_segmentations_on_image(int(frame_index), img_rgb=None, show_masks=True)
            return img
        except Exception as e:
            raise RuntimeError(f'Failed to render frame {frame_index}: {e}')

    def close(self):
        if self.seg is not None:
            try:
                self.seg.close()
            except Exception:
                pass
            self.seg = None

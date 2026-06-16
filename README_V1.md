CETI Segmentation Explorer — V1

But minimal: open segmentations HDF5, optionally open video, display a single frame with segmentations, navigate frames.

Quick start

1) Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # or `.\.venv\Scripts\activate` on Windows
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Create the synthetic test dataset (optional but useful):

```bash
python explorer/data/create_synthetic_dataset.py
```

This will write `data/videos/synthetic_test.mp4` and `data/segmentations/synthetic_test.hdf5`.

4) Run the explorer UI:

```bash
python -m explorer.main
```

Notes

- The UI uses `Segmentations.get_video_frame(..., show_masks=True)` as the primary rendering path and falls back to `visualize_segmentations_on_image` when no video is available.
- If `decord` is missing, video reading may not be available; the UI will still render segmentations on a black image using `visualize_segmentations_on_image`.
- The UI is intentionally minimal (no continuous playback, no GPS/3D, no export).

Files created by this V1:

- `explorer/main.py` — PyQt6 application entrypoint
- `explorer/services/segmentation_service.py` — data access wrapper around `Segmentations`
- `explorer/widgets/image_widget.py` — small image display widget
- `explorer/data/create_synthetic_dataset.py` — script to build a small HDF5 + video for testing


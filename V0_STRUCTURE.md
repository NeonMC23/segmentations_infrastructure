CETI Segmentation Explorer — V0 Structure

Copyright (c) 2025 Joseph DelPreto / MIT CSAIL and Project CETI
Authored for this change by Neon.

Purpose
- Describe the architecture and data flow required for V0 (analysis-only, no GUI).
- Identify minimal APIs to load frames, segmentation contours, bounding boxes, centroids, IDs, timestamps, GPS, altitude, and speed.

1. Objectives for V0
- Validate that segmentation HDF5 files, drone metadata HDF5 files, and associated videos can be read using the existing CETI APIs.
- Provide a concise reference of classes, dependencies, and data flow to support development of V1+ GUI.

2. Main components (files & responsibilities)
- Segmentations (Segmentations.py)
  - Role: Open and read segmentation HDF5 files; provide per-frame access to masks (contours), bounding boxes, centroids, and orientations; expose visualization helpers and video frame access.
  - Key entry points: `Segmentations(h5_filepath, writable=False, video_filepaths=None)`, `get_video_frame(video_key, frame_index, ...)`, `get_masks_contours(frame_index, as_dict=True)`, `get_bounding_boxes_4xy(bounding_box_key, frame_index, as_dict=True)`, `get_centroid_xy(frame_index, whale_index)`, `get_all_id_names()`, `get_id_numbers()`.

- DroneVideos (DroneVideos.py)
  - Role: Open drone metadata HDF5 files per drone, load per-video arrays including aligned timestamps, GPS (lat/lon), altitudes, and estimated speeds; provide synchronization utilities across drones; lazy video reader creation.
  - Key entry points: `DroneVideos(drone_data_hdf5_filepaths, video_dirs=None, custom_epoch_time_s=None)`, `get_drone_data(video_key)`, `get_frame_timestamps_s(video_key)`, `get_frame_timestamps_str(video_key)`, `get_frame_img(video_key, frame_index)`.

- DecordVideoReaderWrapper (DecordVideoReaderWrapper.py)
  - Role: Wrap `decord.VideoReader` to avoid uncontrolled memory growth by periodically seeking and to provide safe `__getitem__` access for frames.

- Helpers (helpers/helpers_various.py)
  - Role: convenience utilities such as `get_video_reader()`, `load_frame()`, `scale_image()`, `draw_text_on_image()`, GPS/time conversions, and other small helpers used by `Segmentations` and `DroneVideos`.

3. Data flow (end-to-end example)
1. Instantiate `Segmentations` with a segmentation HDF5 and optional `video_filepaths` mapping. The constructor opens the HDF5 and references datasets (masks, bounding boxes, centroids, orientations, frames_are_segmented, whale_segmentations_exist, annotations).
2. Instantiate `DroneVideos` with a mapping of drone HDF5 files. The class loads per-video arrays: `timestamps_s`, `timestamps_str`, `latitudes`, `longitudes`, `altitudes_relative_m`, `speed_horizontal_m_s`, etc.
3. To inspect a given `frame_index`:
   - Get timestamp: `ts = droneVideos.get_frame_timestamps_s(video_key)[frame_index]` or `droneVideos.get_frame_timestamps_str(video_key)[frame_index]`.
   - Get telemetry: `drone_data = droneVideos.get_drone_data(video_key)` then `lat = drone_data['latitudes'][frame_index]`, `lon = drone_data['longitudes'][frame_index]`, `alt = drone_data['altitudes_relative_m'][frame_index]`, `speed = drone_data['speed_horizontal_m_s'][frame_index]`.
   - Get video frame: prefer `segmentations.get_video_frame(video_key, frame_index, resize_to_mask_shape=True)` which returns an RGB numpy image aligned to `frame_shape`. Fallback: `droneVideos.get_frame_img(...)` or `get_video_reader()` + `load_frame()`.
   - Get segmentation contours for drawing: `masks = segmentations.get_masks_contours(frame_index, as_dict=True)` returns a dict mapping whale_index->list_of_contours (each contour Px2).
   - Get bounding boxes: `boxes = segmentations.get_bounding_boxes_4xy('full', frame_index, as_dict=True)` returns dict whale_index->8-values (xyxyxyxy) which can be reshaped to (4,2) for polygon drawing.
   - Draw on image using OpenCV (convert RGB to BGR for drawing, then back to RGB if desired).

4. HDF5 dataset conventions (used by Segmentations)
- Masks: `masks` dataset storing contours as int with fillvalue `-1` and shape `[N_frames, N_whales, N_contours, N_points, 2]` (x,y). Use `get_mask_contours()`/`get_masks_contours()` for per-frame access.
- Bounding boxes: `bounding_boxes_<key>_4xy` where `<key>` in `['full','head','tail']`, shape `[N_frames, N_whales, 8]` (xyxyxyxy) with `np.nan` fill for missing boxes.
- Centroids: `centroids_xy` shape `[N_frames, N_whales, 2]` with `np.nan` for missing entries.
- Orientations: `orientations_rad_confidence` shape `[N_frames, N_whales, 2]` (angle_rad, confidence).
- Frames segmented mask: `frames_are_segmented` (uint8) and `whale_segmentations_exist` (uint8) indicate presence.
- Annotations stored under `annotations` group (ids, behaviors, events, notes) with various typed datasets.

5. Minimal entry points (quick reference)
- Open segmentations: `Segmentations(h5_filepath=..., writable=False)`
- Open drone metadata: `DroneVideos(drone_data_hdf5_filepaths={key:path}, video_dirs=None)`
- Get frame timestamps: `droneVideos.get_frame_timestamps_s(video_key)`
- Get drone telemetry: `droneVideos.get_drone_data(video_key)`
- Get video frame: `segmentations.get_video_frame(video_key, frame_index, ...)` or `droneVideos.get_frame_img(...)` or `get_video_reader()` + `load_frame()`
- Get contours: `segmentations.get_masks_contours(frame_index, as_dict=True)`
- Get bounding boxes: `segmentations.get_bounding_boxes_4xy('full', frame_index, as_dict=True)`
- Get IDs: `segmentations.get_all_id_names()`, `segmentations.get_id_numbers()`

6. Dependencies and practical notes
- Required or recommended packages: `h5py`, `numpy`, `opencv-python`, `decord` (recommended for performance), `matplotlib` (optional), `distinctipy` (used for color palettes), `ffmpeg-python` (optional for video encoding tasks).
- Important practical points:
  - `Segmentations.get_video_frame()` returns RGB images; OpenCV drawing requires BGR input — convert explicitly.
  - Many datasets are large; prefer per-frame reads (e.g. `get_masks_contours(frame_index)`) over `get_all_*` reads unless necessary.
  - `DecordVideoReaderWrapper` mitigates memory growth but always close readers when done.

7. Recommendations for V1
- Add small synthetic HDF5 fixtures and unit tests to CI to ensure data-loading APIs remain stable.
- Document `video_key` naming conventions and mapping between segmentation files and drone metadata for reproducible sync.
- Provide helper wrappers for safe drawing and coordinate conversion to centralize RGB/BGR handling.


"""Corrected V0 test script: load segmentation HDF5, drone metadata, retrieve a frame, draw segmentations and print telemetry.

Usage:
  python test_data_loading_fixed.py --segmentations path/to/VIDEO_segmentations.hdf5 --drone_h5 path/to/CETI_metadata.hdf5 --video_key 1688832540499 --frame 3

This variant fixes drawing/text bugs and is defensive when deps/files are missing.
"""
import argparse
import os
import sys

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np

from segmentation_infrastructure.Segmentations import Segmentations
from segmentation_infrastructure.DroneVideos import DroneVideos
from segmentation_infrastructure.helpers.helpers_various import get_video_reader, load_frame, scale_image, draw_text_on_image


def parse_args():
    p = argparse.ArgumentParser(description='V0 test: load segmentation + drone metadata and visualize a frame')
    p.add_argument('--segmentations', required=True, help='Path to segmentations HDF5 file')
    p.add_argument('--drone_h5', required=True, help='Path to drone metadata HDF5 file (single)')
    p.add_argument('--video_key', required=True, help='Video key (e.g. 1688832540499)')
    p.add_argument('--frame', type=int, default=0, help='Frame index to display')
    p.add_argument('--no_display', action='store_true', help='Do not open an image window (headless)')
    p.add_argument('--output', default=None, help='If provided, save annotated image to this path')
    return p.parse_args()


def draw_contours_and_boxes(img_rgb, segmentations, frame_index, bounding_box_key='full'):
    """Draw contours and bounding boxes onto a copy of img_rgb and return RGB image."""
    if img_rgb is None:
        return None
    # Ensure uint8
    try:
        img_uint8 = img_rgb.astype(np.uint8)
    except Exception:
        img_uint8 = img_rgb
    # Convert RGB -> BGR for OpenCV drawing
    if cv2 is not None:
        try:
            out_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        except Exception:
            out_bgr = img_uint8
    else:
        out_bgr = img_uint8

    # Draw contours
    try:
        if segmentations.have_masks():
            masks_dict = segmentations.get_masks_contours(frame_index=frame_index, as_dict=True)
            for whale_index, contours in masks_dict.items():
                if contours is None:
                    continue
                color = tuple(int(x) for x in np.random.randint(0, 255, size=(3,)))
                for cnt in contours:
                    if cnt is None or len(cnt) == 0:
                        continue
                    try:
                        pts = cnt.astype(np.int32).reshape((-1, 1, 2))
                        if cv2 is not None:
                            cv2.drawContours(out_bgr, [pts], -1, color, thickness=2)
                    except Exception:
                        pass
    except Exception:
        pass

    # Draw bounding boxes
    try:
        boxes_dict = segmentations.get_bounding_boxes_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index, as_dict=True)
        for whale_index, box in boxes_dict.items():
            if box is None:
                continue
            try:
                pts = box.reshape((4, 2)).astype(np.int32).reshape((-1, 1, 2))
                if cv2 is not None:
                    cv2.polylines(out_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            except Exception:
                continue
    except Exception:
        pass

    # Convert back to RGB
    try:
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB) if cv2 is not None else out_bgr
    except Exception:
        out_rgb = out_bgr
    return out_rgb


def main():
    args = parse_args()

    if not os.path.exists(args.segmentations):
        print('ERROR: segmentations file not found:', args.segmentations)
        sys.exit(2)
    if not os.path.exists(args.drone_h5):
        print('ERROR: drone HDF5 file not found:', args.drone_h5)
        sys.exit(2)

    if cv2 is None:
        print('WARNING: OpenCV not available (cv2). Display/save may be limited.')

    print('Opening Segmentations:', args.segmentations)
    segmentations = Segmentations(h5_filepath=args.segmentations, writable=False)

    print('Opening DroneVideos for one file:', args.drone_h5)
    drone_map = {args.video_key: args.drone_h5}
    droneVideos = DroneVideos(drone_data_hdf5_filepaths=drone_map, video_dirs=None)

    num_frames = segmentations.get_num_frames_total()
    print('Segmentations reports %d frames total' % num_frames)
    frame_index = args.frame
    if num_frames is not None:
        if frame_index < 0 or frame_index >= num_frames:
            print('Requested frame %d out of bounds; clamping to valid range' % frame_index)
            frame_index = max(0, min(frame_index, num_frames - 1))

    # Get telemetry
    try:
        timestamps_s = droneVideos.get_frame_timestamps_s(video_key=args.video_key)
        timestamps_str = droneVideos.get_frame_timestamps_str(video_key=args.video_key)
        drone_data = droneVideos.get_drone_data(video_key=args.video_key)
    except Exception as e:
        print('ERROR obtaining drone data:', e)
        timestamps_s = None
        timestamps_str = None
        drone_data = None

    if timestamps_s is not None:
        ts = timestamps_s[frame_index]
        ts_str = timestamps_str[frame_index] if timestamps_str is not None else str(ts)
    else:
        ts = None
        ts_str = 'N/A'

    def safe_get(arrayname):
        try:
            return drone_data[arrayname][frame_index]
        except Exception:
            return None

    lat = safe_get('latitudes')
    lon = safe_get('longitudes')
    alt = safe_get('altitudes_relative_m')
    speed = safe_get('speed_horizontal_m_s')

    print('\nFrame: %d | timestamp: %s | lat: %s | lon: %s | alt: %s | speed: %s' % (frame_index, ts_str, str(lat), str(lon), str(alt), str(speed)))

    # Load image (multiple fallbacks)
    img = None
    try:
        img = segmentations.get_video_frame(video_key=args.video_key, frame_index=frame_index, resize_to_mask_shape=False,
                                            show_masks=False, show_centroids=False, show_orientations=False, show_boxes=None)
        if img is not None:
            img = img.astype(np.uint8)
    except Exception:
        img = None

    if img is None:
        try:
            img = droneVideos.get_frame_img(video_key=args.video_key, frame_index=frame_index)
        except Exception:
            img = None

    if img is None and hasattr(segmentations, '_video_filepaths') and isinstance(segmentations._video_filepaths, dict):
        video_path = segmentations._video_filepaths.get(args.video_key, None)
        if video_path is None and len(segmentations._video_filepaths) > 0:
            # pick first available
            video_path = list(segmentations._video_filepaths.values())[0]
            if isinstance(video_path, dict):
                # if mapping keyed by video keys
                try:
                    video_path = list(video_path.values())[0]
                except Exception:
                    video_path = None
        if video_path is not None and os.path.exists(video_path):
            try:
                vr, fps, nframes = get_video_reader(video_path, method='decord_wrapper')
                img = load_frame(vr, frame_index)
                try:
                    if hasattr(vr, 'close'):
                        vr.close()
                except Exception:
                    pass
            except Exception as e:
                print('Fallback video reader failed:', e)

    if img is None:
        print('WARNING: Could not load any video frame; creating a black canvas of frame_shape to draw overlays if possible.')
        frame_shape = segmentations.get_frame_shape() or [1920, 1080]
        w, h = int(frame_shape[0]), int(frame_shape[1])
        img = np.zeros((h, w, 3), dtype=np.uint8)

    # Annotate
    annotated = draw_contours_and_boxes(img, segmentations, frame_index, bounding_box_key='full')

    # Draw overlay text (convert to BGR for helper)
    overlay_text = 'Frame %d | ts %s' % (frame_index, ts_str)
    try:
        if cv2 is not None:
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            draw_text_on_image(annotated_bgr, overlay_text, pos=(10, 10), text_color_bgr=(255, 255, 255), font_scale=1.0)
            annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    # Display or save
    if not args.no_display and cv2 is not None:
        try:
            annotated_disp = scale_image(annotated, target_width=1200, target_height=None)
            cv2.imshow('V0 test annotated', cv2.cvtColor(annotated_disp, cv2.COLOR_RGB2BGR))
            print('Press any key in the image window to close...')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print('Display failed:', e)
    else:
        out_path = args.output or os.path.join(os.getcwd(), 'v0_test_annotated.png')
        if cv2 is not None:
            try:
                cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                print('Saved annotated image to', out_path)
            except Exception as e:
                print('Could not save annotated image:', e)
        else:
            print('OpenCV not available; cannot save annotated image to disk.')

    # Clean up
    try:
        segmentations.close()
    except Exception:
        pass

    try:
        del droneVideos
    except Exception:
        pass


if __name__ == '__main__':
    main()

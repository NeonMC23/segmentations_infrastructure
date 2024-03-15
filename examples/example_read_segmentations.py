
############
#
# Copyright (c) 2024 Joseph DelPreto / MIT CSAIL and Project CETI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Created 2023-2024 by Joseph DelPreto [https://josephdelpreto.com].
# [add additional updates and authors as desired]
#
############

import numpy as np
import cv2
import os

from segmentation_infrastructure.Segmentations import Segmentations
from segmentation_infrastructure.helpers.helpers_various import scale_image

#######################################################################
# Configuration
# TODO: Update the filepath below as desired for the files you want to load.

current_script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_script_dir, '..', 'data')
video_num = 1688829151574
# video_num = 1688830960531
h5_filepath = os.path.join(data_dir, 'segmentations',
                           '%013d_segmentations.hdf5' % video_num)

# If you would not like to read videos, you can just define an empty dictionary:
# video_filepaths = {}
video_filepaths = {
  'original'                : os.path.join(data_dir, '%013d_compressed.mp4' % video_num),
  # 'only_masks'              : os.path.join(data_dir, '%013d_segmentations.mp4' % video_num),
  # 'masks_boxes_vectors'     : os.path.join(data_dir, '%013d_segmentations_boundingBoxes_orientations.mp4' % video_num),
  # 'masks_splitBoxes_vectors': os.path.join(data_dir, '%013d_segmentations_boundingBoxes_head_fluke.mp4' % video_num),
}

# If using videos, will scale example outputs to fit on the monitor.
img_display_screenRatio = 0.95

#######################################################################
# Create a Segmentations instance with the desired input data.
print()
print('-'*50)
print('Opening file %s' % h5_filepath)

# Open the segmentations data.
segmentations = Segmentations(h5_filepath=h5_filepath, video_filepaths=video_filepaths,
                              writable=False)

# Get some basic information about the segmentations.
num_frames = segmentations.get_num_frames_total()
num_frames_segmented = segmentations.get_num_frames_segmented()
num_whales = segmentations.get_num_whales()
frame_shape = segmentations.get_frame_shape()

print(' See %d total frames' % num_frames)
print(' See %d frames that were analyzed' % num_frames_segmented)
print(' See %d maximum whale indexes' % num_whales)
print(' See a single mask shape of', frame_shape)
print()

#######################################################################
# SEGMENTATION DATA OPTION 1: Get data for all frames.
#
# masks will be NxIxHxW
#   N is the number of frames
#   I is the maximum number of whales
#   HxW is the resolution of the video frame corresponding to the mask
#   An entry at mask[frame_index, whale_index, y, x] is 0 or 1 indicating whether the whale is in that pixel
#
# There are multiple bounding boxes per whale, with keys 'full', 'head', and 'tail'.
# For each one, bounding_boxes_4xy will be NxIx8
#   N is the number of frames
#   I is the maximum number of whales
#   The 8 elements represent the bounding box as xyxyxyxy,
#     which is xy for each box corner probably in a CW fashion around the box.
#   bounding_boxes_4xy[frame_index, whale_index, :] will be all np.nan if there was no box for that whale at that frame
#
# centroids_xy will be NxIx2
#   N is the number of frames
#   I is the maximum number of whales
#   The 2 elements represent the pixel position as (x, y)
#   centroids_xy[frame_index, whale_index, :] will be all np.nan if there was no box for that whale at that frame
#
# orientations_rad will be NxIx2
#   N is the number of frames
#   I is the maximum number of whales
#   The 2 elements represents the direction in radians and the confidences.
#   orientations_rad[frame_index, whale_index, :] will be all np.nan if there was no box for that whale at that frame.
#
# frames_are_segmented will be Nx1
#   N is the number of frames
#   Each element is 0 or 1, indicating whether segmentation was applied to that frame.
#   Note that this is seperate from whether any segmentations were found in the frame.
#
# whale_segmentations_exist will be NxI
#   N is the number of frames
#   I is the maximum number of whales
#   Each entry is 0 or 1, indicating whether that whale was found in that frame.
#
# whale_ids will be Ix1
#   I is the maximum number of whales
#   Each entry is the ID (whale name) associated with that whale index.
#
#######################################################################

print('-'*50)
print('Getting data for all frames and whales')

# Specify which bounding box type to use in the below examples.
# Can be 'full', 'head', or 'tail'
bounding_box_keys = segmentations.get_bounding_box_keys() # get the available types for reference
bounding_box_key = 'full'

# Get all masks.
# The entire matrix would be too large to fit in memory, so this will return
#   an h5py Dataset instead of a numpy array.
#   It can be used just like a numpy array, so you don't need to worry about it,
#   but the data will be kept on disk instead of being loaded into memory.
if segmentations.have_masks(): # filtered versions of the data may not have masks
  masks = segmentations.get_all_masks()

# Get all bounding boxes of the desired type.
# Will a numpy array loaded into memory.
bounding_boxes_4xy = segmentations.get_all_bounding_boxes_4xy(bounding_box_key=bounding_box_key)
# If desired, can reshape to have each box corner as a matrix row.
bounding_boxes_4xy_reshaped = bounding_boxes_4xy.reshape((num_frames, num_whales, 4, 2))

# Get all centroids.
# Will a numpy array loaded into memory.
centroids_xy = segmentations.get_all_centroids_xy()

# Get all orientations with confidences.
# Will a numpy array loaded into memory.
orientations_rad_confidence = segmentations.get_all_orientations_rad_confidence()
orientations_rad = orientations_rad_confidence[:, :, 0]
orientations_confidence = orientations_rad_confidence[:, :, 1]

# Get whether frames were segmented.
frames_are_segmented = segmentations.get_frames_are_segmented()

# Get whether segmentations were found for each whale in each frame.
whale_segmentations_exist = segmentations.get_whale_segmentations_exist()

# Get the whale ID for each whale index.
whale_ids = segmentations.get_whale_ids()

print(' See the following matrix shapes')
if segmentations.have_masks(): # filtered versions of the data may not have masks
  print('  masks: type %s of %s | shape' % (type(masks), masks.dtype), masks.shape)
print('  bounding_boxes_4xy: type %s of %s | shape' % (type(bounding_boxes_4xy), bounding_boxes_4xy.dtype), bounding_boxes_4xy.shape)
print('  bounding_boxes_4xy_reshaped: type %s of %s | shape' % (type(bounding_boxes_4xy_reshaped), bounding_boxes_4xy_reshaped.dtype), bounding_boxes_4xy_reshaped.shape)
print('  centroids_xy: type %s of %s | shape' % (type(centroids_xy), centroids_xy.dtype), centroids_xy.shape)
print('  orientations_rad_confidence: type %s of %s | shape' % (type(orientations_rad_confidence), orientations_rad_confidence.dtype), orientations_rad_confidence.shape)
print('  orientations_rad: type %s of %s | shape' % (type(orientations_rad), orientations_rad.dtype), orientations_rad.shape)
print('  orientations_confidence: type %s of %s | shape' % (type(orientations_confidence), orientations_confidence.dtype), orientations_confidence.shape)
print('  frames_are_segmented: type %s of %s | shape' % (type(frames_are_segmented), frames_are_segmented.dtype), frames_are_segmented.shape)
print('  whale_segmentations_exist: type %s of %s | shape' % (type(whale_segmentations_exist), whale_segmentations_exist.dtype), whale_segmentations_exist.shape)
print('  whale_ids: type %s of %s | length' % (type(whale_ids), type(whale_ids[0])), len(whale_ids))

###################################
# Use the matrices spanning all frames to get data for a particular frame and whale instance.
frame_index = 3
whale_index = 1

if segmentations.have_masks(): # filtered versions of the data may not have masks
  # Get the mask for this whale and frame; will have 1s where the whale is present in the frame.
  frame_mask_forWhale = masks[frame_index, whale_index, :,:]
  # Get the masks for all whales for this frame.
  frame_masks_allWhales = np.squeeze(masks[frame_index, :, :,:])

# Get the bounding box of this whale in this frame; will be all np.nan if the whale is not present.
bounding_box_4xy = bounding_boxes_4xy[frame_index, whale_index, :]
bounding_box_4xy_reshaped = bounding_box_4xy.reshape((4,2)) # if desired, can reshape to have each corner as a matrix row
# Check if there actually was a bounding box for this frame/whale
# using the special values or the explicit array.
bounding_box_is_valid = whale_segmentations_exist[frame_index, whale_index]
bounding_box_is_valid = not np.all(np.isnan(bounding_boxes_4xy))

# Get the centroid for this whale in this frame; will be all np.nan if the whale is not present.
centroid_xy = centroids_xy[frame_index, whale_index, :]
# Check if there actually was a bounding box for this frame/whale
# using the special values or the explicit array.
centroid_is_valid = whale_segmentations_exist[frame_index, whale_index]
centroid_is_valid = not np.all(np.isnan(centroid_xy))

# Get the orientation for this whale in this frame; will be all np.nan if the whale is not present.
orientation_rad_confidence = orientations_rad_confidence[frame_index, whale_index, :]
orientation_rad = orientations_rad[frame_index, whale_index]
orientation_confidence = orientations_confidence[frame_index, whale_index]
# Check if there actually was a bounding box for this frame/whale
# using the special values or the explicit array.
orientation_is_valid = whale_segmentations_exist[frame_index, whale_index]
orientation_is_valid = not np.all(np.isnan(orientation_rad_confidence))

# Get whether segmentations were found for this whale in each frame.
whale_segmentations_exist_forWhale = segmentations.whale_segmentation_exists(whale_index, frame_indexes=None)

print()
print(' See the following sample matrices for frame index %d and whale index %d' % (frame_index, whale_index))
if segmentations.have_masks(): # filtered versions of the data may not have masks
  print(' frame_mask_forWhale:', frame_mask_forWhale)
print(' bounding_box_4xy', bounding_box_4xy, ' (is valid? %d)' % bounding_box_is_valid)
print(' centroid_xy', centroid_xy, ' (is valid? %d)' % centroid_is_valid)
print(' orientation_rad', orientation_rad, 'orientation_confidence', orientation_confidence, ' (is valid? %d)' % orientation_is_valid)
print()

#######################################################################
# SEGMENTATION DATA OPTION 2: Get data from a particular frame and, optionally, a particular whale.
# This should be faster for a single frame, since the class does not need to read the
#   whole matrix from disk and convert it to a numpy array (it will only load the desired frame).
# The same comments apply above regarding the format, except it is now easier to know if the segmentation existed:
#   bounding_boxes_4xy, centroids_xy, and orientations_rad will be None
#   if there was no segmentation at a particular frame for a particular whale.
#######################################################################

print('-'*50)
print('Getting data for a particular frame and whale')
frame_index = 3
whale_index = 1

# Check whether the segmentation exists for this whale in this frame.
whale_is_present = whale_segmentations_exist[frame_index, whale_index]

# Get the desired mask.
if segmentations.have_masks(): # filtered versions of the data may not have masks
  frame_mask_forWhale = segmentations.get_mask(frame_index=frame_index, whale_index=whale_index)
  frame_masks_allWhales = segmentations.get_masks(frame_index=frame_index) # can return a matrix, or a dict mapping whale index to mask
  frame_masks_allWhales_dict = segmentations.get_masks(frame_index=frame_index,
                                                       as_dict=True) # can return a matrix, or a dict mapping whale index to mask

# Get the bounding box of this whale in this frame, and check if the segmentation existed in this frame.
bounding_box_4xy = segmentations.get_bounding_box_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index, whale_index=whale_index)
if bounding_box_4xy is not None:
  bounding_box_4xy_reshaped = bounding_box_4xy.reshape((4,2)) # if desired, can reshape to have each corner as a matrix row
else:
  bounding_box_4xy_reshaped = None
# Get bounding boxes for all whales in this frame.
bounding_boxes_4xy_allWhales = segmentations.get_bounding_boxes_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index)
bounding_boxes_4xy_allWhales_dict = segmentations.get_bounding_boxes_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index,
                                                                         as_dict=True)  # can return a matrix, or a dict mapping whale index to bounding box

# Get the centroid of this whale in this frame; will be None if the segmentation was not present.
centroid_xy = segmentations.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
# Get the centroids of all whales in this frame.
centroids_xy_allWhales = segmentations.get_centroids_xy(frame_index=frame_index)
centroids_xy_allWhales_dict = segmentations.get_centroids_xy(frame_index=frame_index,
                                                             as_dict=True)  # can return a matrix, or a dict mapping whale index to centroid

# Get the orientation of this whale in this frame; will be None if the segmentation was not present.
(orientation_rad, orientation_confidence) = segmentations.get_orientation_rad_confidence(frame_index=frame_index, whale_index=whale_index)
# Get the orientations of all whales in this frame.
orientations_rad_confidence_allWhales = segmentations.get_orientations_rad_confidence(frame_index=frame_index)
orientations_rad_confidence_allWhales_dict = segmentations.get_orientations_rad_confidence(frame_index=frame_index,
                                                                                           as_dict=True)  # can return a matrix, or a dict mapping whale index to orientations

print(' See the following matrix shapes for frame index %d and whale index %d' % (frame_index, whale_index))
if segmentations.have_masks(): # filtered versions of the data may not have masks
  print('  frame_mask_forWhale: ', frame_mask_forWhale.shape)
  print('  frame_masks_allWhales: ', frame_masks_allWhales.shape)
  print('  frame_masks_allWhales_dict: dict len %d with entry 1 shape' % len(frame_masks_allWhales_dict),
        frame_masks_allWhales_dict[1].shape if frame_masks_allWhales_dict[1] is not None else None)
print('  bounding_box_4xy: ', bounding_box_4xy.shape if bounding_box_4xy is not None else None)
print('  bounding_box_4xy_reshaped: ', bounding_box_4xy_reshaped.shape if bounding_box_4xy_reshaped is not None else None)
print('  bounding_boxes_4xy_allWhales: ', bounding_boxes_4xy_allWhales.shape)
print('  bounding_boxes_4xy_allWhales_dict: dict len %d with entry 1 shape' % len(bounding_boxes_4xy_allWhales_dict),
      bounding_boxes_4xy_allWhales_dict[1].shape if bounding_boxes_4xy_allWhales_dict[1] is not None else None)
print('  centroid_xy: ', centroid_xy.shape if centroid_xy is not None else None)
print('  centroids_xy_allWhales: ', centroids_xy_allWhales.shape)
print('  centroids_xy_allWhales_dict: dict len %d with entry 1 shape' % len(centroids_xy_allWhales_dict),
      centroids_xy_allWhales_dict[1].shape if centroids_xy_allWhales_dict[1] is not None else None)
print('  orientation_rad: ', type(orientation_rad) if orientation_rad is not None else None)
print('  orientation_confidence: ', type(orientation_confidence) if orientation_confidence is not None else None)
print('  orientations_rad_confidence_allWhales: ', orientations_rad_confidence_allWhales.shape)
print('  orientations_rad_confidence_allWhales_dict: dict len %d with entry 1 type' % len(orientations_rad_confidence_allWhales_dict),
      type(orientations_rad_confidence_allWhales_dict[1]) if orientations_rad_confidence_allWhales_dict[1] is not None else None)
print()
print(' See the following sample matrices for frame index %d and whale index %d' % (frame_index, whale_index))
if segmentations.have_masks(): # filtered versions of the data may not have masks
  print(' frame_mask_forWhale:', frame_mask_forWhale)
print(' bounding_box_4xy', bounding_box_4xy)
print(' centroid_xy', centroid_xy)
print(' orientation_rad', orientation_rad)
print(' orientation_confidence', orientation_confidence)
print()

###############################
# File operations
###############################

# Make a copy of the data.

print('-'*50)
print('Making a copy of the segmentations data')

new_h5_filepath = h5_filepath.replace('.hdf5', '_testCopy.hdf5')

segmentations_copy = segmentations.copy(
    # Specify the filepath for the new HDF5 file.
    new_h5_filepath,
    overwrite_destination_hdf5_file_if_exists=True,
    # Specify whether to include masks in the copied file.
    include_masks=True,
    # Specify whether to create a new Segmentations object pointing to the new data.
    # If True, will return the new Segmentations object that is optionally in a writable mode.
    # Otherwise, will return None.
    open_segmentations_object=True,
    new_segmentations_object_writable=True)
print()

# Remove all masks from the data.

print('-'*50)
print('Removing all masks from a copy of the data')

segmentations_copy.remove_masks_dataset()
print()

#######################################################################
# FILTER AND SMOOTH WHALE INSTANCES
#######################################################################

# Create a version of the data that removes whale instances
#  that are not present in enough frames.

print('-'*50)
print('Filtering the whale instances based on their frame count')

whale_frame_count_threshold = 150

segmentations_filtered = segmentations_copy.filter_whale_instances_byCount(
    # Specify the threshold for how many frames a whale instance should be present.
    # Heuristically, 150 seems to work fairly well.
    # To help choose, you could also use segmentations.get_whale_frame_counts()
    #  which will return the number of frames each whale is present.
    min_frame_count=whale_frame_count_threshold,
    # Specify whether to remove the masks dataset from the HDF5 file.
    # If False, will filter the masks.  This can be slow.
    remove_masks_dataset=True,
    # Specify whether to create a new HDF5 file for the results.
    # If True, will return a Segmentations object for the new HDF5 file.
    # If False, will edit the current HDF5 file (and return None).
    create_new_hdf5_file=True,
    overwrite_destination_hdf5_file_if_exists=True,
)

num_whales_filtered = segmentations_filtered.get_num_whales()
print('See %d whales after filtering with a frame count threshold of %d' % (num_whales_filtered, whale_frame_count_threshold))
print()

#######################################################################
# SMOOTH WHALE INSTANCES
#######################################################################

print('-'*50)
print('Smoothing whale instances')

# print('-'*50)
# print('Smoothing the whale instance masks using a rolling window')
# # Edit the current segmentations to have smoothed masks.  Segmentations must be opened as writable.
# # NOTE: Will edit the current HDF5 file, so you should first copy the file if you want a backup of the original data.
# # For each pixel of each frame, will compute the mean of the masks from window_size_preCenter frames before it
# #  through window_size_postCenter frames after it.
# #  That pixel will be 1 if the mean is at least rolling_mean_threshold, and 0 otherwise.
# segmentations_copy.smooth_masks(
#     # Specify the rolling window size and centering.
#     window_size_preCenter=20, window_size_postCenter=20,
#     # Specify the threshold to use on the rolling mean at each pixel.
#     # NOTE: If it is set to 0.5, the filter effectively becomes a rolling median filter.
#     rolling_mean_threshold=0.5,
#     # If desired, only smooth specified whale indexes.
#     whale_indexes_toSmooth='all',
#     # Print status updates.
#     print_status=True)
# smoothed_masks = segmentations_copy.get_all_masks()

print()
print('Smoothing the centroids using a rolling window')
# Get smoothed versions of the centroids.
smoothed_centroids_xy = segmentations.get_all_centroids_xy(
    # Apply the smoothing filter specified below.
    apply_smoothing_filter=True,
    # Specify the rolling window size and centering.
    smoothing_window_size_preCenter=20, smoothing_window_size_postCenter=20,
    # If desired, only smooth specified whale indexes.
    whale_indexes_toSmooth='all', # a list of indexes or 'all'
    # Specify whether the HDF5 file should be edited,
    #  or whether the computation should only be done in memory.
    smoothing_edits_hdf5_data=False,
    # If desired, plot the original and smoothed centroid coordinates.
    smoothed_whale_indexes_toPlot=[1], # a list of indexes or 'all'
    # Print status updates.
    print_smoothing_status=False)

print()
print('Smoothing the bounding boxes using a rolling window')
# Get smoothed versions of the bounding boxes.
smoothed_bounding_boxes_4xy = segmentations.get_all_bounding_boxes_4xy(
    # Specify which type of box to process.
    bounding_box_key='full', # 'full', 'head', or 'tail'
    # Apply the smoothing filter specified below.
    apply_smoothing_filter=True,
    # Specify the rolling window size and centering.
    smoothing_window_size_preCenter=20, smoothing_window_size_postCenter=20,
    # If desired, only smooth specified whale indexes.
    whale_indexes_toSmooth='all', # a list of indexes or 'all'
    # Specify whether the HDF5 file should be edited,
    #  or whether the computation should only be done in memory.
    smoothing_edits_hdf5_data=False,
    # If desired, plot the original and smoothed bounding box coordinates.
    smoothed_whale_indexes_toPlot=[1], # a list of indexes or 'all'
    # If desired, animate the original and smoothed bounding boxes throughout the video.
    smoothed_whale_indexes_toAnimate=[], # a list of indexes or 'all'
    # Print status updates.
    print_smoothing_status=False)
smoothed_bounding_boxes_4xy_reshaped = smoothed_bounding_boxes_4xy.reshape((num_frames, num_whales, 4, 2)) # if desired, can reshape to have each box corner as a matrix row

print()
print('Smoothing the orientations and confidences using a rolling window')
# Get smoothed versions of the bounding boxes.
smoothed_orientations_rad_confidence = segmentations.get_all_orientations_rad_confidence(
    # Apply the smoothing filter specified below.
    apply_smoothing_filter=True,
    # Specify the rolling window size and centering.
    smoothing_window_size_preCenter=20, smoothing_window_size_postCenter=20,
    # If desired, only smooth specified whale indexes.
    whale_indexes_toSmooth='all', # a list of indexes or 'all'
    # Specify whether the HDF5 file should be edited,
    #  or whether the computation should only be done in memory.
    smoothing_edits_hdf5_data=False,
    # If desired, plot the original and smoothed bounding box coordinates.
    smoothed_whale_indexes_toPlot=[1], # a list of indexes or 'all'
    # Print status updates.
    print_smoothing_status=False)
smoothed_orientations_rad = smoothed_orientations_rad_confidence[:, :, 0]
smoothed_orientations_confidence = smoothed_orientations_rad_confidence[:, :, 1]

print()

#######################################################################
# VIDEO DATA AND VISUALIZATIONS

# Get video frames, optionally annotated with segmentation visualizations.
# Can also visualize segmentations on black images if no videos are handy.

# A returned frame will be HxWx3, where the 3 channels are in RGB order.
# NOTE: H may be slightly greater than the H of the mask and the original drone video,
#  since the MP4 encoding format requires that dimensions are even.
#  For the current segmentation outputs, the drone videos have a height of 1407 so the encoding
#   pads it by adding a row of black pixels at the bottom.
#   The resulting height is then 1408 pixels.
#######################################################################

frame_index_toFetch = 3

print()
print('-'*50)
print('Visualizing segmentations on video frames or on black backgrounds')
for video_key in video_filepaths:
  # Get the original frame from the video.
  print(' Getting a segmented image from video [%s] for frame index %d' % (video_key, frame_index_toFetch))
  img_rgb = segmentations.get_video_frame(video_key=video_key, frame_index=frame_index_toFetch,
                                          resize_to_mask_shape=True,
                                          show_masks=False, show_centroids=False,
                                          show_orientations=False, show_boxes=None)
  print('  See an image with shape', img_rgb.shape)

  # Annotate the video with segmentation visualizations.
  # See the code in the function segmentations.visualize_segmentations() to see how the segmentation is used for this purpose.
  # Note that if masks are not provided in the HDF5 file, the original mask shape is not known
  #  so the video cannot be resized to match and the segmentation locations may be wrong.
  print('  Drawing the centroids, orientations (color-coded by confidence), and bounding boxes (color-coded by type) on the image')
  imgs_rgb_annotated = []
  if segmentations.have_masks(): # filtered versions of the data may not have masks
    imgs_rgb_annotated.append(segmentations.visualize_segmentations(frame_index=frame_index_toFetch,
                                                      show_masks=True,
                                                      show_centroids=False,
                                                      show_orientations=False,
                                                      show_boxes=None,
                                                      show_whale_indexes=True))
  else:
    imgs_rgb_annotated.append(segmentations.get_video_frame(video_key=video_key, frame_index=frame_index_toFetch,
                                                      show_masks=False,
                                                      show_centroids=True,
                                                      show_orientations=False,
                                                      show_boxes=['full'],
                                                      show_whale_indexes=True))
  imgs_rgb_annotated.append(segmentations.get_video_frame(video_key=video_key, frame_index=frame_index_toFetch,
                                                    show_masks=False,
                                                    show_centroids=True,
                                                    show_orientations=True,
                                                    show_boxes=['full'],
                                                    show_whale_indexes=False))
  imgs_rgb_annotated.append(segmentations.get_video_frame(video_key=video_key, frame_index=frame_index_toFetch,
                                                    show_masks=False,
                                                    show_centroids=False,
                                                    show_orientations=False,
                                                    show_boxes=['full', 'head', 'tail'],
                                                    show_whale_indexes=False))

  # Show the original frame and the annotated frames side by side.
  pad_size = round(imgs_rgb_annotated[0].shape[0]*0.01)
  combined_img = cv2.vconcat([
    cv2.hconcat([cv2.copyMakeBorder(img_rgb, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0,0,0)),
                 cv2.copyMakeBorder(imgs_rgb_annotated[0], pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0,0,0))]),
    cv2.hconcat([cv2.copyMakeBorder(imgs_rgb_annotated[1], pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0,0,0)),
                 cv2.copyMakeBorder(imgs_rgb_annotated[2], pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(0,0,0))])
    ]
  )
  # Compute the display size as a ratio of the monitor size.
  cv2.namedWindow('monitorSizeTest', cv2.WINDOW_NORMAL)
  cv2.setWindowProperty('monitorSizeTest', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
  cv2.waitKey(100)
  display_screen_size = cv2.getWindowImageRect('monitorSizeTest')
  cv2.destroyWindow('monitorSizeTest')
  display_screen_width = display_screen_size[2]
  display_screen_height = display_screen_size[3]
  display_screen_aspect_ratio = display_screen_width/display_screen_height
  combined_img_aspect_ratio = combined_img.shape[1]/combined_img.shape[0]
  if display_screen_aspect_ratio > combined_img_aspect_ratio:
    target_height = display_screen_height * img_display_screenRatio
    display_img_width = round(target_height * combined_img_aspect_ratio)
  else:
    target_width = display_screen_width * img_display_screenRatio
    display_img_width = round(target_width)
  # Scale the image accordingly.
  combined_img = scale_image(combined_img, target_width=display_img_width, target_height=None)
  # Show the image.
  window_name = 'Frame at Index %d For Video Key [%s] | Original (top-left) and With Segmentation Annotations' % (frame_index_toFetch, video_key)
  cv2.imshow(window_name, cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
  cv2.moveWindow(window_name, 0, 0)
  print('  Close the image display window to continue')
  cv2.waitKey(0)

  print()

#######################################################################
# Visualize using a graph instead of an image.

import matplotlib.pyplot as plt

print()
print('-'*50)
print('Visualizing segmentations on a graph')

fig = segmentations.visualize_segmentations(frame_index=frame_index_toFetch,
                                            graph=True, fig=None,
                                            show_masks=False,
                                            show_centroids=True,
                                            show_orientations=True,
                                            show_boxes=['full', 'head', 'tail'],
                                            show_whale_indexes=True)
print('  Close the figure to continue')
plt.show(block=True)

#######################################################################
# Visualize whale centroid trajectories.

import matplotlib.pyplot as plt

print()
print('-'*50)
print('Visualizing the centroid trajectories')

fig = segmentations.visualize_centroid_trajectories(
    # Specify frames and whales to visualize.
    # frame_indexes can be 'all' or [start_frame_index, end_frame_index]
    # whale_indexes can be 'all' or a list of whale indexes.
    frame_indexes='all', whale_indexes='all',
    # Filter whales to show based on their duration present if desired.
    # Can be None to show all whales, or a number between 0 and 1
    #   to only show whales that are present in the video for at least that ratio of the video duration.
    whale_duration_ratio_filter=None,
    # Optionally apply a rolling smoothing filter to the centroids.
    apply_smoothing_filter=True,
    smoothing_window_size_preCenter=20,
    smoothing_window_size_postCenter=20)
print('  Close the figure to continue')
plt.show(block=True)

#######################################################################
# Clean up.
#######################################################################
print()
print('-'*50)
print('Closing the files')
segmentations.close()
segmentations_copy.close()
segmentations_filtered.close()

print()
print('-'*50)
print('Happy analyzing!')
print()









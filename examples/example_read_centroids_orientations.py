
############
#
# Copyright (c) 2026 Joseph DelPreto / MIT CSAIL and Project CETI
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
# Created 2023-2026 by Joseph DelPreto [https://josephdelpreto.com].
# [add additional updates and authors as desired]
#
############

import numpy as np
import cv2
import os

from segmentation_infrastructure.Segmentations import Segmentations
from segmentation_infrastructure.helpers.helpers_various import *

#######################################################################
# Configuration

# Point to a file that was downloaded from the Whale Tales annotation dashboard.
h5_filepath = r'raw__2024-10-28__DSWP-DJI_MAVIC3-2__DJI_0323_withAudio_segmentations.hdf5'

#######################################################################
# Create a Segmentations instance with the desired input data.
print()
print('-'*50)
print('Opening file %s' % h5_filepath)

# Open the segmentations data.
segmentations = Segmentations(h5_filepath=h5_filepath, writable=False)

# Get some basic information about the segmentations.
num_frames = segmentations.get_num_frames_total()
num_frames_segmented = segmentations.get_num_frames_segmented()
num_whales = segmentations.get_num_whales()
frame_shape = segmentations.get_frame_shape()

print(' See %d total frames' % num_frames)
print(' See %d frames that were analyzed' % num_frames_segmented)
print(' See %d maximum whale indexes' % num_whales)
print(' See a frame shape of', frame_shape)
print()

#######################################################################
# SEGMENTATION DATA OPTION 1: Get data for all frames.
#
# centroids_xy will be NxIx2
#   N is the number of frames
#   I is the maximum number of whales
#   The 2 elements represent the pixel position as (x, y)
#   centroids_xy[frame_index, whale_index, :] will be all np.nan if there was no box for that whale at that frame
#   The origin is in the top left of the video frame.
#   The x and y maximum limits are given by segmentations.get_frame_shape()
#
# orientations_rad will be NxIx2
#   N is the number of frames
#   I is the maximum number of whales
#   The 2 elements represents the direction in radians and the confidences.
#   orientations_rad[frame_index, whale_index, :] will be all np.nan if there was no box for that whale at that frame.
#   An orientation of 0 points to the right in the frame, and positive orientations rotate CCW from there.
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
# all_id_names and all_id_numbers will be Ix1
#   I is the maximum number of whales
#   Each entry is the ID (whale name) or ID number (persistent) associated with that whale index.
# all_id_species will be Ix1
#   I is the maximum number of whales
#   Each entry is "Sperm Whale", "Pilot Whale", "Frasers Dolphin", or blank
#
#######################################################################

print('-'*50)
print('Getting data for all frames and whales')

# Get all centroids.
centroids_xy = segmentations.get_all_centroids_xy()

# Get all orientations.
orientations_rad_confidence = segmentations.get_all_orientations_rad_confidence()
orientations_rad = orientations_rad_confidence[:, :, 0]
orientations_confidence = orientations_rad_confidence[:, :, 1]

# Get whether frames were segmented.
frames_are_segmented = segmentations.get_frames_are_segmented()

# Get whether segmentations were found for each whale in each frame.
whale_segmentations_exist = segmentations.get_whale_segmentations_exist()

# Get the ID name, number, and species for each whale index.
all_id_names = segmentations.get_all_id_names()
all_id_numbers = segmentations.get_id_numbers()
all_id_species = segmentations.get_all_id_species()

print(' See the following matrix shapes')
print('  centroids_xy: type %s of %s | shape' % (type(centroids_xy), centroids_xy.dtype), centroids_xy.shape)
print('  orientations_rad_confidence: type %s of %s | shape' % (type(orientations_rad_confidence), orientations_rad_confidence.dtype), orientations_rad_confidence.shape)
print('  orientations_rad: type %s of %s | shape' % (type(orientations_rad), orientations_rad.dtype), orientations_rad.shape)
print('  orientations_confidence: type %s of %s | shape' % (type(orientations_confidence), orientations_confidence.dtype), orientations_confidence.shape)
print('  frames_are_segmented: type %s of %s | shape' % (type(frames_are_segmented), frames_are_segmented.dtype), frames_are_segmented.shape)
print('  whale_segmentations_exist: type %s of %s | shape' % (type(whale_segmentations_exist), whale_segmentations_exist.dtype), whale_segmentations_exist.shape)
print('  all_id_names: type %s of %s | length' % (type(all_id_names), type(all_id_names[0])), len(all_id_names))
print('  all_id_numbers: type %s of %s | length' % (type(all_id_numbers), type(all_id_numbers[0])), len(all_id_numbers))
print('  all_id_species: type %s of %s | length %d | unique values %s' % (type(all_id_species), type(all_id_species[0]), len(all_id_species), list(set(all_id_species))))

###################################
# Use the matrices spanning all frames to get data for a particular frame and whale instance.
frame_index = 55
id_number = 6
whale_index = segmentations.get_whale_index_for_id_number(id_number)

# Get the whale name and number (which are shown in Whale Tales).
whale_name = all_id_names[whale_index]
whale_number = all_id_numbers[whale_index]

# Get the centroid for this whale in this frame; will be all np.nan if the whale is not present.
centroid_xy = centroids_xy[frame_index, whale_index, :]
# Check if there actually was a bounding box for this frame/whale
# using the special values or the explicit array.
centroid_is_valid = whale_segmentations_exist[frame_index, whale_index]
centroid_is_valid = not np.all(np.isnan(centroid_xy))

# Get the orientation for this whale in this frame; will be all np.nan if the whale is not present.
orientation_rad_confidence = orientations_rad_confidence[frame_index, whale_index, :]
orientation_rad = orientations_rad[frame_index, whale_index]
# Check if there actually was a bounding box for this frame/whale
# using the special values or the explicit array.
orientation_is_valid = whale_segmentations_exist[frame_index, whale_index]
orientation_is_valid = not np.all(np.isnan(orientation_rad_confidence))

# Get whether segmentations were found for this whale in each frame.
whale_segmentations_exist_forWhale = segmentations.whale_segmentation_exists(whale_index, frame_indexes=None)

print()
print(' See the following samples for frame index %d and whale index %d' % (frame_index, whale_index))
print('  whale_name', whale_name)
print('  whale_number', whale_number)
print('  centroid_xy', centroid_xy, ' (is valid? %d)' % centroid_is_valid)
print('  orientation_rad', orientation_rad, ' (is valid? %d)' % orientation_is_valid)
print('  orientation_deg', np.rad2deg(orientation_rad), ' (is valid? %d)' % orientation_is_valid)
print()


###############################
# Annotations
###############################

print('-'*50)
print('Getting annotations and the edit history')

annotations_ids = segmentations.get_annotations_ids()
annotations_behaviors = segmentations.get_annotations_behaviors()
annotations_notes = segmentations.get_annotations_notes()
history_log = segmentations.get_history()

# print(' See the following annotations of IDs:')
# print_dict(annotations_ids, level=1)
# print(' See the following annotations of behaviors:')
# print_dict(annotations_behaviors, level=1)
# print(' See the following annotations of general notes:')
# print_dict(annotations_notes, level=1)
# print(' See the following history of the HDF5 file:')
# print_dict(history_log, level=1)
print()

#######################################################################
# Clean up.
#######################################################################
print()
print('-'*50)
print('Closing the files')
print('Also trimming datasets and removing unused whale indexes in the copied file')
segmentations.close()

print()
print('-'*50)
print('Happy analyzing!')
print()









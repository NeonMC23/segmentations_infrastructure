
############
#
# Copyright (c) 2025 Joseph DelPreto / MIT CSAIL and Project CETI
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
# Created 2023-2025 by Joseph DelPreto [https://josephdelpreto.com].
# [add additional updates and authors as desired]
#
############

from segmentation_infrastructure.Segmentations import Segmentations
from segmentation_infrastructure.DroneVideos import DroneVideos
from segmentation_infrastructure.helpers.helpers_various import *
import os
current_script_dir = os.path.dirname(os.path.realpath(__file__))

#########################################
# Configuration
#########################################

# Specify the folder with segmentations HDF5 files.
segmentations_data_dir = os.path.join(current_script_dir, '..', 'data', 'segmentations')

# Specify the HDF5 files with extracted drone data.
drone_data_dir = os.path.join(current_script_dir, '..', 'data', 'drones')
drone_data_hdf5_filepaths = {
  'CETI': os.path.join(drone_data_dir, 'CETI-DJI_MAVIC3-1_metadata.hdf5'),
  'DSWP': os.path.join(drone_data_dir, 'DSWP-DJI_MAVIC3-2_metadata.hdf5'),
}

# Define a baby epoch for fetching timestamps in that reference frame.
# Can also be None if it will not be used.
baby_epoch_s = time_str_to_time_s('2023-07-08 11:45:45 -0400')

# Define functions for extracting birth regions.
in_birth_regions_functions = {
  'Before Birth': lambda x: (x < -33*60),
  'During Birth': lambda x: (-33*60 <= x) & (x < 0),
  'After Birth': lambda x: (0 <= x) & (x < 60*(2*60+35)),
  'After After Birth': lambda x: (x >= 60*(2*60+35)),
}

# Specify a video to use for the below examples.
video_key_for_examples = '1688832540499'

#########################################
# Initialization
#########################################
print()

# Open the drone video metadata.
print('Loading preprocessed metadata and sensor data for all videos')
droneVideos = DroneVideos(
    drone_data_hdf5_filepaths=drone_data_hdf5_filepaths,
    video_dirs=None,
    custom_epoch_time_s=baby_epoch_s
)

# Open the segmentations.
print('Loading segmentations for video %s' % video_key_for_examples)
segmentations = Segmentations(
    h5_filepath=os.path.join(segmentations_data_dir, '%s_segmentations.hdf5' % video_key_for_examples)
)

#########################################
# Example usage: whale IDs, numbers, indexes
#########################################

# Get whale IDs.
# These are the whale names, and are ordered by matrix index.
whale_ids = segmentations.get_whale_ids()

# Get whale ID numbers.
# These are the persistent numbers shown in Whale Tales, and are ordered by matrix index.
whale_id_numbers = segmentations.get_whale_id_numbers()

# Helper functions are also available to convert between these.
whale_id = segmentations.get_whale_id(whale_id_number=0)
whale_id = segmentations.get_whale_id(whale_index=0)
whale_index = segmentations.get_whale_index_for_whale_id_number(whale_id_number=0)
whale_id_number = segmentations.get_whale_id_number(whale_index=0)

# Get whether IDs are auto-generated.
ids_is_auto = segmentations.get_whale_ids_is_auto()

# Print whale information to help clarify the above arrays.
print()
print('See the following whale IDs and ID numbers:')
for whale_index in range(len(whale_ids)):
  whale_id_number = whale_id_numbers[whale_index]
  whale_id = whale_ids[whale_index]
  is_auto_id = ids_is_auto[whale_index]
  print('  Matrix index %2d | ID number %2d | auto? %d | ID "%s"' % (whale_index, whale_id_number, is_auto_id, whale_id))

#########################################
# Example usage: timestamps
#########################################

# Get timestamps for each frame of the video.
#   timestamps_s can be standard posix epoch time or baby whale time.
#   timestamps_str will be human-readable strings.
timestamps_s = droneVideos.get_frame_timestamps_s(video_key=video_key_for_examples,
                                                  use_custom_epoch=False) # standard posix epoch time
timestamps_s_babyTime = droneVideos.get_frame_timestamps_s(video_key=video_key_for_examples,
                                                           use_custom_epoch=True) # baby time!
timestamps_str = droneVideos.get_frame_timestamps_str(video_key=video_key_for_examples)

print()
print('See timestamps for video %s' % (video_key_for_examples))
print('  Start time: %s | epoch time %0.3fs | baby time %0.3fs' % (timestamps_str[0], timestamps_s[0], timestamps_s_babyTime[0]))
print('  End time  : %s | epoch time %0.3fs | baby time %0.3fs' % (timestamps_str[-1], timestamps_s[-1], timestamps_s_babyTime[-1]))

#########################################
# Example usage: birth regions
#########################################

print()
print('Checking birth regions for video %s' % (video_key_for_examples))
max_birth_region_str_length = max([len(birth_region) for birth_region in in_birth_regions_functions])
for (birth_region, in_birth_region_function) in in_birth_regions_functions.items():
  is_birth_region = in_birth_regions_functions[birth_region](timestamps_s_babyTime)
  birth_region_indexes = np.where(is_birth_region)
  print('  %s: %5d frames' % (birth_region.ljust(max_birth_region_str_length),
                              np.sum(is_birth_region)))


#########################################
# Example usage: various segmentation helpers
#########################################

# Get a list of whether each frame had segmentation applied
# (useful for videos where only every N frames were processed).
frames_are_segmented = segmentations.get_frames_are_segmented()
frame_index = segmentations.get_closest_frame_index_with_segmentation(frame_index=123, distance_threshold=30)
frame_index = segmentations.get_next_frame_index_with_segmentation(frame_index=123)
frame_index = segmentations.get_previous_frame_index_with_segmentation(frame_index=123)

# Check where particular whales were found in the video.
whale_segmentations_exist = segmentations.get_whale_segmentations_exist()
whale_frame_counts = segmentations.get_whale_frame_counts()
frame_index = segmentations.get_next_frame_index_with_whale_segmentation(frame_index=123, whale_index=0)
frame_index = segmentations.get_previous_frame_index_with_whale_segmentation(frame_index=123, whale_index=0)
[start_frame_index, end_frame_index] = segmentations.get_frame_bounds_for_whale_segmentations(frame_index=123, whale_indexes=[0])

#########################################
# Clean up
#########################################

print()
segmentations.close()

print()
print('Done!')
print()



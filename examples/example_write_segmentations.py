
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
import os

from segmentation_infrastructure.Segmentations import Segmentations

# Configuration
current_script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_script_dir, '..', 'data')
video_num = 0 # e.g. 1688827433752
h5_filepath = os.path.join(data_dir, 'segmentations', '%013d_segmentations.hdf5' % video_num)
video_filepaths = { # the keys can be anything you want
  'masks_boxes_vectors': os.path.join(data_dir, 'segmentations', '%013d_segmentations_boundingBoxes_orientations.mp4' % video_num),
  'only_masks':          os.path.join(data_dir, 'segmentations', '%013d_segmentations.mp4' % video_num),
}
mask_shape = (1407, 2500) # video frame (height, width)

# Remove existing outputs.
if os.path.exists(h5_filepath):
  os.remove(h5_filepath)
for video_filepath in video_filepaths.values():
  if os.path.exists(video_filepath):
    os.remove(video_filepath)

# Open the file.
print('Opening file %s' % h5_filepath)
segmentations = Segmentations(h5_filepath=h5_filepath,
                              writable=True,
                              video_filepaths=video_filepaths,
                              num_video_frames_to_save_as_images=2,
                              frame_shape=mask_shape,
                              output_video_fps=30)

# Write some sample frames.
num_whales_to_add = 5
for frame_index in range(12):
  print('Adding random data for frame index %d' % frame_index)
  
  # For each whale, the mask will be an HxW array of 1 or 0.
  for whale_index in range(num_whales_to_add):
    # Create a sample mask with a rectangular whale.
    mask_matrix = np.zeros(mask_shape, dtype=np.uint8)
    whale_corner = np.random.randint(low=0, high=min(mask_shape), size=2)
    whale_size = np.random.randint(low=0, high=round(min(mask_shape)*0.2), size=2)
    mask_matrix[whale_corner[1]:(whale_corner[1]+whale_size[1]), whale_corner[0]:(whale_corner[0]+whale_size[0])] = 1
    # Add the mask.
    segmentations.add_mask(frame_index=frame_index, whale_index=whale_index, mask_matrix=mask_matrix)
  
  # A bounding box will be an 8-element vector: xy for each box corner.
  # bounding_box_key can currently be 'full', 'head', or 'tail'.
  # If a frame/whale doesn't have a box, simply don't call add_bounding_box for that frame/whale.
  # If a box is added with all coordinates 0, will be treated as a dummy box and ignored.
  for whale_index in range(num_whales_to_add):
    for bounding_box_key in ['full', 'head', 'tail']:
      bounding_box_4xy = np.random.randint(low=0, high=mask_shape[0], size=(8,), dtype=np.int16)
      segmentations.add_bounding_box(bounding_box_key=bounding_box_key, frame_index=frame_index, whale_index=whale_index,
                                     bounding_box_4xy=bounding_box_4xy)
  
  # A centroid will be a 2-element vector.
  # NOTE: it should be in the format [y, x] rather than [x, y].
  # If a frame/whale doesn't have a mask, simply don't call add_centroid for that frame/whale.
  # If a centroid is added with both coordinates 0, will be treated as a dummy centroid and ignored.
  for whale_index in range(num_whales_to_add):
    centroid_yx = np.random.random(size=(2,))*500
    segmentations.add_centroid(frame_index=frame_index, whale_index=whale_index, centroid_yx=centroid_yx)
  
  # An orientation will be a 2-element vector of angle and confidence.
  # If a frame/whale doesn't have a mask, simply don't call add_orientation for that frame/whale.
  for whale_index in range(num_whales_to_add):
    orientation_rad = np.random.random(size=(1,))*3.14
    orientation_confidence = np.random.random(size=(1,))
    segmentations.add_orientation(frame_index=frame_index, whale_index=whale_index,
                                  orientation_rad=orientation_rad, orientation_confidence=orientation_confidence)

  # A segmented image will be an HxWx3 matrix in RGB (or BGR) format.
  # Add to the video with only masks.
  img = np.zeros(shape=(*mask_shape, 3))
  img[:,:,2] = 255*((frame_index % 20)/20) # fade to blue then jump back to black
  segmentations.add_video_frame(video_key='only_masks', frame_index=frame_index, img=img, img_format='rgb',
                                write_frame_index=True)
  # Add to the video with masks and boxes and vectors.
  img = np.zeros(shape=(*mask_shape, 3))
  img[:,:,1] = 255*((frame_index % 20)/20) # fade to green then jump back to black
  segmentations.add_video_frame(video_key='masks_boxes_vectors', frame_index=frame_index, img=img, img_format='rgb',
                                write_frame_index=True)
  
# Close the file.
segmentations.close()










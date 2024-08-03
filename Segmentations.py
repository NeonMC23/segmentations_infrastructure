
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
import scipy
import h5py
import os
import shutil
import distinctipy
import glob
from collections import OrderedDict
import warnings
import json

try:
  import ffmpeg
except:
  pass
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Circle

from csail_data_processing.DecordVideoReaderWrapper import DecordVideoReaderWrapper
from csail_data_processing.helpers.helpers_various import *

class Segmentations:
  ###############################
  # Initialization
  ###############################
  
  def __init__(self, h5_filepath=None, writable=False,
               frame_shape=None,
               video_filepaths=None,  # a dictionary mapping a key to a filepath (example: {'myvid1': my_vid_1.mp4, 'myvid2': my_vid_2.mp4})
               num_video_frames_to_save_as_images=0,  # -1 to save all frames
               output_video_fps=30, video_compression=17, video_preset='veryfast',
               author=''):
    self._h5_filepath = h5_filepath
    self._writable = writable
    self._video_filepaths = video_filepaths if video_filepaths is not None else {}
    self._output_video_fps = output_video_fps
    # Set up saving frames as individual images if desired.
    self._video_frame_image_dirs = {}
    self._num_video_frames_to_save_as_images = num_video_frames_to_save_as_images
    if self._num_video_frames_to_save_as_images != 0:
      for (video_key, video_filepath) in self._video_filepaths.items():
        (video_base_dir, video_filename) = os.path.split(video_filepath)
        self._video_frame_image_dirs[video_key] = os.path.join(video_base_dir, 'frame_images_%s'%
                                                               os.path.splitext(video_filename)[0])
        os.makedirs(self._video_frame_image_dirs[video_key], exist_ok=True)
        for image_filepath in glob.glob(os.path.join(self._video_frame_image_dirs[video_key], '*.jpg')):
          os.remove(image_filepath)
    
    # Specify video compression.
    # Lossless may not be compatible with all players.
    #  If using lossless, preset should probably be ultrafast (faster encoding) or veryslow (better compression).
    #  For visually lossless but not technically lossless, recommend compression of around 17.
    # See https://trac.ffmpeg.org/wiki/Encode/H.264 for more information.
    self._video_compression = video_compression # range 0-51: 0 is lossless, default is 23
    self._video_preset = video_preset # faster easier to play back? [veryslow, slower, slow, medium, fast, veryfast, superfast, ultrafast]

    # Initialize state.
    self._dataset_extra_expansion_size_frameDimension = 300
    self._dataset_extra_expansion_size_whaleDimension = 10
    self._h5_compression_level = 1 # 0-9, default is 4
    self._bounding_box_keys = ['full', 'head', 'tail']
    self._datasets = {
      'masks': None,
      'centroids_xy': None,
      'orientations_rad_confidence': None,
      'frames_are_segmented': None,
      'whale_segmentations_exist': None,
      'annotations': None,
      'history': None,
    }
    self._bounding_box_key_to_name = lambda key: 'bounding_boxes_%s_4xy' % key
    self._bounding_box_names = [self._bounding_box_key_to_name(key) for key in self._bounding_box_keys]
    for bounding_box_name in self._bounding_box_names:
      self._datasets[bounding_box_name] = None
    self._frame_shape = frame_shape
    self._author = author
    
    # Initialize the HDF5 output.
    self._h5_file = None
    if self._h5_filepath is not None:
      # Open the HDF5 file, creating it if it doesn't exist yet.
      using_existing_file = os.path.exists(self._h5_filepath)
      self._h5_file = h5py.File(self._h5_filepath, 'a' if self._writable else 'r')
      # Prune the dataset names to the ones in the existing file.
      if using_existing_file:
        missing_datasets = [dataset_key for dataset_key in self._datasets if dataset_key not in self._h5_file]
        for missing_dataset in missing_datasets:
          del self._datasets[missing_dataset]
        metadata = dict(self._h5_file.attrs.items())
        try:
          self._frame_shape = eval(metadata['frame_shape'])
        except KeyError:
          self._frame_shape = None
          raise
      # Point to existing datasets if this is an existing file,
      #  or create new ones if this is a new file.
      for dataset_key in self._datasets:
        if dataset_key in self._h5_file:
          self._datasets[dataset_key] = self._h5_file[dataset_key]
        elif dataset_key == 'masks' and self._writable and not using_existing_file:
          matrix_shape = [0, 0, 0, 0, 2] # [frame, whale, contour, point, xy]
          max_matrix_shape = [None, None, None, None, 2] # [frame, whale, contour, point, xy]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     matrix_shape,
                                                                     maxshape=max_matrix_shape,
                                                                     dtype='int',
                                                                     fillvalue=-1,
                                                                     chunks=(32,1,16,1024,2),
                                                                     compression='gzip',
                                                                     compression_opts=self._h5_compression_level) # 0-9, default is 4
        elif dataset_key in self._bounding_box_names and self._writable and not using_existing_file:
          matrix_shape = [0, 0, 8] # [frame, whale, 4 xy coordinates]
          max_matrix_shape = [None, None, 8] # [frame, whale, frame_resolution]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     matrix_shape,
                                                                     maxshape=max_matrix_shape,
                                                                     dtype='float',
                                                                     fillvalue=np.nan,
                                                                     chunks=(32,64,8),
                                                                     compression='gzip',
                                                                     compression_opts=self._h5_compression_level) # 0-9, default is 4
        elif dataset_key == 'centroids_xy' and self._writable and not using_existing_file:
          matrix_shape = [0, 0, 2] # [frame, whale, xy]
          max_matrix_shape = [None, None, 2] # [frame, whale, xy]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     matrix_shape,
                                                                     maxshape=max_matrix_shape,
                                                                     dtype='float',
                                                                     fillvalue=np.nan,
                                                                     chunks=(32,64,2),
                                                                     compression='gzip',
                                                                     compression_opts=self._h5_compression_level) # 0-9, default is 4
        elif dataset_key == 'orientations_rad_confidence' and self._writable and not using_existing_file:
          matrix_shape = [0, 0, 2] # [frame, whale, rad-confidence]
          max_matrix_shape = [None, None, 2] # [frame, whale, rad-confidence]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     matrix_shape,
                                                                     maxshape=max_matrix_shape,
                                                                     dtype='float',
                                                                     fillvalue=np.nan,
                                                                     chunks=(32,64,2),
                                                                     compression='gzip',
                                                                     compression_opts=self._h5_compression_level) # 0-9, default is 4
        elif dataset_key == 'frames_are_segmented' and self._writable and not using_existing_file:
          matrix_shape = [0, 1] # [frame, is_segmented]
          max_matrix_shape = [None, 1] # [frame, is_segmented]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     matrix_shape,
                                                                     maxshape=max_matrix_shape,
                                                                     dtype='uint8',
                                                                     chunks=(1024,1),
                                                                     compression='gzip',
                                                                     compression_opts=self._h5_compression_level) # 0-9, default is 4
        elif dataset_key == 'whale_segmentations_exist' and self._writable and not using_existing_file:
          matrix_shape = [0, 0] # [frame, whale]
          max_matrix_shape = [None, None] # [frame, whale]
          self._datasets[dataset_key] = self._h5_file.create_dataset(dataset_key,
                                                                     matrix_shape,
                                                                     maxshape=max_matrix_shape,
                                                                     dtype='uint8',
                                                                     chunks=(128,128),
                                                                     compression='gzip',
                                                                     compression_opts=self._h5_compression_level) # 0-9, default is 4
        elif dataset_key == 'annotations' and self._writable and not using_existing_file:
          self._datasets['annotations'] = self._h5_file.create_group('annotations')
          # Store fields for entering a whale ID.
          whale_ids = self._datasets['annotations'].create_group('whale_ids')
          dataset_kwargs = {
            'compression': 'gzip',
            'compression_opts': self._h5_compression_level, # 0-9, default is 4
          }
          whale_ids.create_dataset('ids', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S128', **dataset_kwargs)
          whale_ids.create_dataset('id_numbers', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='int', **dataset_kwargs)
          whale_ids.create_dataset('is_auto_id', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='uint8', **dataset_kwargs)
          whale_ids.create_dataset('confidences', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float', fillvalue=np.nan, **dataset_kwargs)
          whale_ids.create_dataset('notes', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S256', **dataset_kwargs)
          whale_ids.create_dataset('source_points_xy', [0, 0, 2], maxshape=[None, None, 2], chunks=(32,32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          whale_ids.create_dataset('source_frame_bounds', [0, 2], maxshape=[None, 2], chunks=(32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          whale_ids.create_dataset('timestamps_s', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float64', fillvalue=np.nan, **dataset_kwargs)
          whale_ids.create_dataset('timestamps_str', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S26', **dataset_kwargs)
          whale_ids.create_dataset('authors', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S64', **dataset_kwargs)
          
          # Store fields for entering a behavior.
          behaviors = self._datasets['annotations'].create_group('behaviors')
          dataset_kwargs = {
            'compression': 'gzip',
            'compression_opts': self._h5_compression_level, # 0-9, default is 4
          }
          behaviors.create_dataset('behaviors', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S256', **dataset_kwargs)
          behaviors.create_dataset('confidences', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float', fillvalue=np.nan, **dataset_kwargs)
          behaviors.create_dataset('frame_bounds', [0, 2], maxshape=[None, 2], chunks=(32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          behaviors.create_dataset('whales_involved', [0, 0], maxshape=[None, None], chunks=(32,32), dtype='uint8', fillvalue=-1, **dataset_kwargs)
          behaviors.create_dataset('points_xy', [0, 0, 2], maxshape=[None, None, 2], chunks=(32,32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          behaviors.create_dataset('notes', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S256', **dataset_kwargs)
          behaviors.create_dataset('timestamps_s', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float64', fillvalue=np.nan, **dataset_kwargs)
          behaviors.create_dataset('timestamps_str', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S26', **dataset_kwargs)
          behaviors.create_dataset('authors', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S64', **dataset_kwargs)
         
          # Store fields for entering an event.
          events = self._datasets['annotations'].create_group('events')
          dataset_kwargs = {
            'compression': 'gzip',
            'compression_opts': self._h5_compression_level, # 0-9, default is 4
          }
          events.create_dataset('events', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S512', **dataset_kwargs)
          events.create_dataset('confidences', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float', fillvalue=np.nan, **dataset_kwargs)
          events.create_dataset('frame_bounds', [0, 2], maxshape=[None, 2], chunks=(32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          events.create_dataset('whales_involved', [0, 0], maxshape=[None, None], chunks=(32,32), dtype='uint8', fillvalue=-1, **dataset_kwargs)
          events.create_dataset('points_xy', [0, 0, 2], maxshape=[None, None, 2], chunks=(32,32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          events.create_dataset('notes', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S256', **dataset_kwargs)
          events.create_dataset('timestamps_s', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float64', fillvalue=np.nan, **dataset_kwargs)
          events.create_dataset('timestamps_str', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S26', **dataset_kwargs)
          events.create_dataset('authors', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S64', **dataset_kwargs)
          
          # Store fields for entering a general note.
          notes = self._datasets['annotations'].create_group('notes')
          dataset_kwargs = {
            'compression': 'gzip',
            'compression_opts': self._h5_compression_level, # 0-9, default is 4
          }
          notes.create_dataset('notes', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S256', **dataset_kwargs)
          notes.create_dataset('frame_bounds', [0, 2], maxshape=[None, 2], chunks=(32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          notes.create_dataset('whales_involved', [0, 0], maxshape=[None, None], chunks=(32,32), dtype='uint8', fillvalue=-1, **dataset_kwargs)
          notes.create_dataset('points_xy', [0, 0, 2], maxshape=[None, None, 2], chunks=(32,32,2), dtype='int', fillvalue=-1, **dataset_kwargs)
          notes.create_dataset('timestamps_s', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float64', fillvalue=np.nan, **dataset_kwargs)
          notes.create_dataset('timestamps_str', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S26', **dataset_kwargs)
          notes.create_dataset('authors', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S64', **dataset_kwargs)
        elif dataset_key == 'history' and self._writable and not using_existing_file:
          self._datasets['history'] = self._h5_file.create_group('history')
          dataset_kwargs = {
            'compression': 'gzip',
            'compression_opts': self._h5_compression_level, # 0-9, default is 4
          }
          self._datasets['history'].create_dataset('summaries', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S128', **dataset_kwargs)
          self._datasets['history'].create_dataset('details', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S512', **dataset_kwargs)
          self._datasets['history'].create_dataset('timestamps_s', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='float64', fillvalue=np.nan, **dataset_kwargs)
          self._datasets['history'].create_dataset('timestamps_str', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S26', **dataset_kwargs)
          self._datasets['history'].create_dataset('authors', [0, 1], maxshape=[None, 1], chunks=(32,1), dtype='S64', **dataset_kwargs)
      
      # Store metadata if creating a new file.
      if (not using_existing_file) and self._writable:
        self._update_metadata('frame_shape',  list(frame_shape) if frame_shape is not None else [])
        self._update_metadata('format_version',  9)
        self._update_metadata_dateModified(segmentations=True, annotations=True)
        self.add_history_entry(summary='Created Datasets', details=None, timestamp_s=None, author=self._author)
      else:
        self.add_history_entry(summary='__init__', details=None, timestamp_s=time.time(), author=self._author)
    
    # Store a copy of the existence matrix in memory
    #  to significantly speed up checks later when shortcutting work.
    self._whale_segmentations_exist = self.get_whale_segmentations_exist()
    # Similarly, store a copy of the whale IDs in memory.
    self._whale_ids = self.get_whale_ids(use_local_copy=False)
    self._whale_id_numbers = self.get_whale_id_numbers(use_local_copy=False)
    self._whale_ids_is_auto = self.get_whale_ids_is_auto(use_local_copy=False)
    
    # Store the number of frames.
    self._num_frames = 0
    for (dataset_name, dataset) in self._datasets.items():
      if dataset_name not in ['annotations', 'history']: # ones that are not a dataset with the frame dimension first
        self._num_frames = dataset.shape[0]
        break
    
    # Will store unique colors for each segmentation instance.
    # They take a little time to load though (about 0.09 seconds on XPS laptop),
    #  so will only compute them the first time they are needed.
    self._segmentations_colors = []

    # Open a reader for existing videos, or create an ffmpeg handle to write new ones.
    self._video_readers = {}
    self._ff_procs = {}
    self._num_video_frames = {}
    for (video_key, video_filepath) in self._video_filepaths.items():
      if os.path.exists(video_filepath):
        print('Segmentations opening a video reader to', video_filepath)
        self._video_readers[video_key] = DecordVideoReaderWrapper(video_filepath)
        self._ff_procs[video_key] = None
        self._num_video_frames[video_key] = len(self._video_readers[video_key])
      else:
        self._ff_procs[video_key] = None # will be created when the first frame is provided
        self._video_readers[video_key] = None
        self._num_video_frames[video_key] = 0
  
  # Define unique colors for each segmentation instance.
  def _update_segmentation_colors(self):
    num_colors = 15 # min(15, self.get_num_whales())
    if len(self._segmentations_colors) < num_colors:
      self._segmentations_colors = distinctipy.get_colors(num_colors, exclude_colors=[(0, 0, 1)], rng=6)
      self._segmentations_colors = [np.array(distinctipy.get_rgb256(c), dtype=np.uint8) for c in self._segmentations_colors]
  
  ###############################
  # File operations
  ###############################
  
  # Update a metadata field.
  def _update_metadata(self, key=None, value=None, items=None, clear_all_existing_metadata=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if clear_all_existing_metadata:
      metadata = {}
    else:
      metadata = dict(self._h5_file.attrs.items())
    if key is not None and value is not None:
      items = [(key, value)]
    for (key, value) in items:
      metadata[key] = value
    metadata = convert_dict_values_to_str(metadata, preserve_nested_dicts=False)
    self._h5_file.attrs.update(metadata)
  
  # Update the date modified.
  def _update_metadata_dateModified(self, time_s=None, segmentations=True, annotations=True):
    if not self._writable:
      return
    time_s = time_s or time.time()
    time_str = time_s_to_str(time_s, use_current_utc_time=True)
    dateModified_dict = {}
    if segmentations:
      dateModified_dict['date_modified_segmentations_time_s'] = time_s
      dateModified_dict['date_modified_segmentations_time_str'] = time_str
    if annotations:
      dateModified_dict['date_modified_annotations_time_s'] = time_s
      dateModified_dict['date_modified_annotations_time_str'] = time_str
    if len(dateModified_dict) > 0:
      self._update_metadata(items=dateModified_dict.items())
  
  # Get the date modified.
  def get_date_modified(self, segmentations=True, annotations=True,
                              return_str=True, return_epoch_s=False,
                              date_str_format='%Y-%m-%d %H:%M:%S.%f',
                              date_str_use_local_timezone=False,
                              date_str_include_timezone_offset=True):
    metadata = dict(self._h5_file.attrs.items())
    dates_modified_s = []
    if segmentations:
      try:
        dates_modified_s.append(float(metadata['date_modified_segmentations_time_s']))
      except:
        pass
    if annotations:
      try:
        dates_modified_s.append(float(metadata['date_modified_annotations_time_s']))
      except:
        pass
    import traceback
    try:
      date_modified_s = max(dates_modified_s)
      date_modified_str = time_s_to_str(date_modified_s,
                                        use_current_local_time=date_str_use_local_timezone,
                                        use_current_utc_time=(not date_str_use_local_timezone),
                                        date_str_format=date_str_format,
                                        date_str_include_timezone_offset=date_str_include_timezone_offset)
    except:
      date_modified_s = None
      date_modified_str = None
    if return_str and not return_epoch_s:
      return date_modified_str
    if not return_str and return_epoch_s:
      return return_epoch_s
    if return_str and return_epoch_s:
      return (date_modified_s, date_modified_str)
    return None
  
  # Make a copy of the data and optionally return a Segmentations pointer to it.
  def copy(self, new_h5_filepath, include_masks=True,
           open_segmentations_object=True, new_segmentations_object_writable=False, overwrite_destination_hdf5_file_if_exists=False,
           timestamp_s=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if os.path.exists(new_h5_filepath):
      if overwrite_destination_hdf5_file_if_exists:
        os.remove(new_h5_filepath)
      else:
        raise AssertionError('The target HDF5 filepath already exists: [%s]' % new_h5_filepath)
    self.add_history_entry(summary='copy', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    
    # Create the new HDF5 file.
    new_h5_file = h5py.File(new_h5_filepath, 'w')
    
    # Copy each dataset to the new file.
    for (dataset_name, dataset) in self._h5_file.items():
      if dataset_name == 'masks' and not include_masks:
        continue
      self._h5_file.copy(dataset, new_h5_file,
                         name=None, shallow=False,
                         expand_soft=True, expand_external=True, expand_refs=True,
                         without_attrs=False)
    
    # Copy metadata.
    metadata = dict(self._h5_file.attrs.items())
    metadata = convert_dict_values_to_str(metadata, preserve_nested_dicts=False)
    new_h5_file.attrs.update(metadata)
    
    # Close the HDF5 file.
    new_h5_file.close()
    
    # Open a Segmentations object for it if desired.
    if open_segmentations_object:
      segmentations_kwargs = {
        'h5_filepath': new_h5_filepath,
        'writable': new_segmentations_object_writable,
        'frame_shape': self.get_frame_shape(),
        'video_filepaths': self._video_filepaths,
        'num_video_frames_to_save_as_images': self._num_video_frames_to_save_as_images,
        'output_video_fps': self._output_video_fps,
        'video_compression': self._video_compression,
        'video_preset': self._video_preset,
      }
      new_segmentations = Segmentations(**segmentations_kwargs)
      return new_segmentations
    return None
  
  ###################################
  # Frame sizes, counts, and mappings
  ###################################
  
  # Get a list of whether segmentations were computed for a frame or for all frames.
  def get_frames_are_segmented(self, frame_index=None):
    if self._h5_file is None:
      return None
    if frame_index is None:
      return np.array(self._h5_file['frames_are_segmented'])
    return self._h5_file['frames_are_segmented'][frame_index] == 1
  
  # Get the total number of frames.
  def get_num_frames_total(self):
    # Note that checking the shape of a numpy array is extremely fast,
    #  while checking the shape of a dataset can take a little time.
    #  For example, self._h5_file['centroids_xy'].shape[1] took about 0.125ms with 5628 frames and 157 whales.
    if self._whale_segmentations_exist is not None:
      return self._whale_segmentations_exist.shape[0]
    return None
  
  # Get the number of frames with segmentations computed.
  def get_num_frames_segmented(self):
    if self._h5_file is not None:
      return np.sum(self._h5_file['frames_are_segmented'])
    # There was no data to determine the number of frames with segmentations.
    return None
  
  # Get the last frame index for which a segmentation was added.
  def get_max_frame_index_segmented(self):
    if self._h5_file is not None:
      segmented_frame_indexes = np.where(self._h5_file['frames_are_segmented'])[0]
      if segmented_frame_indexes.size > 0:
        return np.max(segmented_frame_indexes)
      return None
    # There was no data to determine the number of frames with segmentations.
    return None
  
  # Get the shape of a single frame/mask.
  def get_frame_shape(self):
    return self._frame_shape
  
  # Get a matrix entry index with a segmentation close to the desired frame index, within an optional tolerance.
  def get_closest_frame_index_with_segmentation(self, frame_index, distance_threshold=0):
    frames_are_segmented = self.get_frames_are_segmented()
    if frames_are_segmented is None:
      return None
    frame_indexes_with_segmentations = np.where(frames_are_segmented)[0]
    if frame_indexes_with_segmentations.size == 0:
      return None
    if frames_are_segmented.shape[0] == 1:
      # If there is only one entry, consider that the best one.
      best_index = 0
    else:
      # Find the index where the target frame index would be inserted without changing the sort order.
      # This is much faster than using something like numpy.where(), since it can assume the input is sorted.
      next_index_pastTarget = np.searchsorted(frame_indexes_with_segmentations, frame_index)
      # If it returned the length of the array, decrement it to make it a valid index.
      if next_index_pastTarget == frame_indexes_with_segmentations.shape[0]:
        next_index_pastTarget -= 1
      # If it returned the first element, use that as the best index.
      if next_index_pastTarget == 0:
        best_index = 0
      else:
        # We have placed the target between two entry indexes.
        # Now see which one of those two is closer to the target.
        index_candidates = np.array([next_index_pastTarget-1, next_index_pastTarget])
        candidate_distances = np.abs(frame_indexes_with_segmentations[index_candidates] - frame_index)
        if candidate_distances[0] < candidate_distances[1]:
          best_index = index_candidates[0]
        else:
          best_index = index_candidates[1]
    # Check if the closest index is within the threshold region of the target.
    distance = np.abs(frame_indexes_with_segmentations[best_index] - frame_index)
    if distance < distance_threshold:
      # We found a good index!
      return frame_indexes_with_segmentations[best_index]
    # No entry was found to be close enough.
    return None
  
  # Get the next frame index with a segmentation.
  def get_next_frame_index_with_segmentation(self, frame_index):
    frames_are_segmented = self.get_frames_are_segmented()
    if frames_are_segmented is None:
      return None
    future_frame_indexes_with_segmentations = np.where(frames_are_segmented[frame_index+1:])[0]
    if future_frame_indexes_with_segmentations.size == 0:
      return None
    return future_frame_indexes_with_segmentations[0] + frame_index + 1
  
  # Get the previous frame index with a segmentation.
  def get_previous_frame_index_with_segmentation(self, frame_index):
    frames_are_segmented = self.get_frames_are_segmented()
    if frames_are_segmented is None:
      return None
    previous_frame_indexes_with_segmentations = np.where(frames_are_segmented[0:frame_index])[0]
    if previous_frame_indexes_with_segmentations.size == 0:
      return None
    return previous_frame_indexes_with_segmentations[-1]
    
  ###################################
  # Whale indexes
  ###################################
  
  # Get whether a segmentation was found for each whale in each frame.
  def get_whale_segmentations_exist(self):
    if self._h5_file is None:
      return None
    return np.array(self._h5_file['whale_segmentations_exist'])
  
  # Check whether a whale segmentation exists for a particular whale in a frame or all frames.
  def whale_segmentation_exists(self, whale_index, frame_indexes=None):
    if self._whale_segmentations_exist is None:
      return None
    if frame_indexes is None:
      return np.squeeze(self._whale_segmentations_exist[:, whale_index])
    return np.squeeze(self._whale_segmentations_exist[frame_indexes, whale_index])
  
  # Recompute whether whale segmentations exist in each frame,
  #  based on the centroids, boxes, and orientations.
  # Note that it will not consider masks at the moment.
  def _recompute_whale_segmentations_exist(self):
    centroids_xy = self.get_all_centroids_xy()
    orientations_rad_confidence = self.get_all_orientations_rad_confidence()
    bounding_boxes_4xy = dict([(box_key, self.get_all_bounding_boxes_4xy(box_key)) for box_key in self.get_bounding_box_keys()])
    for whale_index in range(self.get_num_whales()):
      frames_with_whale = []
      frames_with_whale.append(np.all(~np.isnan(centroids_xy[:, whale_index, :]), axis=1))
      frames_with_whale.append(np.all(~np.isnan(orientations_rad_confidence[:, whale_index, :]), axis=1))
      for box_key in bounding_boxes_4xy:
        frames_with_whale.append(np.all(~np.isnan(bounding_boxes_4xy[box_key][:, whale_index, :]), axis=1))
      frames_with_whale = np.all(np.stack(frames_with_whale, axis=1), axis=1)
      self._whale_segmentations_exist[:, whale_index] = frames_with_whale
      self._datasets['whale_segmentations_exist'][:, whale_index] = frames_with_whale
    
  # Get the number of whales used.
  def get_num_whales(self):
    # Note that checking the shape of a numpy array is extremely fast,
    #  while checking the shape of a dataset can take a little time.
    #  For example, self._h5_file['centroids_xy'].shape[1] took about 0.125ms with 5628 frames and 157 whales.
    if self._whale_segmentations_exist is not None:
      return self._whale_segmentations_exist.shape[1]
    return None
  
  # Get the number of frames each whale is present.
  # Will return a numpy array where the list index is the whale index.
  def get_whale_frame_counts(self):
    if self._whale_segmentations_exist is None:
      return None
    return np.sum(self._whale_segmentations_exist, axis=0).T
  
  # Get a whale index corresponding to the desired whale ID.
  def get_whale_index_for_whale_id(self, whale_id):
    whale_ids = self.get_whale_ids()
    if whale_ids is None:
      return None
    try:
      return whale_ids.index(whale_id)
    except ValueError:
      return None
  
  # Get a whale index corresponding to the desired whale ID number.
  def get_whale_index_for_whale_id_number(self, whale_id_number):
    whale_id_numbers = self.get_whale_id_numbers()
    if whale_id_numbers is None:
      return None
    try:
      return np.where(whale_id_numbers == whale_id_number)[0][0]
    except IndexError:
      return None
    
  # Get the next frame index with a segmentation.
  def get_next_frame_index_with_whale_segmentation(self, frame_index, whale_index):
    whale_segmentations_exist = self._whale_segmentations_exist[:, whale_index]
    future_frame_indexes_with_whale = np.where(whale_segmentations_exist[frame_index+1:])[0]
    if future_frame_indexes_with_whale.size == 0:
      return None
    return future_frame_indexes_with_whale[0] + frame_index + 1
  
  # Get the previous frame index with a segmentation.
  def get_previous_frame_index_with_whale_segmentation(self, frame_index, whale_index):
    whale_segmentations_exist = self._whale_segmentations_exist[:, whale_index]
    previous_frame_indexes_with_whale = np.where(whale_segmentations_exist[0:frame_index])[0]
    if previous_frame_indexes_with_whale.size == 0:
      return None
    return previous_frame_indexes_with_whale[-1]
  
  # Get the frame bounds for the current instance of one or more whale segmentation.
  # If multiple whales are specified, will return bounds over which they are all present.
  def get_frame_bounds_for_whale_segmentations(self, frame_index, whale_indexes):
    if not isinstance(whale_indexes, (list, tuple, np.ndarray)):
      whale_indexes = [whale_indexes]
    whale_segmentations_exist = np.all(self._whale_segmentations_exist[:, whale_indexes], axis=1)
    if not whale_segmentations_exist[frame_index]:
      return (None, None)
    previous_frame_indexes_without_whales = np.where(~whale_segmentations_exist[0:frame_index])[0]
    if previous_frame_indexes_without_whales.size == 0:
      segmentations_start_frame_index = 0
    else:
      segmentations_start_frame_index = previous_frame_indexes_without_whales[-1]+1
    next_frame_indexes_without_whales = np.where(~whale_segmentations_exist[frame_index+1:])[0]
    if next_frame_indexes_without_whales.size == 0:
      segmentations_end_frame_index = whale_segmentations_exist.shape[0]-1
    else:
      segmentations_end_frame_index = next_frame_indexes_without_whales[0]-1+frame_index+1
    return (segmentations_start_frame_index, segmentations_end_frame_index)
    
  # Get a color for a specified whale index.
  def get_whale_color(self, whale_index, scale_range=255):
    self._update_segmentation_colors() # compute colors if needed
    whale_color = self._segmentations_colors[whale_index % len(self._segmentations_colors)]
    if scale_range == 255:
      whale_color = [int(x) for x in whale_color]
    else:
      whale_color = whale_color/255 * scale_range
    return whale_color
  
  ###############################
  # Annotations
  ###############################
  
  # Get annotation information.
  
  def get_annotations_whale_ids(self, source_frame_start_index=None, source_frame_end_index=None,
                                source_frame_indexes_includes=None,
                                fieldnames=None):
    if self._h5_file is None:
      return None
    if fieldnames is not None:
      if 'source_frame_bounds' not in fieldnames:
        if (source_frame_start_index is not None or source_frame_end_index is not None or source_frame_indexes_includes is not None):
          fieldnames.append('source_frame_bounds')
    annotation_info = {}
    for key in ['ids', 'notes', 'authors', 'timestamps_str']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = self._decode_utf8_values([value[0] for value in self._datasets['annotations']['whale_ids'][key]])
    for key in ['id_numbers', 'is_auto_id', 'confidences', 'source_points_xy', 'source_frame_bounds', 'timestamps_s']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = np.array(self._datasets['annotations']['whale_ids'][key])
    if len(annotation_info) == 0:
      return annotation_info
    num_annotations = len(annotation_info[list(annotation_info.keys())[0]])
    annotation_info['whale_index'] = np.arange(0, num_annotations)
    # Filter based on frame bounds if desired.
    in_frame_bounds = np.ones(shape=(num_annotations,)) == 1
    if source_frame_start_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['source_frame_bounds'][:,0] >= source_frame_start_index)
    if source_frame_end_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['source_frame_bounds'][:,1] <= source_frame_end_index)
    if source_frame_indexes_includes is not None:
      in_frame_bounds = (annotation_info['source_frame_bounds'][:,0] <= source_frame_indexes_includes) & (annotation_info['source_frame_bounds'][:,1] >= source_frame_indexes_includes)
    for key in annotation_info:
      if isinstance(annotation_info[key], (list, tuple)):
        annotation_info[key] = [x for (i, x) in enumerate(annotation_info[key]) if in_frame_bounds[i]]
      else:
        annotation_info[key] = annotation_info[key][in_frame_bounds]
    return annotation_info
  def get_num_annotations_whale_ids(self, only_manual_ids=False, only_auto_ids=False):
    if self._whale_ids is None:
      return None
    ids = self.get_whale_ids()
    id_exists = [len(id) > 0 for id in ids]
    is_auto = self.get_whale_ids_is_auto()
    is_manual = [not id_is_auto for id_is_auto in is_auto]
    if only_manual_ids:
      ids_filtered = [id for (i, id) in enumerate(self.get_whale_ids()) if id_exists[i] and is_manual[i]]
    elif only_auto_ids:
      ids_filtered = [id for (i, id) in enumerate(self.get_whale_ids()) if id_exists[i] and is_auto[i]]
    else:
      ids_filtered = [id for (i, id) in enumerate(self.get_whale_ids()) if id_exists[i]]
    return len(ids_filtered)
  
  def get_annotations_notes(self, frame_start_index=None, frame_end_index=None,
                            frame_indexes_includes=None,
                            fieldnames=None):
    if self._h5_file is None:
      return None
    if fieldnames is not None:
      if 'frame_bounds' not in fieldnames:
        if (frame_start_index is not None or frame_end_index is not None or frame_indexes_includes is not None):
          fieldnames.append('frame_bounds')
    annotation_info = {}
    for key in ['notes', 'authors', 'timestamps_str']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = self._decode_utf8_values([value[0] for value in self._datasets['annotations']['notes'][key]])
    for key in ['frame_bounds', 'whales_involved', 'points_xy', 'timestamps_s']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = np.array(self._datasets['annotations']['notes'][key])
    if len(annotation_info) == 0:
      return annotation_info
    num_annotations = len(annotation_info[list(annotation_info.keys())[0]])
    annotation_info['annotation_index'] = np.arange(0, num_annotations)
    # Filter based on frame bounds if desired.
    in_frame_bounds = np.ones(shape=(num_annotations,)) == 1
    if frame_start_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['frame_bounds'][:,0] >= frame_start_index)
    if frame_end_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['frame_bounds'][:,1] <= frame_end_index)
    if frame_indexes_includes is not None:
      in_frame_bounds = (annotation_info['frame_bounds'][:,0] <= frame_indexes_includes) & (annotation_info['frame_bounds'][:,1] >= frame_indexes_includes)
    for key in annotation_info:
      if isinstance(annotation_info[key], (list, tuple)):
        annotation_info[key] = [x for (i, x) in enumerate(annotation_info[key]) if in_frame_bounds[i]]
      else:
        annotation_info[key] = annotation_info[key][in_frame_bounds]
    return annotation_info
  def get_num_annotations_notes(self):
    if self._h5_file is None:
      return None
    return self._datasets['annotations']['notes']['timestamps_s'].shape[0]
  
  def get_annotations_behaviors(self, frame_start_index=None, frame_end_index=None,
                                frame_indexes_includes=None,
                                fieldnames=None):
    if self._h5_file is None:
      return None
    if fieldnames is not None:
      if 'frame_bounds' not in fieldnames:
        if (frame_start_index is not None or frame_end_index is not None or frame_indexes_includes is not None):
          fieldnames.append('frame_bounds')
    annotation_info = {}
    for key in ['behaviors', 'notes', 'authors', 'timestamps_str']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = self._decode_utf8_values([value[0] for value in self._datasets['annotations']['behaviors'][key]])
    for key in ['frame_bounds', 'whales_involved', 'points_xy', 'confidences', 'timestamps_s']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = np.array(self._datasets['annotations']['behaviors'][key])
    if len(annotation_info) == 0:
      return annotation_info
    num_annotations = len(annotation_info[list(annotation_info.keys())[0]])
    annotation_info['annotation_index'] = np.arange(0, num_annotations)
    # Filter based on frame bounds if desired.
    in_frame_bounds = np.ones(shape=(num_annotations,)) == 1
    if frame_start_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['frame_bounds'][:,0] >= frame_start_index)
    if frame_end_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['frame_bounds'][:,1] <= frame_end_index)
    if frame_indexes_includes is not None:
      in_frame_bounds = (annotation_info['frame_bounds'][:,0] <= frame_indexes_includes) & (annotation_info['frame_bounds'][:,1] >= frame_indexes_includes)
    for key in annotation_info:
      if isinstance(annotation_info[key], (list, tuple)):
        annotation_info[key] = [x for (i, x) in enumerate(annotation_info[key]) if in_frame_bounds[i]]
      else:
        annotation_info[key] = annotation_info[key][in_frame_bounds]
    return annotation_info
  def get_num_annotations_behaviors(self):
    if self._h5_file is None:
      return None
    return self._datasets['annotations']['behaviors']['timestamps_s'].shape[0]
    
  def get_annotations_events(self, frame_start_index=None, frame_end_index=None,
                             frame_indexes_includes=None,
                             fieldnames=None):
    if self._h5_file is None:
      return None
    if fieldnames is not None:
      if 'frame_bounds' not in fieldnames:
        if (frame_start_index is not None or frame_end_index is not None or frame_indexes_includes is not None):
          fieldnames.append('frame_bounds')
    annotation_info = {}
    for key in ['events', 'notes', 'authors', 'timestamps_str']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = self._decode_utf8_values([value[0] for value in self._datasets['annotations']['events'][key]])
    for key in ['frame_bounds', 'whales_involved', 'points_xy', 'confidences', 'timestamps_s']:
      if fieldnames is not None and key not in fieldnames:
        continue
      annotation_info[key] = np.array(self._datasets['annotations']['events'][key])
    if len(annotation_info) == 0:
      return annotation_info
    num_annotations = len(annotation_info[list(annotation_info.keys())[0]])
    annotation_info['annotation_index'] = np.arange(0, num_annotations)
    # Filter based on frame bounds if desired.
    in_frame_bounds = np.ones(shape=(num_annotations,)) == 1
    if frame_start_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['frame_bounds'][:,0] >= frame_start_index)
    if frame_end_index is not None:
      in_frame_bounds = in_frame_bounds & (annotation_info['frame_bounds'][:,1] <= frame_end_index)
    if frame_indexes_includes is not None:
      in_frame_bounds = (annotation_info['frame_bounds'][:,0] <= frame_indexes_includes) & (annotation_info['frame_bounds'][:,1] >= frame_indexes_includes)
    for key in annotation_info:
      if isinstance(annotation_info[key], (list, tuple)):
        annotation_info[key] = [x for (i, x) in enumerate(annotation_info[key]) if in_frame_bounds[i]]
      else:
        annotation_info[key] = annotation_info[key][in_frame_bounds]
    return annotation_info
  def get_num_annotations_events(self):
    if self._h5_file is None:
      return None
    return self._datasets['annotations']['events']['timestamps_s'].shape[0]
  
  def get_history(self, time_start_s=0, fieldnames=None):
    if self._h5_file is None:
      return None
    history_info = {}
    # Determine the first row that matches the desired timestamp filter.
    row_index_start = 0
    timestamps_s = np.array(self._datasets['history']['timestamps_s'])
    if len(timestamps_s) > 0:
      matching_row_indexes = np.where(timestamps_s >= time_start_s)[0]
      if len(matching_row_indexes) > 0:
        row_index_start = matching_row_indexes[0]
      else:
        return history_info
    # Load the data.
    for key in ['summaries', 'details', 'authors', 'timestamps_str']:
      if fieldnames is not None and key not in fieldnames:
        continue
      history_info[key] = self._decode_utf8_values([value[0] for value in self._datasets['history'][key][row_index_start:]])
    for key in ['timestamps_s']:
      if fieldnames is not None and key not in fieldnames:
        continue
      if key == 'timestamps_s':
        history_info[key] = timestamps_s
      else:
        history_info[key] = np.array(self._datasets['history'][key])
    return history_info
  
  # Get the whale ID mapping.
  def get_whale_ids(self, use_local_copy=True):
    if use_local_copy:
      return self._whale_ids
    if self._h5_file is not None:
      return self._decode_utf8_values([whale_id[0] for whale_id in self._datasets['annotations']['whale_ids']['ids']])
    return None
  
  def get_whale_id_numbers(self, use_local_copy=True):
    if use_local_copy:
      return self._whale_id_numbers
    if self._h5_file is not None:
      return np.squeeze(self._datasets['annotations']['whale_ids']['id_numbers'])
    return None
  
  def get_whale_ids_is_auto(self, use_local_copy=True):
    if use_local_copy:
      return self._whale_ids_is_auto
    if self._h5_file is not None:
      return [bool(is_auto) for is_auto in self._datasets['annotations']['whale_ids']['is_auto_id']]
    return None
  
  # Get the whale ID for a specific whale index or ID number.
  def get_whale_id(self, whale_id_number=None, whale_index=None):
    if whale_id_number is not None:
      whale_index = self.get_whale_index_for_whale_id_number(whale_id_number)
    if whale_index is None or whale_index < 0 or whale_index >= self.get_num_whales():
      return None
    whale_ids = self.get_whale_ids()
    if whale_ids is not None:
      return whale_ids[whale_index]
    return None
  
  def get_whale_id_number(self, whale_index):
    if whale_index is None or whale_index < 0 or whale_index >= self.get_num_whales():
      return None
    return self.get_whale_id_numbers()[whale_index]
  
  def get_whale_id_is_auto(self, whale_id_number=None, whale_index=None):
    if whale_id_number is not None:
      whale_index = self.get_whale_index_for_whale_id_number(whale_id_number)
    if whale_index is None or whale_index < 0 or whale_index >= self.get_num_whales():
      return None
    return self.get_whale_ids_is_auto()[whale_index]
  
  # Refresh the whale ID numbers to make them sequential.
  def reassign_whale_id_numbers(self):
    whale_id_numbers = self.get_whale_id_numbers()
    if whale_id_numbers is None:
      return
    if not self._writable:
      return
    new_id_numbers = list(range(whale_id_numbers.shape[0]))
    self._datasets['annotations']['whale_ids']['id_numbers'][:,0] = new_id_numbers
    self._whale_id_numbers = self.get_whale_id_numbers(use_local_copy=False)
    
  # Assign a whale ID to a whale index.
  def add_annotation_whale_id(self, whale_index, whale_id, is_auto_id=False, frame_bounds=None, confidence=np.nan, notes='', points=None, timestamp_s=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='add_annotation_whale_id', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    # Expand the points dataset as needed.
    if points is not None:
      num_points = len(points)
      dataset = self._h5_file['annotations']['whale_ids']['source_points_xy']
      new_shape = list(dataset.shape)
      new_shape[1] = max(new_shape[1], num_points)
      dataset.resize(new_shape)
    # Update the annotation information.
    self._h5_file['annotations']['whale_ids']['ids'][whale_index] = str(whale_id) if whale_id is not None else ''
    self._h5_file['annotations']['whale_ids']['is_auto_id'][whale_index] = is_auto_id
    self._h5_file['annotations']['whale_ids']['confidences'][whale_index] = confidence
    self._h5_file['annotations']['whale_ids']['notes'][whale_index] = notes
    self._h5_file['annotations']['whale_ids']['source_frame_bounds'][whale_index, :] = frame_bounds if frame_bounds is not None else -1
    if points is not None:
      points = np.array(points)
      self._h5_file['annotations']['whale_ids']['source_points_xy'][whale_index, 0:points.shape[0], :] = points
    else:
      self._h5_file['annotations']['whale_ids']['source_points_xy'][whale_index, :, :] = -1
    timestamp_s = timestamp_s if timestamp_s is not None else time.time()
    self._h5_file['annotations']['whale_ids']['timestamps_s'][whale_index] = timestamp_s
    self._h5_file['annotations']['whale_ids']['timestamps_str'][whale_index] = time_s_to_str(timestamp_s, use_current_utc_time=True)
    self._h5_file['annotations']['whale_ids']['authors'][whale_index] = author
    # Update the date modified.
    self._update_metadata_dateModified(segmentations=False, annotations=True)
    # Update the local copy of the whale IDs.
    self._whale_ids = self.get_whale_ids(use_local_copy=False)
    self._whale_id_numbers = self.get_whale_id_numbers(use_local_copy=False)
    self._whale_ids_is_auto = self.get_whale_ids_is_auto(use_local_copy=False)
  
  # Add a behavior annotation.
  def add_annotation_behavior(self, behavior, frame_bounds, whale_indexes_involved, confidence=np.nan, notes='', points=None, timestamp_s=None, author='',
                              annotation_index=None):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='add_annotation_behavior', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    # Resize datasets if a new annotation is being added.
    if annotation_index is None:
      annotation_index = self._h5_file['annotations']['behaviors']['behaviors'].shape[0]
      for (dataset_key, dataset) in self._h5_file['annotations']['behaviors'].items():
        new_shape = list(dataset.shape)
        new_shape[0] = new_shape[0]+1
        dataset.resize(new_shape)
    # Expand the points dataset as needed.
    if points is not None:
      num_points = len(points)
      dataset = self._h5_file['annotations']['behaviors']['points_xy']
      new_shape = list(dataset.shape)
      new_shape[1] = max(new_shape[1], num_points)
      dataset.resize(new_shape)
    # Ensure indexes are in increasing order.
    if isinstance(whale_indexes_involved, np.ndarray):
      whale_indexes_involved = whale_indexes_involved.tolist()
    whale_indexes_involved.sort()
    # Update the annotation information.
    self._h5_file['annotations']['behaviors']['behaviors'][annotation_index] = behavior
    self._h5_file['annotations']['behaviors']['frame_bounds'][annotation_index, :] = frame_bounds
    self._h5_file['annotations']['behaviors']['whales_involved'][annotation_index, :] = 0
    self._h5_file['annotations']['behaviors']['whales_involved'][annotation_index, whale_indexes_involved] = 1
    self._h5_file['annotations']['behaviors']['confidences'][annotation_index] = confidence
    self._h5_file['annotations']['behaviors']['notes'][annotation_index] = notes
    if points is not None:
      points = np.array(points)
      self._h5_file['annotations']['behaviors']['points_xy'][annotation_index, 0:points.shape[0], :] = points
    else:
      self._h5_file['annotations']['behaviors']['points_xy'][annotation_index, :, :] = -1
    timestamp_s = timestamp_s if timestamp_s is not None else time.time()
    self._h5_file['annotations']['behaviors']['timestamps_s'][annotation_index] = timestamp_s
    self._h5_file['annotations']['behaviors']['timestamps_str'][annotation_index] = time_s_to_str(timestamp_s, use_current_utc_time=True)
    self._h5_file['annotations']['behaviors']['authors'][annotation_index] = author
    # Update the date modified.
    self._update_metadata_dateModified(segmentations=False, annotations=True)
  
  # Add an event annotation.
  def add_annotation_event(self, event, frame_bounds, whale_indexes_involved, confidence=np.nan, notes='', points=None, timestamp_s=None, author='',
                           annotation_index=None):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='add_annotation_event', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    # Resize datasets if a new annotation is being added.
    if annotation_index is None:
      annotation_index = self._h5_file['annotations']['events']['events'].shape[0]
      for (dataset_key, dataset) in self._h5_file['annotations']['events'].items():
        new_shape = list(dataset.shape)
        new_shape[0] = new_shape[0]+1
        dataset.resize(new_shape)
    # Expand the points dataset as needed.
    if points is not None:
      num_points = len(points)
      dataset = self._h5_file['annotations']['events']['points_xy']
      new_shape = list(dataset.shape)
      new_shape[1] = max(new_shape[1], num_points)
      dataset.resize(new_shape)
    # Ensure indexes are in increasing order.
    if isinstance(whale_indexes_involved, np.ndarray):
      whale_indexes_involved = whale_indexes_involved.tolist()
    whale_indexes_involved.sort()
    # Update the annotation information.
    self._h5_file['annotations']['events']['events'][annotation_index] = event
    self._h5_file['annotations']['events']['frame_bounds'][annotation_index, :] = frame_bounds
    self._h5_file['annotations']['events']['whales_involved'][annotation_index, :] = 0
    self._h5_file['annotations']['events']['whales_involved'][annotation_index, whale_indexes_involved] = 1
    self._h5_file['annotations']['events']['confidences'][annotation_index] = confidence
    self._h5_file['annotations']['events']['notes'][annotation_index] = notes
    if points is not None:
      points = np.array(points)
      self._h5_file['annotations']['events']['points_xy'][annotation_index, 0:points.shape[0], :] = points
    else:
      self._h5_file['annotations']['events']['points_xy'][annotation_index, :, :] = -1
    timestamp_s = timestamp_s if timestamp_s is not None else time.time()
    self._h5_file['annotations']['events']['timestamps_s'][annotation_index] = timestamp_s
    self._h5_file['annotations']['events']['timestamps_str'][annotation_index] = time_s_to_str(timestamp_s, use_current_utc_time=True)
    self._h5_file['annotations']['events']['authors'][annotation_index] = author
    # Update the date modified.
    self._update_metadata_dateModified(segmentations=False, annotations=True)
  
  # Add a general note annotation.
  def add_annotation_note(self, notes, frame_bounds, whale_indexes_involved, points=None, timestamp_s=None, author='',
                          annotation_index=None):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='add_annotation_note', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    # Resize datasets if a new annotation is being added.
    if annotation_index is None:
      annotation_index = self._h5_file['annotations']['notes']['notes'].shape[0]
      for (dataset_key, dataset) in self._h5_file['annotations']['notes'].items():
        new_shape = list(dataset.shape)
        new_shape[0] = new_shape[0]+1
        dataset.resize(new_shape)
    # Expand the points dataset as needed.
    if points is not None:
      num_points = len(points)
      dataset = self._h5_file['annotations']['notes']['points_xy']
      new_shape = list(dataset.shape)
      new_shape[1] = max(new_shape[1], num_points)
      dataset.resize(new_shape)
    # Ensure indexes are in increasing order.
    if isinstance(whale_indexes_involved, np.ndarray):
      whale_indexes_involved = whale_indexes_involved.tolist()
    whale_indexes_involved.sort()
    # Update the annotation information.
    self._h5_file['annotations']['notes']['notes'][annotation_index] = notes
    self._h5_file['annotations']['notes']['frame_bounds'][annotation_index, :] = frame_bounds
    self._h5_file['annotations']['notes']['whales_involved'][annotation_index, :] = 0
    self._h5_file['annotations']['notes']['whales_involved'][annotation_index, whale_indexes_involved] = 1
    if points is not None:
      points = np.array(points)
      self._h5_file['annotations']['notes']['points_xy'][annotation_index, 0:points.shape[0], :] = points
    else:
      self._h5_file['annotations']['notes']['points_xy'][annotation_index, :, :] = -1
    timestamp_s = timestamp_s if timestamp_s is not None else time.time()
    self._h5_file['annotations']['notes']['timestamps_s'][annotation_index] = timestamp_s
    self._h5_file['annotations']['notes']['timestamps_str'][annotation_index] = time_s_to_str(timestamp_s, use_current_utc_time=True)
    self._h5_file['annotations']['notes']['authors'][annotation_index] = author
    # Update the date modified.
    self._update_metadata_dateModified(segmentations=False, annotations=True)
  
  # Add a history entry.
  def add_history_entry(self, summary, details, timestamp_s=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      return
    author = author or self._author
    annotation_index = self._h5_file['history']['summaries'].shape[0]
    for (dataset_key, dataset) in self._h5_file['history'].items():
      dataset.resize((dataset.shape[0]+1, *dataset.shape[1:]))
    self._h5_file['history']['summaries'][annotation_index] = summary
    if isinstance(details, dict):
      whale_id_numbers = self.get_whale_id_numbers()
      if 'whale_index' in details:
        details['whale_id_number'] = whale_id_numbers[details['whale_index']]
      if 'whale_indexes_involved' in details:
        details['whale_id_numbers_involved'] = [whale_id_numbers[w] for w in details['whale_indexes_involved']]
      if 'whale_index_toMove' in details:
        details['whale_id_number_toMove'] = whale_id_numbers[details['whale_index_toMove']]
      if 'whale_index_1' in details:
        details['whale_id_number_1'] = whale_id_numbers[details['whale_index_1']]
      if 'whale_index_source' in details:
        details['whale_id_number_source'] = whale_id_numbers[details['whale_index_source']]
    try:
      details = json.dumps(details)
    except TypeError:
      details = str(details)
    details = details[0:self._h5_file['history']['details'].dtype.itemsize]
    self._h5_file['history']['details'][annotation_index] = details
    timestamp_s = timestamp_s if timestamp_s is not None else time.time()
    self._h5_file['history']['timestamps_s'][annotation_index] = timestamp_s
    self._h5_file['history']['timestamps_str'][annotation_index] = time_s_to_str(timestamp_s, use_current_utc_time=True)
    self._h5_file['history']['authors'][annotation_index] = author
  
  # Delete a whale ID annotation.
  def delete_annotation_whale_id(self, whale_index, timestamp_s=None, author=''):
    self.add_annotation_whale_id(whale_index, whale_id='', is_auto_id=False, frame_bounds=None, confidence=np.nan, notes='', points=None, timestamp_s=timestamp_s, author=author)
  
  # Delete a behavior annotation.
  def delete_annotation_behavior(self, annotation_index_toRemove, timestamp_s=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='delete_annotation_behavior', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    # Get the current annotation information.
    annotations_info = self.get_annotations_behaviors()
    del annotations_info['annotation_index']
    
    # Determine a permutation of the current indexes that puts the one to remove at the end.
    # And determine the annotation indexes to keep.
    num_annotations_original = len(annotations_info['timestamps_s'])
    annotation_indexes_original = list(range(num_annotations_original))
    annotation_indexes_toKeep = [index for index in annotation_indexes_original if index != annotation_index_toRemove]
    permuted_annotation_indexes = np.array(annotation_indexes_toKeep + [annotation_index_toRemove])
    num_annotations_toKeep = len(annotation_indexes_toKeep)
    need_to_permute = not np.array_equal(permuted_annotation_indexes, np.arange(0, num_annotations_original))
    
    # Update the annotation information.
    for key in annotations_info:
      dataset = self._h5_file['annotations']['behaviors'][key]
      if need_to_permute:
        dataset[0:num_annotations_toKeep, :] = dataset[annotation_indexes_toKeep, :]
      matrix_shape = list(dataset.shape)
      matrix_shape[0] = num_annotations_toKeep
      dataset.resize(matrix_shape)
    # Update the date modified.
    self._update_metadata_dateModified(segmentations=False, annotations=True)
  
  # Delete an event annotation.
  def delete_annotation_event(self, annotation_index_toRemove, timestamp_s=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='delete_annotation_event', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    # Get the current annotation information.
    annotations_info = self.get_annotations_events()
    del annotations_info['annotation_index']
    
    # Determine a permutation of the current indexes that puts the one to remove at the end.
    # And determine the annotation indexes to keep.
    num_annotations_original = len(annotations_info['timestamps_s'])
    annotation_indexes_original = list(range(num_annotations_original))
    annotation_indexes_toKeep = [index for index in annotation_indexes_original if index != annotation_index_toRemove]
    permuted_annotation_indexes = np.array(annotation_indexes_toKeep + [annotation_index_toRemove])
    num_annotations_toKeep = len(annotation_indexes_toKeep)
    need_to_permute = not np.array_equal(permuted_annotation_indexes, np.arange(0, num_annotations_original))
    
    # Update the annotation information.
    for key in annotations_info:
      dataset = self._h5_file['annotations']['events'][key]
      if need_to_permute:
        dataset[0:num_annotations_toKeep, :] = dataset[annotation_indexes_toKeep, :]
      matrix_shape = list(dataset.shape)
      matrix_shape[0] = num_annotations_toKeep
      dataset.resize(matrix_shape)
    # Update the date modified.
    self._update_metadata_dateModified(segmentations=False, annotations=True)
  
  # Delete a general notes annotation.
  def delete_annotation_note(self, annotation_index_toRemove, timestamp_s=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='delete_annotation_note', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=timestamp_s, author=author)
    # Get the current annotation information.
    annotations_info = self.get_annotations_notes()
    del annotations_info['annotation_index']
    
    # Determine a permutation of the current indexes that puts the one to remove at the end.
    # And determine the annotation indexes to keep.
    num_annotations_original = len(annotations_info['timestamps_s'])
    annotation_indexes_original = list(range(num_annotations_original))
    annotation_indexes_toKeep = [index for index in annotation_indexes_original if index != annotation_index_toRemove]
    permuted_annotation_indexes = np.array(annotation_indexes_toKeep + [annotation_index_toRemove])
    num_annotations_toKeep = len(annotation_indexes_toKeep)
    need_to_permute = not np.array_equal(permuted_annotation_indexes, np.arange(0, num_annotations_original))
    
    # Update the annotation information.
    for key in annotations_info:
      dataset = self._h5_file['annotations']['notes'][key]
      if need_to_permute:
        dataset[0:num_annotations_toKeep, :] = dataset[annotation_indexes_toKeep, :]
      matrix_shape = list(dataset.shape)
      matrix_shape[0] = num_annotations_toKeep
      dataset.resize(matrix_shape)
    # Update the date modified.
    self._update_metadata_dateModified(segmentations=False, annotations=True)
    
  ###############################
  # Dataset management helpers
  ###############################
  
  def _expand_datasets(self, frame_index, whale_index,
                       dataset_extra_expansion_size_frameDimension=None,
                       dataset_extra_expansion_size_whaleDimension=None):
    if dataset_extra_expansion_size_frameDimension is None:
      dataset_extra_expansion_size_frameDimension = self._dataset_extra_expansion_size_frameDimension
    if dataset_extra_expansion_size_whaleDimension is None:
      dataset_extra_expansion_size_whaleDimension = self._dataset_extra_expansion_size_whaleDimension
    for (dataset_name, dataset) in self._datasets.items():
      # Skip annotations and history since they will be processed below.
      if dataset_name in ['annotations', 'history']:
        continue
      # Specify which dimension is used for frames and whales.
      # Most datasets have frame as dimension 0 and whales as dimension 1, but there are a few exceptions.
      frame_dimension = 0
      whale_dimension = 1
      if dataset_name == 'frames_are_segmented':
        whale_dimension = None
      # Expand the dataset along the frame dimension if needed.
      if frame_dimension is not None:
        if dataset.shape[frame_dimension] < (frame_index+1):
          new_shape = list(dataset.shape)
          new_shape[frame_dimension] = (frame_index+1) + dataset_extra_expansion_size_frameDimension
          dataset.resize(new_shape)
      # Expand the dataset along the whale dimension if needed.
      if whale_dimension is not None:
        if dataset.shape[whale_dimension] < (whale_index+1):
          new_shape = list(dataset.shape)
          new_shape[whale_dimension] = (whale_index+1) + dataset_extra_expansion_size_whaleDimension
          dataset.resize(new_shape)
    # Update annotations datasets.
    for (dataset_key, dataset) in self._h5_file['annotations']['whale_ids'].items():
      whale_dimension = 0
      if dataset.shape[whale_dimension] < (whale_index+1):
        new_shape = list(dataset.shape)
        new_shape[whale_dimension] = (whale_index+1) + dataset_extra_expansion_size_whaleDimension
        dataset.resize(new_shape)
    max_whale_id_number = max(self._whale_id_numbers) if len(self._whale_id_numbers) > 0 else -1
    for whale_index in range(len(self._whale_id_numbers), self._datasets['annotations']['whale_ids']['id_numbers'].shape[0]):
      self._datasets['annotations']['whale_ids']['id_numbers'][whale_index] = max_whale_id_number+1
      max_whale_id_number += 1
    self._whale_ids = self.get_whale_ids(use_local_copy=False)
    self._whale_id_numbers = self.get_whale_id_numbers(use_local_copy=False)
    self._whale_ids_is_auto = self.get_whale_ids_is_auto(use_local_copy=False)
    for group_key in ['notes', 'behaviors', 'events']:
      whale_dimension = 1
      dataset = self._h5_file['annotations'][group_key]['whales_involved']
      if dataset.shape[whale_dimension] < (whale_index+1):
        new_shape = list(dataset.shape)
        new_shape[whale_dimension] = (whale_index+1) + dataset_extra_expansion_size_whaleDimension
        dataset.resize(new_shape)
    # Update the local existence matrix.
    if not np.array_equal(self._whale_segmentations_exist.shape, self._h5_file['whale_segmentations_exist'].shape):
      self._whale_segmentations_exist = self.get_whale_segmentations_exist()
  
  def _decode_utf8_values(self, values):
    values_decoded = []
    for value in values:
      try:
        values_decoded.append(value.decode('utf-8'))
      except:
        values_decoded.append('')
    return values_decoded
  
  ###############################
  # Masks
  ###############################
  
  # Check if masks are present in the HDF5 file.
  def have_masks(self):
    return 'masks' in self._datasets
  
  # Remove all masks from the data.
  def remove_masks_dataset(self, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='remove_masks_dataset', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    # Delete the masks dataset if it exists.
    # Note that using "del" will delete the name but not reclaim space,
    #  so instead will copy all except the masks to a temporary HDF5 file.
    if self.have_masks():
      # Copy the file to a temporary file, excluding masks.
      h5_filepath_temp = '%s_TEMP' % self._h5_filepath
      self.copy(h5_filepath_temp, include_masks=False, open_segmentations_object=False, overwrite_destination_hdf5_file_if_exists=True)
      # Remove the original file, and rename the new one to its original name.
      self._h5_file.close()
      os.remove(self._h5_filepath)
      os.rename(h5_filepath_temp, self._h5_filepath)
      # Update internal state.
      del self._datasets['masks']
      self._h5_file = h5py.File(self._h5_filepath, 'a' if self._writable else 'r')
      self._frame_shape = None
      for dataset_key in self._datasets.keys():
        self._datasets[dataset_key] = self._h5_file[dataset_key]
      # Update the date modified.
      self._update_metadata_dateModified(segmentations=True, annotations=False)
  
  # Add a mask for the desired frame index.
  # If mask_contours is provided, will use that and ignore mask_matrix.
  #   mask_contours should be a list of numpy arrays, matching the output of cv2.findcontours
  #   mask_matrix should be a matrix of 0 and 1 whose shape matches the frame shape.
  def add_mask(self, frame_index, whale_index, mask_matrix, mask_contours=None, contour_area_threshold_ratio=(1.125/100*1.125/100), log_in_history=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    if not self.have_masks():
      raise AssertionError('The provided HDF5 file does not have masks.')
    # Add a log entry for the action.
    author = author or self._author
    if log_in_history:
      self.add_history_entry(summary='add_mask', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                             timestamp_s=time.time(), author=author)
    # Expand datasets if needed for the number of frames and whales.
    self._expand_datasets(frame_index, whale_index)
    # Compute the mask contours if a binary mask was provided.
    if mask_matrix is not None and mask_contours is None:
      mask_contours, _ = cv2.findContours(mask_matrix.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Filter the contours by area.
    if contour_area_threshold_ratio is not None and self._frame_shape is not None:
      mask_contours = [contour for contour in mask_contours if cv2.contourArea(contour) >= (contour_area_threshold_ratio*np.prod(self._frame_shape))]
    # Check if at least one contour was found in the mask; if not, no writing needs to be done.
    num_contours = len(mask_contours)
    if num_contours > 0:
      # Fetch the masks dataset.
      dataset = self._datasets['masks']
      # Expand the masks dataset if needed for the number of contours and length of contours.
      new_shape = list(dataset.shape)
      contour_lengths = [contour.shape[0] for contour in mask_contours]
      new_shape[2] = max(new_shape[2], num_contours)
      new_shape[3] = max(new_shape[3], max(contour_lengths))
      if not np.array_equal(dataset.shape, new_shape):
        print('\t RESIZING masks dataset old shape %s new shape %s' % (dataset.shape, new_shape))
        dataset.resize(new_shape)
      # Write the new entry.
      mask_contours = [np.squeeze(contour) for contour in mask_contours]
      padding_lengths = [new_shape[3] - contour.shape[0] for contour in mask_contours]
      mask_contours = [np.pad(contour, ((0, padding_lengths[c]), (0,0)), 'constant', constant_values=-1) for (c, contour) in enumerate(mask_contours)]
      mask_contours = np.stack(mask_contours)
      mask_contours = np.pad(mask_contours, ((0, new_shape[2] - num_contours), (0,0), (0,0)), 'constant', constant_values=-1)
      dataset[frame_index, whale_index, :, :, :] = mask_contours
      # for (contour_index, contour_points) in enumerate(mask_contours):
      #   dataset[frame_index, whale_index, contour_index, 0:contour_points.shape[0], :] = np.squeeze(contour_points)
    # Update metadata arrays.
    self._datasets['frames_are_segmented'][frame_index] = 1
    self._datasets['whale_segmentations_exist'][frame_index, whale_index] = num_contours > 0
    self._whale_segmentations_exist[frame_index, whale_index] = num_contours > 0
    self._num_frames = max(self._num_frames, frame_index+1)
    self._update_metadata_dateModified(segmentations=True, annotations=False)
    
  # Get a mask for a desired frame and whale.
  # If the whale was not segmented in this frame, will return None.
  # Otherwise, will return a list of matrices that are each Px2, where
  #  P is the points in the contour
  #  2 is [x, y]
  def get_mask_contours(self, frame_index, whale_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self.have_masks():
      raise AssertionError('The provided HDF5 file does not have masks.')
    # Check if the whale segmentation exists for this frame.
    if not self.whale_segmentation_exists(whale_index=whale_index, frame_indexes=frame_index):
      return None
    # Load the array into memory.
    mask_contours = np.array(self._datasets['masks'][frame_index, whale_index, :, :, :])
    # Convert from a matrix to a list of matrices, filtering out nan values.
    mask_contours = [mask_contours[c, mask_contours[c,:,0] >= 0, :] for c in range(mask_contours.shape[0])]
    mask_contours = [mask_contour for mask_contour in mask_contours if mask_contour.size > 0]
    return mask_contours
  
  # Convert a contour to a dense mask, which is the shape of the frame and all 0s or 1s.
  def convert_mask_contours_to_dense_mask(self, mask_contours):
    if self._frame_shape is None:
      return None
    # Convert from an np matrix to a list of contours if needed.
    if isinstance(mask_contours, np.ndarray):
      mask_contours = [mask_contours[c, mask_contours[c,:,0] >= 0, :] for c in range(mask_contours.shape[0])]
      mask_contours = [mask_contour for mask_contour in mask_contours if mask_contour.size > 0]
    # Fill in each contour.
    dense_mask = np.zeros(self._frame_shape, dtype=np.uint8)
    dense_mask = cv2.drawContours(dense_mask, mask_contours,
                                  -1, # -1 means to draw all contours in the given list
                                  (1,), # value to fill
                                  -1) # -1 means fill rather than only outline
    return dense_mask
  
  # Check if a point is inside a mask.
  # Can provide one of the following combinations:
  #   mask_contours: will check those specific contours.
  #   frame_index and whale_index: will check the mask for that frame and whale
  #   frame_index: will return the whale indexes containing the point if there are any
  def is_point_inside_segmentation(self, point_xy, mask_contours=None, frame_index=None, whale_index=None):
    if isinstance(point_xy, np.ndarray):
      point_xy = point_xy.tolist()
    if mask_contours is not None:
      # Convert from an np matrix to a list of contours if needed.
      if isinstance(mask_contours, np.ndarray):
        mask_contours = [mask_contours[c, mask_contours[c,:,0] >= 0, :] for c in range(mask_contours.shape[0])]
        mask_contours = [mask_contour for mask_contour in mask_contours if mask_contour.size > 0]
      # Check if the point is in any of the contours.
      for contour in mask_contours:
        if cv2.pointPolygonTest(contour, point_xy, False) >= 0: # inside or on the edge
          return True
      return False
    elif frame_index is not None and whale_index is not None:
      mask_contours = self.get_mask_contours(frame_index, whale_index)
      return self.is_point_inside_segmentation(point_xy, mask_contours=mask_contours)
    elif frame_index is not None:
      whale_indexes = []
      for whale_index in range(self.get_num_whales()):
        if self.is_point_inside_segmentation(point_xy, frame_index=frame_index, whale_index=whale_index):
          whale_indexes.append(whale_index)
      return whale_indexes
    return None
  
  # Get all masks for a desired frame.
  # If as_dict is True, will return a dictionary mapping whale index to mask.
  #   Each mask is a list of contours, where a contour is a Px2 numpy array of xy points.
  #   Values will be None if there was no segmentation for that whale index.
  # Otherwise, will return an IxCxPx2 matrix of the mask contours at this frame where I is whale and C is contour.  Unused values will be -1.
  def get_masks_contours(self, frame_index, as_dict=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self.have_masks():
      raise AssertionError('The provided HDF5 file does not have masks.')
    # Check if segmentations were actually created for this frame.
    if not self.get_frames_are_segmented(frame_index):
      return None
    # Fetch the masks for this frame.
    if as_dict:
      masks_contours = OrderedDict()
      for whale_index in range(self.get_num_whales()):
        masks_contours[whale_index] = self.get_mask_contours(frame_index=frame_index, whale_index=whale_index)
    else:
      dataset = self._datasets['masks']
      if frame_index < 0 or frame_index >= dataset.shape[0]:
        return None
      # Load the array into memory and return it.
      masks_contours = np.squeeze(dataset[frame_index, :, :, :, :])
    return masks_contours
  
  # Get masks for all frames and whales.
  # Will return an NxIxCxPx2 matrix, where
  #   N is the number of frames
  #   I is the number of whales
  #   C is the number of contours
  #   P is the number points defining the contour
  #   2 is [x, y]
  # Will return an HDF5 dataset pointer instead of a numpy array, in case the matrix is large.
  def get_all_masks_contours(self):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self.have_masks():
      raise AssertionError('The provided HDF5 file does not have masks.')
    # Will not cast it to an array / copy it / squeeze it, since we do not want
    #  to force it to load into memory in case it is large.
    return self._datasets['masks']
  
  # Set the masks for a desired whale in the desired frames.
  # masks_contours should be NxCxPx2 where
  #   N matches len(frame_indexes)
  #   C matches the max number of contours
  #   P matches the max number of points per contour
  #   2 is [x,y]
  def set_masks_contours(self, end_frame_index, start_frame_index, whale_index, masks_contours, frames_are_segmented=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self.have_masks():
      raise AssertionError('The provided HDF5 file does not have masks.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='set_masks_contours', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self', 'masks_contours']]),
                           timestamp_s=time.time(), author=author)
    # Get a pointer to the current dataset.
    dataset = self._datasets['masks']
    # Verify the new shape and type.
    if masks_contours.shape[0] != end_frame_index - start_frame_index + 1:
      raise AssertionError('The new mask matrix has %d frames, but will be assigned to %d indexes.' % (masks_contours.shape[0], end_frame_index - start_frame_index + 1))
    if (masks_contours.shape[2] != dataset.shape[2]) or (masks_contours.shape[3] != dataset.shape[3]):
      raise AssertionError('The new mask matrix has frame shape %s, but the current frame shape is %s.' % (list(masks_contours.shape[2:]), list(dataset.shape[2:])))
    if masks_contours.dtype != dataset.dtype:
      raise AssertionError('The new mask matrix has type %s, but the dataset on disk has type %s.' % (masks_contours.dtype, dataset.dtype))
    if frames_are_segmented is not None and not frames_are_segmented.shape[0] == masks_contours.shape[0]:
      raise AssertionError('The new mask matrix array has %d frames but the frames_are_segmented array has %d entries.' % (masks_contours.shape[0], frames_are_segmented.shape[0]))
    # Assign the new masks.
    dataset[start_frame_index:end_frame_index+1, whale_index, :, :, :] = masks_contours
    # Update metadata arrays.
    if frames_are_segmented is not None:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = frames_are_segmented
    else:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = 1
    for (mask_index, frame_index) in enumerate(range(start_frame_index, end_frame_index+1)):
      mask_contours = masks_contours[mask_index, :, :, :]
      self._datasets['whale_segmentations_exist'][frame_index, whale_index] = np.any(mask_contours >= 0)
      self._whale_segmentations_exist[frame_index, whale_index] = np.any(mask_contours >= 0)
    self._update_metadata_dateModified(segmentations=True, annotations=False)
  
  
  ###############################
  # Bounding boxes
  ###############################
  
  # Get the available types of bounding boxes.
  def get_bounding_box_keys(self):
    return self._bounding_box_keys
  
  # Add a bounding box for the desired frame and whale index.
  # bounding_box_4xy is 8 numbers: xy of each box corner in order base, leftUpper, top, rightUpper
  def add_bounding_box(self, bounding_box_key, frame_index, whale_index, bounding_box_4xy, log_in_history=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    if log_in_history:
      self.add_history_entry(summary='add_bounding_box', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                             timestamp_s=time.time(), author=author)
    # Expand datasets if needed.
    self._expand_datasets(frame_index, whale_index)
    # Get the dataset pointer.
    dataset_name = self._bounding_box_key_to_name(bounding_box_key)
    dataset = self._datasets[dataset_name]
    # Write the new entry, if it is not a dummy entry.
    if np.any(bounding_box_4xy > 0):
      dataset[frame_index, whale_index, :] = np.array(bounding_box_4xy)
    # Update metadata arrays.
    self._datasets['frames_are_segmented'][frame_index] = 1
    self._datasets['whale_segmentations_exist'][frame_index, whale_index] = np.any(bounding_box_4xy > 0)
    self._whale_segmentations_exist[frame_index, whale_index] = np.any(bounding_box_4xy > 0)
    self._num_frames = max(self._num_frames, frame_index+1)
  
  # Get a bounding box for a desired frame and whale index.
  # bounding_boxes_4xy is 8 numbers: xy for each box corner
  # Will return None if there were no segmentations computed for the frame.
  # Will be all nan if there was no bounding box for this whale in this frame.
  def get_bounding_box_4xy(self, bounding_box_key, frame_index, whale_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if whale_index >= self.get_num_whales():
      raise AssertionError('The specified whale index of %d is greater than the maximum of %d' % (whale_index, self.get_num_whales()-1))
    # Check whether the whale should exist in this frame.
    if not self.whale_segmentation_exists(whale_index, frame_index):
      return None
    # Fetch the dataset pointer.
    dataset_name = self._bounding_box_key_to_name(bounding_box_key)
    dataset = self._datasets[dataset_name]
    if frame_index < 0 or frame_index >= dataset.shape[0]:
      return None
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    bounding_boxes_4xy = np.squeeze(dataset[frame_index, whale_index, :])
    return bounding_boxes_4xy
  
  # Check if a point is inside a bounding box.
  # Can provide one of the following combinations:
  #   box_corners: will check the box specified by those corners, where corners is a 4x2 numpy array
  #   frame_index and whale_index: will check the mask for that frame and whale
  #   frame_index: will return the whale indexes containing the point if there are any
  def is_point_inside_bounding_box(self, point_xy, box_corners=None, box_key='full', frame_index=None, whale_index=None):
    if isinstance(point_xy, np.ndarray):
      point_xy = point_xy.tolist()
    if box_corners is not None:
      # Check if the point is in the box.
      if cv2.pointPolygonTest(box_corners.astype(int), point_xy, False) >= 0: # inside or on the edge
        return True
      return False
    elif frame_index is not None and whale_index is not None:
      bounding_box_4xy = self.get_bounding_box_4xy(box_key, frame_index, whale_index)
      if bounding_box_4xy is not None:
        return self.is_point_inside_bounding_box(point_xy, box_corners=bounding_box_4xy.reshape((-1, 2)))
      else:
        return None
    elif frame_index is not None:
      whale_indexes = []
      for whale_index in range(self.get_num_whales()):
        if self.is_point_inside_bounding_box(point_xy, box_key=box_key, frame_index=frame_index, whale_index=whale_index):
          whale_indexes.append(whale_index)
      return whale_indexes
    return None
  
  # Get all bounding boxes for a desired frame.
  # Will return None if no segmentations were computed for this frame.
  # If as_dict is True, will return a dictionary mapping whale index to bounding box.
  #   Each bounding box is 8 numbers: xy for each box corner
  #   Values will be nan if this whale was not found in this frame.
  # Otherwise, will return an Ix8 matrix where I is the max number of whales.
  #   result[whale, :] will be all nan if there was no bounding box for that frame index and whale index.
  def get_bounding_boxes_4xy(self, bounding_box_key, frame_index, as_dict=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Check if segmentations were actually created for this frame.
    if not self.get_frames_are_segmented(frame_index):
      return None
    # Fetch the bounding boxes for this frame.
    if as_dict:
      bounding_boxes_4xy = OrderedDict()
      for whale_index in range(self.get_num_whales()):
        bounding_boxes_4xy[whale_index] = self.get_bounding_box_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index, whale_index=whale_index)
    else:
      dataset_name = self._bounding_box_key_to_name(bounding_box_key)
      dataset = self._datasets[dataset_name]
      if frame_index < 0 or frame_index >= dataset.shape[0]:
        return None
      # Squeeze the matrix, which will also force the matrix to be loaded into memory.
      # To continue using it from the disk instead, just return the slice directly.
      bounding_boxes_4xy = np.squeeze(dataset[frame_index, :, :])
    return bounding_boxes_4xy
  
  # Get all bounding boxes, and optionally apply a smoothing filter.
  # Will return an NxIx8 matrix, where N is the number of frames and I is the max whale index.
  # result[frame, whale, :] will be all nan if there was no bounding box for that frame index and whale index.
  # If smoothing is desired:
  #   Can optionally affect the current HDF5 file.
  #   The window size and centering is defined by window_size_preCenter and window_size_postCenter.
  def get_all_bounding_boxes_4xy(self, bounding_box_key, apply_smoothing_filter=False,
                                       smoothing_window_size_preCenter=20, smoothing_window_size_postCenter=20,
                                       whale_indexes_toSmooth='all',
                                       smoothing_edits_hdf5_data=False,
                                       smoothed_whale_indexes_toPlot=None, smoothed_whale_indexes_toAnimate=None,
                                       print_smoothing_status=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not apply_smoothing_filter:
      # Load the array into memory and return it.
      dataset_name = self._bounding_box_key_to_name(bounding_box_key)
      return np.array(self._datasets[dataset_name])
    else:
      return self.smooth_bounding_boxes(bounding_box_key=bounding_box_key,
                                        window_size_preCenter=smoothing_window_size_preCenter,
                                        window_size_postCenter=smoothing_window_size_postCenter,
                                        whale_indexes_toSmooth=whale_indexes_toSmooth,
                                        edit_hdf5_data=smoothing_edits_hdf5_data,
                                        whale_indexes_toPlot=smoothed_whale_indexes_toPlot,
                                        whale_indexes_toAnimate=smoothed_whale_indexes_toAnimate,
                                        print_status=print_smoothing_status)
  
  # Set the bounding boxes for a desired whale in the desired frames.
  # Entries without real bounding boxes should use nan.
  def set_bounding_boxes_4xy(self, bounding_box_key, start_frame_index, end_frame_index, whale_index, bounding_boxes_4xy, frames_are_segmented=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='set_bounding_boxes_4xy', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    # Get a pointer to the current dataset.
    dataset_name = self._bounding_box_key_to_name(bounding_box_key)
    dataset = self._datasets[dataset_name]
    # Verify the new shape and type.
    if bounding_boxes_4xy.ndim == 1:
      if bounding_boxes_4xy.shape[0] != 8:
        raise AssertionError('The new bounding box array has 1 frame and %d entries, but should have 8 entries (x,y,x,y,x,y,x,y).' % (bounding_boxes_4xy.shape[0]))
      if frames_are_segmented is not None and not frames_are_segmented.shape[0] == 1:
        raise AssertionError('The new bounding box array has 1 frame but the frames_are_segmented array has %d entries.' % (frames_are_segmented.shape[0]))
    else:
      if bounding_boxes_4xy.ndim != 2:
        raise AssertionError('The new bounding box matrix has %d dimensions, but should have 2.' % (bounding_boxes_4xy.ndims))
      if bounding_boxes_4xy.shape[0] != end_frame_index - start_frame_index + 1:
        raise AssertionError('The new bounding box matrix has %d frames, but will be assigned to %d indexes.' % (bounding_boxes_4xy.shape[0], end_frame_index - start_frame_index + 1))
      if (bounding_boxes_4xy.shape[1] != dataset.shape[2]):
        raise AssertionError('The new bounding box matrix has shape %s for each frame/whale, but the shape should be %s.' % (list(bounding_boxes_4xy.shape[1:]), list(dataset.shape[2:])))
      if frames_are_segmented is not None and not frames_are_segmented.shape[0] == bounding_boxes_4xy.shape[0]:
        raise AssertionError('The new bounding box array has %d frames but the frames_are_segmented array has %d entries.' % (bounding_boxes_4xy.shape[0], frames_are_segmented.shape[0]))
    # Assign the new bounding boxes.
    dataset[start_frame_index:end_frame_index+1, whale_index, :] = bounding_boxes_4xy
    # Update metadata arrays.
    if frames_are_segmented is not None:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = frames_are_segmented
    else:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = 1
    if bounding_boxes_4xy.ndim == 1:
      self._datasets['whale_segmentations_exist'][start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(bounding_boxes_4xy))
      self._whale_segmentations_exist[start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(bounding_boxes_4xy))
    else:
      self._datasets['whale_segmentations_exist'][start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(bounding_boxes_4xy), axis=1)
      self._whale_segmentations_exist[start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(bounding_boxes_4xy), axis=1)
    self._update_metadata_dateModified(segmentations=True, annotations=False)
    
  ###############################
  # Centroids
  ###############################
  
  # Add a centroid of the mask for the desired frame and whale index.
  # centroid_yx is 2 numbers: (y, x)
  #  This can be the direct output of props.centroid if using skimage.measure.regionprops
  def add_centroid(self, frame_index, whale_index, centroid_yx=None, centroid_xy=None, log_in_history=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    if centroid_yx is None and centroid_xy is None:
      raise AssertionError('No centroid was provided')
    # Add a log entry for the action.
    author = author or self._author
    if log_in_history:
      self.add_history_entry(summary='add_centroid', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                             timestamp_s=time.time(), author=author)
    # Expand datasets if needed.
    self._expand_datasets(frame_index, whale_index)
    # Fetch the dataset pointer.
    dataset = self._datasets['centroids_xy']
    # Write the new entry.
    if centroid_yx is not None:
      centroid_xy = np.array(centroid_yx)[[1,0]]
    elif centroid_xy is not None:
      centroid_xy = np.array(centroid_xy)
    dataset[frame_index, whale_index, :] = centroid_xy
    # Update metadata arrays.
    self._datasets['frames_are_segmented'][frame_index] = 1
    self._datasets['whale_segmentations_exist'][frame_index, whale_index] = ~np.any(np.isnan(centroid_xy))
    self._whale_segmentations_exist[frame_index, whale_index] = ~np.any(np.isnan(centroid_xy))
    self._num_frames = max(self._num_frames, frame_index+1)
  
  # Get a centroid for a desired frame and whale index.
  # Will return None if there were no segmentations computed for this frame.
  # Entries will be nan if there was no centroid found for this whale in this frame.
  def get_centroid_xy(self, frame_index, whale_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if whale_index >= self.get_num_whales():
      raise AssertionError('The specified whale index of %d is greater than the maximum of %d' % (whale_index, self.get_num_whales()-1))
    # Check whether the whale should exist in this frame.
    if not self.whale_segmentation_exists(whale_index, frame_index):
      return None
    # Fetch the dataset pointer.
    dataset = self._datasets['centroids_xy']
    if frame_index < 0 or frame_index >= dataset.shape[0]:
      return None
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    centroid_xy = np.squeeze(dataset[frame_index, whale_index, :])
    return centroid_xy
  
  # Get all centroids for a desired frame.
  # Will return None if no segmentations were computed for this frame.
  # If as_dict is True, will return a dictionary mapping whale index to centroid.
  #   Values will be nan if this whale was not found in this frame.
  # Otherwise, will return an Ix2 matrix where I is the max number of whales.
  #  result[whale, :] will be all nan if there was no segmentation for that whale index.
  def get_centroids_xy(self, frame_index, as_dict=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Check if segmentations were actually created for this frame.
    if not self.get_frames_are_segmented(frame_index):
      return None
    # Fetch the centroids for this frame.
    if as_dict:
      centroids_xy = OrderedDict()
      for whale_index in range(self.get_num_whales()):
        centroids_xy[whale_index] = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
    else:
      dataset = self._datasets['centroids_xy']
      if frame_index < 0 or frame_index >= dataset.shape[0]:
        return None
      # Squeeze the matrix, which will also force the matrix to be loaded into memory.
      # To continue using it from the disk instead, just return the slice directly.
      centroids_xy = np.squeeze(dataset[frame_index, :, :])
    return centroids_xy
  
  # Get all centroids, optionally with a smoothing filter applied.
  # Will return an NxIx2 matrix, where N is the number of frames and I is the max whale index.
  # Each centroid is (x,y)
  # result[frame, whale, :] will be all nan if there was no segmentation for that whale index.
  # If smoothing is desired:
  #   Can optionally affect the current HDF5 file.
  #   The window size and centering is defined by window_size_preCenter and window_size_postCenter.
  def get_all_centroids_xy(self, apply_smoothing_filter=False,
                                 smoothing_window_size_preCenter=20, smoothing_window_size_postCenter=20,
                                 whale_indexes_toSmooth='all',
                                 smoothing_edits_hdf5_data=False,
                                 smoothed_whale_indexes_toPlot=None,
                                 print_smoothing_status=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not apply_smoothing_filter:
      # Load the array into memory and return it.
      return np.array(self._datasets['centroids_xy'])
    else:
      return self.smooth_centroids(window_size_preCenter=smoothing_window_size_preCenter,
                                   window_size_postCenter=smoothing_window_size_postCenter,
                                   whale_indexes_toSmooth=whale_indexes_toSmooth,
                                   edit_hdf5_data=smoothing_edits_hdf5_data,
                                   whale_indexes_toPlot=smoothed_whale_indexes_toPlot,
                                   print_status=print_smoothing_status)
  
  # Set the centroids for a desired whale in the desired frames.
  # Entries without real centroids should use nan.
  def set_centroids_xy(self, start_frame_index, end_frame_index, whale_index, centroids_xy, frames_are_segmented=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='set_centroids_xy', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    # Get a pointer to the current dataset.
    dataset = self._datasets['centroids_xy']
    # Verify the new shape and type.
    if centroids_xy.ndim == 1:
      if centroids_xy.shape[0] != 2:
        raise AssertionError('The new centroid array has 1 frame and %d entries, but should have 8 entries (x,y,x,y,x,y,x,y).' % (centroids_xy.shape[0]))
      if frames_are_segmented is not None and not frames_are_segmented.shape[0] == 1:
        raise AssertionError('The new centroid array has 1 frame but the frames_are_segmented array has %d entries.' % (frames_are_segmented.shape[0]))
    else:
      if centroids_xy.ndim != 2:
        raise AssertionError('The new centroid matrix has %d dimensions, but should have 2.' % (centroids_xy.ndims))
      if centroids_xy.shape[0] != end_frame_index - start_frame_index + 1:
        raise AssertionError('The new centroid matrix has %d frames, but will be assigned to %d indexes.' % (centroids_xy.shape[0], end_frame_index - start_frame_index + 1))
      if (centroids_xy.shape[1] != dataset.shape[2]):
        raise AssertionError('The new centroid matrix has shape %s for each frame/whale, but the shape should be %s.' % (list(centroids_xy.shape[1:]), list(dataset.shape[2:])))
      if frames_are_segmented is not None and not frames_are_segmented.shape[0] == centroids_xy.shape[0]:
        raise AssertionError('The new centroid matrix has %d frames but the frames_are_segmented array has %d entries.' % (centroids_xy.shape[0], frames_are_segmented.shape[0]))
    # Assign the new centroids.
    dataset[start_frame_index:end_frame_index+1, whale_index, :] = centroids_xy
    # Update metadata arrays.
    if frames_are_segmented is not None:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = frames_are_segmented
    else:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = 1
    if centroids_xy.ndim == 1:
      self._datasets['whale_segmentations_exist'][start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(centroids_xy))
      self._whale_segmentations_exist[start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(centroids_xy))
    else:
      self._datasets['whale_segmentations_exist'][start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(centroids_xy), axis=1)
      self._whale_segmentations_exist[start_frame_index:end_frame_index+1, whale_index] = ~np.any(np.isnan(centroids_xy), axis=1)
    self._update_metadata_dateModified(segmentations=True, annotations=False)
    
  ###############################
  # Orientations
  ###############################
  
  # Add an orientation angle of the mask for the desired frame and whale index.
  def add_orientation(self, frame_index, whale_index, orientation_rad, orientation_confidence, log_in_history=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    if log_in_history:
      self.add_history_entry(summary='add_orientation', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                             timestamp_s=time.time(), author=author)
    # Expand datasets if needed.
    self._expand_datasets(frame_index, whale_index)
    # Fetch the datset pointer.
    dataset = self._datasets['orientations_rad_confidence']
    # Write the new entry.
    dataset[frame_index, whale_index, :] = np.squeeze(np.array([orientation_rad, orientation_confidence]))
    # Update metadata arrays.
    # Will not update whale_segmentations_exist, since the orientation and confidence both being 0 might be valid.
    self._datasets['frames_are_segmented'][frame_index] = 1
    self._num_frames = max(self._num_frames, frame_index+1)
  
  # Get an orientation angle for a desired frame and whale index.
  # Will return (orientation_rad, orientation_confidence)
  # Will return (None, None) if no segmentations were computed for this frame.
  # Values will be nan if this whale was not found in this frame.
  def get_orientation_rad_confidence(self, frame_index, whale_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if whale_index >= self.get_num_whales():
      raise AssertionError('The specified whale index of %d is greater than the maximum of %d' % (whale_index, self.get_num_whales()-1))
    # Check whether the whale should exist in this frame.
    if not self.whale_segmentation_exists(whale_index, frame_index):
      return (None, None)
    # Fetch the dataset pointer.
    dataset = self._datasets['orientations_rad_confidence']
    if frame_index < 0 or frame_index >= dataset.shape[0]:
      return (None, None)
    # Squeeze the matrix, which will also force the matrix to be loaded into memory.
    # To continue using it from the disk instead, just return the slice directly.
    (orientation_rad, orientation_confidence) = np.squeeze(dataset[frame_index, whale_index, :])
    return (orientation_rad, orientation_confidence)
  
  # Get all orientations for a desired frame.
  # Will return None if no segmentations were computed for this frame.
  # If as_dict is True, will return a dictionary mapping whale index to (orientation_rad, orientation_confidence).
  #   Values will be nan if there was no direction vector for that whale index.
  # Otherwise, will return an Ix2 matrix where I is the max number of whales,
  #   result[whale, :] will be all nan if there was no vector found for that whale index.
  def get_orientations_rad_confidence(self, frame_index, as_dict=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Check if segmentations were actually created for this frame.
    if not self.get_frames_are_segmented(frame_index):
      return None
    # Fetch the orientation data for this frame.
    if as_dict:
      orientations_rad_confidence = OrderedDict()
      for whale_index in range(self.get_num_whales()):
        orientations_rad_confidence[whale_index] = self.get_orientation_rad_confidence(frame_index=frame_index, whale_index=whale_index)
    else:
      dataset = self._datasets['orientations_rad_confidence']
      if frame_index < 0 or frame_index >= dataset.shape[0]:
        return None
      # Squeeze the matrix, which will also force the matrix to be loaded into memory.
      # To continue using it from the disk instead, just return the slice directly.
      orientations_rad_confidence = np.squeeze(dataset[frame_index, :, :])
    return orientations_rad_confidence
  
  # Get orientations for a desired frame range for a specified whale.
  # Will return an Fx2 matrix where F is the number of frames,
  #   result[f, :] will be all nan if there was no vector found for the whale in that frame.
  def get_whale_orientations_rad_confidence(self, whale_index, start_frame_index, end_frame_index):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Fetch the desired orientation data.
    dataset = self._datasets['orientations_rad_confidence']
    orientations_rad_confidence = dataset[start_frame_index:end_frame_index+1, whale_index, :]
    return orientations_rad_confidence
  
  # Get all orientations, and optionally apply a smoothing filter.
  # Will return an NxIx2 matrix, where N is the number of frames, I is the max whale index,
  #   and 2 elements are (orientation_rad, orientation_confidence)
  # result[frame, whale, :] will be all 0 if there was no segmentation for that frame index and whale index.
  #  So if an entry is 0, also do get_centroid_xy for that entry; if that is None, then the orientation is a dummy.
  # If smoothing is desired:
  #   Can optionally affect the current HDF5 file.
  #   The window size and centering is defined by window_size_preCenter and window_size_postCenter.
  #   The window size must be odd; if an even window size is provided, it will be expanded by 1.
  def get_all_orientations_rad_confidence(self, apply_smoothing_filter=False,
                                                smoothing_window_size_preCenter=20, smoothing_window_size_postCenter=20,
                                                whale_indexes_toSmooth='all',
                                                smoothing_edits_hdf5_data=False,
                                                smoothed_whale_indexes_toPlot=None,
                                                print_smoothing_status=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not apply_smoothing_filter:
      # Load the array into memory and return it.
      return np.array(self._datasets['orientations_rad_confidence'])
    else:
      return self.smooth_orientations(window_size_preCenter=smoothing_window_size_preCenter,
                                      window_size_postCenter=smoothing_window_size_postCenter,
                                      whale_indexes_toSmooth=whale_indexes_toSmooth,
                                      edit_hdf5_data=smoothing_edits_hdf5_data,
                                      whale_indexes_toPlot=smoothed_whale_indexes_toPlot,
                                      print_status=print_smoothing_status)
  
  # Set the orientations for a desired whale in the desired frames.
  # Entries without real orientations should use nan.
  def set_orientations_rad_confidence(self, start_frame_index, end_frame_index, whale_index, orientations_rad_confidence, frames_are_segmented=None, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='set_orientations_rad_confidence', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    # Get a pointer to the current dataset.
    dataset = self._datasets['orientations_rad_confidence']
    # Verify the new shape and type.
    if orientations_rad_confidence.ndim == 1:
      if orientations_rad_confidence.shape[0] != 2:
        raise AssertionError('The new orientations array has 1 frame and %d entries, but should have 2 entries (angle and confidence).' % (orientations_rad_confidence.shape[0]))
      if frames_are_segmented is not None and not frames_are_segmented.shape[0] == 1:
        raise AssertionError('The new orientations array has 1 frame but the frames_are_segmented array has %d entries.' % (frames_are_segmented.shape[0]))
    else:
      if orientations_rad_confidence.ndim != 2:
        raise AssertionError('The new orientations matrix has %d dimensions, but should have 2.' % (orientations_rad_confidence.ndims))
      if orientations_rad_confidence.shape[0] != end_frame_index - start_frame_index + 1:
        raise AssertionError('The new orientations matrix has %d frames, but will be assigned to %d indexes.' % (orientations_rad_confidence.shape[0], end_frame_index - start_frame_index + 1))
      if (orientations_rad_confidence.shape[1] != dataset.shape[2]):
        raise AssertionError('The new orientations matrix has shape %s for each frame/whale, but the shape should be %s.' % (list(orientations_rad_confidence.shape[1:]), list(dataset.shape[2:])))
      if frames_are_segmented is not None and not frames_are_segmented.shape[0] == orientations_rad_confidence.shape[0]:
        raise AssertionError('The new orientations array has %d frames but the frames_are_segmented array has %d entries.' % (orientations_rad_confidence.shape[0], frames_are_segmented.shape[0]))
    # Assign the new orientations.
    dataset[start_frame_index:end_frame_index+1, whale_index, :] = orientations_rad_confidence
    # Update metadata arrays.
    if frames_are_segmented is not None:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = frames_are_segmented
    else:
      self._datasets['frames_are_segmented'][start_frame_index:end_frame_index+1] = 1
    if orientations_rad_confidence.ndim == 1:
      self._datasets['whale_segmentations_exist'][start_frame_index:end_frame_index+1, whale_index] = ~np.isnan(orientations_rad_confidence[0])
      self._whale_segmentations_exist[start_frame_index:end_frame_index+1, whale_index] = ~np.isnan(orientations_rad_confidence[0])
    else:
      self._datasets['whale_segmentations_exist'][start_frame_index:end_frame_index+1, whale_index] = ~np.isnan(orientations_rad_confidence[:, 0])
      self._whale_segmentations_exist[start_frame_index:end_frame_index+1, whale_index] = ~np.isnan(orientations_rad_confidence[:, 0])
    self._update_metadata_dateModified(segmentations=True, annotations=False)
  
  ###############################
  # Edit segmentations
  ###############################
  
  # Determine the region of frames within the requested window where a whale actually exists.
  # Optionally provide a window of frames to search.
  # Will return (frame_index_start, frame_index_end) or (None, None) if the whale was not found.
  def get_frame_indexes_with_whale_segmentation(self, whale_index, frame_index_start=None, frame_index_end=None):
    if frame_index_start is None:
      frame_index_start = 0
    if frame_index_end is None:
      frame_index_end = self.get_num_frames_total()-1
    whale_exists = self.whale_segmentation_exists(whale_index)[frame_index_start:frame_index_end+1]
    whale_exists_indexes = np.where(whale_exists)[0]
    if whale_exists_indexes.size == 0:
      return (None, None)
    frame_index_end = frame_index_start + whale_exists_indexes[-1]
    frame_index_start = frame_index_start + whale_exists_indexes[0]
    return (frame_index_start, frame_index_end)
    
  # Remove a segmentation for a whale index in the desired frames.
  # Will update the masks, bounding boxes, centroids, and orientations.
  def remove_segmentation(self, whale_index, frame_index_start, frame_index_end, print_status=False, author=''):
    if print_status: print('Removing segmentations for whale index %d from frames [%d, %d])' % (whale_index, frame_index_start, frame_index_end))
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='remove_segmentation', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    
    # Determine the region of frames within the requested window where the whale actually exists.
    # Outside of that window, no data needs to be copied since all are already np.nan (or -1 for masks).
    (frame_index_start, frame_index_end) = self.get_frame_indexes_with_whale_segmentation(
        whale_index, frame_index_start=frame_index_start, frame_index_end=frame_index_end)
    # If the whale never exists, no editing is needed.
    if frame_index_start is None:
      return
    
    # Remove from masks.
    if self.have_masks():
      if print_status: print(' Updating masks; this may take some time and memory if many frames are selected')
      # Note that loading the submatrix, assigning the value in memory, and then assigning the submatrix
      #  was *much* faser than simply assigning the value to the dataset slice.
      dataset = self._datasets['masks']
      masks = np.empty((frame_index_end-frame_index_start+1, *dataset.shape[2:]), dtype=dataset.dtype)
      masks[:] = -1
      dataset[frame_index_start:frame_index_end+1, whale_index, :, :, :] = masks
    # Remove from bounding boxes.
    if print_status: print(' Updating bounding boxes')
    for box_key in self._bounding_box_keys:
      self._datasets[self._bounding_box_key_to_name(box_key)][frame_index_start:frame_index_end+1, whale_index, :] = np.nan
    # Remove from centroids.
    if print_status: print(' Updating centroids')
    self._datasets['centroids_xy'][frame_index_start:frame_index_end+1, whale_index, :] = np.nan
    # Remove from orientations.
    if print_status: print(' Updating orientations')
    self._datasets['orientations_rad_confidence'][frame_index_start:frame_index_end+1, whale_index, :] = np.nan
    # Update metadata arrays.
    self._datasets['whale_segmentations_exist'][frame_index_start:frame_index_end+1, whale_index] = 0
    self._whale_segmentations_exist[frame_index_start:frame_index_end+1, whale_index] = 0
  
  # Remove a whale index entirely.
  def remove_whale_indexes(self, whale_indexes_toRemove, print_status=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='remove_whale_indexes', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    # Check inputs.
    if isinstance(whale_indexes_toRemove, int):
      whale_indexes_toRemove = [whale_indexes_toRemove]
    whale_indexes_toRemove = list(set(whale_indexes_toRemove))
    whale_indexes_toRemove.sort()
    
    if print_status: print('Removing the following whale indexes: %s' % whale_indexes_toRemove)
    
    # If no whale indexes are provided, nothing needs to be done.
    if len(whale_indexes_toRemove) == 0:
      return
    
    # Do a series of swaps to put the indexes to remove at the end, then resize to remove them.
    # This will likely be faster for the masks.
    num_whales = self.get_num_whales()
    removalZone_indexes = list(range(num_whales-len(whale_indexes_toRemove), num_whales))
    indexes_toKeep = [whale_index for whale_index in range(num_whales) if whale_index not in whale_indexes_toRemove]
    indexes_toKeep_inRemovalZone = [whale_index for whale_index in indexes_toKeep if whale_index in removalZone_indexes]
    for whale_index_toRemove in whale_indexes_toRemove:
      # If this index is already set to be removed, nothing needs to be done.
      if whale_index_toRemove in removalZone_indexes:
        continue
      # Swap the whale with an index that is set to be removed but that shouldn't be.
      whale_index_toSwap = indexes_toKeep_inRemovalZone[0]
      self.swap_whale_indexes(whale_index_toRemove, whale_index_toSwap,
                              0, self.get_num_frames_total()-1,
                              swap_whale_ids=True, author='[subcall from "remove_whale_indexes"]')
      indexes_toKeep_inRemovalZone.pop(indexes_toKeep_inRemovalZone.index(whale_index_toSwap))
    # The indexes to remove are now the trailing indexes.
    whale_indexes_toRemove = removalZone_indexes
      
    # Determine a permutation of the current whale indexes that puts the ones to remove at the end.
    # And determine the whale indexes to keep.
    # Note that this is no longer needed since swaps are performed above to put them at the end already.
    #  But the code is kept for reference in case it is later determined to be better to avoid swapping
    #  and since it shouldn't add noticeable overhead (note that need_to_permute will simply be False below).
    num_whales_original = self.get_num_whales()
    whale_indexes_original = list(range(num_whales_original))
    whale_indexes_toKeep = [whale_index for whale_index in whale_indexes_original if whale_index not in whale_indexes_toRemove]
    permuted_whale_indexes = np.array(whale_indexes_toKeep + whale_indexes_toRemove)
    num_whales_toKeep = len(whale_indexes_toKeep)
    need_to_permute = not np.array_equal(permuted_whale_indexes, np.arange(0, num_whales_original))
    assert (not need_to_permute)
    
    # Update masks.
    # The entire matrix might be large, so load it in batches if needed.
    #  In each batch, put the indexes to remove at the end.
    # If the indexes to remove happen to already be the trailing indexes,
    #  then can skip the slow matrix swapping.
    if self.have_masks():
      dataset = self._datasets['masks']
      if need_to_permute:
        max_gb_to_allocate = 10
        gb_per_frame_per_whale = (np.prod(dataset.shape[2:])*4)/1024/1024/1024 # 4 bytes per entry
        num_frames_to_load = round(max_gb_to_allocate/(gb_per_frame_per_whale*num_whales_original))
        start_frame_index = 0
        while start_frame_index < self.get_num_frames_total():
          end_frame_index = min(self.get_num_frames_total()-1, start_frame_index+num_frames_to_load-1)
          if print_status: print('  Updating masks for frame indexes [%d, %d] out of %d' % (start_frame_index, end_frame_index, self.get_num_frames_total()))
          masks_permuted = np.array(dataset[start_frame_index:end_frame_index+1, :, :, :, :])
          masks_permuted = masks_permuted[:, permuted_whale_indexes, :, :, :]
          dataset[start_frame_index:end_frame_index+1, :, :, :, :] = masks_permuted
          start_frame_index = end_frame_index+1
        #   masks_permuted = np.zeros((end_frame_index-start_frame_index+1, *dataset.shape[1:]), dtype=dataset.dtype)
        #   dataset.read_direct(masks_permuted, np.s_[start_frame_index:end_frame_index+1, :, :, :])
        #   masks_permuted = masks_permuted[:, permuted_whale_indexes, :, :]
        #   masks_permuted = np.ascontiguousarray(masks_permuted)
        #   dataset.write_direct(masks_permuted, None, np.s_[start_frame_index:end_frame_index+1, :, :, :])
      # Now remove the ones at the end by resizing the dataset.
      if print_status: print('  Resizing the mask matrix to trim the removed whales')
      matrix_shape = list(dataset.shape)
      matrix_shape[1] = num_whales_toKeep
      dataset.resize(matrix_shape)
    
    # Update centroids.
    # This can be done entirely in memory.
    if print_status: print('  Updating centroids')
    dataset = self._datasets['centroids_xy']
    if need_to_permute:
      dataset[:, 0:num_whales_toKeep, :] = dataset[:, whale_indexes_toKeep, :]
    matrix_shape = list(dataset.shape)
    matrix_shape[1] = num_whales_toKeep
    dataset.resize(matrix_shape)
    
    # Update bounding boxes.
    # This can be done entirely in memory.
    if print_status: print('  Updating bounding boxes')
    for bounding_box_key in self.get_bounding_box_keys():
      dataset_name = self._bounding_box_key_to_name(bounding_box_key)
      dataset = self._datasets[dataset_name]
      if need_to_permute:
        dataset[:, 0:num_whales_toKeep, :] = dataset[:, whale_indexes_toKeep, :]
      matrix_shape = list(dataset.shape)
      matrix_shape[1] = num_whales_toKeep
      dataset.resize(matrix_shape)
      
    # Update orientations.
    # This can be done entirely in memory.
    if print_status: print('  Updating orientations')
    dataset = self._datasets['orientations_rad_confidence']
    if need_to_permute:
      dataset[:, 0:num_whales_toKeep, :] = dataset[:, whale_indexes_toKeep, :]
    matrix_shape = list(dataset.shape)
    matrix_shape[1] = num_whales_toKeep
    dataset.resize(matrix_shape)
    
    # Update which frames have which whales.
    dataset = self._datasets['whale_segmentations_exist']
    if need_to_permute:
      dataset[:, 0:num_whales_toKeep] = dataset[:, whale_indexes_toKeep]
    matrix_shape = list(dataset.shape)
    matrix_shape[1] = num_whales_toKeep
    dataset.resize(matrix_shape)
    self._whale_segmentations_exist = self.get_whale_segmentations_exist()
    
    # Update annotations datasets.
    for (dataset_key, dataset) in self._h5_file['annotations']['whale_ids'].items():
      if need_to_permute:
        dataset[0:num_whales_toKeep, :] = dataset[whale_indexes_toKeep, :]
      matrix_shape = list(dataset.shape)
      matrix_shape[0] = num_whales_toKeep
      dataset.resize(matrix_shape)
    self._whale_ids = self.get_whale_ids(use_local_copy=False)
    self._whale_id_numbers = self.get_whale_id_numbers(use_local_copy=False)
    self._whale_ids_is_auto = self.get_whale_ids_is_auto(use_local_copy=False)
    for group_key in ['notes', 'behaviors', 'events']:
      dataset = self._h5_file['annotations'][group_key]['whales_involved']
      if need_to_permute:
        dataset[:, 0:num_whales_toKeep] = dataset[:, whale_indexes_toKeep]
      matrix_shape = list(dataset.shape)
      matrix_shape[1] = num_whales_toKeep
      dataset.resize(matrix_shape)
    
    # Update metadata.
    self._update_metadata_dateModified(segmentations=True, annotations=False)
  
  # Swap two whale indexes in the desired frames.
  # Optionally also swap the whale ID mapping for these indexes.
  def swap_whale_indexes(self, whale_index_1, whale_index_2, frame_index_start, frame_index_end, swap_whale_ids=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='swap_whale_indexes', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    
    # First swap the IDs, since they do not depend on the frames.
    if swap_whale_ids:
      for (dataset_key, dataset) in self._h5_file['annotations']['whale_ids'].items():
        data_1 = dataset[whale_index_1, :]
        dataset[whale_index_1, :] = dataset[whale_index_2, :]
        dataset[whale_index_2, :] = data_1
      self._whale_ids = self.get_whale_ids(use_local_copy=False)
      self._whale_id_numbers = self.get_whale_id_numbers(use_local_copy=False)
      self._whale_ids_is_auto = self.get_whale_ids_is_auto(use_local_copy=False)
      for group_key in ['notes', 'behaviors', 'events']:
        dataset = self._h5_file['annotations'][group_key]['whales_involved']
        data_1 = dataset[:, whale_index_1]
        dataset[:, whale_index_1] = dataset[:, whale_index_2]
        dataset[:, whale_index_2] = data_1
      
    # Determine the region of frames within the requested window where either whale actually exists.
    # Outside of that window, no data needs to be copied since all are already np.nan (or -1 for masks).
    whale_1_exists = self.whale_segmentation_exists(whale_index_1)[frame_index_start:frame_index_end+1]
    whale_2_exists = self.whale_segmentation_exists(whale_index_2)[frame_index_start:frame_index_end+1]
    either_whale_exists = whale_1_exists | whale_2_exists
    either_whale_exists_indexes = np.where(either_whale_exists)[0]
    # If neither whale ever exists, no swapping is needed.
    if either_whale_exists_indexes.size == 0:
      return
    frame_index_end = frame_index_start + either_whale_exists_indexes[-1]
    frame_index_start = frame_index_start + either_whale_exists_indexes[0]
    
    # Update masks.
    # Assume that the matrix for a single whale over the frame range is small enough to load into memory.
    if self.have_masks():
      dataset = self._datasets['masks']
      masks_1 = np.array(dataset[frame_index_start:frame_index_end+1, whale_index_1, :, :, :])
      masks_2 = np.array(dataset[frame_index_start:frame_index_end+1, whale_index_2, :, :, :])
      dataset[frame_index_start:frame_index_end+1, whale_index_1, :, :, :] = masks_2
      dataset[frame_index_start:frame_index_end+1, whale_index_2, :, :, :] = masks_1
    
    # Update centroids, orientations, and bounding boxes.
    dataset_names = ['centroids_xy', 'orientations_rad_confidence'] + [self._bounding_box_key_to_name(bounding_box_key) for bounding_box_key in self.get_bounding_box_keys()]
    for dataset_name in dataset_names:
      dataset = self._datasets[dataset_name]
      data_1 = dataset[frame_index_start:frame_index_end+1, whale_index_1, :]
      dataset[frame_index_start:frame_index_end+1, whale_index_1, :] = dataset[frame_index_start:frame_index_end+1, whale_index_2, :]
      dataset[frame_index_start:frame_index_end+1, whale_index_2, :] = data_1
    
    # Update metadata arrays.
    dataset = self._datasets['whale_segmentations_exist']
    data_1 = dataset[frame_index_start:frame_index_end+1, whale_index_1]
    dataset[frame_index_start:frame_index_end+1, whale_index_1] = dataset[frame_index_start:frame_index_end+1, whale_index_2]
    dataset[frame_index_start:frame_index_end+1, whale_index_2] = data_1
    self._whale_segmentations_exist = self.get_whale_segmentations_exist()
    self._update_metadata_dateModified(segmentations=True, annotations=False)
  
  # Change a whale index to another index in the desired frames.
  # Will clobber any existing segmentation data for the destination whale in those frames.
  # Will mark the segmentation in the source whale index as no longer being present.
  def change_whale_index(self, whale_index_source, whale_index_destination,
                               frame_index_start, frame_index_end, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='change_whale_index', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    
    # If the source and destination are the same, nothing should be done.
    if whale_index_source == whale_index_destination:
      return
    
    # Determine the region of frames within the requested window where either whale actually exists.
    # Outside of that window, no data needs to be copied since all are already np.nan (or -1 for masks).
    whale_source_exists = self.whale_segmentation_exists(whale_index_source)[frame_index_start:frame_index_end+1]
    whale_destination_exists = self.whale_segmentation_exists(whale_index_destination)[frame_index_start:frame_index_end+1]
    either_whale_exists = whale_source_exists | whale_destination_exists
    either_whale_exists_indexes = np.where(either_whale_exists)[0]
    # If neither whale ever exists, no adjustments are needed.
    if either_whale_exists_indexes.size == 0:
      return
    frame_index_end = frame_index_start + either_whale_exists_indexes[-1]
    frame_index_start = frame_index_start + either_whale_exists_indexes[0]
    
    # Update masks.
    # Assume that the matrix for a single whale over the frame range is small enough to load into memory.
    if self.have_masks():
      # Note that loading the submatrix, assigning the value in memory, and then assigning the submatrix
      #  was *much* faser than simply assigning the value to the dataset slice.
      dataset = self._datasets['masks']
      masks = np.array(dataset[frame_index_start:frame_index_end+1, whale_index_source, :, :, :])
      dataset[frame_index_start:frame_index_end+1, whale_index_destination, :, :, :] = masks
      masks = np.empty((frame_index_end-frame_index_start+1, *dataset.shape[2:]), dtype=dataset.dtype)
      masks[:] = -1
      dataset[frame_index_start:frame_index_end+1, whale_index_source, :, :, :] = masks
    
    # Update centroids, orientations, and bounding boxes.
    dataset_names = ['centroids_xy', 'orientations_rad_confidence'] + [self._bounding_box_key_to_name(bounding_box_key) for bounding_box_key in self.get_bounding_box_keys()]
    for dataset_name in dataset_names:
      dataset = self._datasets[dataset_name]
      dataset[frame_index_start:frame_index_end+1, whale_index_destination, :] = dataset[frame_index_start:frame_index_end+1, whale_index_source, :]
      dataset[frame_index_start:frame_index_end+1, whale_index_source, :] = np.nan
    
    # Update metadata arrays.
    dataset = self._datasets['whale_segmentations_exist']
    dataset[frame_index_start:frame_index_end+1, whale_index_destination] = dataset[frame_index_start:frame_index_end+1, whale_index_source]
    dataset[frame_index_start:frame_index_end+1, whale_index_source] = 0
    self._whale_segmentations_exist = self.get_whale_segmentations_exist()
    self._update_metadata_dateModified(segmentations=True, annotations=False)
    # Whale IDs and related annotations fields will remain the same
    
  # Create a new whale index for a desired whale in desired frames.
  def move_to_new_whale_index(self, whale_index_toMove, frame_index_start, frame_index_end, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action.
    author = author or self._author
    self.add_history_entry(summary='move_to_new_whale_index', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                           timestamp_s=time.time(), author=author)
    
    # Expand datasets for a new whale at the end.
    new_whale_index = self.get_num_whales()
    self._expand_datasets(0, new_whale_index, dataset_extra_expansion_size_whaleDimension=0)
    
    # Change the index.
    self.change_whale_index(whale_index_toMove, new_whale_index,
                            frame_index_start, frame_index_end, author='[subcall from "move_to_new_whale_index"]')
    
    # Return the new index.
    return new_whale_index
    
  #################################
  # Filter and smooth whales
  #################################
  
  # Filter the instances to only keep whales that are found in at least a threshold number of frames.
  # If create_new_hdf5_file is True, will edit a copy of this file instead of editing in place.
  def filter_whale_instances_byCount(self, min_frame_count=150, remove_masks_dataset=False, create_new_hdf5_file=False, overwrite_destination_hdf5_file_if_exists=False, author='', print_status=False):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not create_new_hdf5_file and not self._writable:
      raise AssertionError('The Segmentations object must be created with the "writable" argument to filter instances in place')
    original_locals = dict([(k,v) for (k,v) in locals().items() if k not in ['self']])
    
    # Determine the whale indexes to remove.
    whale_frame_counts = self.get_whale_frame_counts()
    whale_indexes_toRemove = np.where(whale_frame_counts < min_frame_count)[0]
    if whale_indexes_toRemove.size == 0:
      return
    
    # Create a new HDF5 file for the filtered results if desired.
    if create_new_hdf5_file:
      h5_filepath_filtered = '%s_filtered%dframes.hdf5' % (os.path.splitext(self._h5_filepath)[0], min_frame_count)
      segmentations_filtered = self.copy(h5_filepath_filtered, include_masks=True, open_segmentations_object=True, new_segmentations_object_writable=True, overwrite_destination_hdf5_file_if_exists=overwrite_destination_hdf5_file_if_exists)
    # Otherwise, edit data in the current file.
    else:
      segmentations_filtered = self
    
    # Add a log entry for the action.
    author = author or self._author
    segmentations_filtered.add_history_entry(summary='filter_whale_instances_byCount', details=original_locals,
                                             timestamp_s=time.time(), author=author)
    
    # Remove masks if desired.
    if remove_masks_dataset:
      segmentations_filtered.remove_masks_dataset(author='[subcall from "filter_whale_instances_byCount"]')
    
    # Remove the whales that did not meet the threshold.
    t0 = time.time()
    segmentations_filtered.remove_whale_indexes(whale_indexes_toRemove, print_status=print_status, author='[subcall from "filter_whale_instances_byCount"]')
    if print_status: print('Removed %d whale indexes in %0.2fs: %s' % (len(whale_indexes_toRemove), time.time()-t0, whale_indexes_toRemove))
    whale_frame_counts = self.get_whale_frame_counts()
    
    # Close the file if a new one was created.
    if create_new_hdf5_file:
      segmentations_filtered.close()
      
    # If a new file was created, close it to clean it up then return a pointer to it.
    if create_new_hdf5_file:
      segmentations_filtered.close()
      segmentations_kwargs = {
        'h5_filepath': h5_filepath_filtered,
        'writable': self._writable,
        'frame_shape': self.get_frame_shape(),
        'video_filepaths': self._video_filepaths,
        'num_video_frames_to_save_as_images': self._num_video_frames_to_save_as_images,
        'output_video_fps': self._output_video_fps,
        'video_compression': self._video_compression,
        'video_preset': self._video_preset,
      }
      segmentations_filtered = Segmentations(**segmentations_kwargs)
      return segmentations_filtered
    # Otherwise, return nothing.
    return None
    
  
  # Return a version of the centroids that has been smoothed using a rolling filter.
  # Can optionally edit the current HDF5 file.
  # The window size and centering is defined by window_size_preCenter and window_size_postCenter.
  def smooth_centroids(self, window_size_preCenter=20, window_size_postCenter=20,
                             whale_indexes_toSmooth='all', edit_hdf5_data=False,
                             whale_indexes_toPlot=None, print_status=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if edit_hdf5_data and not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action if this file is being edited.
    author = author or self._author
    if edit_hdf5_data:
      self.add_history_entry(summary='smooth_centroids', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                             timestamp_s=time.time(), author=author)
    # Initialize.
    start_time_s = time.time()
    centroids = np.array(self.get_all_centroids_xy()) # load the whole matrix into memory rather than editing the HDF5 file
    if isinstance(whale_indexes_toSmooth, str) and whale_indexes_toSmooth.lower().strip() == 'all':
      whale_indexes_toSmooth = None
    if whale_indexes_toPlot is None:
      whale_indexes_toPlot = []
    elif isinstance(whale_indexes_toPlot, str) and whale_indexes_toPlot.lower().strip() == 'all':
      whale_indexes_toPlot = list(range(centroids.shape[1]))
    def print_ifDesired(args, kwargs=None):
      if isinstance(args, str):
        args = [args]
      if print_status:
        if kwargs is not None:
          print(*args, **kwargs)
        else:
          print(*args)
    
    # Process each whale index.
    print_ifDesired('Smoothing centroids with window size [%d, %d]' % (window_size_preCenter, window_size_postCenter))
    for whale_index in range(centroids.shape[1]):
      if whale_indexes_toSmooth is not None and whale_index not in whale_indexes_toSmooth:
        continue
      t0 = time.time()
      print_ifDesired('  Filtering centroids for whale index %d... ' % whale_index, {'end':''})
      centroids_forWhale = np.copy(centroids[:, whale_index, :])
      centroids_forWhale[np.isnan(centroids_forWhale)] = -1
      centroids_forWhale_filtered = scipy.ndimage.median_filter(centroids_forWhale,
                                                                size=(window_size_preCenter+window_size_postCenter+1),
                                                                mode='reflect',
                                                                origin=0,
                                                                axes=[0])
      centroids_forWhale[centroids_forWhale == -1] = np.nan
      centroids_forWhale_filtered[centroids_forWhale_filtered == -1] = np.nan
      centroids_forWhale_filtered[np.any(np.isnan(centroids_forWhale_filtered), axis=1)] = np.nan
      # Store the result in memory.
      centroids[:, whale_index, :] = centroids_forWhale_filtered
      print_ifDesired('completed in %0.2fs' % (time.time() - t0))
      
      # Plot if desired.
      if whale_index in whale_indexes_toPlot:
        fig, axs = plt.subplots(nrows=2, ncols=1,
                                 squeeze=False, # if False, always return 2D array of axes
                                 sharex=True, sharey=False,
                                 subplot_kw={'frame_on': True},
                                 figsize=(4,6),
                                 )
        plt.get_current_fig_manager().window.showMaximized()
        axs[0][0].grid(True, color='lightgray')
        axs[1][0].grid(True, color='lightgray')
        axs[0][0].plot(centroids_forWhale[:,0])
        axs[0][0].plot(centroids_forWhale_filtered[:,0])
        axs[1][0].plot(centroids_forWhale[:,1])
        axs[1][0].plot(centroids_forWhale_filtered[:,1])
        axs[0][0].set_title('Centroid X')
        axs[1][0].set_title('Centroid Y')
        axs[1][0].set_xlabel('Frame Index')
        axs[1][0].set_ylabel('Mask Coordinate')
        axs[0][0].set_ylabel('Mask Coordinate')
        plt.suptitle('Original and Smoothed Centroids for Whale Index %d' % whale_index)
    
    print_ifDesired('  Finished smoothing the centroids in %0.2fs' % (time.time() - start_time_s))
    print_ifDesired('')
    if len(whale_indexes_toPlot) > 0:
      plt.show(block=True)
    
    # Update the HDF5 file if desired.
    if edit_hdf5_data:
      self._datasets['centroids_xy'][:,:,:] = centroids
      # Recompute whether whales exist in each frame.
      self._recompute_whale_segmentations_exist()
      # Update metadata.
      self._update_metadata_dateModified(segmentations=True, annotations=False)
      
    return centroids
  
  # Return a version of the orientations that has been smoothed using a rolling filter.
  # Can optionally affect the current HDF5 file.
  # The window size and centering is defined by window_size_preCenter and window_size_postCenter.
  # The window size must be odd; if an even window size is provided, it will be expanded by 1.
  def smooth_orientations(self, window_size_preCenter=20, window_size_postCenter=20,
                          whale_indexes_toSmooth='all', edit_hdf5_data=False,
                          whale_indexes_toPlot=None, print_status=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if edit_hdf5_data and not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action if this file is being edited.
    author = author or self._author
    if edit_hdf5_data:
      self.add_history_entry(summary='smooth_orientations', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                             timestamp_s=time.time(), author=author)
    
    # Initialize.
    start_time_s = time.time()
    orientations_rad_confidence = np.array(self.get_all_orientations_rad_confidence()) # load the whole matrix into memory rather than editing the HDF5 file
    if (window_size_preCenter+window_size_postCenter+1) % 2 == 0:
      window_size_preCenter += 1
    if isinstance(whale_indexes_toSmooth, str) and whale_indexes_toSmooth.lower().strip() == 'all':
      whale_indexes_toSmooth = None
    if whale_indexes_toPlot is None:
      whale_indexes_toPlot = []
    elif isinstance(whale_indexes_toPlot, str) and whale_indexes_toPlot.lower().strip() == 'all':
      whale_indexes_toPlot = list(range(orientations_rad_confidence.shape[1]))
    def print_ifDesired(args, kwargs=None):
      if isinstance(args, str):
        args = [args]
      if print_status:
        if kwargs is not None:
          print(*args, **kwargs)
        else:
          print(*args)
    
    # Process each whale index.
    print_ifDesired('Smoothing orientations and confidences with window size [%d, %d]' % (window_size_preCenter, window_size_postCenter))
    for whale_index in range(orientations_rad_confidence.shape[1]):
      if whale_indexes_toSmooth is not None and whale_index not in whale_indexes_toSmooth:
        continue
      t0 = time.time()
      print_ifDesired('  Filtering orientations and confidences for whale index %d... ' % whale_index, {'end':''})
      orientations_rad_confidence_forWhale = np.copy(orientations_rad_confidence[:, whale_index, :])
      orientations_rad_confidence_forWhale[np.isnan(orientations_rad_confidence_forWhale)] = -1
      orientations_rad_confidence_forWhale_filtered = scipy.ndimage.median_filter(orientations_rad_confidence_forWhale,
                                                                size=(window_size_preCenter+window_size_postCenter+1),
                                                                mode='reflect',
                                                                origin=0,
                                                                axes=[0])
      orientations_rad_confidence_forWhale[orientations_rad_confidence_forWhale == -1] = np.nan
      orientations_rad_confidence_forWhale_filtered[orientations_rad_confidence_forWhale_filtered == -1] = np.nan
      # Store the result in memory.
      orientations_rad_confidence[:, whale_index, :] = orientations_rad_confidence_forWhale_filtered
      print_ifDesired('completed in %0.2fs' % (time.time() - t0))
      
      # Plot if desired.
      if whale_index in whale_indexes_toPlot:
        fig, axs = plt.subplots(nrows=2, ncols=1,
                                 squeeze=False, # if False, always return 2D array of axes
                                 sharex=True, sharey=False,
                                 subplot_kw={'frame_on': True},
                                 figsize=(4,6),
                                 )
        plt.get_current_fig_manager().window.showMaximized()
        axs[0][0].grid(True, color='lightgray')
        axs[1][0].grid(True, color='lightgray')
        axs[0][0].plot(np.degrees(orientations_rad_confidence_forWhale[:,0]))
        axs[0][0].plot(np.degrees(orientations_rad_confidence_forWhale_filtered[:,0]))
        axs[1][0].plot(orientations_rad_confidence_forWhale[:,1])
        axs[1][0].plot(orientations_rad_confidence_forWhale_filtered[:,1])
        axs[0][0].set_title('Orientation Angle')
        axs[1][0].set_title('Head/Tail Confidence')
        axs[1][0].set_xlabel('Frame Index')
        axs[1][0].set_ylabel('Confidence')
        axs[0][0].set_ylabel('Angle [degrees]')
        plt.suptitle('Original and Smoothed Orientations for Whale Index %d' % whale_index)
    
    print_ifDesired('  Finished smoothing the orientations and confidences in %0.2fs' % (time.time() - start_time_s))
    print_ifDesired('')
    if len(whale_indexes_toPlot) > 0:
      plt.show(block=True)
    
    # Update the HDF5 file if desired.
    if edit_hdf5_data:
      self._datasets['orientations_rad_confidence'][:,:,:] = orientations_rad_confidence
      # Recompute whether whales exist in each frame.
      self._recompute_whale_segmentations_exist()
      # Update metadata.
      self._update_metadata_dateModified(segmentations=True, annotations=False)
    
    return orientations_rad_confidence
    
  # Return a version of bounding boxes that has been smoothed using a rolling filter.
  # Can optionally edit the current HDF5 file.
  # The window size and centering is defined by window_size_preCenter and window_size_postCenter.
  def smooth_bounding_boxes(self, bounding_box_key='full',
                            window_size_preCenter=20, window_size_postCenter=20,
                            whale_indexes_toSmooth='all', edit_hdf5_data=False,
                            whale_indexes_toPlot=None, whale_indexes_toAnimate=None,
                            print_status=False, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if edit_hdf5_data and not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Add a log entry for the action if this file is being edited.
    author = author or self._author
    if edit_hdf5_data:
      self.add_history_entry(summary='smooth_bounding_boxes', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
                             timestamp_s=time.time(), author=author)
    
    # Initialize.
    start_time_s = time.time()
    bounding_boxes = np.array(self.get_all_bounding_boxes_4xy(bounding_box_key=bounding_box_key)) # Load into memory instead of editing the current HDF5 file
    if isinstance(whale_indexes_toSmooth, str) and whale_indexes_toSmooth.lower().strip() == 'all':
      whale_indexes_toSmooth = None
    if whale_indexes_toPlot is None:
      whale_indexes_toPlot = []
    elif isinstance(whale_indexes_toPlot, str) and whale_indexes_toPlot.lower().strip() == 'all':
      whale_indexes_toPlot = list(range(bounding_boxes.shape[1]))
    if whale_indexes_toAnimate is None:
      whale_indexes_toAnimate = []
    elif isinstance(whale_indexes_toAnimate, str) and whale_indexes_toAnimate.lower().strip() == 'all':
      whale_indexes_toAnimate = list(range(bounding_boxes.shape[1]))
    showed_plots = False
    def print_ifDesired(args, kwargs=None):
      if isinstance(args, str):
        args = [args]
      if print_status:
        if kwargs is not None:
          print(*args, **kwargs)
        else:
          print(*args)
    
    # Ignore warnings about dividing by zero or invalid values,
    #  since those cases will be handled explicitly after the division.
    with np.errstate(divide='ignore', invalid='ignore'):
      print_ifDesired('Smoothing bounding boxes with window size [%d, %d]' % (window_size_preCenter, window_size_postCenter))
      # Process each whale index.
      for whale_index in range(bounding_boxes.shape[1]):
        t0 = time.time()
        print_ifDesired('Filtering bounding boxes [%s] for whale index %d... ' % (bounding_box_key, whale_index), {'end':''})
        # Extract the data, and set invalid entries to nan.
        boxes_forWhale = np.squeeze(np.copy(bounding_boxes[:, whale_index, :]).reshape((-1, 4, 2))).astype(float)
        boxes_forWhale[np.all(boxes_forWhale == 0, axis=(1,2)),:,:] = np.nan
        # Compute the center of each box, and the half-diagonals from the center to two consecutive corners.
        box_centers_forWhale = np.nanmean(boxes_forWhale, axis=1)
        box_halfDiagonals0_forWhale = boxes_forWhale[:, 0, :] - box_centers_forWhale
        box_halfDiagonals1_forWhale = boxes_forWhale[:, 1, :] - box_centers_forWhale
        # Apply a median filter to each of these quantities.
        box_centers_forWhale[np.isnan(box_centers_forWhale)] = -1
        box_halfDiagonals0_forWhale[np.isnan(box_halfDiagonals0_forWhale)] = -1
        box_halfDiagonals1_forWhale[np.isnan(box_halfDiagonals1_forWhale)] = -1
        box_centers_forWhale = scipy.ndimage.median_filter(box_centers_forWhale,
                                                            size=(window_size_preCenter+window_size_postCenter+1),
                                                            mode='reflect', origin=0, axes=[0])
        box_halfDiagonals0_forWhale = scipy.ndimage.median_filter(box_halfDiagonals0_forWhale,
                                                            size=(window_size_preCenter+window_size_postCenter+1),
                                                            mode='reflect', origin=0, axes=[0])
        box_halfDiagonals1_forWhale = scipy.ndimage.median_filter(box_halfDiagonals1_forWhale,
                                                            size=(window_size_preCenter+window_size_postCenter+1),
                                                            mode='reflect', origin=0, axes=[0])
        box_centers_forWhale[box_centers_forWhale == -1] = np.nan
        box_halfDiagonals0_forWhale[box_halfDiagonals0_forWhale == -1] = np.nan
        box_halfDiagonals1_forWhale[box_halfDiagonals1_forWhale == -1] = np.nan
        # Initialize the filtered results.
        boxes_forWhale_filtered = np.zeros_like(boxes_forWhale)
        
        # The first two consecutive corners can be computed directly from the filtered half-diagonals and the filtered centers.
        boxes_forWhale_filtered[:, 0, :] = box_centers_forWhale + box_halfDiagonals0_forWhale
        boxes_forWhale_filtered[:, 1, :] = box_centers_forWhale + box_halfDiagonals1_forWhale
        
        # The other two corners will be the first two corners reflected about a line.
        #  The reflection line will pass through the center,
        #  and be parallel to the side connecting the first two corners.
        
        # First, compute the slope and intercept of the box side connecting the two known corners.
        slopes = (boxes_forWhale_filtered[:, 0, 1] - boxes_forWhale_filtered[:, 1, 1]) / (boxes_forWhale_filtered[:, 0, 0] - boxes_forWhale_filtered[:, 1, 0])
        intercepts = box_centers_forWhale[:, 1] - slopes*box_centers_forWhale[:, 0]
        
        # Now compute the slope of the line perpendicular to this side.
        reflection_slopes = -1/slopes
        
        # For the third corner, compute the line that goes through the previous corner.
        reflection_intercepts = boxes_forWhale_filtered[:, 1, 1] - reflection_slopes*boxes_forWhale_filtered[:, 1, 0]
        # Then find the x coordinate of the midpoint of this side of the box.
        #   It will be the point that intersects two lines:
        #     1) The line parallel to the first side that passes through the center
        #     2) The line parallel to the current side (perpendicular to the first side) passing through the previous corner.
        side_centers_x = (reflection_intercepts - intercepts) / (slopes - reflection_slopes)
        # Compute the delta-x from this side midpoint to the previous corner.
        # The third corner will be this delta-x from the side midpoint.
        boxes_forWhale_filtered[:, 2, 0] = side_centers_x + (side_centers_x - boxes_forWhale_filtered[:, 1, 0])
        # Use the equation of the perpendicular line to compute the third corner y.
        boxes_forWhale_filtered[:, 2, 1] = reflection_slopes*boxes_forWhale_filtered[:, 2, 0] + reflection_intercepts
        
        # Do the same for the fourth corner, but using the first corner as the reflection reference point.
        reflection_intercepts = boxes_forWhale_filtered[:, 0, 1] - reflection_slopes*boxes_forWhale_filtered[:, 0, 0]
        side_centers_x = (reflection_intercepts - intercepts) / (slopes - reflection_slopes)
        boxes_forWhale_filtered[:, 3, 0] = side_centers_x + (side_centers_x - boxes_forWhale_filtered[:, 0, 0])
        boxes_forWhale_filtered[:, 3, 1] = reflection_slopes*boxes_forWhale_filtered[:, 3, 0] + reflection_intercepts
        
        # If the first side was horizontal, the perpendicular slope is infinite.
        # In this case, the third and fourth corners can simply copy the x coordinates of the first two corners
        #  and have their y coordinates be computed based on the delta-y from the first side to the center.
        reflection_slope_is_inf = np.isinf(reflection_slopes)
        if np.any(reflection_slope_is_inf):
          boxes_forWhale_filtered[reflection_slope_is_inf, 2, 0] = boxes_forWhale_filtered[reflection_slope_is_inf, 1, 0]
          boxes_forWhale_filtered[reflection_slope_is_inf, 3, 0] = boxes_forWhale_filtered[reflection_slope_is_inf, 0, 0]
          boxes_forWhale_filtered[reflection_slope_is_inf, 2, 1] = box_centers_forWhale[reflection_slope_is_inf, 1] + (box_centers_forWhale[reflection_slope_is_inf, 1] - boxes_forWhale_filtered[reflection_slope_is_inf, 1, 1])
          boxes_forWhale_filtered[reflection_slope_is_inf, 3, 1] = boxes_forWhale_filtered[reflection_slope_is_inf, 2, 1]
        # If the first side was vertical, the perpendicular slope is zero.
        # In this case, the third and fourth corners can simply copy the y coordinates of the first two corners
        #  and have their x coordinates be computed based on the delta-x from the first side to the center.
        reflection_slope_is_zero = reflection_slopes == 0
        if np.any(reflection_slope_is_zero):
          boxes_forWhale_filtered[reflection_slope_is_zero, 2, 1] = boxes_forWhale_filtered[reflection_slope_is_zero, 1, 1]
          boxes_forWhale_filtered[reflection_slope_is_zero, 3, 1] = boxes_forWhale_filtered[reflection_slope_is_zero, 0, 1]
          boxes_forWhale_filtered[reflection_slope_is_zero, 2, 0] = box_centers_forWhale[reflection_slope_is_zero, 0] + (box_centers_forWhale[reflection_slope_is_zero, 0] - boxes_forWhale_filtered[reflection_slope_is_zero, 1, 0])
          boxes_forWhale_filtered[reflection_slope_is_zero, 3, 0] = boxes_forWhale_filtered[reflection_slope_is_zero, 2, 0]
        
        # Store the result in memory.
        bounding_boxes[:, whale_index, :] = boxes_forWhale_filtered.reshape((-1, 8)).astype(float)
        print_ifDesired('completed in %0.2fs' % (time.time() - t0))
        
        # Plot the original and smoothed coordinates if desired.
        if whale_index in whale_indexes_toPlot:
          fig, axs = plt.subplots(nrows=4, ncols=2,
                                     squeeze=False, # if False, always return 2D array of axes
                                     sharex=True, sharey=False,
                                     subplot_kw={'frame_on': True},
                                     figsize=(4,6),
                                     )
          plt.get_current_fig_manager().window.showMaximized()
          for row_index in range(4):
            for col_index in range(2):
              axs[row_index][col_index].plot(boxes_forWhale[:,row_index,col_index])
              axs[row_index][col_index].plot(boxes_forWhale_filtered[:,row_index,col_index])
              axs[row_index][col_index].grid(True, color='lightgray')
              axs[row_index][col_index].set_title('%s For Corner %d' % ('X' if col_index == 0 else 'Y', row_index))
              if row_index == 3:
                axs[row_index][col_index].set_xlabel('Frame Index')
            axs[row_index][0].set_ylabel('Mask Coordinate')
          plt.suptitle('Original and Smoothed Boxes Type [%s] for Whale Index %d' % (bounding_box_key, whale_index))
          showed_plots = True
        
        # Animate the original and smoothed boxes if desired.
        if whale_index in whale_indexes_toAnimate:
          plt.figure()
          plt.grid(True, color='lightgray')
          plt.xlabel('Mask X Coordinate')
          plt.ylabel('Mask Y Coordinate')
          plt.show(block=False)
          for frame_index in range(boxes_forWhale_filtered.shape[0]):
            plt.title('Original and Smoothed Boxes Type [%s] for Whale Index %d Frame Index %d' % (bounding_box_key, whale_index, frame_index))
            box = boxes_forWhale[frame_index]
            box_filtered = boxes_forWhale_filtered[frame_index]
            del plt.gca().lines[:]
            plt.plot(box[:,0], box[:,1], 'k.-')
            plt.plot(box_filtered[:,0], box_filtered[:,1], 'm.-')
            plt.plot(box_filtered[0,0], box_filtered[0,1], 'c.')
            plt.plot(box_centers_forWhale[frame_index,0], box_centers_forWhale[frame_index,1], 'm.')
            plt.gca().set_aspect('equal', 'box')
            plt.xlim([0, self.get_frame_shape()[1]])
            plt.ylim([0, self.get_frame_shape()[0]])
            plt.draw()
            cv2.waitKey(1)
        
    print_ifDesired('  Finished smoothing the bounding boxes in %0.2fs' % (time.time() - start_time_s))
    print_ifDesired('')
    if showed_plots:
      plt.show(block=True)
    
    # Update the HDF5 file if desired.
    if edit_hdf5_data:
      self._datasets[self._bounding_box_key_to_name(bounding_box_key)][:,:,:] = bounding_boxes
      # Recompute whether whales exist in each frame.
      self._recompute_whale_segmentations_exist()
      # Update metadata.
      self._update_metadata_dateModified(segmentations=True, annotations=False)
    
    return bounding_boxes
  
  # Smooth the masks using a rolling filter.
  # Will edit the current HDF5 file.
  # The window size and centering is defined by window_size_preCenter and window_size_postCenter.
  # For each pixel of each frame, will compute the mean of the masks from window_size_preCenter frames before it
  #  through window_size_postCenter frames after it.
  #  That pixel will be 1 if the mean is at least rolling_mean_threshold, and 0 otherwise.
  # NOTE: If rolling_mean_threshold is set to 0.5, this is effectively a rolling median filter.
  def smooth_masks(self, window_size_preCenter=20, window_size_postCenter=20,
                         rolling_mean_threshold=0.5,
                         whale_indexes_toSmooth='all',
                         print_status=True, author=''):
    if self._h5_file is None:
      raise AssertionError('No HDF5 filepath was provided.')
    if not self._writable:
      raise AssertionError('The Segmentations object must be created with the "writable" argument to enable smoothing masks')
    
    raise NotImplementedError('Mask smoothing is not yet implemented for the new contour-based storage format')
    
    # # Add a log entry for the action if this file is being edited.
    # author = author or self._author
    # self.add_history_entry(summary='smooth_masks', details=dict([(k,v) for (k,v) in locals().items() if k not in ['self']]),
    #                        timestamp_s=time.time(), author=author)
    #
    # # Initialize.
    # start_time_s = time.time()
    # masks = self.get_all_masks()
    # centroids = self.get_all_centroids_xy()
    # bounding_boxes = self.get_all_bounding_boxes_4xy(bounding_box_key='full')
    # if isinstance(whale_indexes_toSmooth, str) and whale_indexes_toSmooth.lower().strip() == 'all':
    #   whale_indexes_toSmooth = None
    # def print_ifDesired(args, kwargs=None):
    #   if isinstance(args, str):
    #     args = [args]
    #   if print_status:
    #     if kwargs is not None:
    #       print(*args, **kwargs)
    #     else:
    #       print(*args)
    #
    # # Process each whale index.
    # print_ifDesired('Smoothing masks with window size [%d, %d] and threshold %g' % (window_size_preCenter, window_size_postCenter, rolling_mean_threshold))
    # for whale_index in range(masks.shape[1]):
    #   if whale_indexes_toSmooth is not None and whale_index not in whale_indexes_toSmooth:
    #     continue
    #   print_ifDesired('  Smoothing masks for whale index %d/%d' % (whale_index, masks.shape[1]-1))
    #
    #   # Determine the frame entries that contain this whale instance.
    #   centroids_forWhale = centroids[:, whale_index, :]
    #   entries_have_whale = np.all(centroids_forWhale > 0, axis=1)
    #   if np.sum(entries_have_whale) == 0:
    #     continue
    #   entries_with_whale = np.where(entries_have_whale)[0]
    #   first_entry_with_whale = entries_with_whale[0]
    #   last_entry_with_whale = entries_with_whale[-1]
    #   # Determine the span of frame entries that should be processed for filtering.
    #   # Will start and end beyond the first and last instance according to the filtering window,
    #   #   since the span of the filtered version may change based on the filter threshold.
    #   first_entry_toProcess = max(0, first_entry_with_whale - window_size_preCenter)
    #   last_entry_toProcess = min(len(masks) - 1, last_entry_with_whale + window_size_postCenter)
    #   print_ifDesired('    Will process %d frames, from index %d to %d' % (last_entry_toProcess-first_entry_toProcess+1, first_entry_toProcess, last_entry_toProcess))
    #
    #   # Determine the span of the image that contains the whale, across all frames.
    #   bounding_boxes_forWhale = bounding_boxes[:, whale_index, :].reshape((-1, 4, 2))
    #   x_min = int(np.nanmin(bounding_boxes_forWhale[entries_have_whale, :, 0]))
    #   x_max = int(np.nanmax(bounding_boxes_forWhale[entries_have_whale, :, 0]))
    #   y_min = int(np.nanmin(bounding_boxes_forWhale[entries_have_whale, :, 1]))
    #   y_max = int(np.nanmax(bounding_boxes_forWhale[entries_have_whale, :, 1]))
    #   x_range = float(x_max - x_min)
    #   y_range = float(y_max - y_min)
    #   print_ifDesired('    Will process %d pixels, from (%d, %d) to (%d, %d); a %0.1f reduction factor' % (x_range*y_range, x_min, y_min, x_max, y_max, float(np.prod(masks.shape[2:]))/float(x_range*y_range)))
    #
    #   # Load the cropped masks for the active frames into memory.
    #   print_ifDesired('    Loading cropped masks for this whale... ', {'end': ''})
    #   t0 = time.time()
    #   # masks_toFilter = np.zeros((last_entry_toProcess-first_entry_toProcess+1, 1, y_max-y_min+1, x_max-x_min+1), dtype=masks.dtype)
    #   # masks.read_direct(masks_toFilter, np.s_[first_entry_toProcess:last_entry_toProcess+1, whale_index, y_min:y_max+1, x_min:x_max+1])
    #   # masks_toFilter = np.squeeze(masks_toFilter)
    #   masks_toFilter = np.squeeze(masks[first_entry_toProcess:last_entry_toProcess+1, whale_index, y_min:y_max+1, x_min:x_max+1])
    #   print_ifDesired('completed in %0.2fs | matrix shape [%d, %d, %d]' % (time.time() - t0, *masks_toFilter.shape))
    #
    #   # Compute a fast rolling median.
    #   # Taking and thresholding the median on a vector of 0 and 1 is equivalent to checking if the mean is above 0.5
    #   #   and computing a sum is much faster than computing a median.
    #   # To speed up computing the sum, will store the sum of the current window and then simply
    #   #   add/subtract the next/previous frame instead of recomputing an entire sum for each window.
    #   print_ifDesired('    Computing the rolling filter')
    #   masks_filtered = np.zeros_like(masks_toFilter) # will be the end result
    #   mask_window_sum = None # the sum of the mask window
    #   reached_end = False # whether the end of the window has reached the end of the frames
    #   t0 = time.time()
    #   last_print_time = t0
    #   for frame_index in range(masks_toFilter.shape[0]):
    #     if time.time() - last_print_time > 5 or frame_index == masks_toFilter.shape[0]-1:
    #       print_ifDesired('      Processing frame index %5d/%d | elapsed time so far: %0.2fs' % (frame_index, masks_toFilter.shape[0]-1, time.time() - t0))
    #       last_print_time = time.time()
    #     # Determine the start/end of the current moving window.
    #     window_start_frame_index = max(0, frame_index - window_size_preCenter)
    #     window_end_frame_index = min(len(masks_toFilter) - 1, frame_index + window_size_postCenter)
    #     num_frames_in_sum = (window_end_frame_index - window_start_frame_index + 1)
    #     # Determine the sum of the masks in the window.
    #     # If this is the first iteration, compute the sum for the whole window.
    #     if mask_window_sum is None:
    #       mask_window_sum = np.sum(masks_toFilter[window_start_frame_index:window_end_frame_index+1, :, :], axis=0)
    #     # If this is not the first iteration, adjust the previous sum instead of processing the entire window.
    #     else:
    #       # If the window has not reached the end of the data, then its end just advanced by one so add that new frame.
    #       if not reached_end:
    #         mask_window_sum += masks_toFilter[window_end_frame_index, :, :]
    #       reached_end = (window_end_frame_index == (len(masks)-1))
    #       # If the window was not pushed against the start of the data, then its start just advanced by one so subtract the previous frame.
    #       if window_start_frame_index > 0:
    #         mask_window_sum -= masks_toFilter[window_start_frame_index-1, :, :]
    #     # Filter based on the mean!
    #     frame_filtered = (mask_window_sum/num_frames_in_sum) >= rolling_mean_threshold
    #     # Store the result in memory.
    #     masks_filtered[frame_index, :, :] = frame_filtered
    #   print_ifDesired('    Completed filtering in %0.2fs' % (time.time() - t0))
    #   print_ifDesired('    Updating the dataset on disk... ', {'end':''})
    #   t0 = time.time()
    #   # masks.write_direct(masks_filtered, None, np.s_[first_entry_toProcess:last_entry_toProcess+1, whale_index, y_min:y_max+1, x_min:x_max+1])
    #   masks[first_entry_toProcess:last_entry_toProcess+1, whale_index, y_min:y_max+1, x_min:x_max+1] = masks_filtered
    #   print_ifDesired('completed in %0.2fs' % (time.time() - t0))
    #   print_ifDesired('    Total elapsed time: %0.2fs' % (time.time() - start_time_s))
    #
    # # Update metadata.
    # self._update_metadata_dateModified(segmentations=True, annotations=False)
    #
    # print_ifDesired('  Finished smoothing the whale instance masks')
  
  #################################
  # Video Frames and Visualizations
  #################################
  
  # Add a frame to a desired video.
  def add_video_frame(self, video_key, frame_index, img, img_format='rgb', write_frame_index=True):
    # if self._video_readers[video_key] is not None:
    #   raise AssertionError('Cannot currently add to a video that existed at startup')
    if self._video_filepaths[video_key] is None:
      raise AssertionError('No video filepath was provided for key [%s]' % video_key)
    if not self._writable:
      raise AssertionError('Segmentations was opened in read-only mode')
    # Load the image if a filepath was provided.
    if isinstance(img, str):
      img = load_image(img)
      img = np.squeeze(img[:,:,0:3])
      img = np.ascontiguousarray(img)
      img_format = 'rgb'
    # Only use the first three image channels.
    img = np.squeeze(img[:,:,0:3])
    # Create the FFMPEG process if needed.
    (img_height, img_width, img_depth) = img.shape
    if self._ff_procs[video_key] is None:
      self._ff_procs[video_key] = (
            ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='%sx%s'%(int(img_width), int(img_height)),
                   r=self._output_video_fps,  # assume a constant frame rate. NOTE: If put "r" as an output argument, ffmpeg will add/drop frames to achieve the target rate
                   )
            .output(self._video_filepaths[video_key],
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    crf=self._video_compression,
                    preset=self._video_preset,
                    # h264 needs dimensions divisible by 2,
                    # so add a filter that pads the bottom and right as needed
                    vf="pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
                    )
            .run_async(pipe_stdin=True)
        )
    # Add the frame number if desired.
    if write_frame_index:
      draw_text_on_image(img, 'Frame index: %6d' % frame_index,
                         pos=(-1, -1),
                         font_scale=None, text_width_ratio=0.2,
                         font_thickness=1, font=cv2.FONT_HERSHEY_DUPLEX,
                         text_color_bgr=(0, 0, 0),
                         text_bg_color_bgr=(200, 200, 200),
                         text_bg_outline_color_bgr=None,
                         text_bg_pad_width_ratio=0.03,
                         preview_only=False,
                         )
    # Convert the image if needed.
    if img_format.lower().strip() == 'bgr':
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Write black frames if needed to get to the target frame index.
    while self._num_video_frames[video_key] < frame_index:
      img_black = 0*img
      self._ff_procs[video_key].stdin.write(img_black.astype(np.uint8).tobytes())
      self._num_video_frames[video_key] += 1
    # Write the new frame to the video.
    self._ff_procs[video_key].stdin.write(img.astype(np.uint8).tobytes())
    self._num_video_frames[video_key] += 1
    
    # Save the frame as an image if desired.
    if video_key in self._video_frame_image_dirs:
      images_dir = self._video_frame_image_dirs[video_key]
      # Delete the oldest image if there are more than desired in the folder.
      images_saved = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
      if self._num_video_frames_to_save_as_images > 0 and len(images_saved) >= self._num_video_frames_to_save_as_images:
        os.remove(images_saved[0])
      # Save the new image.
      image_filepath = os.path.join(images_dir, '%s_frame_%06d.jpg' % (video_key, frame_index))
      cv2.imwrite(image_filepath, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
  # Get a desired video frame.
  def get_video_frame(self, video_key, frame_index, resize_to_mask_shape=False,
                      show_masks=False, show_centroids=False,
                      show_orientations=False, show_boxes=None,
                      show_whale_id_numbers=False, show_whale_indexes=False):
    if self._ff_procs[video_key] is not None:
      raise AssertionError('Cannot currently read a video while it is being created')
    if self._video_filepaths[video_key] is None:
      raise AssertionError('No video filepath was provided for key [%s]' % video_key)
    if self._video_readers[video_key] is None:
      raise AssertionError('No video reader was opened for key [%s]' % video_key)
    if frame_index < 0 or frame_index >= self._num_video_frames[video_key]:
      raise ValueError('Invalid frame index %d for video [%s] with %d total frames'%(frame_index, video_key, self._num_video_frames[video_key]))
    # Load the video frame.
    img_rgb = self._video_readers[video_key][frame_index].asnumpy()
    
    # Resize the image to match the mask shape if desired.
    annotating_img = show_masks or show_centroids or show_orientations or (show_boxes is not None) or show_whale_id_numbers or show_whale_indexes
    if (resize_to_mask_shape or annotating_img) and self.have_masks():
      img_rgb = scale_image(img_rgb, target_width=self._frame_shape[1], target_height=self._frame_shape[0], maintain_aspect_ratio=False)
    
    # Add any desired visualizations to the image.
    if annotating_img:
      img_rgb = self.visualize_segmentations(frame_index=frame_index, img_rgb=img_rgb,
                                             show_masks=show_masks,
                                             show_centroids=show_centroids,
                                             show_orientations=show_orientations,
                                             show_boxes=show_boxes,
                                             show_whale_id_numbers=show_whale_id_numbers,
                                             show_whale_indexes=show_whale_indexes)
    
    # Return the result.
    return img_rgb
  
  # Add segmentation visualizations to a frame or a graph.
  # If the image frame is None, will draw on a black image.
  # If graph is True, will return a Matplotlib graph instead of drawing on an image.
  #  If graphing, will use the provided figure if one is provided.
  #  If the image Frame is None and no masks are stored in the segmentations, graph will be True.
  def visualize_segmentations(self, frame_index, img_rgb=None, graph=False, fig=None,
                              show_masks=False, show_centroids=False,
                              show_orientations=False, show_boxes=None,
                              show_whale_id_numbers=False, show_whale_indexes=False):
    # If no image is provided and there are no masks, use graphing.
    if img_rgb is None and not self.have_masks():
      graph = True
    
    # Create an image or a plot with the visualizations.
    if not graph:
      return self.visualize_segmentations_on_image(frame_index=frame_index, img_rgb=img_rgb,
                                                    show_masks=show_masks, show_centroids=show_centroids,
                                                    show_orientations=show_orientations, show_boxes=show_boxes,
                                                    show_whale_id_numbers=show_whale_id_numbers, show_whale_indexes=show_whale_indexes)
    else:
      return self.visualize_segmentations_on_graph(frame_index=frame_index, fig=fig,
                                                    show_masks=show_masks, show_centroids=show_centroids,
                                                    show_orientations=show_orientations, show_boxes=show_boxes,
                                                    show_whale_id_numbers=show_whale_id_numbers, show_whale_indexes=show_whale_indexes)
  
  # Add segmentation visualizations to a frame.
  # If the image frame is None, will draw on a black image.
  def visualize_segmentations_on_image(self, frame_index, img_rgb=None,
                                        show_masks=False, show_centroids=False,
                                        show_orientations=False, show_boxes=None,
                                        show_whale_id_numbers=False, show_whale_indexes=False):
    # Create a black image if no frame was provided.
    if img_rgb is None:
      img_rgb = np.zeros((*self.get_frame_shape(), 3), dtype=np.uint8)
      
    # Resize the image to match the mask shape.
    if self.have_masks():
      img_rgb = scale_image(img_rgb, target_width=self._frame_shape[1], target_height=self._frame_shape[0], maintain_aspect_ratio=False)
    
    # Define sizes based on the frame size.
    linewidth_thicker = round(img_rgb.shape[0]*0.01)
    linewidth_thinner = round(linewidth_thicker*3/4)
    circle_radius = round(img_rgb.shape[0]*0.01)
    mask_alpha = 0.5
    
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr_annotated = img_bgr.copy()
    for whale_index in range(self.get_num_whales()):
      # Draw the bounding boxes.
      if show_boxes is not None:
        bounding_box_colors = {'full': (255,255,255), 'head': (255,255,0), 'tail':(255,0,255)}
        for bounding_box_key in show_boxes:
          bounding_box_4xy = self.get_bounding_box_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index, whale_index=whale_index)
          if bounding_box_4xy is not None:
            bounding_box_points = np.array([bounding_box_4xy[0:2], bounding_box_4xy[2:4],
                                            bounding_box_4xy[4:6], bounding_box_4xy[6:8]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_bgr_annotated, [bounding_box_points], True,
                          bounding_box_colors[bounding_box_key],
                          linewidth_thicker if bounding_box_key == 'full' else linewidth_thinner)
          else:
            # The segmentation did not exist for this whale in this frame.
            pass
      # Color the masks.
      if show_masks and self.have_masks():
        mask_contours = self.get_mask_contours(frame_index=frame_index, whale_index=whale_index)
        if mask_contours is not None:
          whale_color = self.get_whale_color(whale_index)
          color_mask = np.zeros_like(img_bgr_annotated)
          color_mask = cv2.drawContours(color_mask, mask_contours,
                                        -1, # -1 means to draw all contours in the given list
                                        whale_color, # value to fill
                                        -1) # -1 means fill rather than only outline
          img_bgr_annotated = cv2.addWeighted(img_bgr_annotated, 1, color_mask, mask_alpha, 0)
      # Draw the centroid.
      if show_centroids:
        centroid_xy = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
        if centroid_xy is not None:
          cv2.circle(img_bgr_annotated, [round(point) for point in centroid_xy],
                     circle_radius, (255, 255, 255), -1)
      # Draw the orientation color-coded by orientation confidence.
      if show_orientations:
        (orientation_rad, orientation_confidence) = self.get_orientation_rad_confidence(frame_index=frame_index, whale_index=whale_index)
        if orientation_rad is not None:
          # Compute the orientation vector length as the length of the bounding box.
          bounding_box_4xy = self.get_bounding_box_4xy(bounding_box_key='full', frame_index=frame_index, whale_index=whale_index)
          bounding_boxes_4xy_reshaped = bounding_box_4xy.reshape((-1, 2))
          orientation_length = max(np.linalg.norm(np.diff(bounding_boxes_4xy_reshaped, axis=0), axis=1))/2
          # Compute the orientation start/end point.
          centroid_xy = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
          orientation_start_point = [round(coord) for coord in centroid_xy]
          orientation_end_point = np.array(orientation_start_point) \
                                  + orientation_length*np.array([np.cos(orientation_rad), -np.sin(orientation_rad)])
          orientation_end_point = [round(coord) for coord in orientation_end_point]
          # Compute the orientation color to represent the confidence.
          orientation_color_red = round(255*(1-orientation_confidence))
          orientation_color_green = round(255*orientation_confidence)
          orientation_color = (0, orientation_color_green, orientation_color_red)
          # Draw the start point.
          cv2.circle(img_bgr_annotated, orientation_start_point,
                     round(circle_radius*0.8), orientation_color, -1)
          # Draw the orientation vector.
          cv2.line(img_bgr_annotated, orientation_start_point, orientation_end_point,
                   orientation_color,
                   linewidth_thicker)
        else:
          # The segmentation did not exist for this whale in this frame.
          pass
      # Write the whale indexes or ID numbers.
      if show_whale_id_numbers or show_whale_indexes:
        centroid_xy = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
        if centroid_xy is not None:
          centroid_xy = centroid_xy.astype(int)
          bounding_box_4xy = self.get_bounding_box_4xy(bounding_box_key='full', frame_index=frame_index, whale_index=whale_index)
          bounding_boxes_4xy_reshaped = bounding_box_4xy.reshape((-1, 2))
          # Find a font scale.
          if show_whale_id_numbers:
            text_str = '%s' % self.get_whale_id_numbers()[whale_index]
          else:
            text_str = '%s' % whale_index
          (text_w, text_h, font_scale, _) = draw_text_on_image(
            img_bgr_annotated,
            text_str,
            pos=(0.5, 0.5),
            font_scale=None,
            text_height_ratio=0.02,#(box_width)/img.shape[1],
            font_thickness=2, font=cv2.FONT_HERSHEY_DUPLEX,
            text_color_bgr=(255,255,255),
            text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None,
            text_bg_pad_width_ratio=0.05,
            preview_only=True,
            )
          # Adjust the text position based on the text size at this font scale.
          text_xy = centroid_xy + np.array([-text_w, text_h])
          draw_text_on_image(
            img_bgr_annotated,
            text_str,
            pos=text_xy,
            font_scale=font_scale,
            text_height_ratio=None,
            font_thickness=2, font=cv2.FONT_HERSHEY_DUPLEX,
            text_color_bgr=(255,255,255),
            text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None,
            text_bg_pad_width_ratio=0.05,
            preview_only=False,
            )
    img_rgb = cv2.cvtColor(img_bgr_annotated, cv2.COLOR_BGR2RGB)
    
    # Return the result.
    return img_rgb
  
  # Visualize segmentations on a graph.
  def visualize_segmentations_on_graph(self, frame_index, fig=None,
                              show_masks=False, show_centroids=False,
                              show_orientations=False, show_boxes=None,
                              show_whale_id_numbers=False, show_whale_indexes=False):
    # Create a figure if none was provided.
    if fig is None:
      fig = plt.figure()
      plt.get_current_fig_manager().window.showMaximized()
      
    # Define sizes based on the frame size.
    linewidth_thicker = 3
    linewidth_thinner = 2
    circle_radius = 10
    mask_alpha = 0.1
    
    for whale_index in range(self.get_num_whales()):
      # Color the masks.
      if show_masks and self.have_masks():
        mask_contours = self.get_mask_contours(frame_index=frame_index, whale_index=whale_index)
        if mask_contours is not None:
          for contour in mask_contours:
            coords_xy = np.atleast_2d(np.squeeze(contour))
            # self._update_segmentation_colors() # compute colors if needed
            # plt.plot(coords_xy[:,0], coords_xy[:,1], '.-', alpha=mask_alpha,
            #        color=self._segmentations_colors[whale_index % len(self._segmentations_colors)]/255)
            plt.gca().add_patch(Polygon(
                  coords_xy,
                  closed=True,
                  edgecolor='none',
                  facecolor=self.get_whale_color(whale_index, 1),
                  fill=True))
      # Draw the bounding boxes.
      if show_boxes is not None:
        bounding_box_colors = {'full': (0,0,0), 'head': (0,1,1), 'tail':(1,0,1)}
        for bounding_box_key in show_boxes:
          bounding_box_4xy = self.get_bounding_box_4xy(bounding_box_key=bounding_box_key, frame_index=frame_index, whale_index=whale_index)
          if bounding_box_4xy is not None:
            bounding_boxes_4xy_reshaped = bounding_box_4xy.reshape((4, 2))
            plt.gca().add_patch(Polygon(
                bounding_boxes_4xy_reshaped,
                closed=True,
                edgecolor=bounding_box_colors[bounding_box_key],
                linewidth=linewidth_thicker if bounding_box_key == 'full' else linewidth_thinner,
                fill=False))
          else:
            # The segmentation did not exist for this whale in this frame.
            pass
      # Draw the centroid.
      if show_centroids:
        centroid_xy = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
        if centroid_xy is not None:
          plt.gca().add_patch(Circle(
                centroid_xy,
                radius=circle_radius,
                edgecolor='none',
                facecolor=(0,0,0),
                fill=True))
      # Draw the orientation color-coded by orientation confidence.
      if show_orientations:
        (orientation_rad, orientation_confidence) = self.get_orientation_rad_confidence(frame_index=frame_index, whale_index=whale_index)
        if orientation_rad is not None:
          # Clip the confidence to be between 0 and 1.
          orientation_confidence = max(0, min(1, orientation_confidence))
          # Compute the orientation vector length as the length of the bounding box.
          bounding_box_4xy = self.get_bounding_box_4xy(bounding_box_key='full', frame_index=frame_index, whale_index=whale_index)
          bounding_boxes_4xy_reshaped = bounding_box_4xy.reshape((-1, 2))
          orientation_length = max(np.linalg.norm(np.diff(bounding_boxes_4xy_reshaped, axis=0), axis=1))/2
          # Compute the orientation start/end point.
          centroid_xy = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
          orientation_start_point = [round(coord) for coord in centroid_xy]
          orientation_end_point = np.array(orientation_start_point) \
                                  + orientation_length*np.array([np.cos(orientation_rad), -np.sin(orientation_rad)])
          orientation_end_point = [round(coord) for coord in orientation_end_point]
          # Compute the orientation color to represent the confidence.
          orientation_color_red = round(255*(1-orientation_confidence))
          orientation_color_green = round(255*orientation_confidence)
          orientation_color = (orientation_color_red, orientation_color_green, 0)
          orientation_color = np.array(orientation_color)/255
          # Draw the start point.
          plt.gca().add_patch(Circle(
                orientation_start_point,
                radius=round(circle_radius*0.8),
                edgecolor='none',
                facecolor=orientation_color,
                fill=True))
          # Draw the orientation vector.
          plt.plot([orientation_start_point[0], orientation_end_point[0]],
                   [orientation_start_point[1], orientation_end_point[1]],
                   '-', color=orientation_color, linewidth=linewidth_thicker)
        else:
          # The segmentation did not exist for this whale in this frame.
          pass
      # Write the whale indexes or ID numbers.
      if show_whale_id_numbers or show_whale_indexes:
        centroid_xy = self.get_centroid_xy(frame_index=frame_index, whale_index=whale_index)
        if centroid_xy is not None:
          centroid_xy = centroid_xy.astype(int)
          if show_whale_id_numbers:
            text_str = '%s' % self.get_whale_id_numbers()[whale_index]
          else:
            text_str = '%s' % whale_index
          text_xy = centroid_xy
          plt.text(text_xy[0], text_xy[1], text_str)
    
    # Set the axis limits to the frame size if known.
    if self.have_masks():
      frame_shape = self.get_frame_shape()
      plt.xlim([0, frame_shape[1]])
      plt.ylim([0, frame_shape[0]])
    # Flip the y axis to match image coordinates, which has the origin in the upper left.
    plt.gca().invert_yaxis()
    # Set an equal aspect ratio.
    plt.gca().set_aspect('equal', 'box')
    # Add labels.
    plt.title('Segmentations for Frame %d' % frame_index)
    plt.xlabel('Image Coordinate X')
    plt.ylabel('Image Coordinate Y')
    
    # Return the figure.
    return fig
  
  # Visualize centroids over time.
  # frame_indexes can be 'all' or [start_index, end_index].
  # whale_indexes can be 'all' or a list of whale indexes to process.
  # whale_duration_ratio_filter can be None to show all whales, or a number between 0 and 1
  #   to only show whales that are present in the video for at least that ratio of the video duration.
  def visualize_centroid_trajectories(self, frame_indexes='all', whale_indexes='all',
                                            whale_duration_ratio_filter=None,
                                            apply_smoothing_filter=False,
                                            smoothing_window_size_preCenter=20, smoothing_window_size_postCenter=20):
    # Get the centroids, with an optional smoothing filter.
    centroids_xy = self.get_all_centroids_xy(apply_smoothing_filter=apply_smoothing_filter,
                                              smoothing_window_size_preCenter=smoothing_window_size_preCenter, smoothing_window_size_postCenter=smoothing_window_size_postCenter,
                                              whale_indexes_toSmooth=whale_indexes, smoothed_whale_indexes_toPlot=[],
                                              print_smoothing_status=False)
    centroids_xy = np.array(centroids_xy) # Load them into memory
    # Initialize.
    if isinstance(whale_indexes, str) and whale_indexes.lower().strip() == 'all':
      whale_indexes = None
    if isinstance(frame_indexes, str) and frame_indexes.lower().strip() == 'all':
      frame_indexes = [0, centroids_xy.shape[0]]
    if whale_duration_ratio_filter is None:
      whale_duration_ratio_filter = 0
    fig = plt.figure()
    plt.get_current_fig_manager().window.showMaximized()
  
    # Define plotting parameters.
    trajectory_linewidth = 2
    trajectory_markersize = 5
    trajectory_start_markersize = 15
    
    # Visualize the trajectory for each whale.
    for whale_index in range(self.get_num_whales()):
      if whale_indexes is not None and whale_index not in whale_indexes:
        continue
      # Get the color for this whale index.
      whale_color = self.get_whale_color(whale_index, 1)
      # Get the trajectory for this whale.
      x = centroids_xy[frame_indexes[0]:frame_indexes[-1]+1, whale_index, 0]
      y = centroids_xy[frame_indexes[0]:frame_indexes[-1]+1, whale_index, 1]
      # Determine the frame indexes with this whale active.
      frame_indexes_withWhale = np.where(np.all(~np.isnan(centroids_xy[frame_indexes[0]:frame_indexes[-1]+1, whale_index, :]), axis=1))[0]
      # Filter based on the duration this whale is active.
      if whale_duration_ratio_filter is not None and len(frame_indexes_withWhale)/centroids_xy.shape[0] < whale_duration_ratio_filter:
        continue
      if len(frame_indexes_withWhale) == 0:
        continue
      # Plot the starting position.
      x0 = x[frame_indexes_withWhale[0]]
      y0 = y[frame_indexes_withWhale[0]]
      plt.plot(x0, y0,
               '*', color=whale_color, markersize=trajectory_start_markersize)
      # Plot the trajectory.
      plt.plot(x[frame_indexes_withWhale], y[frame_indexes_withWhale],
               '.-', color=whale_color, markersize=trajectory_markersize, linewidth=trajectory_linewidth)
      # Label the whale index.
      text_str = '%s' % self.get_whale_id_numbers()[whale_index]
      plt.text(x0, y0, text_str, backgroundcolor=(0.8, 0.8, 0.8))
    # Format the plot.
    plt.title('Centroid Trajectories')
    plt.xlabel('Mask X Coordinate')
    plt.ylabel('Mask Y Coordinate')
    plt.grid(True, color='lightgray')
    plt.xlim([0, self.get_frame_shape()[1]])
    plt.ylim([0, self.get_frame_shape()[0]])
  
  
  ###############################
  # Cleanup
  ###############################
  
  def quit(self, resize_frame_dimension=False, resize_whale_dimension=False, remove_unused_whale_indexes=False,
                 num_frames_total=None, num_whales_total=None, author=''):
    if self._h5_file is not None:
      if self._writable:
        # Add a log entry.
        author = author or self._author
        method_kwargs = dict([(k,v) for (k,v) in locals().items() if k not in ['self']])
        self.add_history_entry(summary='quit', details=method_kwargs,
                               timestamp_s=time.time(), author=author)
        # Resize the datasets to remove any extra empty frames or whale indexes
        # or to add frames if desired (i.e. if there  were no segmentations added for trailing frames in the video).
        num_frames = None
        max_frame_index_segmented = self.get_max_frame_index_segmented()
        if max_frame_index_segmented is not None:
          num_frames = max_frame_index_segmented+1
        if num_frames_total is not None:
          num_frames = num_frames_total
        if num_frames is None:
          resize_frame_dimension = False
        whale_frame_counts = self.get_whale_frame_counts()
        whale_indexes_with_segmentations = np.where(whale_frame_counts > 0)[0]
        if whale_indexes_with_segmentations.size == 0:
          print('*** WARNING: It seems like none of the whale indexes have associated segmentations. ')
          num_whales = None
        else:
          num_whales = np.max(whale_indexes_with_segmentations)+1
        if num_whales_total is not None:
          num_whales = num_whales_total
        edited_datasets = False
        for (dataset_name, dataset) in self._datasets.items():
          # Skip annotations and history since they will be processed below.
          if dataset_name in ['annotations', 'history']:
            continue
          # Specify which dimension is used for frames and whales.
          # Most datasets have frame as dimension 0 and whales as dimension 1, but there are a few exceptions.
          frame_dimension = 0
          whale_dimension = 1
          if dataset_name == 'frames_are_segmented':
            whale_dimension = None
          # Trim (or expand) the frame dimension.
          if resize_frame_dimension and frame_dimension is not None:
            new_shape = list(dataset.shape)
            new_shape[frame_dimension] = num_frames
            if not np.array_equal(new_shape, dataset.shape):
              edited_datasets = True
              dataset.resize(new_shape)
          # Trim (or expand) the whale dimension.
          if resize_whale_dimension and whale_dimension is not None and num_whales is not None:
            new_shape = list(dataset.shape)
            new_shape[whale_dimension] = num_whales
            if not np.array_equal(new_shape, dataset.shape):
              edited_datasets = True
              dataset.resize(new_shape)
        self._num_frames = num_frames
        # Update annotations datasets.
        if resize_whale_dimension and num_whales is not None:
          for (dataset_key, dataset) in self._h5_file['annotations']['whale_ids'].items():
            whale_dimension = 0
            new_shape = list(dataset.shape)
            new_shape[whale_dimension] = num_whales
            if not np.array_equal(new_shape, dataset.shape):
              edited_datasets = True
              dataset.resize(new_shape)
          max_whale_id_number = max(self._whale_id_numbers) if len(self._whale_id_numbers) > 0 else -1
          for whale_index in range(len(self._whale_id_numbers), self._h5_file['annotations']['whale_ids']['id_numbers'].shape[0]):
            self._h5_file['annotations']['whale_ids']['id_numbers'][whale_index] = max_whale_id_number+1
            max_whale_id_number += 1
          self._whale_ids = self.get_whale_ids(use_local_copy=False)
          self._whale_id_numbers = self.get_whale_id_numbers(use_local_copy=False)
          self._whale_ids_is_auto = self.get_whale_ids_is_auto(use_local_copy=False)
          for group_key in ['notes', 'behaviors', 'events']:
            whale_dimension = 1
            dataset = self._h5_file['annotations'][group_key]['whales_involved']
            new_shape = list(dataset.shape)
            new_shape[whale_dimension] = num_whales
            if not np.array_equal(new_shape, dataset.shape):
              edited_datasets = True
              dataset.resize(new_shape)
        # Update the local existence matrix.
        if not np.array_equal(self._whale_segmentations_exist.shape, self._h5_file['whale_segmentations_exist'].shape):
          self._whale_segmentations_exist = self.get_whale_segmentations_exist()
        # Remove any whale indexes that are no longer used.
        if remove_unused_whale_indexes:
          edited_datasets = np.any(self.get_whale_frame_counts() < 1)
          self.filter_whale_instances_byCount(min_frame_count=1, remove_masks_dataset=False, create_new_hdf5_file=False, author='[subcall from "quit"]')
        # Update metadata if needed.
        if edited_datasets:
          self._update_metadata_dateModified(segmentations=True, annotations=True)
          # self.add_history_entry(summary='quit', details=method_kwargs,
          #                        timestamp_s=time.time(), author=author)
        
      # Close the file
      try:
        self._h5_file.close()
      except ModuleNotFoundError: # something goes out of scope related to calling filter_whale_instances?
        pass
      self._h5_file = None
    # Close any FFMpeg processes and wait for them to finish.
    try:
      for ff_proc in self._ff_procs.values():
        if ff_proc is not None:
          ff_proc.stdin.close()
          ff_proc.wait()
    except AttributeError:
      pass # The class probably didn't finish initializing and create the self._ff_procs variable
    # Close any video readers.
    try:
      for (video_key, video_reader) in self._video_readers.items():
        if video_reader is not None:
          video_reader.close()
          video_reader = None
          self._video_readers[video_key] = None
      del self._video_readers
    except AttributeError:
      pass # The class probably didn't finish initializing and create the self._video_readers variable
    
  def close(self, resize_frame_dimension=False, resize_whale_dimension=False, remove_unused_whale_indexes=False,
                  num_frames_total=None, num_whales_total=None, author=''):
    self.quit(resize_frame_dimension=resize_frame_dimension, resize_whale_dimension=resize_whale_dimension,
              remove_unused_whale_indexes=remove_unused_whale_indexes,
              num_frames_total=num_frames_total, num_whales_total=num_whales_total, author=author)

  def __del__(self):
    # print('Closing the Segmentations since the object is being deleted')
    self.close(author='[object_deleted] [%s]' % self._author)

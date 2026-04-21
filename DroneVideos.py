
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
# [can add additional updates and authors as desired]
#
############

import numpy as np
import h5py
import cv2
import re
import fnmatch
import os
import glob
import copy
from collections import OrderedDict
import gc
try:
  import ctypes
  libc = ctypes.CDLL("libc.so.6")
  libc.malloc_trim(0) # https://stackoverflow.com/a/64879094
  can_use_libc = True
except:
  can_use_libc = False

try:
  from csail_data_processing.DecordVideoReaderWrapper import DecordVideoReaderWrapper
  can_use_videos = True
except:
  can_use_videos = False

###############################################
# Helpers
###############################################

# Find a timestamp from a device that most closely matches a target timestamp.
# Will return the index of that matched timestamp within the device's array of timestamps.
# If there is no such timestamp within a specified threshold of the target, returns None.
def get_index_for_time_s(timestamps_s, target_time_s, timestamp_to_target_thresholds_s):
  if timestamps_s.shape[0] == 1:
    # If there is only one timestamp, consider that the best one.
    best_index = 0
  else:
    # Find the index where the target timestamp would be inserted without changing the sort order.
    # This is much faster than using something like numpy.where(), since it can assume the input is sorted.
    next_index_pastTarget = timestamps_s.searchsorted(target_time_s)
    # If it returned the length of the array, decrement it to make it a valid index.
    if next_index_pastTarget == timestamps_s.shape[0]:
      next_index_pastTarget -= 1
    # If it returned the first element, use that as the best index.
    if next_index_pastTarget == 0:
      best_index = 0
    else:
      # We have placed the target between two device timestamps.
      # Now see which one of those two is closer to the target.
      index_candidates = np.array([next_index_pastTarget-1, next_index_pastTarget])
      dt_candidates = np.abs(timestamps_s[index_candidates] - target_time_s)
      if dt_candidates[0] < dt_candidates[1]:
        best_index = index_candidates[0]
      else:
        best_index = index_candidates[1]
  # Check if the closest timestamp is within the threshold region of the target.
  if timestamps_s[best_index] < (target_time_s - timestamp_to_target_thresholds_s[0]):
    return None
  if timestamps_s[best_index] > (target_time_s + timestamp_to_target_thresholds_s[1]):
    return None
  # We found a good timestamp! Return its index.
  return best_index

###############################################
# DroneVideos class
###############################################
class DroneVideos:
  # Initialize the class.
  # drone_data_hdf5_filepaths is a dict mapping drone key to an HDF5 file.
  # video_filepaths is a dict mapping drone key to a list of video filepaths.
  def __init__(self, drone_data_hdf5_filepaths, video_dirs=None,
               custom_epoch_time_s=None):
    import time
    t0 = time.time()
    self._data_filepaths = drone_data_hdf5_filepaths
    self._video_dirs = video_dirs
    self._custom_epoch_time_s = custom_epoch_time_s
    
    self._drone_keys = list(drone_data_hdf5_filepaths.keys())
    self._drone_datas = dict([(key, OrderedDict()) for key in self._drone_keys])
    self._video_readers = dict([(key, OrderedDict()) for key in self._drone_keys])
    self._video_readers_fullResolution = dict([(key, OrderedDict()) for key in self._drone_keys])
    self._video_filepaths = dict([(key, OrderedDict()) for key in self._drone_keys])
    self._video_frame_shapes = dict([(key, OrderedDict()) for key in self._drone_keys])
    self._target_width = None
    self._target_height = None
    
    # Store video filepaths, and initialize empty entries for video readers for each video.
    # Will actually create readers only when a video needs to be accessed.
    print('DroneVideos initializing empty video readers')
    if self._video_dirs is not None:
      for drone_key in self._drone_keys:
        # Look for versions with various resolutions.
        all_filepaths = glob.glob(os.path.join(self._video_dirs[drone_key], '**', '*'), recursive=True)
        ignore_extension_case = True
        for extension_pattern in ['*.%s' % extension for extension in ['lrf', 'mp4']]:
          rule = re.compile(fnmatch.translate(extension_pattern), re.IGNORECASE) if ignore_extension_case \
                   else re.compile(fnmatch.translate(extension_pattern))
          video_filepaths = [filepath for filepath in all_filepaths if rule.match(filepath)]
          print('  See %2d videos for drone [%s]' % (len(video_filepaths), drone_key))
          for video_filepath in video_filepaths:
            video_key = os.path.splitext(os.path.basename(video_filepath))[0]
            self._video_filepaths[drone_key].setdefault(video_key, [])
            self._video_frame_shapes[drone_key].setdefault(video_key, [])
            self._video_filepaths[drone_key][video_key].append(video_filepath)
            self._video_frame_shapes[drone_key][video_key].append(None)
            self._video_readers[drone_key][video_key] = None
            self._video_readers_fullResolution[drone_key][video_key] = None
    
    # Load the drone metadata.
    print('DroneVideos loading drone data series')
    for drone_key in self._drone_keys:
      # Open the HDF5 file.
      data_filepath = self._data_filepaths[drone_key]
      h5file = h5py.File(data_filepath, 'r')
      # The top-level keys of the file are the corresponding video filenames.
      video_filenames = list(h5file.keys())
      print('  See %2d entries of video data for drone [%s]' % (len(video_filenames), drone_key))
      # Load the data for each video file.
      for video_filename in video_filenames:
        # Load the global timestamps for each frame of the video.
        # Timestamps have been adjusted to be synchronized based on the current manual estimates.
        timestamps_s   = np.array(h5file[video_filename]['time']['aligned_timestamp_s'])
        timestamps_str = np.array(h5file[video_filename]['time']['aligned_timestamp_str'])
        # The HDF5 file also has original timestamps recorded by the drone during filming.
        # These are not aligned using the manual offsets, and they may not account for time zone offsets.
        original_timestamps_s   = np.array(h5file[video_filename]['time']['original_timestamp_s'])
        original_timestamps_str = np.array(h5file[video_filename]['time']['original_timestamp_str'])
        
        # Load data about the drone's position.
        latitudes            = np.array(h5file[video_filename]['position']['latitude'])
        longitudes           = np.array(h5file[video_filename]['position']['longitude'])
        altitudes_relative_m = np.array(h5file[video_filename]['position']['altitude_relative_m'])
        #altitudes_absolute_m = np.array(h5file[video_filename]['position']['altitude_absolute_m'])
        
        # Load data about the drone's speed that was estimated from the GPS positions.
        speed_horizontal_m_s = np.array(h5file[video_filename]['speed']['estimated_speed_horizontal_fromGPS_m_s'])
        is_stationary_horizontal = np.array(h5file[video_filename]['speed']['estimated_isStationary_horizontal_fromGPS'])
        # Load data about the drone's speed that was estimated from altitude measurements.
        speed_vertical_m_s = np.array(h5file[video_filename]['speed']['estimated_speed_vertical_fromAltitude_m_s'])
        is_stationary_vertical = np.array(h5file[video_filename]['speed']['estimated_isStationary_vertical_fromAltitude'])
        
        # Note that there is 1 frame where the latitude was 0;
        #  presumably this is used to indicate a sensor error.
        # Mark any such frames as NaN for clarity.
        speed_horizontal_m_s[np.where((latitudes == 0) | (longitudes == 0))[0]] = np.nan
        is_stationary_horizontal[np.where((latitudes == 0) | (longitudes == 0))[0]] = np.nan
        speed_vertical_m_s[np.where((latitudes == 0) | (longitudes == 0))[0]] = np.nan
        is_stationary_vertical[np.where((latitudes == 0) | (longitudes == 0))[0]] = np.nan
        latitudes[np.where(latitudes == 0)[0]] = np.nan
        longitudes[np.where(longitudes == 0)[0]] = np.nan
        
        # Load data about the camera settings.
        color_modes        = np.array(h5file[video_filename]['camera']['color_mode'])
        color_temperatures = np.array(h5file[video_filename]['camera']['color_temperature'])
        exposure_values    = np.array(h5file[video_filename]['camera']['exposure_value'])
        f_numbers          = np.array(h5file[video_filename]['camera']['f_number'])
        focal_lengths      = np.array(h5file[video_filename]['camera']['focal_length'])
        isos               = np.array(h5file[video_filename]['camera']['iso'])
        shutters           = np.array(h5file[video_filename]['camera']['shutter'])
        
        # Store the data.
        self._drone_datas[drone_key][video_filename] = {
          'timestamps_s': timestamps_s,
          'timestamps_str': timestamps_str,
          'latitudes': latitudes,
          'longitudes': longitudes,
          'altitudes_relative_m': altitudes_relative_m,
          'speed_horizontal_m_s': speed_horizontal_m_s,
          'is_stationary_horizontal': is_stationary_horizontal,
          'speed_vertical_m_s': speed_vertical_m_s,
          'is_stationary_vertical': is_stationary_vertical,
          'color_modes': color_modes,
          'color_temperatures': color_temperatures,
          'exposure_values': exposure_values,
          'f_numbers': f_numbers,
          'focal_lengths': focal_lengths,
          'isos': isos,
          'shutters': shutters,
        }
      
      # Release the file reader.
      h5file.close()
    # print()
    # print('DroneVideos loaded in %0.3fs' % (time.time() - t0))
    
  def __del__(self):
    self.clear_video_readers()
  
  # Clear video readers to reduce memory usage.
  def clear_video_readers(self, clear_fullResolution_readers=True):
    print('DroneVideos clearing video readers')
    for drone_key in self._video_filepaths:
      for video_key in self._video_filepaths[drone_key]:
        if self._video_readers[drone_key][video_key] is not None:
          print('  Deleting video reader for', video_key)
          self._video_readers[drone_key][video_key].close()
          gc.collect()
          if can_use_libc: libc.malloc_trim(0)
          del self._video_readers[drone_key][video_key]
          gc.collect()
          if can_use_libc: libc.malloc_trim(0)
          self._video_readers[drone_key][video_key] = None
        if clear_fullResolution_readers and self._video_readers_fullResolution[drone_key][video_key] is not None:
          print('  Deleting full-resolution video reader for', video_key)
          self._video_readers_fullResolution[drone_key][video_key].close()
          gc.collect()
          if can_use_libc: libc.malloc_trim(0)
          del self._video_readers_fullResolution[drone_key][video_key]
          gc.collect()
          if can_use_libc: libc.malloc_trim(0)
          self._video_readers_fullResolution[drone_key][video_key] = None
    
  # Update the target video resolution.
  # Will clear all existing video readers.
  def set_target_video_resolution(self, target_width=None, target_height=None):
    if (target_width != self._target_width) or (target_height != self._target_height):
      self.clear_video_readers(clear_fullResolution_readers=False)
    self._target_width = target_width
    self._target_height = target_height
  
  # Check if a video file was found for a given video key.
  def video_file_exists(self, video_key):
    drone_key = self.get_drone_key_for_video_key(video_key)
    if drone_key is None:
      return False
    return video_key in self._video_readers[drone_key]
  
  # Get the possible video filepaths for a video key.
  def get_video_filepaths(self, video_key):
    video_key = str(video_key)
    drone_key = self.get_drone_key_for_video_key(video_key)
    if video_key in self._video_filepaths[drone_key]:
      return self._video_filepaths[drone_key][video_key]
    return []
  
  # Get the possible video resolutions for a video key.
  def get_video_frame_shapes(self, video_key):
    video_key = str(video_key)
    drone_key = self.get_drone_key_for_video_key(video_key)
    if video_key in self._video_frame_shapes[drone_key]:
      for (video_index, video_frame_shape) in enumerate(self._video_frame_shapes[drone_key][video_key]):
        if video_frame_shape is None:
          # Note that CV2 seems to be faster than decord for just checking the resolution.
          v = cv2.VideoCapture(self._video_filepaths[drone_key][video_key][video_index])
          video_frame_shape = [v.get(cv2.CAP_PROP_FRAME_HEIGHT), v.get(cv2.CAP_PROP_FRAME_WIDTH)]
          v.release()
          self._video_frame_shapes[drone_key][video_key][video_index] = video_frame_shape
      return self._video_frame_shapes[drone_key][video_key]
    return []
    
  # Get the video keys associated with a drone.
  # If drone_key is None, will return a dictionary with video keys for each drone.
  def get_video_keys(self, drone_key=None):
    if drone_key is None:
      video_keys = {}
      for drone_key in list(self._drone_datas.keys()):
        video_keys[drone_key] = self.get_video_keys(drone_key=drone_key)
      return video_keys
    return list(self._drone_datas[drone_key].keys())
  
  # Get the drone key that has a desired video key.
  def get_drone_key_for_video_key(self, video_key):
    video_key = str(video_key)
    for drone_key in self._drone_keys:
      if video_key in self._drone_datas[drone_key]:
        return drone_key
    return None
    
  # Get the number of videos associated with a drone.
  def get_num_videos(self, drone_key):
    return len(self.get_video_keys(drone_key))
  
  # Get the number of frames in a video.
  # Can either specify the video_key OR the drone_key and the video_index.
  def get_num_frames(self, video_key=None, drone_key=None, video_index=None):
    timestamps_s = self.get_frame_timestamps_s(video_key=video_key, drone_key=drone_key, video_index=video_index)
    if timestamps_s is None:
      return None
    return len(timestamps_s)
  
  # Get drone data for a given video.
  # Can either specify the video_key OR the drone_key and the video_index.
  def get_drone_data(self, video_key=None, drone_key=None, video_index=None):
    if video_key is not None:
      if isinstance(video_key, (int, float)):
        video_key = str(video_key)
      for (drone_key, drone_datas) in self._drone_datas.items():
        if video_key in drone_datas:
          return drone_datas[video_key]
    else:
      video_key = self.get_video_keys(drone_key)[video_index]
      return self._drone_datas[drone_key][video_key]
    return None
  
  # Get a video reader for a given video.
  # Can either specify the video_key OR the drone_key and the video_index.
  def get_video_reader(self, video_key=None, drone_key=None, video_index=None, force_full_resolution=False):
    if not can_use_videos:
      raise AssertionError('Videos are not available since DecordVideoReaderWrapper could not be imported')
    if video_key is None:
      video_key = self.get_video_keys(drone_key)[video_index]
    if isinstance(video_key, (int, float)):
      video_key = str(video_key)
    for (drone_key, video_readers) in self._video_readers.items():
      if video_key in video_readers:
        # Get the frame shapes of available videos if not discovered already.
        for (video_index, video_frame_shape) in enumerate(self._video_frame_shapes[drone_key][video_key]):
          if video_frame_shape is None:
            # Note that CV2 seems to be faster than decord for just checking the resolution.
            v = cv2.VideoCapture(self._video_filepaths[drone_key][video_key][video_index])
            video_frame_shape = [v.get(cv2.CAP_PROP_FRAME_HEIGHT), v.get(cv2.CAP_PROP_FRAME_WIDTH)]
            v.release()
            self._video_frame_shapes[drone_key][video_key][video_index] = video_frame_shape
        # Create the video reader if it hasn't been created already,
        # or if full resolution is desired and the current reader is not full resolution.
        if force_full_resolution:
          video_reader = self._video_readers_fullResolution[drone_key][video_key]
          if video_reader is None:
            max_width = -1
            for (video_index, video_frame_shape) in enumerate(self._video_frame_shapes[drone_key][video_key]):
              if video_frame_shape[1] > max_width:
                video_filepath = self._video_filepaths[drone_key][video_key][video_index]
                max_width = video_frame_shape[1]
            print('DroneVideos opening videoreader to', video_filepath, 'with full resolution')
            video_reader = DecordVideoReaderWrapper(video_filepath, width=-1, height=-1)
            self._video_readers_fullResolution[drone_key][video_key] = video_reader
          return video_reader
        else:
          video_reader = video_readers[video_key]
          if video_reader is None:
            target_width = -1
            target_height = -1
            if self._target_width is not None and self._target_height is not None:
              target_width = self._target_width
              target_height = self._target_height
            elif self._target_width is not None or self._target_height is not None:
              sample_frame_shape = self._video_frame_shapes[drone_key][video_key][0]
              if self._target_height is None:
                target_width = self._target_width
                target_height = int(target_width * sample_frame_shape[0]/sample_frame_shape[1])
              elif self._target_width is None:
                target_height = self._target_height
                target_width = int(target_height * sample_frame_shape[1]/sample_frame_shape[0])
            if target_width == -1 and target_height == -1 and self._video_readers_fullResolution[drone_key][video_key] is not None:
              video_reader = self._video_readers_fullResolution[drone_key][video_key]
              self._video_readers[drone_key][video_key] = video_reader
            else:
              frame_widths = [shape[1] for shape in self._video_frame_shapes[drone_key][video_key]]
              if target_width == -1:
                # Use the largest video.
                max_width_index = frame_widths.index(max(frame_widths))
                video_filepath = self._video_filepaths[drone_key][video_key][max_width_index]
              else:
                # Choose the smallest video resolution greater than the specified width.
                frame_widths_sorted = frame_widths.copy()
                frame_widths_sorted.sort()
                frame_widths_sufficient = [frame_width for frame_width in frame_widths_sorted if frame_width >= target_width]
                if len(frame_widths_sufficient) > 0:
                  frame_width = frame_widths_sufficient[0]
                else:
                  # If all available videos are too small, use the largest one.
                  frame_width = frame_widths[-1]
                  target_width = -1
                  target_height = -1
                video_index = frame_widths.index(frame_width)
                video_filepath = self._video_filepaths[drone_key][video_key][video_index]
              print('DroneVideos opening videoreader to %s with target resolution (%d, %d)' % (video_filepath, target_width, target_height))
              video_reader = DecordVideoReaderWrapper(video_filepath, width=target_width, height=target_height)
              self._video_readers[drone_key][video_key] = video_reader
              if target_width == -1 and target_height == -1:
                self._video_readers_fullResolution[drone_key][video_key] = video_reader
          return video_reader
    return None

  # Get frame timestamps for a given drone video.
  # Can either specify the video_key OR the drone_key and the video_index.
  def get_frame_timestamps_s(self, video_key=None, drone_key=None, video_index=None, use_custom_epoch=False):
    drone_data = self.get_drone_data(video_key=video_key, drone_key=drone_key, video_index=video_index)
    if drone_data is None:
      return None
    timestamps_s = drone_data['timestamps_s']
    if use_custom_epoch:
      if self._custom_epoch_time_s is not None:
        timestamps_s = timestamps_s - self._custom_epoch_time_s
      else:
        raise AssertionError('No custom epoch time was provided')
    return timestamps_s
  def get_frame_timestamps_str(self, video_key=None, drone_key=None, video_index=None):
    drone_data = self.get_drone_data(video_key=video_key, drone_key=drone_key, video_index=video_index)
    if drone_data is None:
      return None
    timestamps_str = drone_data['timestamps_str']
    timestamps_str = [timestamp_str.decode('utf-8') for timestamp_str in timestamps_str]
    return timestamps_str
  
  # Get the best frame index for a desired time.
  def get_frame_index_for_time_s(self, drone_key, target_time_s, timestamp_to_target_thresholds_s=None):
    for (video_index, video_key) in enumerate(self.get_video_keys(drone_key)):
      timestamps_s = self.get_frame_timestamps_s(video_key=video_key)
      if timestamp_to_target_thresholds_s is None:
        video_fps = (timestamps_s.shape[0]-1)/(timestamps_s[-1] - timestamps_s[0])
        timestamp_to_target_thresholds_s = (1/video_fps*0.6, 1/video_fps*0.6)
      frame_index = get_index_for_time_s(timestamps_s, target_time_s, timestamp_to_target_thresholds_s)
      if frame_index is not None:
        return (video_key, frame_index)
    return (None, None)
  
  # Get a frame from a drone that corresponds to a frame from a different drone.
  # For the source, can either specify the video_key OR the drone_key and the video_index.
  # Must specify the source_frame_index.
  def get_frame_index_from_other_drone(self, source_video_key=None, source_drone_key=None, source_video_index=None,
                                             source_frame_index=None,
                                             target_drone_key=None):
    timestamps_s = self.get_frame_timestamps_s(video_key=source_video_key, drone_key=source_drone_key, video_index=source_video_index)
    target_time_s = timestamps_s[source_frame_index]
    if target_drone_key is None:
      if source_drone_key is None:
        source_drone_key = self.get_drone_key_for_video_key(source_video_key)
      for target_drone_key in self._drone_keys:
        if target_drone_key != source_drone_key:
          break
    return self.get_frame_index_for_time_s(target_drone_key, target_time_s)
  
  # Get a frame image a drone that corresponds to a frame from a different drone.
  # For the source, can either specify the video_key OR the drone_key and the video_index.
  # Must specify the source_frame_index.
  def get_frame_img_from_other_drone(self, source_video_key=None, source_drone_key=None, source_video_index=None,
                                           source_frame_index=None,
                                           target_drone_key=None,
                                           force_full_resolution=False):
    (other_video_key, other_frame_index) = self.get_frame_index_from_other_drone(source_video_key=source_video_key, source_drone_key=source_drone_key, source_video_index=source_video_index, source_frame_index=source_frame_index, target_drone_key=target_drone_key)
    if other_video_key is not None:
      return self.get_frame_img(video_key=other_video_key, frame_index=other_frame_index, force_full_resolution=force_full_resolution)
    return None
    
  # Can either specify the video_key OR the drone_key and the video_index.
  # Must specify the frame_index.
  def get_frame_img(self, video_key=None, drone_key=None, video_index=None, frame_index=None, force_full_resolution=False):
    video_reader = self.get_video_reader(video_key=video_key, drone_key=drone_key, video_index=video_index, force_full_resolution=force_full_resolution)
    if video_reader is not None:
      frame = video_reader[frame_index]
      if frame is not None:
        return frame.asnumpy()
    return None
  
  
###############################################
# Testing
###############################################
if __name__ == '__main__':
  drone_data_hdf5_filepaths = {
    'CETI': 'data/drones/CETI-DJI_MAVIC3-1_metadata.hdf5',
    'DSWP': 'data/drones/DSWP-DJI_MAVIC3-2_metadata.hdf5',
  }
  video_dirs = {
    'CETI': 'F:/CETI-DJI_MAVIC3-1',
    'DSWP': 'F:/DSWP-DJI_MAVIC3-2',
  }
  
  droneVideos = DroneVideos(
      drone_data_hdf5_filepaths=drone_data_hdf5_filepaths,
      video_dirs=video_dirs,
  )
  
  print(droneVideos.get_video_keys('CETI'))
  print(droneVideos.get_video_keys('DSWP'))
  print()
  print(droneVideos.get_drone_data(video_key='1688826012514'))
  print(droneVideos.get_frame_timestamps_s(video_key='1688826012514'))
  print(droneVideos.get_frame_timestamps_str(video_key='1688826012514'))
  print(droneVideos.get_num_frames(video_key='1688826012514'))
  print(droneVideos.get_num_frames(drone_key='CETI', video_index=0))
  print(droneVideos.get_num_videos(drone_key='CETI'))
  print(droneVideos.get_num_videos(drone_key='DSWP'))
  
  timestamps_s = droneVideos.get_frame_timestamps_s(video_key='1688829190202')
  target_time_s = timestamps_s[12]
  print('target_time_s', target_time_s)
  (ceti_video_key, ceti_frame_index) = droneVideos.get_frame_index_for_time_s('CETI', target_time_s=target_time_s)
  (dswp_video_key, dswp_frame_index) = droneVideos.get_frame_index_for_time_s('DSWP', target_time_s=target_time_s)
  print('ceti_frame_index', ceti_video_key, ceti_frame_index)
  print('dswp_frame_index', dswp_video_key, dswp_frame_index)
  if ceti_frame_index is not None and dswp_frame_index is not None:
    print(droneVideos.get_frame_timestamps_s(video_key=ceti_video_key)[ceti_frame_index])
    print(droneVideos.get_frame_timestamps_s(video_key=dswp_video_key)[dswp_frame_index])
  print(droneVideos.get_frame_index_from_other_drone(source_video_key=ceti_video_key,
                                                     source_frame_index=ceti_frame_index))
  
  
  print()
  video_key = '1688829190202'
  frame_index = 12
  (other_video_key, other_frame_index) = droneVideos.get_frame_index_from_other_drone(source_video_key=video_key, source_frame_index=frame_index)
  print(video_key, frame_index)
  print(other_video_key, other_frame_index)
  frame = droneVideos.get_frame_img(video_key=video_key, frame_index=frame_index)
  
  print()
  import cv2
  video_key = '1688829190202'
  for frame_index in range(600):
    print(frame_index)
    frame_img_drone0 = droneVideos.get_frame_img(video_key=video_key, frame_index=frame_index)
    frame_img_drone1 = droneVideos.get_frame_img_from_other_drone(source_video_key=video_key, source_frame_index=frame_index)
    if frame_img_drone0 is None or frame_img_drone1 is None:
      print('  skip!')
      continue
    frame_img_drone0 = cv2.cvtColor(cv2.resize(frame_img_drone0, (500, 280)), cv2.COLOR_RGB2BGR)
    frame_img_drone1 = cv2.cvtColor(cv2.resize(frame_img_drone1, (500, 280)), cv2.COLOR_RGB2BGR)
    cv2.imshow('drone0', frame_img_drone0)
    cv2.imshow('drone1', frame_img_drone1)
    cv2.waitKey(60)

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

import copy

import decord
from threading import Thread
from threading import Lock
import time
import gc
try:
  import ctypes
  libc = ctypes.CDLL("libc.so.6")
  libc.malloc_trim(0) # https://stackoverflow.com/a/64879094
  can_use_libc = True
except:
  can_use_libc = False

###########################################
# Observations:
#
# Creating the video reader does not immediately cause memory to increase.
# As soon as a frame is read, memory will start increasing extremely quickly.
# If frames are continuously read, memory will be kept in check.
# Whenever seek is called, memory will drop back down.
# It does not seem to matter if seek is called with 0 or a random index.
# After frame reading ends, a single seek is enough to keep memory down - it is not needed periodically.
#
# By default, it can read an uncompressed drone video at 350Hz (sequential) or 55Hz (random).
# With periodic resets to a random index every 5 seconds, the rates are the same.
# With periodic resets to a random index every 2 seconds, the rates are 340Hz/52Hz.
# With periodic resets to a random index every 1 seconds, the rates are 340Hz/52Hz.
# With periodic resets to a random index every 0.5 seconds, the rates are 345Hz/55Hz.
# With periodic resets to a random index every 0.1 seconds, the rates are 330Hz/55Hz.
# With seeking after every frame retrieval, the rates are 60Hz/45Hz.
# With an initial seek after 0.5 second, memory usage got to about 0.8GB.
# With an initial seek after 1 second, memory usage got to about 1GB.
# With an initial seek after 2 second, memory usage got to about 3.2GB.
#
# So the paradigm implemented below is to reset by seeking after frame reading ends.
#   If an application is reading frames faster than the reset interval, everything behaves normally.
#   But if there is a lull in reading frames, there will be a seek afterwards to keep memory usage low.
###########################################

class DecordVideoReaderWrapper(decord.VideoReader):
  
  def __init__(self, *args, **kwargs):
    keys_for_wrapper = ['reset_seek_initial_period_s',
                        'reset_seek_period_s',
                        'reset_seek_period_frameReads',]
    kwargs_for_super = copy.deepcopy(kwargs)
    for key in keys_for_wrapper:
      if key in kwargs_for_super:
        del kwargs_for_super[key]
    super().__init__(*args, **kwargs_for_super)
    self._video_filepath = args[0]
    self._video_reader_mutex = Lock()
    # Periodically seek the video reader to 0 to limit memory usage.
    # Will trigger this based on number of frame reads and based on time.
    self._reset_seek_initial_period_s = 0.1
    self._reset_seek_period_s = 0.5
    self._reset_seek_period_frameReads = 150
    if 'reset_seek_initial_period_s' in kwargs:
      self._reset_seek_initial_period_s = kwargs['reset_seek_initial_period_s']
    if 'reset_seek_period_s' in kwargs:
      self._reset_seek_period_s = kwargs['reset_seek_period_s']
    if 'reset_seek_period_frameReads' in kwargs:
      self._reset_seek_period_frameReads = kwargs['reset_seek_period_frameReads']
    self._frame_read_counter = 0
    self._last_reset_poll_time_s = 0
    self._last_frameRead_time_s = 0
    # Start a thread to periodically seek to 0 based on time.
    self._stop_reset_seek_thread = False
    self._reset_seek_thread = Thread(target=self._reset_seek_thread_fn, args=())
    self._reset_seek_thread.start()
  
  def close(self, timeout_s=2):
    self._stop_reset_seek_thread = True
    start_wait_time_s = time.time()
    while self._reset_seek_thread.is_alive() and (time.time() - start_wait_time_s < timeout_s):
      time.sleep(0.05)
    
  def __getitem__(self, key):
    if not self._video_reader_mutex.acquire(timeout=1):
      return None
    # Read the frame(s).
    frames = None
    for i in range(3):
      try:
        frames = super().__getitem__(key)
        break
      except IndexError:
        self._video_reader_mutex.release()
        self.close()
        raise
      except:
        pass
    if frames is None:
      self._video_reader_mutex.release()
      self.close()
      raise ValueError('Error retrieving frames for key %s' % key)
    # Try to limit memory usage.
    # If the frame read threshold is reached, seek back to 0.
    if self._reset_seek_period_frameReads is not None:
      # Increment the read counter by the number of frames read.
      # If a single frame was read, frames will be that single frame matrix.
      # Otherwise, frames will be (N x Height x Width x Channels).
      if len(frames.shape) == 4:
        self._frame_read_counter += frames.shape[0]
      else:
        self._frame_read_counter += 1
      # Seek the video reader if the threshold has been reached.
      if self._frame_read_counter >= self._reset_seek_period_frameReads:
        # print('seeking to 0 based on read count')
        self.seek(0)
        self._frame_read_counter = 0
        gc.collect()
        if can_use_libc: libc.malloc_trim(0)
    # Record this time as a time when the video reader was reset for memory purposes.
    self._last_frameRead_time_s = time.time()
    self._video_reader_mutex.release()
    return frames
  
  def _reset_seek_thread_fn(self):
    time.sleep(self._reset_seek_initial_period_s)
    # print('seeking to 0 based on time [initial]', self._video_filepath)
    if self._video_reader_mutex.acquire(timeout=2):
      self.seek(0)
      self._last_reset_poll_time_s = time.time()
      self._video_reader_mutex.release()
    did_reset_after_frameReading = False
    while not self._stop_reset_seek_thread:
      while (time.time() - self._last_reset_poll_time_s) < self._reset_seek_period_s \
          and not self._stop_reset_seek_thread:
        time.sleep(min([0.1, self._reset_seek_period_s]))
      # If a frame read has occurred recently, consider that as the reset.
      # But mark that frame reading is being performed.
      if time.time() - self._last_frameRead_time_s < self._reset_seek_period_s:
        did_reset_after_frameReading = False
      else:
        # If this is the first time after frame reading paused, do a reset.
        if not did_reset_after_frameReading:
          if not self._video_reader_mutex.acquire(timeout=2):
            continue
          # print('seeking to 0 based on time', self._video_filepath)
          self.seek(0)
          gc.collect()
          if can_use_libc: libc.malloc_trim(0)
          self._video_reader_mutex.release()
          did_reset_after_frameReading = True
      self._last_reset_poll_time_s = time.time()
    
    # Clean up one last time before exiting.
    if self._video_reader_mutex.acquire(timeout=2):
      self.seek(0)
      gc.collect()
      if can_use_libc: libc.malloc_trim(0)
      self._video_reader_mutex.release()
      
###########################################
# TESTING
###########################################
if __name__ == '__main__':
  import os
  import time
  import cv2
  import numpy as np
  
  video_filepath = 'test_video_filepath'
  # video_reader = decord.VideoReader(video_filepath)
  video_reader = DecordVideoReaderWrapper(video_filepath)
  frame_shape = video_reader[0].asnumpy().shape
  cv2.waitKey(3000)
  
  # start_time_s = time.time()
  # frame_index = 0
  # while time.time() - start_time_s < 30:
  #   cv2.waitKey(100)
  #   # print('reading frame index', frame_index)
  #   # frame_shape = video_reader[frame_index].asnumpy().shape
  #   # frame_index = (frame_index+1) % len(video_reader)
  
  print('Starting read speed tests')
  N = 2000
  frame_indexes = list(range(0,N))
  # frame_indexes = np.random.randint(0, len(video_reader), size=(N, 1))
  start_time_s = time.time()
  for frame_index in frame_indexes:
    frame = video_reader[frame_index].asnumpy()
  duration_s = time.time() - start_time_s
  print('Read %3d sequential frames in %6.3fs | %5.1f Hz' % (N, duration_s, (N-1)/duration_s))
  N = 200
  # frame_indexes = list(range(0,N))
  frame_indexes = np.random.randint(0, len(video_reader), size=(N,))
  start_time_s = time.time()
  for frame_index in frame_indexes:
    frame = video_reader[frame_index].asnumpy()
  duration_s = time.time() - start_time_s
  print('Read %3d random     frames in %6.3fs | %5.1f Hz' % (N, duration_s, (N-1)/duration_s))
  
  cv2.waitKey(10000)
  print('closing video reader')
  video_reader.close()


























############
#
# Copyright (c) 2023 Joseph DelPreto / MIT CSAIL and Project CETI
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
# Created 2023 by Joseph DelPreto [https://josephdelpreto.com].
# [add additional updates and authors as desired]
#
############

import decord
import cv2
import csv
import distinctipy
import re
import os
import glob
import time
import datetime
from collections import OrderedDict
import numpy as np

from segmentation_infrastructure.Segmentations import Segmentations
from segmentation_infrastructure.helpers.helpers_various import *

####################################
# CONFIGURATION
####################################

current_script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_script_dir, '..', 'data')

video_num = 1688829151574
# video_num = 1688829190202
raw_video_filepath = os.path.join(data_dir, '%s_compressed.mp4' % (str(video_num)))
segmentations_filepath = os.path.join(data_dir, 'segmentations',
                                      '%s_segmentations.hdf5' % (str(video_num)))
output_filepath = os.path.join(data_dir, 'interactive_segmentation_visualizer_annotator',
                               '%s_manual_annotations.csv' \
                  % (str(video_num)))

# Will store a buffer of frames to speed up rewinding.
rewind_buffer_duration_s = 30
use_rewind_buffer = True

# Will resize images before displaying them so they fit on the monitor.
display_img_screenRatio = 0.7

# Define colors.
border_color_neutral        = (50, 50, 50) # BGR
border_color_freeformInput  = {
  'whale':      (250, 250, 250), # BGR
  'confidence': (250, 250, 250), # BGR
  'notes':      (250, 250, 250), # BGR
  'frame':      (250, 250, 0), # BGR
  'delete':     (0, 0, 250), # BGR
}
border_color_getFrame       = (250, 250, 0) # BGR
banner_frame_text_color          = (250, 250, 0) # BGR
banner_annotation_text_color     = (250, 0, 250) # BGR
banner_instructions_text_color   = (250, 250, 250) # BGR
annotation_color                 = (250, 0, 250) # BGR
annotation_color_badSegmentation = (0, 0, 250) # BGR
annotation_select_color          = (0, 250, 250) # BGR
annotation_text_color = (250, 0, 250) # BGR
annotation_text_bg_color = None
bounding_box_colors = {'full': (255,255,255), 'head': (255,255,0), 'tail':(0,255,255)}

annotation_circle_size_imgRatio = 0.006
annotation_click_distance_threshold_circleFactor = 2

banner_height_ratio = 0.03
banner_frame_font_heightRatio = banner_height_ratio*0.75
banner_frame_fontScale = None # will be computed on the first iteration
banner_annotation_font_heightRatio = banner_height_ratio*0.75
banner_annotation_fontScale = None # will be computed on the first iteration
bounding_box_width_imgRatio = 0.003

window_name = 'Happy Birthday!'

gui_images_dir = os.path.join(current_script_dir, 'gui_images')
trackbar_height_ratio = 0.015
gui_divider_height_ratio = trackbar_height_ratio*0.2
gui_colors = {
  'bg': (200, 200, 200), # BGR
  'divider': (243, 227, 218), # BGR
  'playhead': (0, 0, 0), # BGR
  'trackbar_annotations': (255, 0, 255), # BGR
}
gui_button_bounds_x_max = 199
gui_button_bounds_y_max = 3
gui_button_bounds_xy = {
  # 'display_segmentations': [(3, 34), (0.15, 0.85)],
  'display_segmentations_box': [(16.5, 21), (0.15, 0.85)],
  'display_segmentations_spotlight': [(22, 26.5), (0.15, 0.85)],
  'display_segmentations_colors': [(28, 32), (0.15, 0.85)],
  'display_orientations': [(33.5, 37.5), (0.15, 0.85)],
  'display_whale_indexes': [(38.5, 42), (0.15, 0.85)],
  'pause': [(45, 51.5), (0.15, 0.85)],
  'play_reverse': [(63, 66), (0.15, 0.85)],
  'play_forward': [(68, 71), (0.15, 0.85)],
  'seek_reverse': [(86, 89.5), (0.15, 0.85)],
  'seek_forward': [(91.5, 95), (0.15, 0.85)],
  'seek_annotation_reverse': [(122, 126), (0.15, 0.85)],
  'seek_annotation_forward': [(128, 132), (0.15, 0.85)],
  'frame_skip_down': [(156, 160), (0.15, 0.85)],
  'frame_skip_up': [(162, 166), (0.15, 0.85)],
  'frame_jump': [(175, 196), (0.15, 0.85)],
  
  'delete_annotation': [(3, 34), (1.15, 1.85)],
  'whale_id': [(53, 84), (1.15, 1.85)],
  'whale_confidence': [(103, 140), (1.15, 1.85)],
  'notes': [(160, 196), (1.15, 1.85)],
  
  'bad_segmentation': [(3, 34), (2.15, 2.85)],
  'delete_segmentation_currentFrame': [(82, 105), (2.15, 2.85)],
  'delete_segmentation_allFrames': [(109, 132), (2.15, 2.85)],
  'delete_segmentation_currentInstance': [(136, 160), (2.15, 2.85)],
  'delete_segmentation_enterFrames': [(164, 196), (2.15, 2.85)],
}
gui_controls_y_min = None
gui_controls_y_max = None
gui_frame_y_min = None

####################################
# MENU
####################################
print()
print('*'*75)
print(' '*16, end='')
print('Welcome to the interactive annotation tool!')
print('*'*75)
print('''
=========================================
USAGE SUMMARY
=========================================
1) Seek through the video to find where whales can be identified.
   -- You can increase the playback speed, or play it in reverse.
   -- You can pause the video and then seek frame-by-frame (or jump by a number of frames).
   -- You can jump between frames that have manual annotations.
   -- You can toggle whether automatic segmentations are shown on the video.
2) Enter manual annotations on a chosen frame.
   -- You can click on one or more points in the frame.
   -- For each point, you can enter a whale ID, notes on ID confidence, and/or general notes.
   -- You can edit or delete previously added annotation points.
   -- You can indicate that there is a bad automatic segmentation.
   -- You can delete automatic segmentations.
3) Share the output CSV file.
   -- Results will be saved whenever annotations are updated, so data is saved if something crashes.
   -- On startup it will load annotations from the CSV if it exists so you can continue working.

=========================================
DETAILED COMMANDS TO USE
WHILE THE VIDEO WINDOW IS ACTIVE
=========================================

VIDEO PLAYBACK AND FRAME-BY-FRAME SEEKING
-----------------------------------------
  space    Play or pause
  r        Reverse playback/seeking direction
            If the printed Frame Skip is negative, playback/seek is reversed
  ]        Increase playback/seeking speed
  [        Decrease playback/seeking speed
  .        Seek forward by frame_skip frames (when paused)
  ,        Seek backward by frame_skip frames (when paused)

  a        Jump to the next frame with a manual annotation,
            or the previous one if playback/seeking is reversed

  f        Start typing a frame index to jump to
            The image border will be cyan, but typed numbers are not shown
  enter    Finish typing a frame index to jump to

ANNOTATING AFTER PAUSING THE VIDEO
----------------------------------
  Left-Click    Add an annotation point or select an existing point
                 Selected points will be outlined in yellow
                 The bottom banner will print annotation information
  Right-Click   Delete an annotation point

When an annotation point is selected:
  w             Start typing a whale ID for the point
  c             Start typing confidence notes for the ID
  n             Start typing general notes for the point
  enter         Finish typing the ID, confidence, or note
                 The image border will be white while typing
                 
  b             Toggle whether the segmentation at the point is bad
  d             Delete the containing segmentation; start typing deletion type:
  f               Only delete the segmentation in the current frame
  a               Delete this whale index segmentation in all frames
  i               Delete this whale index in the current segmentation series
  #1, #2          Delete this whale index from #1 frames back to #2 frames ahead
  enter           Confirm deletion option
  esc             Cancel deletion
  
SEGMENTATIONS OVERLAY
---------------------
  o             Toggle orientation indication
  i             Toggle ID display
  s             Start typing a segmentation display option:
  color         Color each whale
  box           Bounding boxes around whales
  spotlight     Spotlight each whale
  

MISCELLANEOUS COMMANDS AND NOTES
--------------------------------
  q        Quit

  The following colors may be used:
    magenta point         A manual annotation
    yellow point border   A manual annotation that is selected/active
    red point             A manual annotation with a bad segmentation

  NOTE: Make sure the output CSV file is not open when the script is run,
        since otherwise the script will be denied permission to it.

''')
print('*'*75)
print('*'*75)
print()
print()

###################################
# SETUP AND HELPERS
###################################

# Open the video.
print('Opening the video %s' % raw_video_filepath)
video_reader = decord.VideoReader(raw_video_filepath)

# Open the segmentations.
print('Opening the segmentations data %s' % segmentations_filepath)
segmentations = Segmentations(h5_filepath=segmentations_filepath, writable=True)
segmentations_num_whales = segmentations.get_num_whales()
segmentations_colors = distinctipy.get_colors(min(15, segmentations_num_whales), exclude_colors=[(0, 0, 1)], rng=6)
segmentations_colors = [np.array(distinctipy.get_rgb256(c), dtype=np.uint8) for c in segmentations_colors]

# Initialize state.
print('Initializing')
current_frame_index = 0
prev_frame_index = 0
frame_skip = 1
user_input = ''
freeform_input_mode = None # 'whale', 'confidence', 'notes', 'frame', 'segmentations'
target_frame_index_str = ''
segmentations_display_option_str = ''
deletion_option_str = ''
selected_annotation_index = None
selected_annotation_data = None
latest_image_mouseEvent = {}
latest_trackbar_mouseEvent = {}
latest_gui_clickedItem = None
paused = False
show_orientation = False
show_whale_index = False

# Determine the frame size used by the segmentations.
mask_shape = segmentations.get_frame_shape()

# Determine display image size based on screen size.
gui_aspect_ratio = 14.01/1.9 # copied from PowerPoint; does not include trackbar, but close enough
frame_aspect_ratio = mask_shape[1]/mask_shape[0]
combined_aspect_ratio = gui_aspect_ratio * frame_aspect_ratio / (gui_aspect_ratio + frame_aspect_ratio)
cv2.namedWindow('monitorSizeTest', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('monitorSizeTest', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.waitKey(100)
display_screen_size = cv2.getWindowImageRect('monitorSizeTest')
cv2.destroyWindow('monitorSizeTest')
display_screen_width = display_screen_size[2]
display_screen_height = display_screen_size[3]
display_screen_aspect_ratio = display_screen_width/display_screen_height
if display_screen_aspect_ratio > combined_aspect_ratio:
  target_height = display_screen_height * display_img_screenRatio
  display_img_width = round(target_height * combined_aspect_ratio)
else:
  target_width = display_screen_width * display_img_screenRatio
  display_img_width = round(target_width)

# Determine the image size and corresponding border/banner/annotation sizes.
sample_display_img = video_reader[0].asnumpy()
sample_display_img = scale_image(sample_display_img, target_width=display_img_width, target_height=None)
border_width = 0 #int(sample_display_img.shape[1]*0.02) # NOTE if this is used again, mouse click logic for the GUI buttons will need to be updated
banner_height = int(sample_display_img.shape[1]*banner_height_ratio)
banner_width = sample_display_img.shape[1] + 2*border_width
banner_divider_height = int(banner_height*0.1)
annotation_circle_size = int(display_img_width*annotation_circle_size_imgRatio)
annotation_click_distance_threshold = annotation_circle_size*annotation_click_distance_threshold_circleFactor
bounding_box_width = int(display_img_width*bounding_box_width_imgRatio)

trackbar_height = int(sample_display_img.shape[1]*trackbar_height_ratio)
gui_divider_height = int(sample_display_img.shape[1]*gui_divider_height_ratio)

# Load GUI interface images and scale them to match the displayed frame width.
gui_images = {
  'playback': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_playback.jpg'))[0:-5,:,:],
  'annotate': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_annotate.jpg'))[0:-5,:,:],
  'bad_segmentation': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_badSegmentation.jpg'))[0:-5,:,:],
  'instruction_clickImage': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_instruction_clickImage.jpg'))[0:-5,:,:],
  'instruction_enterWhale': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_instruction_enterWhale.jpg'))[0:-5,:,:],
  'instruction_enterConfidence': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_instruction_enterConfidence.jpg'))[0:-5,:,:],
  'instruction_enterNotes': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_instruction_enterNotes.jpg'))[0:-5,:,:],
  'instruction_enterFrame': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_instruction_enterFrame.jpg'))[0:-5,:,:],
  'instruction_enterDeleteFrames': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_instruction_enterDeleteFrames.jpg'))[0:-5,:,:],
  'instruction_enterSegmentationsOption': cv2.imread(os.path.join(gui_images_dir, 'gui_interface_instruction_enterSegmentationsOption.jpg'))[0:-5,:,:],
}
for (key, img) in gui_images.items():
  gui_images[key] = scale_image(img, target_width=sample_display_img.shape[1], target_height=None)

# Determine the video frame rate.
frame_rate = video_reader.get_avg_fps()

# Will store a buffer of frames to speed up rewinding.
sample_img = video_reader[0].asnumpy()
if use_rewind_buffer:
  rewind_buffer_length = round(rewind_buffer_duration_s * frame_rate)
  rewind_buffer = np.zeros((rewind_buffer_length, *sample_img.shape), dtype=sample_img.dtype)
  rewind_buffer_frame_indexes = -1*np.ones((rewind_buffer_length,))
  rewind_buffer_next_index = 0

# Determine the ratio from display to raw image sizes.
display_to_mask_ratio_x = mask_shape[1]/sample_display_img.shape[1]
display_to_mask_ratio_y = mask_shape[0]/sample_display_img.shape[0]

# Initialize records of manual assessments.
# Will be a list of dictionaries.
annotations_data = []
def get_annotation_data_newFrame(frame_index, frame_skip):
  return OrderedDict([
    ('frame_index', frame_index),
    ('frame_skip', frame_skip),
    ('x_display', None),
    ('y_display', None),
    ('x_mask', None),
    ('y_mask', None),
    ('whale_id', ''),
    ('whale_id_confidence', ''),
    ('bad_segmentation', 0),
    ('notes', ''),
    ('display_size', '(%d %d)' % (sample_display_img.shape[1], sample_display_img.shape[0])),
    ('frame_size', '(%d %d)' % (sample_img.shape[1], sample_img.shape[0])),
    ('mask_size', '(%d %d)' % (mask_shape[1], mask_shape[0])),
    ('border_width', border_width),
    ('video_filename', os.path.basename(raw_video_filepath)),
    ('segmentations_filename', os.path.basename(segmentations_filepath)),
    ('date_created', datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")),
    ('date_modified', datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")),
  ])
# Determine the headers for the CSV file.
headers_str = ','.join(list(get_annotation_data_newFrame(frame_index=0, frame_skip=0).keys()))

# Check if the output file already exists.
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
if os.path.exists(output_filepath):
  # overwrite_file = input('Output CSV file exists - overwrite it or load it? [o/l] ').lower().strip() == 'o'
  # Load the CSV data.
  print('Loading existing annotation data from %s' % output_filepath)
  fin = open(output_filepath, 'r')
  csv_reader = csv.reader(fin)
  csv_rows = list(csv_reader)
  fin.close()
  # Convert to a list of annotation dictionaries.
  for row_data in csv_rows[1:]:
    frame_index = int(row_data[0])
    annotations_data.append(OrderedDict([
      ('frame_index', int(row_data[0])),
      ('frame_skip', int(row_data[1])),
      ('x_display', int(row_data[2])),
      ('y_display', int(row_data[3])),
      ('x_mask', int(row_data[4])),
      ('y_mask', int(row_data[5])),
      ('whale_id', str(row_data[6])),
      ('whale_id_confidence', str(row_data[7])),
      ('bad_segmentation', int(row_data[8])),
      ('notes', str(row_data[9])),
      ('display_size', str(row_data[10])),
      ('frame_size', str(row_data[11])),
      ('mask_size', str(row_data[12])),
      ('border_width', str(row_data[13])),
      ('video_filename', str(row_data[14])),
      ('segmentations_filename', str(row_data[15])),
      ('date_created', str(row_data[16])),
      ('date_modified', str(row_data[17])),
    ]))

# Define a helper to write information below the image.
def add_info_banner_frame(img, frame_index):
  global banner_frame_fontScale
  # Add the bottom banner shading.
  img = cv2.copyMakeBorder(img, 0, banner_height, 0, 0,
                           cv2.BORDER_CONSTANT, value=(0,0,0))
  
  # Specify the text to write.
  banner_str = '%s | Frame Skip (Speed): %d | Frame Index: %06d | Annotation Count: %d' % (
    os.path.basename(raw_video_filepath),
    frame_skip, current_frame_index, len([x for x in annotations_data if x['frame_index'] == frame_index])
  )
  
  # Draw the text on the image.
  # If this is the first time, compute a font size to use that makes the text fit well on the banner.
  (_, _, banner_frame_fontScale, _) = draw_text_on_image(
                       img, banner_str, pos=(0.5, 0.98),
                       font_scale=banner_frame_fontScale,
                       text_height_ratio=banner_frame_font_heightRatio,
                       font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                       text_color_bgr=banner_frame_text_color,
                       text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None, text_bg_pad_width_ratio=0,
                       preview_only=False,
                       )
  return img

# Define a helper to write information below the image.
def add_info_banner_annotation(img, selected_annotation_data):
  global banner_annotation_fontScale
  # Add a divider on top.
  img = cv2.copyMakeBorder(img, 0, banner_divider_height, 0, 0,
                           cv2.BORDER_CONSTANT, value=(100,100,100))
  # Add the first row bottom banner shading.
  img = cv2.copyMakeBorder(img, 0, banner_height, 0, 0,
                           cv2.BORDER_CONSTANT, value=(0,0,0))
  # Specify the text to write.
  if selected_annotation_data is not None:
    banner_str = 'whale: %s (confidence %s) | bad segmentation? %d | xy: display (%d, %d) mask (%d, %d)' % (
      selected_annotation_data['whale_id'] if len(selected_annotation_data['whale_id']) > 0 else '--',
      selected_annotation_data['whale_id_confidence'] if len(selected_annotation_data['whale_id_confidence']) > 0 else '--',
      selected_annotation_data['bad_segmentation'],
      selected_annotation_data['x_display'], selected_annotation_data['y_display'],
      selected_annotation_data['x_mask'], selected_annotation_data['y_mask'],
    )
  else:
    banner_str = ' '
  # Draw the text on the image.
  # If this is the first time, compute a font size to use that makes the text fit well on the banner.
  (_, _, banner_annotation_fontScale, _) = draw_text_on_image(
                       img, banner_str, pos=(0.5, 0.98),
                       font_scale=banner_annotation_fontScale,
                       text_height_ratio=banner_annotation_font_heightRatio,
                       font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                       text_color_bgr=banner_annotation_text_color,
                       text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None, text_bg_pad_width_ratio=0,
                       preview_only=False,
                       )
  
  # Add the second row bottom banner shading.
  img = cv2.copyMakeBorder(img, 0, banner_height, 0, 0,
                           cv2.BORDER_CONSTANT, value=(0,0,0))
  # Specify the text to write.
  if selected_annotation_data is not None:
    banner_str = 'notes: %s' % (
      selected_annotation_data['notes'] if len(selected_annotation_data['notes']) > 0 else '--',
    )
  else:
    banner_str = ' '
  
  # Draw the text on the image.
  # If this is the first time, compute a font size to use that makes the text fit well on the banner.
  (_, _, banner_annotation_fontScale, _) = draw_text_on_image(
                       img, banner_str, pos=(0.5, 0.98),
                       font_scale=banner_annotation_fontScale,
                       text_height_ratio=banner_annotation_font_heightRatio,
                       font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                       text_color_bgr=banner_annotation_text_color,
                       text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None, text_bg_pad_width_ratio=0,
                       preview_only=False,
                       )
  return img

# Define a helper to write instructions below the image.
def add_info_banner_instructions(img, selected_annotation_data):
  global banner_annotation_fontScale
  # Determine the instructions to write.
  text_rows = ['']*4
  if not paused:
    # Video is playing.
    text_rows[0] =   'PLAYBACK:  space Pause  |  [] Speed  |  r Reverse  |  s Toggle segmentations'
  else:
    if freeform_input_mode is not None:
      if freeform_input_mode == 'delete':
        # Paused, with an annotation selected, freeform typing segmentation deletion options.
        text_rows[0] = 'TYPE DELETION OPTION:  f Only this frame  |  a All frames  |  i Whole segmentation instance'
        text_rows[1] = '                        #1, #2 Back #1 frames forward #2 frames'
        text_rows[2] = '                        enter Submit option  |  esc Cancel deletion'
      # Paused, with an annotation selected, freeform typing but not deleting a segmentation.
      elif freeform_input_mode == 'frame':
        text_rows[0] = 'TYPING TARGET FRAME INDEX:    enter Finish typing'
      elif freeform_input_mode == 'whale':
        text_rows[0] = 'TYPING WHALE NAME:    enter Finish typing'
      elif freeform_input_mode == 'confidence':
        text_rows[0] = 'TYPING CONFIDENCE:    enter Finish typing'
      elif freeform_input_mode == 'notes':
        text_rows[0] = 'TYPING GENERAL NOTES:    enter Finish typing'
      elif freeform_input_mode == 'segmentations':
        text_rows[0] = 'TYPING SEGMENTATION OPTION [box/color/spotlight/outline]:    enter Finish typing'
      else:
        text_rows[0] = 'TYPING:    enter Finish typing'
    elif selected_annotation_data is None:
      # Paused, with no annotation selected.
      text_rows[0] = 'PLAYBACK:  space Play  |  ,. Seek  |  [] Speed  |  a Next Annotated  |  f To frame  |  s Toggle segmentations'
      text_rows[1] = 'ANNOTATE:  left-click Add/Select annotation point  |  right-click Delete annotation point'
    else:
      # Paused, with an annotation selected, not freeform typing.
      text_rows[0] = 'PLAYBACK:  space Play  |  ,. Seek  |  [] Speed  |  a Next Annotated  |  f To frame  |  s Toggle segmentations'
      text_rows[1] = 'ANNOTATE:  left-click Add/Select/Deselect annotation point  |  right-click Delete annotation point'
      text_rows[2] = '            w Whale ID  |  c Confidence  |  n General notes'
      text_rows[3] = 'SEGMENT:   b Bad segmentation  |  d Delete segmentation'
      
      
  # Make all rows the same number of characters.
  max_length = max([len(text_row) for text_row in text_rows])
  text_rows = [text_row.ljust(max_length, ' ') for text_row in text_rows]
  
  # Add a divider on top.
  img = cv2.copyMakeBorder(img, 0, banner_divider_height, 0, 0,
                           cv2.BORDER_CONSTANT, value=(100,100,100))
  
  # Add each row of text.
  for text_row in text_rows:
    # Add the bottom banner shading.
    img = cv2.copyMakeBorder(img, 0, banner_height, 0, 0,
                             cv2.BORDER_CONSTANT, value=(0,0,0))
    # Draw the text on the image.
    # If this is the first time, compute a font size to use that makes the text fit well on the banner.
    (_, _, banner_annotation_fontScale, _) = draw_text_on_image(
                         img, text_row, pos=(int(border_width*0.5), 0.98),
                         font_scale=banner_annotation_fontScale,
                         text_height_ratio=banner_annotation_font_heightRatio,
                         font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                         text_color_bgr=banner_instructions_text_color,
                         text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None, text_bg_pad_width_ratio=0,
                         preview_only=False,
                         )
  return img
  
# Define a helper to make a GUI interface below the image.
def add_gui(img):
  global gui_controls_y_min, gui_controls_y_max, gui_trackbar_y_min, gui_trackbar_y_max
  
  def add_gui_divider(img):
    return cv2.copyMakeBorder(img, 0, gui_divider_height, 0, 0, cv2.BORDER_CONSTANT, value=gui_colors['divider'])
  
  # Add a trackbar.
  img = add_gui_divider(img)
  gui_trackbar_y_min = img.shape[0]
  img = cv2.copyMakeBorder(img, 0, trackbar_height, 0, 0, cv2.BORDER_CONSTANT, value=gui_colors['bg'])
  # Indicate the play position.
  current_position = round(current_frame_index/len(video_reader) * img.shape[1])
  circle_radius = round(trackbar_height*0.4)
  cv2.circle(img, [current_position, round(img.shape[0]-trackbar_height/2)],
                  circle_radius, gui_colors['playhead'], -1)
  # Indicate annotation positions.
  for annotation_data in annotations_data:
    x = round(annotation_data['frame_index']/len(video_reader) * img.shape[1])
    y = img.shape[0]-trackbar_height
    width = round(trackbar_height*0.1)
    height = trackbar_height
    cv2.rectangle(img, (x,y), (x + width, y + height), gui_colors['trackbar_annotations'], -1)
  gui_trackbar_y_max = img.shape[0]
  # Add padding below the trackbar.
  img = add_gui_divider(img)
  img = add_gui_divider(img)
  
  gui_controls_y_min = img.shape[0]
  # Add instructions or additional controls.
  if freeform_input_mode == 'whale':
    img = add_gui_divider(img)
    img = cv2.vconcat([img, gui_images['instruction_enterWhale']])
  elif freeform_input_mode == 'confidence':
    img = add_gui_divider(img)
    img = cv2.vconcat([img, gui_images['instruction_enterConfidence']])
  elif freeform_input_mode == 'notes':
    img = add_gui_divider(img)
    img = cv2.vconcat([img, gui_images['instruction_enterNotes']])
  elif freeform_input_mode == 'frame':
    img = add_gui_divider(img)
    img = cv2.vconcat([img, gui_images['instruction_enterFrame']])
  elif freeform_input_mode == 'delete':
    img = add_gui_divider(img)
    img = cv2.vconcat([img, gui_images['instruction_enterDeleteFrames']])
  elif freeform_input_mode == 'segmentations':
    img = add_gui_divider(img)
    img = cv2.vconcat([img, gui_images['instruction_enterSegmentationsOption']])
  else:
    # Add playback controls.
    img = add_gui_divider(img)
    img = cv2.vconcat([img, gui_images['playback']])
    # Add instructions to add/select an annotation.
    if not paused or selected_annotation_data is None:
      img = add_gui_divider(img)
      img = cv2.vconcat([img, gui_images['instruction_clickImage']])
    else:
      # Add annotation controls.
      img = add_gui_divider(img)
      img = cv2.vconcat([img, gui_images['annotate']])
      # Add bad segmentation controls.
      img = add_gui_divider(img)
      img = cv2.vconcat([img, gui_images['bad_segmentation']])
  gui_controls_y_max = img.shape[0]
  
  # Return the image with a GUI.
  return img

###################################
# Function to handle mouse events.
def process_mouse_event(event, x, y, flags, param):
  global latest_image_mouseEvent, latest_trackbar_mouseEvent, latest_gui_clickedItem
  if event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
    # Check if the user clicked on the displayed frame.
    if y >= 0 and y < gui_frame_y_min:
      latest_image_mouseEvent = {
        'event': event,
        'x': x,
        'y': y,
      }
    # Check if the user clicked on the trackbar.
    if y >= gui_trackbar_y_min and y < gui_trackbar_y_max:
      latest_trackbar_mouseEvent = {
        'event': event,
        'x': x,
        'y': y,
      }
    # Check if the user clicked on a GUI item.
    latest_gui_clickedItem = None
    if y >= gui_controls_y_min and y <= gui_controls_y_max:
      x = x/img.shape[1]*gui_button_bounds_x_max
      y = (y-gui_controls_y_min)/(gui_controls_y_max-gui_controls_y_min)*gui_button_bounds_y_max
      for (gui_item, (gui_x, gui_y)) in gui_button_bounds_xy.items():
        if x >= gui_x[0] and x <= gui_x[1] \
          and y >= gui_y[0] and y <= gui_y[1]:
          latest_gui_clickedItem = gui_item
          break
cv2.namedWindow(window_name)
cv2.moveWindow(window_name, 0, 0)
cv2.setMouseCallback(window_name, process_mouse_event)

# Define a helper to delete segmentations based on the desired window of frames.
def delete_segmentation(deletion_option_str, whale_index_toDelete):
  deletion_option_str = deletion_option_str.lower().strip()
  frame_index_start = None
  frame_index_end = None
  try:
    # See if a range of frame indexes was entered, separated by a space.
    (frames_back, frames_forward) = [int(x.strip()) for x in deletion_option_str.replace(',',' ').replace('(','').replace(')','').split()]
    frame_index_start = current_frame_index - frames_back
    frame_index_end = current_frame_index + frames_forward
  except:
    # Only delete from the current frame.
    if deletion_option_str == 'f':
      frame_index_start = current_frame_index
      frame_index_end = current_frame_index
    # Delete from all frames.
    elif deletion_option_str == 'a':
      frame_index_start = 0
      frame_index_end = len(video_reader)-1
    # Delete from the current instance of that whale index segmentation.
    elif deletion_option_str == 'i':
      print('Finding the segmentation start; this may take some time and memory if many frames are included')
      # Determine the first frame before the current that has this whale.
      frame_index_start = current_frame_index
      while frame_index_start >= 0:
        mask = segmentations.get_mask(frame_index_start, whale_index=whale_index_toDelete)
        mask_has_whale = mask is not None
        if not mask_has_whale:
          frame_index_start += 1
          break
        frame_index_start -= 1
      frame_index_start = max(0, frame_index_start)
      print('Finding the segmentation stop; this may take some time and memory if many frames are included')
      # Determine the last frame after the current that has this whale.
      frame_index_end = current_frame_index
      while frame_index_end < len(video_reader):
        mask = segmentations.get_mask(frame_index_end, whale_index=whale_index_toDelete)
        mask_has_whale = mask is not None
        if not mask_has_whale:
          frame_index_end -= 1
          break
        frame_index_end += 1
      frame_index_end = min(len(video_reader)-1, frame_index_end)
  # Delete the whale index segmentation between the selected frames.
  if frame_index_start is not None and frame_index_end is not None:
    segmentations.remove_segmentation(list(range(frame_index_start, frame_index_end+1)),
                                      whale_index_toDelete)

###################################
# MAIN LOOP
###################################
print('Running!')
while user_input != 'q':
  start_loop_time_s = time.time()
  should_write_img_annotations_data = False
  # Get the image to show, using the buffer or the video reader.
  current_frame_index = min(max(0, current_frame_index), len(video_reader)-1)
  if use_rewind_buffer and np.any(current_frame_index == rewind_buffer_frame_indexes):
    img = np.squeeze(rewind_buffer[np.where(rewind_buffer_frame_indexes == current_frame_index)[0]])
  else:
    img = video_reader[current_frame_index].asnumpy()
    if use_rewind_buffer:
      rewind_buffer[rewind_buffer_next_index, :] = img
      rewind_buffer_frame_indexes[rewind_buffer_next_index] = current_frame_index
      rewind_buffer_next_index = (rewind_buffer_next_index+1) % rewind_buffer_length
  img = scale_image(img, target_width=display_img_width, target_height=None)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  # Deselect points outside of this frame.
  if selected_annotation_data is not None and selected_annotation_data['frame_index'] != current_frame_index:
    selected_annotation_index = None
    selected_annotation_data = None
    
  # Select the first annotation in this frame by default.
  if selected_annotation_data is None and prev_frame_index != current_frame_index:
    for (annotation_index, annotation_data) in enumerate(annotations_data):
      if annotation_data['frame_index'] == current_frame_index:
        selected_annotation_index = annotation_index
        selected_annotation_data = annotation_data
        break
  prev_frame_index = current_frame_index
  
  # Show segmentation overlays if desired.
  if segmentations_display_option_str == 'box':
    for whale_index in range(segmentations_num_whales):
      bounding_box_4xy = segmentations.get_bounding_box_4xy(bounding_box_key='full', frame_index=current_frame_index, whale_index=whale_index)
      if bounding_box_4xy is not None:
        bounding_boxes_4xy_reshaped = bounding_box_4xy.reshape((-1, 2))
        bounding_boxes_4xy_reshaped[:, 0] = bounding_boxes_4xy_reshaped[:, 0]/display_to_mask_ratio_x
        bounding_boxes_4xy_reshaped[:, 1] = bounding_boxes_4xy_reshaped[:, 1]/display_to_mask_ratio_y
        bounding_boxes_4xy_reshaped[:, 0] = bounding_boxes_4xy_reshaped[:, 0] + border_width
        bounding_boxes_4xy_reshaped[:, 1] = bounding_boxes_4xy_reshaped[:, 1] + border_width
        bounding_box_4xy = np.squeeze(bounding_boxes_4xy_reshaped.reshape((-1, 1)))
        bounding_box_points = np.array([bounding_box_4xy[0:2], bounding_box_4xy[2:4],
                                        bounding_box_4xy[4:6], bounding_box_4xy[6:8]], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [bounding_box_points], True, bounding_box_colors['full'], bounding_box_width)
  elif segmentations_display_option_str == 'color':
    color_mask = np.zeros_like(img)
    for whale_index in range(segmentations.get_num_whales()):
      mask = segmentations.get_mask(frame_index=current_frame_index, whale_index=whale_index)
      if mask is not None:
        mask = scale_image(mask, target_width=img.shape[1], target_height=img.shape[0], maintain_aspect_ratio=False)
        color_mask[mask == 1] = segmentations_colors[whale_index % len(segmentations_colors)]
    img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)
  elif segmentations_display_option_str == 'outline':
    color_mask = np.zeros_like(img)
    for whale_index in range(segmentations.get_num_whales()):
      mask = segmentations.get_mask(frame_index=current_frame_index, whale_index=whale_index)
      if mask is not None:
        mask = scale_image(mask, target_width=img.shape[1], target_height=img.shape[0], maintain_aspect_ratio=False)
        mask = np.abs(np.concatenate([np.zeros((1,mask.shape[1])), np.diff(mask, axis=0)])).astype(np.uint8)
        color_mask[mask == 1] = segmentations_colors[whale_index % len(segmentations_colors)]
    img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)
  elif segmentations_display_option_str == 'spotlight':
    mask = np.any(segmentations.get_masks(frame_index=current_frame_index), axis=0).astype(np.uint8)
    mask = scale_image(mask, target_width=img.shape[1], target_height=img.shape[0], maintain_aspect_ratio=False)
    # color_mask = 255*np.ones_like(img)
    # color_mask[mask > 0] = (0,0,0)
    # img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)
    img[mask == 0] = img[mask == 0]*0.5
  if show_orientation:
    for whale_index in range(segmentations_num_whales):
      bounding_box_4xy = segmentations.get_bounding_box_4xy(bounding_box_key='full', frame_index=current_frame_index, whale_index=whale_index)
      if bounding_box_4xy is not None:
        bounding_boxes_4xy_reshaped = bounding_box_4xy.reshape((-1, 2))
        bounding_boxes_4xy_reshaped[:, 0] = bounding_boxes_4xy_reshaped[:, 0]/display_to_mask_ratio_x
        bounding_boxes_4xy_reshaped[:, 1] = bounding_boxes_4xy_reshaped[:, 1]/display_to_mask_ratio_y
        (orientation_rad, orientation_confidence) = segmentations.get_orientation_rad_confidence(frame_index=current_frame_index, whale_index=whale_index)
        centroid_xy = segmentations.get_centroid_xy(frame_index=current_frame_index, whale_index=whale_index)
        centroid_xy[0] /= display_to_mask_ratio_x
        centroid_xy[1] /= display_to_mask_ratio_y
        centroid_xy = centroid_xy.astype(int)
        orientation_length = max(np.linalg.norm(np.diff(bounding_boxes_4xy_reshaped, axis=0), axis=1))/2
        orientation_start_point = [round(coord) for coord in centroid_xy]
        orientation_end_point = np.array(orientation_start_point) \
                                + orientation_length*np.array([np.cos(orientation_rad), -np.sin(orientation_rad)])
        orientation_end_point = [round(coord) for coord in orientation_end_point]
        orientation_color_red = round(255*(1-orientation_confidence))
        orientation_color_green = round(255*orientation_confidence)
        orientation_color = (0, orientation_color_green, orientation_color_red)
        cv2.circle(img, centroid_xy, bounding_box_width*2, orientation_color, -1)
        cv2.line(img, orientation_start_point, orientation_end_point,
                 orientation_color, bounding_box_width)
  if show_whale_index:
    for whale_index in range(segmentations_num_whales):
      centroid_xy = segmentations.get_centroid_xy(frame_index=current_frame_index, whale_index=whale_index)
      if centroid_xy is not None:
        centroid_xy[0] /= display_to_mask_ratio_x
        centroid_xy[1] /= display_to_mask_ratio_y
        centroid_xy = centroid_xy.astype(int)
        bounding_box_4xy = segmentations.get_bounding_box_4xy(bounding_box_key='full', frame_index=current_frame_index, whale_index=whale_index)
        bounding_boxes_4xy_reshaped = bounding_box_4xy.reshape((-1, 2))
        bounding_boxes_4xy_reshaped[:, 0] = bounding_boxes_4xy_reshaped[:, 0]/display_to_mask_ratio_x
        bounding_boxes_4xy_reshaped[:, 1] = bounding_boxes_4xy_reshaped[:, 1]/display_to_mask_ratio_y
        (orientation_rad, orientation_confidence) = segmentations.get_orientation_rad_confidence(frame_index=current_frame_index, whale_index=whale_index)
        box_width = min(np.linalg.norm(np.diff(bounding_boxes_4xy_reshaped, axis=0), axis=1))
        
        (text_w, text_h, font_scale, _) = draw_text_on_image(
          img,
          '%s' % whale_index,
          pos=(0.5, 0.5),
          font_scale=None,
          text_height_ratio=0.02,#(box_width)/img.shape[1],
          font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
          text_color_bgr=(255,255,255),
          text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None,
          text_bg_pad_width_ratio=0.05,
          preview_only=True,
          )
        text_xy = centroid_xy + np.array([-text_w, text_h])
        draw_text_on_image(
          img,
          '%s' % whale_index,
          pos=text_xy,
          font_scale=font_scale,
          text_height_ratio=None,
          font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
          text_color_bgr=(255,255,255),
          text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None,
          text_bg_pad_width_ratio=0.05,
          preview_only=False,
          )
  
  # # Add a colored border to indicate status.
  # if freeform_input_mode is not None:
  #   border_color = border_color_freeformInput[freeform_input_mode]
  # else:
  #   border_color = border_color_neutral
  # img = cv2.copyMakeBorder(
  #                img,
  #                border_width, border_width, border_width, border_width, # top, bottom, left, right
  #                cv2.BORDER_CONSTANT,
  #                value=border_color,
  #             )
  # Add information in a bottom banner.
  gui_frame_y_min = img.shape[0]
  img = add_info_banner_frame(img, current_frame_index)
  img = add_info_banner_annotation(img, selected_annotation_data)
  # img = add_info_banner_instructions(img, selected_annotation_data)
  img = add_gui(img)
  
  # Draw manual annotations on the image.
  for (annotation_index, annotation_data) in enumerate(annotations_data):
    if annotation_data['frame_index'] != current_frame_index:
      continue
    if annotation_data['bad_segmentation']:
      circle_color = annotation_color_badSegmentation
    else:
      circle_color = annotation_color
    cv2.circle(img, [annotation_data['x_display']+border_width, annotation_data['y_display']+border_width], annotation_circle_size, circle_color, -1)
    if selected_annotation_index is not None and selected_annotation_index == annotation_index:
      cv2.circle(img, [annotation_data['x_display']+border_width, annotation_data['y_display']+border_width],
                 annotation_circle_size, annotation_select_color, 2)
  
  # Show the augmented image.
  cv2.imshow(window_name, img)
  
  # Get the user input.
  if paused:
    user_input = cv2.waitKey(1)
  else:
    user_input = cv2.waitKey(1)
  
  # Enter a whale ID
  if freeform_input_mode == 'whale' and user_input >= 0:
    if user_input == 8: # backspace
      selected_annotation_data['whale_id'] = selected_annotation_data['whale_id'][0:-1]
    elif user_input == 13: # enter
      selected_annotation_data['date_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
      freeform_input_mode = None
      should_write_img_annotations_data = True
    else:
      selected_annotation_data['whale_id'] += chr(user_input)
    selected_annotation_data['whale_id'] = selected_annotation_data['whale_id'].replace(',',';')
    user_input = ''
  # Enter a whale ID confidence
  elif freeform_input_mode == 'confidence' and user_input >= 0:
    if user_input == 8: # backspace
      selected_annotation_data['whale_id_confidence'] = selected_annotation_data['whale_id_confidence'][0:-1]
    elif user_input == 13: # enter
      selected_annotation_data['date_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
      freeform_input_mode = None
      should_write_img_annotations_data = True
    else:
      selected_annotation_data['whale_id_confidence'] += chr(user_input)
    selected_annotation_data['whale_id_confidence'] = selected_annotation_data['whale_id_confidence'].replace(',',';')
    user_input = ''
  # Enter a segmentation deletion option.
  elif freeform_input_mode == 'delete' and user_input >= 0:
    if user_input == 8: # backspace
      deletion_option_str = deletion_option_str[0:-1]
    elif user_input == 13: # enter
      whale_index_toDelete = None
      for whale_index in range(segmentations.get_num_whales()):
        mask_current = segmentations.get_mask(frame_index=current_frame_index, whale_index=whale_index)
        if mask_current is not None:
          if mask_current[selected_annotation_data['y_mask'], selected_annotation_data['x_mask']]:
            whale_index_toDelete = whale_index
            break
      if whale_index_toDelete is not None:
        delete_segmentation(deletion_option_str, whale_index_toDelete)
      deletion_option_str = ''
      freeform_input_mode = None
      # Remove the annotation.
      del annotations_data[selected_annotation_index]
      selected_annotation_index = None
      selected_annotation_data = None
      should_write_img_annotations_data = True
    elif user_input == 27: # escape
      freeform_input_mode = None
      deletion_option_str = ''
    else:
      deletion_option_str += chr(user_input)
    user_input = ''
  # Enter a segmentation display option
  elif freeform_input_mode == 'segmentations' and user_input >= 0:
    if user_input == 8: # backspace
      segmentations_display_option_str = segmentations_display_option_str[0:-1]
    elif user_input == 13: # enter
      freeform_input_mode = None
      try:
        segmentations_display_option_str = segmentations_display_option_str.lower().strip()
      except:
        pass
    else:
      segmentations_display_option_str += chr(user_input)
    user_input = ''
  # Enter general notes
  elif freeform_input_mode == 'notes' and user_input >= 0:
    if user_input == 8: # backspace
      selected_annotation_data['notes'] = selected_annotation_data['notes'][0:-1]
    elif user_input == 13: # enter
      selected_annotation_data['date_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
      freeform_input_mode = None
      should_write_img_annotations_data = True
    else:
      selected_annotation_data['notes'] += chr(user_input)
    selected_annotation_data['notes'] = selected_annotation_data['notes'].replace(',',';')
    user_input = ''
  # Enter a target frame index
  elif freeform_input_mode == 'frame' and user_input >= 0:
    if user_input == 8: # backspace
      target_frame_index_str = target_frame_index_str[0:-1]
    elif user_input == 13: # enter
      freeform_input_mode = None
      try:
        target_frame_index = int(target_frame_index_str)
        current_frame_index = max(0, min(target_frame_index, len(video_reader) - 1))
      except:
        pass
      target_frame_index_str = ''
    else:
      target_frame_index_str += chr(user_input)
    user_input = ''
  
  # Quit by pressing q or by closing the window.
  try:
    window_is_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    if window_is_visible == 0:
      user_input = ord('q')
  except:
    user_input = ord('q')
  if user_input == ord('q'):
    user_input = 'q'
  # Play/pause
  elif user_input == ord(' '):
    paused = not paused
  # Reverse
  elif user_input == ord('r'):
    frame_skip = frame_skip*-1
  # Increase/decrease playback speed
  elif user_input == ord(']'):
    frame_skip = int((frame_skip/abs(frame_skip)) * (abs(frame_skip)+1))
  elif user_input == ord('['):
    frame_skip = int((frame_skip/abs(frame_skip)) * (abs(frame_skip)-1))
    frame_skip = max(1, frame_skip)
  # Seek forward/backward
  elif user_input == ord('.'):
    current_frame_index = min(current_frame_index + abs(frame_skip), len(video_reader) - 1)
  elif user_input == ord(','):
    current_frame_index = max(0, current_frame_index - abs(frame_skip))
  elif user_input == ord('a'):
    num_annotations = 0
    new_current_frame_index = current_frame_index
    while num_annotations == 0 and new_current_frame_index < len(video_reader) and new_current_frame_index >= 0:
      new_current_frame_index = new_current_frame_index + int(frame_skip/abs(frame_skip))
      num_annotations = len([x for x in annotations_data if x['frame_index'] == new_current_frame_index])
    if num_annotations > 0:
      current_frame_index = new_current_frame_index
  # Start entering a whale ID.
  elif user_input == ord('w') and selected_annotation_index is not None:
    freeform_input_mode = 'whale'
  # Start entering a whale ID confidence.
  elif user_input == ord('c') and selected_annotation_index is not None:
    freeform_input_mode = 'confidence'
  # Start entering annotation notes.
  elif user_input == ord('n') and selected_annotation_index is not None:
    freeform_input_mode = 'notes'
  # Mark a segmentation as bad.
  elif user_input == ord('b') and selected_annotation_index is not None:
    selected_annotation_data['bad_segmentation'] = int(not selected_annotation_data['bad_segmentation'])
    selected_annotation_data['date_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    should_write_img_annotations_data = True
  # Delete a segmentation.
  elif user_input == ord('d') and selected_annotation_index is not None:
    freeform_input_mode = 'delete'
  # Start entering a target frame index.
  elif user_input == ord('f'):
    freeform_input_mode = 'frame'
  # Start entering a segmentation display option.
  elif user_input == ord('s'):
    freeform_input_mode = 'segmentations'
    segmentations_display_option_str = ''
  # Toggle whether orientation is indicated.
  elif user_input == ord('o'):
    show_orientation = not show_orientation
  # Toggle whether to print the whale index
  elif user_input == ord('i'):
    show_whale_index = not show_whale_index
  
  # Process GUI controls.
  if latest_gui_clickedItem is not None:
    if latest_gui_clickedItem == 'display_segmentations':
      freeform_input_mode = 'segmentations'
      segmentations_display_option_str = ''
    elif latest_gui_clickedItem == 'display_segmentations_box':
      segmentations_display_option_str = 'box' if segmentations_display_option_str != 'box' else ''
    elif latest_gui_clickedItem == 'display_segmentations_colors':
      segmentations_display_option_str = 'color' if segmentations_display_option_str != 'color' else ''
    elif latest_gui_clickedItem == 'display_segmentations_spotlight':
      segmentations_display_option_str = 'spotlight' if segmentations_display_option_str != 'spotlight' else ''
    elif latest_gui_clickedItem == 'display_whale_indexes':
      show_whale_index = not show_whale_index
    elif latest_gui_clickedItem == 'display_orientations':
      show_orientation = not show_orientation
    elif latest_gui_clickedItem == 'pause':
      paused = True
    elif latest_gui_clickedItem == 'play_forward':
      paused = False
      frame_skip = abs(frame_skip)
    elif latest_gui_clickedItem == 'play_reverse':
      paused = False
      frame_skip = abs(frame_skip)*-1
    elif latest_gui_clickedItem == 'seek_forward':
      paused = True
      current_frame_index = min(current_frame_index + abs(frame_skip), len(video_reader) - 1)
    elif latest_gui_clickedItem == 'seek_reverse':
      paused = True
      current_frame_index = max(0, current_frame_index - abs(frame_skip))
    elif latest_gui_clickedItem == 'seek_annotation_forward':
      num_annotations = 0
      new_current_frame_index = current_frame_index
      while num_annotations == 0 and new_current_frame_index < len(video_reader) and new_current_frame_index >= 0:
        new_current_frame_index = new_current_frame_index + 1
        num_annotations = len([x for x in annotations_data if x['frame_index'] == new_current_frame_index])
      if num_annotations > 0:
        current_frame_index = new_current_frame_index
    elif latest_gui_clickedItem == 'seek_annotation_reverse':
      num_annotations = 0
      new_current_frame_index = current_frame_index
      while num_annotations == 0 and new_current_frame_index < len(video_reader) and new_current_frame_index >= 0:
        new_current_frame_index = new_current_frame_index - 1
        num_annotations = len([x for x in annotations_data if x['frame_index'] == new_current_frame_index])
      if num_annotations > 0:
        current_frame_index = new_current_frame_index
    elif latest_gui_clickedItem == 'frame_skip_up':
      frame_skip = int((frame_skip/abs(frame_skip)) * (abs(frame_skip)+1))
    elif latest_gui_clickedItem == 'frame_skip_down':
      frame_skip = int((frame_skip/abs(frame_skip)) * (abs(frame_skip)-1))
      frame_skip = max(1, frame_skip)
    elif latest_gui_clickedItem == 'frame_jump':
      freeform_input_mode = 'frame'
    elif latest_gui_clickedItem == 'delete_annotation' and selected_annotation_index is not None:
      del annotations_data[selected_annotation_index]
      selected_annotation_index = None
      selected_annotation_data = None
      should_write_img_annotations_data = True
    elif latest_gui_clickedItem == 'whale_id' and selected_annotation_index is not None:
      freeform_input_mode = 'whale'
    elif latest_gui_clickedItem == 'whale_confidence' and selected_annotation_index is not None:
      freeform_input_mode = 'confidence'
    elif latest_gui_clickedItem == 'notes' and selected_annotation_index is not None:
      freeform_input_mode = 'notes'
    elif latest_gui_clickedItem == 'bad_segmentation' and selected_annotation_index is not None:
      selected_annotation_data['bad_segmentation'] = int(not selected_annotation_data['bad_segmentation'])
      selected_annotation_data['date_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
      should_write_img_annotations_data = True
    elif 'delete_segmentation' in latest_gui_clickedItem and selected_annotation_index is not None:
      if latest_gui_clickedItem == 'delete_segmentation_enterFrames':
        freeform_input_mode = 'delete'
      else:
        whale_index_toDelete = None
        for whale_index in range(segmentations.get_num_whales()):
          mask_current = segmentations.get_mask(frame_index=current_frame_index, whale_index=whale_index)
          if mask_current is not None:
            if mask_current[selected_annotation_data['y_mask'], selected_annotation_data['x_mask']]:
              whale_index_toDelete = whale_index
              break
        if whale_index_toDelete is not None:
          if latest_gui_clickedItem == 'delete_segmentation_currentFrame':
            delete_segmentation('f', whale_index_toDelete)
          if latest_gui_clickedItem == 'delete_segmentation_allFrames':
            delete_segmentation('a', whale_index_toDelete)
          if latest_gui_clickedItem == 'delete_segmentation_currentInstance':
            delete_segmentation('i', whale_index_toDelete)
        # Remove the annotation.
        del annotations_data[selected_annotation_index]
        selected_annotation_index = None
        selected_annotation_data = None
        should_write_img_annotations_data = True
  latest_gui_clickedItem = None
  
  # Process GUI trackbar seeking.
  if len(latest_trackbar_mouseEvent) > 0:
    target_frame_index = int(latest_trackbar_mouseEvent['x']/img.shape[1]*len(video_reader))
    current_frame_index = max(0, min(target_frame_index, len(video_reader) - 1))
  latest_trackbar_mouseEvent = {}
  
  # Process mouse events on the image.
  if len(latest_image_mouseEvent) > 0 and paused:
    if latest_image_mouseEvent['event'] in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
      # Find the existing annotation closest to the click.
      x = latest_image_mouseEvent['x'] - border_width
      y = latest_image_mouseEvent['y'] - border_width
      mouse_point = np.array([x,y])
      closest_annotation_index = None
      closest_annotation_distance = None
      prev_selected_annotation_index = selected_annotation_index
      selected_annotation_index = None
      selected_annotation_data = None
      for (annotation_index, annotation_data) in enumerate(annotations_data):
        annotation_point = np.array([annotation_data['x_display'], annotation_data['y_display']])
        distance = np.linalg.norm(mouse_point - annotation_point)
        if closest_annotation_distance is None or distance < closest_annotation_distance:
          closest_annotation_distance = distance
          closest_annotation_index = annotation_index
      if closest_annotation_distance is not None and closest_annotation_distance < annotation_click_distance_threshold:
        selected_annotation_index = closest_annotation_index
        selected_annotation_data = annotations_data[selected_annotation_index]
      # Process the click.
      if latest_image_mouseEvent['event'] == cv2.EVENT_LBUTTONUP:
        if selected_annotation_index is None:
          # Create a new annotation at the desired point.
          annotations_data.append(get_annotation_data_newFrame(current_frame_index, frame_skip))
          annotations_data[-1]['x_display'] = x
          annotations_data[-1]['y_display'] = y
          annotations_data[-1]['x_mask'] = round(x*display_to_mask_ratio_x)
          annotations_data[-1]['y_mask'] = round(y*display_to_mask_ratio_y)
          selected_annotation_index = len(annotations_data)-1
          selected_annotation_data = annotations_data[-1]
          should_write_img_annotations_data = True
        elif selected_annotation_index == prev_selected_annotation_index:
          # Deselect the point.
          selected_annotation_index = None
          selected_annotation_data = None
      if latest_image_mouseEvent['event'] == cv2.EVENT_RBUTTONUP:
        if selected_annotation_index is not None:
          # Delete the annotation.
          del annotations_data[selected_annotation_index]
          selected_annotation_index = None
          selected_annotation_data = None
          should_write_img_annotations_data = True
  latest_image_mouseEvent = {}
  
  # Write annotations to the output file.
  if should_write_img_annotations_data:
    fout = None
    try:
      fout = open(output_filepath, 'w')
    except PermissionError:
      # Show a red image with error text.
      print()
      print('*** Error opening the CSV file. Make sure it is not open in another program.')
      print()
      img[:,:,:] = (0,0,250) # BGR
      draw_text_on_image(
          img,
          'Error opening output file [%s]' % os.path.basename(output_filepath),
          pos=(0.5, 0.4),
          font_scale=banner_frame_fontScale,
          text_height_ratio=banner_frame_font_heightRatio,
          font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
          text_color_bgr=(255,255,255),
          text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None,
          text_bg_pad_width_ratio=0.05,
          preview_only=False,
          )
      draw_text_on_image(
          img,
          'To continue: close the file in all other programs, or press q to abort and quit.',
          pos=(0.5, 0.6),
          font_scale=banner_frame_fontScale,
          text_height_ratio=banner_frame_font_heightRatio,
          font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
          text_color_bgr=(255,255,255),
          text_bg_color_bgr=(0,0,0), text_bg_outline_color_bgr=None,
          text_bg_pad_width_ratio=0.05,
          preview_only=False,
          )
      cv2.imshow(window_name, img)
      # Wait for the file to be writable or for the user to quit.
      while fout is None and user_input != 'q':
        user_input = cv2.waitKey(100)
        if user_input == ord('q'):
          user_input = 'q'
        else:
          try:
            fout = open(output_filepath, 'w')
          except:
            pass
    # Write the annotations to the output file.
    if fout is not None:
      fout.write(headers_str)
      for (annotation_index, annotation_data) in enumerate(annotations_data):
        fout.write('\n')
        fout.write(','.join([str(value) for value in annotation_data.values()]))
      fout.close()
  
  # Advance the next frame index.
  if not paused:
    current_frame_index += frame_skip
  # Delay to implement real-time video playback.
  frame_rate_sleep_s = (1/frame_rate) - (time.time() - start_loop_time_s)
  if frame_rate_sleep_s > 0.001:
    time.sleep(frame_rate_sleep_s)
  
###################################
# CLEANUP
###################################

# All done!
segmentations.close()
print()
print('Done!')
print()
print()



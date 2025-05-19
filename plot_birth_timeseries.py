
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

import matplotlib
default_matplotlib_backend = 'qt5agg' #matplotlib.rcParams['backend']
# Avoid type 3 fonts, which can cause issues in some systems (and some paper submission systems).
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import distinctipy
import dateutil.parser
import numpy as np
import os

############################################
# Misc helpers
############################################

# Get the next multiple of a number above a specified target.
# For example, the next multiple of 5 above 23 would be 25.
def next_multiple(value, multiple_of):
  if int(value/multiple_of) == value/multiple_of:
    return value
  return (np.floor(value/multiple_of) + 1)*multiple_of

# Get the previous multiple of a number below a specified target.
# For example, the previous multiple of 5 below 23 would be 20.
def previous_multiple(value, multiple_of):
  if int(value/multiple_of) == value/multiple_of:
    return value
  return np.floor(value/multiple_of)*multiple_of

# Convert from a human-readable time string to time as seconds since epoch.
# The time string should include a timezone offset if applicable, for example '0400' for EDT.
def time_str_to_time_s(time_str):
  time_datetime = dateutil.parser.parse(time_str)
  return time_datetime.timestamp()

###################################################
# Configuration
###################################################

# Define a baby epoch for fetching timestamps in that reference frame.
# Can also be None if it will not be used.
baby_epoch_s = time_str_to_time_s('2023-07-08 11:45:45 -0400')

# Define functions for extracting analysis regions.
analysis_region_startEnd_times_babyEpoch_s = {
  'Before Birth': [-1440*60, -33*60],
  'During Birth': [-33*60, 0],
  'After Birth': [0, 60*(2*60+35)],
  'After After Birth': [60*(2*60+35), 1440*60],
}
in_analysis_regions_functions = {
  'Before Birth': lambda x: (x < -33*60),
  'During Birth': lambda x: (-33*60 <= x) & (x < 0),
  'After Birth': lambda x: (0 <= x) & (x < 60*(2*60+35)),
  'After After Birth': lambda x: (x >= 60*(2*60+35)),
}

# Define plot colors for the analysis regions.
# analysis_region_colors = {
#   'Before Birth'     : 'tab:blue',
#   'During Birth'     : 'tab:orange',
#   'After Birth'      : 'tab:green',
#   'After After Birth': 'tab:red',
# }
analysis_region_colors = {
  'Before Birth'     : 'tab:blue',
  'During Birth'     : 'tab:orange',
  'After Birth'      : list(np.array([44, 160, 137])/255), # 'tab:green'
  'After After Birth': list(np.array([214, 71, 40])/255), # 'tab:red',
}

# Define default axis regions for the broken plot.
# Times are in baby epoch format.
# See the function recompute_birth_timeseries_breakpoints() to recompute if desired.
plot_regions_xlims_s = [
  [-3780.0, -3120.0],
  [-2280.0, -1680.0],
  [-480.0, 900.0],
  [1320.0, 2160.0],
  [10380.0, 10740.0]
]

###################################################
# The main plotting function
###################################################

# Parts of the below are adapted from https://stackoverflow.com/a/32186074
def plot_birth_timeseries(
    # The times and the values to plot.
    time_s, y,
    # The y label text.
    ylabel=None,
    # The plot title.
    title=None,
    # Whether to divide and color data according to analysis regions (before/during/after/after-after birth).
    use_analysis_regions=True,
    # What this series should be labeled in the legend.
    # If None and analysis regions are being used, the legend will reflect the region names.
    legend_label=None,
    # The output filepath to use for saving this plot.
    # If None, the plot will be shown instead of saved.
    output_filepath=None,
    # Recommend to hide the figure window if saving the plot.
    hide_figure_window=False,
    # The color of this trace.
    # If None and using analysis regions, will use the analysis region colors.
    # If None and not using analysis regions, will be the default matplotlib color sequence.
    color=None,
    # The total number of subplot rows, and the current row index to use.
    num_subplot_rows=1,
    subplot_row_index=0,
    # The size of the figure window.
    figure_size_inches=[13, 7],
    # Whether to show the grid.
    show_grid=True,
    # Font sizes.
    legend_fontsize=14, label_fontsize=16,
    title_fontsize=16, tick_fontsize=14,
    # Will split the series into multiple series if there are gaps greater than this threshold.
    gap_threshold_s = 30,
    # Whether to show spines and gray shading between broken subplots.
    show_gap_spines=False,
    show_gap_shading=True,
    show_axis_break_slants=True,
    # Custom y axis tick labels.
    y_tick_custom_label_dict=None,
    # The maximum number of columns in the legend.
    legend_ncols=6,
    # The previous output of this function, to add this series to that graph.
    # If None, will create a new figure.
    previous_handles=None,
    # Any additional arguments to pass to plt.plot()
    **plot_kwargs):
  
  if hide_figure_window:
    matplotlib.use('Agg')
  else:
    matplotlib.use(default_matplotlib_backend)
  
  # Set parameters that used to be arguments but that should now be hard-coded.
  legend_outside = True
  axis_index_for_legend=0
  
  # Create subplots if needed.
  if previous_handles is None:
    # Determine subplots that actually have data.
    ax_regions_xlims_s = []
    for xlims_s in plot_regions_xlims_s:
      if np.any((time_s >= xlims_s[0]) & (time_s <= xlims_s[-1])):
        time_s_in_region = time_s[(time_s >= xlims_s[0]) & (time_s <= xlims_s[-1])]
        min_time_s_in_region = np.min(time_s_in_region)
        max_time_s_in_region = np.max(time_s_in_region)
        ax_regions_xlims_s.append([previous_multiple(min_time_s_in_region, 60),
                                   next_multiple(max_time_s_in_region, 60)])
    # Adjust subplot widths to match the amount of time they represent.
    subplot_durations_s = np.squeeze(np.diff(ax_regions_xlims_s, axis=1))
    subplot_width_ratios = subplot_durations_s/np.min(subplot_durations_s)
    subplot_width_ratios = subplot_width_ratios.tolist()
    gridspec_kw={'width_ratios': subplot_width_ratios}
    # Create the subplots.
    (fig, axs_all) = plt.subplots(num_subplot_rows, len(ax_regions_xlims_s), sharey=True,
                                  squeeze=False, # if False, always return 2D array of axes
                                  facecolor='w', gridspec_kw=gridspec_kw)
    axs = axs_all[subplot_row_index]
    if not isinstance(axs, (list, np.ndarray)):
      axs = [axs]
    # Set the figure size and resolution.
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.waitforbuttonpress(timeout=1)
    fig.set_size_inches(*figure_size_inches)
    fig.set_dpi(300)
    legend_lines = []
    legend_labels = []
    xlabel_text = None
    ylabel_text = None
  else:
    (fig, axs_all, ax_regions_xlims_s, subplot_width_ratios, legend_lines, legend_labels, xlabel_text, ylabel_text) = previous_handles
    axs = axs_all[subplot_row_index]
    previous_legend_labels = legend_labels.copy()
  is_last_subplot_row = subplot_row_index == num_subplot_rows-1
  
  # Convert time to minutes.
  time_m = time_s/60
  ax_regions_xlims_m = [[x[0]/60, x[1]/60] for x in ax_regions_xlims_s]
  
  # Split the data into series if there are gaps.
  dt_s = np.diff(time_m)*60
  if np.any(dt_s > gap_threshold_s):
    split_time_m = []
    split_time_s = []
    split_y = []
    gap_indexes = np.where(dt_s > gap_threshold_s)[0]
    gap_indexes = np.append(gap_indexes, time_s.size-1-1)
    previous_start_index = 0
    for gap_index in gap_indexes:
      split_indexes = [previous_start_index, gap_index]
      split_time_s.append(time_s[split_indexes[0]:split_indexes[-1]+1])
      split_time_m.append(time_m[split_indexes[0]:split_indexes[-1]+1])
      split_y.append(y[split_indexes[0]:split_indexes[-1]+1])
      previous_start_index = gap_index+1
  else:
    split_time_s = [time_s]
    split_time_m = [time_m]
    split_y = [y]
  
  # Plot all data on each subplot.
  axs_has_data = [False]*len(axs)
  for (axis_index, ax) in enumerate(axs):
    xlims_m = ax_regions_xlims_m[axis_index]
    split_color = None
    analysis_regions_with_legend_label = []
    # Iterate over each split of the data which was based on time gaps.
    for split_index in range(len(split_time_m)):
      # Split the data into analysis regions if desired.
      if use_analysis_regions:
        for (analysis_region, in_analysis_regions_function) in in_analysis_regions_functions.items():
          in_analysis_region = in_analysis_regions_function(split_time_s[split_index])
          if np.any(in_analysis_region):
            time_m_toPlot = split_time_m[split_index][in_analysis_region]
            y_toPlot = split_y[split_index][in_analysis_region]
            label = analysis_region if analysis_region not in analysis_regions_with_legend_label else None
            h, = ax.plot(time_m_toPlot, y_toPlot, label=label,
                         color=analysis_region_colors[analysis_region], **plot_kwargs)
            if color is not None:
              h.set_color(color)
            if axis_index == axis_index_for_legend:
              if label is not None and label not in legend_labels:
                legend_lines.append(h)
                legend_labels.append(label)
              if not legend_outside:
                ax.legend()
            analysis_regions_with_legend_label.append(analysis_region)
            if time_m_toPlot.size > 0:
              axs_has_data[axis_index] = True
      # Otherwise, plot all given data as a single series.
      else:
        show_legend = legend_label is not None
        time_m_toPlot = split_time_m[split_index]
        y_toPlot = split_y[split_index]
        h, = ax.plot(time_m_toPlot, y_toPlot, label=legend_label, **plot_kwargs)
        if split_color is not None:
          h.set_color(split_color)
        else:
          split_color = h.get_color()
        if color is not None:
          h.set_color(color)
        if show_legend and axis_index == axis_index_for_legend and legend_label not in legend_labels:
          legend_lines.append(h)
          legend_labels.append(legend_label)
          if not legend_outside:
            ax.legend()
        if time_m_toPlot.size > 0:
          axs_has_data[axis_index] = True
  
  # Format the plot, if not already done.
  if num_subplot_rows == 1 or is_last_subplot_row:
    axis_shift_y = 0.08*num_subplot_rows if legend_outside else 0
    axis_scale_y_each_legend_row = 0.97
    num_legend_rows = ((len(legend_labels)-1)//legend_ncols+1)
    # axis_shifts_y = np.array([axis_shift_y_base]*num_legend_rows)*(axis_scale_each_legend_row**np.arange(0, num_legend_rows))
    axis_scale_y = axis_scale_y_each_legend_row ** num_legend_rows
    if previous_handles is None or (len(legend_labels) > legend_ncols and len(previous_legend_labels) <= legend_ncols) or (num_subplot_rows > 1 and is_last_subplot_row):
      # Move the legend outside, if one has not already been added.
      def shrink_axis(ax, shift, scale):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height*shift,
                         box.width, box.height*scale])
      # def add_legend_below(ax, legend_y):
      #   ax.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y),
      #             fancybox=True, shadow=True, ncol=10, fontsize=legend_fontsize)
      if legend_outside:
        for s in range(len(axs_all)):
          for (axis_index, ax) in enumerate(axs_all[s]):
            shrink_axis(ax, shift=axis_shift_y, scale=axis_scale_y)
    if previous_handles is None or legend_labels != previous_legend_labels:
      if legend_outside:
        legend = fig.legend(legend_lines, legend_labels,
                            loc='lower center', bbox_to_anchor=(0.5, 0),
                            frameon=True, ncol=legend_ncols,
                            fancybox=True, shadow=True, fontsize=legend_fontsize)
  
  # # Add axis labels.
  # if ylabel is not None:
  #   axs[0].set_ylabel(ylabel, fontsize=label_fontsize)
  # Format the x axes and the subplot gaps.
  for (axis_index, ax) in enumerate(axs):
    # Set the x limits.
    ax.set_xlim(ax_regions_xlims_m[axis_index])
    # Force integer ticks.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,
                                           nbins=int(1.5*subplot_width_ratios[axis_index]),
                                           min_n_ticks=1))
    # Hide the spines between axes.
    # But keep the first left spine and the last right spine.
    if not show_gap_spines:
      if axis_index == 0:
        ax.spines['right'].set_visible(False)
      elif axis_index == len(axs)-1:
        ax.spines['left'].set_visible(False)
      else:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # Hide the ticks on the y axes, except for the first plot.
    if axis_index > 0:
      for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
    # Turn on the grid.
    if show_grid:
      ax.grid(True, color='lightgray')
    
    # Adjust tick font sizes.
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
  
    # Add shading between axis breaks.
    # In axes coordinates, which are always between 0-1,
    # spine endpoints are at these locations: (0, 0), (0, 1), (1, 0), and (1, 1).
    if show_gap_shading and axis_index < len(axs)-1:
      dx = 0.15/subplot_width_ratios[axis_index]
      width = 0.2/subplot_width_ratios[axis_index]
      ax.add_patch(Rectangle((1+dx, 0), width, 1,
                              facecolor=[0.7, 0.7, 0.7], fill=True,
                              transform=ax.transAxes, clip_on=False))
    
    # Add the cut-out diagonal lines.
    # In axes coordinates, which are always between 0-1,
    # spine endpoints are at these locations: (0, 0), (0, 1), (1, 0), and (1, 1).
    if show_axis_break_slants:
      dy = 0.01  # how big to make the diagonal lines in axes coordinates
      slant_angle_deg = 45
      aspect_ratio = np.diff(ax.get_ylim())[0] / np.diff(ax.get_xlim())[0]
      dx = dy / np.tan(np.deg2rad(slant_angle_deg)) * aspect_ratio
      diag_plot_kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
      if axis_index < len(axs)-1:
        ax.plot((1-dx, 1+dy), (-dy, +dy), **diag_plot_kwargs)
        ax.plot((1-dx, 1+dy), (1-dy, 1+dy), **diag_plot_kwargs)
      if axis_index > 0:
        ax.plot((-dx, +dy), (1-dy, 1+dy), **diag_plot_kwargs)
        ax.plot((-dx, +dy), (-dy, +dy), **diag_plot_kwargs)
    
    # Adjust spacing between subplots.
    # f.subplots_adjust(hspace=...) or plt.subplot_tool()
    
    # Hide x axis tick labels if not the last subplot.
    if num_subplot_rows > 1 and not is_last_subplot_row:
      ax.set_xticklabels([])
      
    # Set custom y tick labels if desired.
    if y_tick_custom_label_dict is not None:
      y_lim = ax.get_ylim()
      yticks = ax.get_yticks().tolist()
      custom_labels = [f"{int(t)}" for t in yticks]  # Default labels
      for (y_tick_value, y_tick_label) in y_tick_custom_label_dict.items():
        if y_tick_value in yticks:
          modified_label_index = yticks.index(y_tick_value)
          custom_labels[modified_label_index] = y_tick_label  # Modify the desired label
      ax.set_yticks(yticks)
      ax.set_yticklabels(custom_labels)
      ax.set_ylim(y_lim)
      # ax.yaxis.labelpad = 20 # Increase spacing between y-axis label and tick labels
      # ax.yaxis.position = (0, 0.2) # Increase spacing between y-axis label and tick labels
      
    if title is not None:
      plt.suptitle(title, fontsize=title_fontsize)
  
    
  if num_subplot_rows == 1 or is_last_subplot_row:
    # Get the bottom of the tick labels.
    tick_labels_bbox = ax.get_xticklabels()[0].get_window_extent(renderer=fig.canvas.get_renderer())
    tick_labels_coord = tick_labels_bbox.transformed(fig.transFigure.inverted())
    tick_labels_bottom_y = tick_labels_coord.y0
    # Add an x label between the legend and the x tick labels.
    if xlabel_text is not None:
      xlabel_text.set_visible(False)
    xlabel_text = fig.text(0.5, tick_labels_bottom_y-0.01,
                           'Time Since Birth Completed [minutes]',
                           ha='center', va='top', fontsize=label_fontsize)
    # Add a y label centered on the side.
    if ylabel is not None:
      if ylabel_text is not None:
        ylabel_text.set_visible(False)
      tick_labels_bbox = axs[0].get_yticklabels()[0].get_window_extent(renderer=fig.canvas.get_renderer())
      tick_labels_coord = tick_labels_bbox.transformed(fig.transFigure.inverted())
      tick_labels_left_x = tick_labels_coord.x0
      ylabel_text = fig.text(tick_labels_left_x-0.03*(y_tick_custom_label_dict is not None), 0.5,
                             ylabel, rotation='vertical',
                             ha='right', va='center', fontsize=label_fontsize)
    
  # Save the plot.
  if output_filepath is not None:
    output_dir = os.path.realpath(os.path.split(output_filepath)[0])
    if len(output_dir) > 0:
      os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_filepath, dpi=300)
  
  # Return the handles for future updates.
  return (fig, axs_all, ax_regions_xlims_s, subplot_width_ratios, legend_lines, legend_labels, xlabel_text, ylabel_text)

###################################################
# Compute the breakpoints for the plot regions.
###################################################

def recompute_birth_timeseries_breakpoints(gap_threshold_s=5*60,
                                           xlim_buffer_s=60,
                                           plot_limits_minute_multiple=1,
                                           drone_data_hdf5_filepaths=None,
                                           segmentations_data_dir=None):
  global plot_regions_xlims_s
  from csail_data_processing.Segmentations import Segmentations
  from csail_data_processing.DroneVideos import DroneVideos
  import os
  import glob
  
  # Set default data paths if none were provided.
  current_script_dir = os.path.dirname(os.path.realpath(__file__))
  if drone_data_hdf5_filepaths is None:
    drone_data_dir = os.path.join(current_script_dir, '..', 'data', 'drones')
    drone_data_hdf5_filepaths = {
      'CETI': os.path.join(drone_data_dir, 'CETI-DJI_MAVIC3-1_metadata.hdf5'),
      'DSWP': os.path.join(drone_data_dir, 'DSWP-DJI_MAVIC3-2_metadata.hdf5'),
    }
  if segmentations_data_dir is None:
    segmentations_data_dir = os.path.join(current_script_dir, '..', 'data', 'segmentations')
  
  # Open an instance for getting video metadata including timestamps.
  droneVideos = DroneVideos(
    drone_data_hdf5_filepaths=drone_data_hdf5_filepaths,
    video_dirs=None,
    custom_epoch_time_s=baby_epoch_s
  )
  # Get the time vectors for each existing segmentations.
  time_s_byVideo = {}
  for segmentation_filepath in glob.glob(os.path.join(segmentations_data_dir, '*_segmentations.hdf5')):
    # Get the timestamp for all frames.
    video_key = os.path.basename(segmentation_filepath).split('_segmentations')[0]
    times_s = droneVideos.get_frame_timestamps_s(video_key=video_key, use_custom_epoch=True)
    # Ignore frames that did have segmentation applied.
    segmentations = Segmentations(h5_filepath=segmentation_filepath)
    indexes_with_segmentation = np.squeeze(segmentations.get_frames_are_segmented())
    segmentations.close()
    times_s = times_s[indexes_with_segmentation.astype(bool)]
    time_s_byVideo[video_key] = times_s
  
  # Find gaps in the timestamps to use as axis breakpoints.
  plot_regions_xlims_s = []
  for (video_index, video_time_s) in time_s_byVideo.items():
    start_time_s = video_time_s[0]
    end_time_s = video_time_s[-1]
    # If no plots have been created yet, create one matching this video.
    if len(plot_regions_xlims_s) == 0:
      plot_regions_xlims_s.append([start_time_s-xlim_buffer_s, end_time_s+xlim_buffer_s])
      continue
    # Extend the previous plot or create a new one for this video.
    prev_xlims = plot_regions_xlims_s[-1]
    if start_time_s - prev_xlims[1] < gap_threshold_s:
      plot_regions_xlims_s[-1][1] = end_time_s+xlim_buffer_s
    else:
      plot_regions_xlims_s.append([start_time_s-xlim_buffer_s, end_time_s+xlim_buffer_s])
  # Round limits to a nice round number of minutes.
  for xlims_index in range(len(plot_regions_xlims_s)):
    plot_regions_xlims_s[xlims_index][0] = previous_multiple(plot_regions_xlims_s[xlims_index][0]/60, plot_limits_minute_multiple)*60
    plot_regions_xlims_s[xlims_index][1] = next_multiple(plot_regions_xlims_s[xlims_index][1]/60, plot_limits_minute_multiple)*60
  # Print the results.
  print()
  print('Computed new axis breakpoints:')
  for (xlims_index, xlims_s) in enumerate(plot_regions_xlims_s):
    print('  Region %d: [%g, %g] seconds' % (xlims_index, *xlims_s))
  print()



############################################
# Testing
############################################
if __name__ == '__main__':
  # If desired, recompute the breakpoints between axes.
  recompute_birth_timeseries_breakpoints()
  
  #----------------------------------
  # Generate random data for testing.
  #----------------------------------
  only_use_time_since_birth = False
  segmentations_time_bounds_s = [
    [-3710.4494099617004, -3483.2894101142883],
    [-3483.222409963608, -3295.501410007477],
    [-3257.1294100284576, -3198.4384100437164],
    [-2179.0574100017548, -1977.2574100494385],
    # [-1987.3287811279297, -1947.2577810287476],
    [-1953.9994101524353, -1798.0444099903107],
    # [-1945.8867809772491, -1789.0317809581757],
    [-409.6196699142456, -182.39367008209229],
    # [-410.21341013908386, -182.98741006851196],
    [-182.36067008972168, 44.933330059051514],
    # [-182.9534101486206, 44.30658984184265],
    [45.00032997131348, 66.35433006286621],
    [85.87133002281189, 223.31032991409302],
    [232.0663299560547, 458.90733003616333],
    [459.7743299007416, 687.2813301086426],
    [687.549329996109, 808.3073298931122],
    [1416.349590063095, 1623.222589969635],
    [1623.5905900001526, 1850.483589887619],
    [1850.8845899105072, 2077.776589870453],
    [10473.916701078415, 10637.482701063156],
    ]
  time_s = []
  y = []
  for test_bounds in segmentations_time_bounds_s:
    if not only_use_time_since_birth or test_bounds[0] > 0:
      time_s_forVideo = np.linspace(test_bounds[0], test_bounds[-1], 100)
      time_s.extend(time_s_forVideo)
      y.extend(10*np.sin(time_s_forVideo/100) + 5*np.sin(time_s_forVideo/10) + 4*np.random.random(size=(time_s_forVideo.size, )))
      # y.extend(np.linspace(0, len(time_s_forVideo), len(time_s_forVideo)))
      # y.extend(np.random.random(size=(time_s_forVideo.size, 1)))
  time_s = np.array(time_s)
  y = np.array(y)
  
  #----------------------------------
  # An example of plotting based on analysis regions.
  #----------------------------------
  plot_kwargs = dict()
  previous_handles = None
  output_filepath_1 = 'example_plot_using_analysis_regions.pdf' # None to not save
  num_series_to_plot = 2
  for series_index in range(num_series_to_plot):
    previous_handles = plot_birth_timeseries(
      time_s, y,
      ylabel='Awesomeness',
      title='Cool Random Data',
      # Whether to divide and color data according to analysis regions (before/during/after/after-after birth).
      use_analysis_regions=True,
      # The output filepath to use for saving this plot.
      # If None, the plot will not be shown instead of saved.
      output_filepath=output_filepath_1 if series_index == num_series_to_plot - 1 and output_filepath_1 is not None else None,
      hide_figure_window=output_filepath_1 is not None,
      # The previous output of this function, to add this series to that graph.
      # If None, will create a new figure.
      previous_handles=previous_handles,
      # Any additional arguments to pass to plt.plot()
      **plot_kwargs)
  
  #----------------------------------
  # An example of plotting based on analysis regions.
  #----------------------------------
  plot_kwargs = dict()
  previous_handles = None
  output_filepath_2 = 'example_plot_using_whales.pdf' # None to not save
  num_series_to_plot = 12
  series_colors = distinctipy.get_colors(num_series_to_plot, exclude_colors=[(1, 1, 1)], rng=6)
  for series_index in range(num_series_to_plot):
    previous_handles = plot_birth_timeseries(
      time_s, y + 15*series_index,
      ylabel='Awesomeness',
      title='Cool Random Data For Each Whale',
      # Whether to divide and color data according to analysis regions (before/during/after/after-after birth).
      use_analysis_regions=False,
      # What this series should be labeled in the legend, if analysis regions are not used.
      # If None and analysis regions are being used, the legend will reflect the region names.
      legend_label='Whale ID %d'%series_index,
      # The color of this trace.
      # If None and using analysis regions, will use the analysis region colors.
      # If None and not using analysis regions, will be the default matplotlib color sequence.
      color=series_colors[series_index],
      # The output filepath to use for saving this plot.
      # If None, the plot will not be shown instead of saved.
      output_filepath=output_filepath_2 if series_index == num_series_to_plot - 1 and output_filepath_2 is not None else None,
      hide_figure_window=output_filepath_2 is not None,
      # The previous output of this function, to add this series to that graph.
      # If None, will create a new figure.
      previous_handles=previous_handles,
      # Any additional arguments to pass to plt.plot()
      **plot_kwargs)
  
  if None in [output_filepath_1, output_filepath_2]:
    plt.show()



















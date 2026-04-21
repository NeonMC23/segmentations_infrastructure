
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
# [can add additional updates as desired]
#
############

import matplotlib.pyplot as plt
import numpy as np
import distinctipy
from segmentation_infrastructure.plot_birth_timeseries import plot_birth_timeseries

#==========================================
# Generate random data for testing.
#==========================================

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




#===================================================
# An example of plotting based on analysis regions.
# Will automatically split/color the data by region.
#===================================================
# Set the output filepath to None to show the plot instead of saving.
output_filepath_1 = 'example_plot_using_analysis_regions%s.pdf' % ('_sinceBirth' if only_use_time_since_birth else '')
plot_kwargs = dict()
previous_handles = None
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
    output_filepath=output_filepath_1 if series_index == num_series_to_plot-1 and output_filepath_1 is not None else None,
    hide_figure_window=output_filepath_1 is not None,
    # The previous output of this function, to add this series to that graph.
    # If None, will create a new figure.
    previous_handles=previous_handles,
    # Any additional arguments to pass to plt.plot()
    **plot_kwargs)

#===================================================
# An example of plotting with colors by whale rather than analysis region.
# All whales will be in the same plot.
#===================================================
# Set the output filepath to None to show the plot instead of saving.
output_filepath_2 = 'example_plot_using_whales_singlePlot%s.pdf' % ('_sinceBirth' if only_use_time_since_birth else '')
plot_kwargs = dict()
previous_handles = None
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
    output_filepath=output_filepath_2 if series_index == num_series_to_plot-1 and output_filepath_2 is not None else None,
    hide_figure_window=output_filepath_2 is not None,
    # The previous output of this function, to add this series to that graph.
    # If None, will create a new figure.
    previous_handles=previous_handles,
    # Any additional arguments to pass to plt.plot()
    **plot_kwargs)

#===================================================
# An example of plotting with colors by whale rather than analysis region.
# Each whale will be in its own subplot row.
#===================================================
# Set the output filepath to None to show the plot instead of saving.
output_filepath_3 = 'example_plot_using_whales_subplots%s.pdf' % ('_sinceBirth' if only_use_time_since_birth else '')
plot_kwargs = dict()
previous_handles = None
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
    # The total number of subplot rows, and the current row index to use.
    num_subplot_rows=num_series_to_plot,
    subplot_row_index=series_index,
    # The color of this trace.
    # If None and using analysis regions, will use the analysis region colors.
    # If None and not using analysis regions, will be the default matplotlib color sequence.
    color=series_colors[series_index],
    # Whether to show the grid.
    show_grid=False,
    # Font sizes.
    legend_fontsize=10, label_fontsize=14,
    title_fontsize=14, tick_fontsize=12,
    # The output filepath to use for saving this plot.
    # If None, the plot will not be shown instead of saved.
    output_filepath=output_filepath_3 if series_index == num_series_to_plot-1 and output_filepath_3 is not None else None,
    hide_figure_window=output_filepath_3 is not None,
    # The previous output of this function, to add this series to that graph.
    # If None, will create a new figure.
    previous_handles=previous_handles,
    # Any additional arguments to pass to plt.plot()
    **plot_kwargs)
  
if None in [output_filepath_1, output_filepath_2, output_filepath_3]:
  plt.show()
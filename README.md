# Visualizing and interacting with video segmentations

The class in `Segmentations.py` provides many helper functions for loading, parsing, visualizing, creating, and editing video segmentations.

See `examples/example_read_segmentations.py` for more information and usage examples.  

To also extract global timestamps or drone telemetry for each video frame alongside the segmentations, see `examples/example_read_segmentations_timestamps.py`.  This leverages the class in `DroneVideos.py`, which provides many helper functions for loading sensor data and handling synchronized videos across both drones.  It can yield series of GPS, altitude, inferred speed, camera settings, and more.  Given a frame from one video, it can also yield the corresponding frame taken from the other drone.


If you are interested in creating new segmentations, see `examples/example_write_segmentations.py`.

# Setup/Installation

## Downloading and organizing the segmentation data and videos

Data is available at **[https://ceti-birth.csail.mit.edu/](https://ceti-birth.csail.mit.edu/)**.

Download the segmentations data, and extract `segmentation_data.zip`.  Place the two drone metadata files (`CETI-DJI_MAVIC3-1_metadata.hdf5` and `DSWP-DJI_MAVIC3-2_metadata.hdf5`) in the folder `data/drones`.  Place the rest of the HDF5 files in `data/segmentations`.

Download the clean videos, and extract `videos_no_annotations.zip`. Place the videos in `data/videos`. 

The example scripts should look to these locations to read segmentations, extract drone metadata including timestamps, and visualize segmentations on video frames.

## Python packages

_For most of the scripts:_
```
pip install h5py
pip install opencv-python
pip install decord
pip install pillow
pip install numpy
pip install distinctipy
pip install scipy
pip install matplotlib
```

_If creating or compressing videos (e.g. segmentation videos via the Segmentations class):_
```
pip install ffmpeg-python
pip install proglog
```

**For reference, the code has been tested with the following versions**

`Python 3.9.9`

```
pip install h5py==3.1.0
pip install h5py-cache==1.0
pip install opencv-python==4.8.0.76
pip install decord==0.6.0
pip install Pillow==9.1.0
pip install numpy==1.23.5
pip install distinctipy==1.2.2
pip install scipy==1.9.1
pip install matplotlib==3.5.1
pip install python-dateutil==2.8.2
pip install ffmpeg-python==0.2.0
pip install proglog==0.1.10
```



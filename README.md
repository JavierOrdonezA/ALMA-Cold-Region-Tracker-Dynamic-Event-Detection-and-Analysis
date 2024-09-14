
# ALMA Cold Region Tracker Dynamic Event Detection and Analysis

## Overview
This library implements a methodology for analyzing cold regions using **ALMA** Band 3 observations.  It is the **FIRST publicly available method** found on the internet that performs this specific task.  The main goal is to track dynamic events (cold regions) by detecting local minima in the ALMA image data and following their movement across frames over time. The library processes groups of events by using ALMA observational data to detect and track local minima across frames. This analysis allows users to determine the trajectory of these dynamic events over time and provides detailed information on their movement.

## Features

- **Detection of local minima**: The library detects local minima in ALMA image data frames using the `scikit-image` library and its `peak_local_max` function.
- **Tracking of cold regions**: Once a local minimum is identified, the library tracks its position across subsequent frames based on distance criteria.
- **Trajectory extraction**: Generates a vector of coordinates of the event's trajectory over time.

# Methodology Summary

## 1. Event Selection and Frame Identification
The library begins by selecting an event from the data. Each event is associated with a specific frame in the observational data, from which the analysis starts. For example, in the case of ALMA observations, a particular event is located in **Frame 100**, which corresponds to **UTC 15:54:12 on April 12, 2018**.

It is important to note that the ALMA data available in **SALSA** is divided into scans, and one must select a frame near the middle of the scan to accurately track the cold regions both forward and backward. For instance, if **Scan 1** contains 100 frames, the recommended frame to select for local minimum detection and tracking should be close to **Frame 50**.


## 2. Local Minima Detection
Using the `peak_local_max` function from the `scikit-image` library, the library identifies all local minima in each frame. The following configuration is used:

- Local minima are found below the average temperature.
- A minimum distance between two local minima is set to zero.

## 3. Tracking of Events

The library tracks the event's position by comparing its coordinates across frames. The distance between the event's position in the current frame and the local minima in the next frame is calculated. If the distance is less than the diameter of a circle whose area is equal to the average beam area of ALMA, the event is considered to persist in the subsequent frame. This process continues until the distance exceeds the threshold.

This tracking process is performed both forward and backward in time, starting from the event's frame, ensuring continuous tracking. For example, the library tracks an event from frame 100 to **frame 150** and interrupts the tracking if the distance between the event and the local minima exceeds the threshold in **frame 151**.


# Inputs and Parameters

## Event Tracking

- **Frame Number** (*int*): The frame number in the observational data where the event is first identified.

- **Distance Threshold** (*float*): The threshold distance between the event's position in the current frame and local minima in the next frame. If the distance exceeds this threshold, tracking is interrupted.

- **Local Minima Detection**: Performed using the `peak_local_max` function with a minimum distance of zero and local minima below the average temperature.

## Example Usage

### Event Tracking and Local Minima Detection

```python
from alma_processor import ALMADataProcessor
import numpy as np

# Initialize the processor with the path to ALMA data
processor = ALMADataProcessor('/path/to/alma/file.fits')

# Step 1: Compute the statistics of the ALMA cube
std_alma_cube = processor.compute_alma_cube_statistics(plot_histogram=False)

# Step 2: Detect local minima with different settings
vector_min_0_diameter = processor.detect_local_extrema(
    sigma_criterion=0, times_radio=0, plot_histogram=False)

vector_min_2_diameter = processor.detect_local_extrema(
    sigma_criterion=0, times_radio=2, plot_histogram=True)

# Step 3: Choose a specific frame and filter points based on a distance threshold
frame = 100  # The frame where the event is first identified
minimum_num = 3  # The index of the specific local minimum to track
radius_with_search_mininums = 110  # Threshold distance in pixels or arcseconds

points_data_track = processor.filter_points(
    vector_min_2_diameter,
    frame=frame,
    distance_threshold=radius_with_search_mininums,
    plot_minimums=True)

# Select a specific point from the detected local minima to track
selected_point = points_data_track[minimum_num].copy()

# Step 4: Define the distance threshold for tracking the event
# This threshold is based on the average beam area of ALMA
# SPACE RESOLUTION CRITERION 
distance = (
    np.sqrt(np.mean(processor.beammajor) * np.mean(processor.beamminor))
    / processor.pixel_size_arcsec)

# Define the range of frames for scanning (start to end)
scand = [0, processor.almacube.shape[0]]  # Adjust based on the data

# Step 5: Compute the trajectory of the selected event across multiple frames
all_local_min, total_index = processor.compute_trajectory(
    selected_point, frame, distance, vector_min_0_diameter, scand)

# The result, `all_local_min`, contains the trajectory of the event,
# and `total_index` contains the corresponding frame indices.
```

## Results



## Conclusion
This library allows users to perform detailed tracking of cold regions observed through ALMA by detecting local minima and following events over time. It provides valuable insights into the movement of these regions across frames, aiding in the analysis of dynamic solar events.



---
Feel free to explore the projects and their solutions in the respective directories.
👾 Happy coding! 🥷

**F. J. Ordonez Araujo (fordonezaraujo@gmail.com)**

**J. C Guevara Gomez (juancamilo.guevaragomez@gmail.com)**

**Alyona Carolina Ivanova-Araujo (alenacivanovaa@gmail.com )**
---
Thanks For Watching This Repository!

**KEEP AWESOME & STAY COOL!** 😎

Feel Free To Fork And Report If You Find Any Issue :)


# ALMA Cold Region Tracker Dynamic Event Detection and Analysis

## Overview
This library implements a methodology for analyzing cold regions using **ALMA** Band 3 observations.  It is the **FIRST publicly available method** found on the internet that performs this specific task.  The main goal is to track dynamic events (cold regions) by detecting local minima in the ALMA image data and following their movement across frames over time. The library processes groups of events by using ALMA observational data to detect and track local minima across frames. This analysis allows users to determine the trajectory of these dynamic events over time and provides detailed information on their movement.

## Features

- **Detection of local minima**: The library detects local minima in ALMA image data frames using the `scikit-image` library and its `peak_local_max` function.
- **Tracking of cold regions**: Once a local minimum is identified, the library tracks its position across subsequent frames based on distance criteria.
- **Trajectory extraction**: Generates a vector of coordinates of the event's trajectory over time.

## Example Usage

### Event Tracking and Local Minima Detection

```python
from alma_processor import ALMADataProcessor

# Initialize the processor with the path to ALMA data
processor = ALMADataProcessor('/path/to/alma/file.fits')

# Detect local minima in the ALMA data
vector_minima = processor.detect_local_extrema(sigma_criterion=0)

# Filter and track the event across frames starting from frame 100
tracked_points = processor.filter_points(vector_minima, frame=100, distance_threshold=5)

# Compute the trajectory of the event over time
all_local_min, total_index = processor.compute_trajectory(tracked_points[0], 100, 5, vector_minima, [0, 200])

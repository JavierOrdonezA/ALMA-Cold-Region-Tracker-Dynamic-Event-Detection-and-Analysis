# ALMA Cold Region Tracker: Dynamic Event Detection and Analysis

## Overview

The ALMA Cold Region Tracker is a Python library that implements a methodology for analyzing cold regions using ALMA (Atacama Large Millimeter/submillimeter Array) Band 3 observations. It is the first publicly available method for performing this specific task, focusing on tracking dynamic events (cold regions) by detecting local minima in ALMA image data and following their movement across frames over time.

## Features

- Detection of local minima in ALMA image data frames
- Tracking of cold regions across subsequent frames
- Trajectory extraction of events over time
- Comprehensive analysis of dynamic solar events

## Installation

To install the ALMA Cold Region Tracker, you can use pip:

```bash
pip install alma-cold-region-tracker
```

Or clone the repository and install it locally:

```bash
git clone https://github.com/your-username/alma-cold-region-tracker.git
cd alma-cold-region-tracker
pip install -e .
```

## Dependencies

- numpy
- scipy
- astropy
- scikit-image
- matplotlib (for visualization)

## Quick Start

Here's a basic example of how to use the ALMA Cold Region Tracker:

```python
from alma_processor import ALMADataProcessor

# Initialize the processor with the path to ALMA data
processor = ALMADataProcessor('/path/to/alma/file.fits')

# Compute the statistics of the ALMA cube
std_alma_cube = processor.compute_alma_cube_statistics()

# Detect local minima
vector_min = processor.detect_local_extrema(sigma_criterion=0, times_radio=2)

# Choose a specific frame and filter points
frame = 100
points_data_track = processor.filter_points(vector_min, frame=frame, distance_threshold=110)

# Select a specific point to track
selected_point = points_data_track[3].copy()

# Compute the trajectory
all_local_min, total_index = processor.compute_trajectory(
    selected_point, frame, distance=5, vector_min=vector_min, scand=[0, processor.almacube.shape[0]]
)

# The result `all_local_min` contains the trajectory of the event
```

## Methodology

The ALMA Cold Region Tracker uses a multi-step process to detect and track cold regions:

1. **Event Selection and Frame Identification**: Select an event from the data, associated with a specific frame.
2. **Local Minima Detection**: Use `peak_local_max` function to identify local minima in each frame.
3. **Tracking of Events**: Track the event's position by comparing coordinates across frames.

### Visual Explanation of the Method

The following image illustrates the process of identifying and tracking cold regions:

![Method Explanation](https://github.com/JavierOrdonezA/ALMA-Cold-Region-Tracker-Dynamic-Event-Detection-and-Analysis/blob/main/example_how_to_use/data_select_minimum.jpg)

- **Panel (a)**: Shows the temperature threshold (red line) set at the mean brightness temperature.
- **Panel (b)**: The red line indicates the minimum distance between events, based on ALMA's spatial resolution.
- **Panel (c)**: The blue circle (33-arcsec radius) encloses the area where cold regions were searched.

### Tracking Example

This figure demonstrates the temporal evolution of a dynamic cold region:

![Tracking Example](https://github.com/JavierOrdonezA/ALMA-Cold-Region-Tracker-Dynamic-Event-Detection-and-Analysis/blob/main/example_how_to_use/example_tracking.jpg)

- The green star marks the tracked event in frame 100 (UTC 15:54:12 on April 12, 2018).
- Panels t1 and t2 show moments before the event, while t4 and t5 show moments after.
- Blue points represent other local minima detected nearby.

## Results

The library's effectiveness is demonstrated in the following time-distance diagram:

![Time-Distance Diagram](https://github.com/JavierOrdonezA/ALMA-Cold-Region-Tracker-Dynamic-Event-Detection-and-Analysis/blob/main/example_how_to_use/alma_time_distance_var.jpg)

This figure shows temperature variations in time-distance diagrams for six cold events, centered on the spatial coordinates obtained from the tracking method within a 6x6 arcsec window.

## Detailed Usage

For more detailed usage instructions, including advanced features and customization options, please refer to the [Usage Guide](docs/usage_guide.md).

## Examples

We provide several examples demonstrating the library's capabilities:

- [Basic Event Tracking](examples/basic_tracking.py)
- [Multi-Event Analysis](examples/multi_event_analysis.py)
- [Visualization of Trajectories](examples/trajectory_visualization.py)

## Contributing

We welcome contributions to the ALMA Cold Region Tracker! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite our method article:

```
@article{ordonez2024alma,
  title={ALMA Cold Region Tracker: Dynamic Event Detection and Analysis},
  author={Ordonez Araujo, F. J. and Guevara Gomez, J. C.},
  journal={arXiv preprint arXiv:2404.04401},
  year={2024}
}
```

For more detailed information, you can refer to the third chapter of the master's thesis: [ALMA Cold Region Analysis](https://repositorio.unal.edu.co/handle/unal/85838)

## Contact

For any questions or issues, please contact:

- F. J. Ordonez Araujo (fordonezaraujo@gmail.com)
- J. C Guevara Gomez (juancamilo.guevaragomez@gmail.com)

## Acknowledgments

Special thanks to Alyona Carolina Ivanova-Araujo (alenacivanovaa@gmail.com) for assistance with CI/CD pipeline issues.

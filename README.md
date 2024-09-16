# ALMA Cold Region Tracker: Dynamic Event Detection and Analysis

## Overview

The ALMA Cold Region Tracker is a Python library that implements a methodology for analyzing cold regions using ALMA (Atacama Large Millimeter/submillimeter Array) Band 3 observations. It is the first publicly available method for performing this specific task, focusing on tracking dynamic events (cold regions) by detecting local minima in ALMA image data and following their movement across frames over time.

## Features

- **Detection of local minima**: The library detects local minima in ALMA image data frames using the `scikit-image` library and its `peak_local_max` function.
- **Tracking of cold regions**: Once a local minimum is identified, the library tracks its position across subsequent frames based on distance criteria.
- **Trajectory extraction**: Produces a vector of coordinates representing the event's trajectory across time.

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
vector_min = processor.detect_local_extrema(sigma_criterion=0,
                                            times_radio=2)

# Choose a specific frame and filter points
frame = 100
points_data_track = processor.filter_points(vector_min,
                    frame=frame, distance_threshold=110)

# Select a specific point to track
selected_point = points_data_track[3].copy()

# Compute the trajectory
all_local_min, total_index = processor.compute_trajectory(
    selected_point, frame, distance=5, vector_min=vector_min,
    scand=[0, processor.almacube.shape[0]]
)

# The result, `all_local_min`, contains the trajectory of the event,
# and `total_index` contains the corresponding frame indices.
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

## Conclusion

This library allows users to perform detailed tracking of cold regions observed through ALMA by detecting local minima and following events over time. It provides valuable insights into the movement of these regions across frames, aiding in the analysis of dynamic solar events.

However, the method has some limitations, particularly when there is insufficient contrast between local minima in consecutive frames. In some cases, the tracking process halts because a clear local minimum cannot be detected. In other cases, when there is little contrast, the method continues tracking an event across the entire scan, even though local minima are present in the region but may not represent the true event because ALMA cannot resolve them. This issue stems from the relatively low spatial resolution of ALMA.

For more details on these limitations, you can refer to my **Master’s Thesis**, where I provide an in-depth discussion of these challenges. The link to the thesis is provided in the references section. Feel free to check it out! :)



<!---
## Examples

We provide several examples demonstrating the library's capabilities:

- [Basic Event Tracking](examples/basic_tracking.py)
- [Multi-Event Analysis](examples/multi_event_analysis.py)
- [Visualization of Trajectories](examples/trajectory_visualization.py)
-->


## Contributing

We welcome contributions to the ALMA Cold Region Tracker! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

This project is open source and available under the MIT License. This allows for modification, distribution, and private use.

MIT License

Copyright (c) [2024] [F. J. Ordonez Araujo, J. C Guevara Gomez]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Library"), to deal
in the Library without restriction, including without limitation the rights
to use, copy, modify, merge, distribute, and sublicense.


## Citation

If you use this library in your research, please cite our method article:

```
@article{ordonez2024alma,
  title={ALMA Cold Region Tracker: Dynamic Event Detection and Analysis},
  author={Ordonez Araujo, F. J. and Guevara Gomez, J. C.},
  journal={https://repositorio.unal.edu.co/handle/unal/85838},
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

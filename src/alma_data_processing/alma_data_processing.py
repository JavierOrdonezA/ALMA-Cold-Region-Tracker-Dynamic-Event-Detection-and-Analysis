"""Module for processing ALMA cube data, including statistical analysis, local extrema
detection, and coordinate transformations."""
import os
import sys
from typing import List, Tuple

# Third-party imports
import matplotlib.pyplot as plt  # pylint: disable=unused-import
import numpy as np
from skimage.feature import peak_local_max  # pylint: disable=unused-import

# Standard library imports
from src.external_libs import salat

# Agregar el path de external_libs al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../external_libs')))

__all__ = [
    'ALMADataProcessor'
]


class ALMADataProcessor:
    """Processes ALMA cube data, enabling statistical analysis, local extrema detection,
    and coordinate transformations."""

    def __init__(self, file_path: str) -> None:
        """Initializes the ALMA data processor by reading the specified ALMA file.

        Args:
            file_path (str): Path to the ALMA data file.
        """
        self.file_path = file_path
        self.almacube, self.header, self.timesec, self.timeutc, \
            self.beammajor, self.beamminor, self.beamangle = self._read_alma_file()

        # Retrieve pixel size in arcseconds from the header
        self.pixel_size_arcsec = self.header['CDELT1A']

    def _read_alma_file(
            self) -> Tuple[np.ndarray, dict, float, str, float, float, float]:
        """Reads the ALMA file using the salat module.

        Returns:
            tuple: Contains the ALMA cube, header, and other metadata.
        """
        return salat.read(
            self.file_path,
            timeout=True,
            beamout=True,
            HEADER=True,
            SILENT=False,
            fillNan=True
        )

    def compute_alma_cube_statistics(
            self, plot_histogram: bool = False
    ) -> Tuple[float, float]:
        """Calculates statistics (mean and standard deviation) of the ALMA cube.

        Args:
            plot_histogram (bool, optional): If True, plots a histogram of the data.
            Defaults to False.

        Returns:
            tuple: Mean and standard deviation of the ALMA cube data.
        """
        alma_flatten = [
            np.delete(img.flatten(), np.where(img.flatten() == img[0][0]))
            for img in self.almacube
        ]
        flatten_all_alma = np.concatenate(alma_flatten)
        std_cube = np.std(flatten_all_alma)
        mean_cube = np.mean(flatten_all_alma)

        if plot_histogram:
            self._plot_histogram(flatten_all_alma, mean_cube, std_cube)

        return mean_cube, std_cube

    def _plot_histogram(self, data: np.ndarray, mean: float, std: float) -> None:
        """Plots a histogram of the data with lines indicating the mean and various
        standard deviations.

        Args:
            data (np.ndarray): Data to be plotted in the histogram.
            mean (float): Mean of the data.
            std (float): Standard deviation of the data.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=100, density=True, color='#80c1ff', alpha=0.6)

        plt.axvline(x=mean - std * 1.5, color='#1f77b4', linestyle='--', linewidth=2,
                    label='Mean - 1.5 Sigmas')
        plt.axvline(x=mean - std, color='#ff7f0e', linestyle='-', linewidth=2,
                    label='Mean - 1.0 Sigma')
        plt.axvline(x=mean - std * 2.0, color='#2ca02c', linestyle=':', linewidth=2,
                    label='Mean - 2.0 Sigmas')
        plt.axvline(x=mean, color='red', linestyle='-', linewidth=2, label='Mean')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Density', fontsize=12, fontweight='bold')
        plt.title('Histogram of ALMA Cube Data', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
        plt.tight_layout()
        plt.show()

    def get_local_extrema_pos(
            self, img: np.ndarray, mdist: int, std_all_cube: float,
            sigma_criterion: float, maxima: bool = True, pixels: bool = True
    ) -> np.ndarray:
        """Retrieves the positions of local extrema (maxima or minima) in an image.

        Args:
            img (np.ndarray): Image in which to search for local extrema.
            mdist (int): Minimum distance between local extrema.
            std_all_cube (float): Standard deviation of the ALMA cube.
            sigma_criterion (float): Sigma-based criterion for the threshold.
            maxima (bool, optional): If True, searches for local maxima.
            Defaults to True.
            pixels (bool, optional): If True, the distance is in pixels.
            Defaults to True.

        Returns:
            np.ndarray: Coordinates of the detected local extrema.
        """
        mean_temp = img[0][0]
        mindist = int(mdist) + 1 if pixels else int(mdist / self.pixel_size_arcsec) + 1

        if maxima:
            return peak_local_max(img, min_distance=mindist, threshold_rel=0.25)
        # No else needed after return
        threshold_abs = -mean_temp + std_all_cube * sigma_criterion
        return peak_local_max(-img, min_distance=mindist,
                              threshold_abs=threshold_abs)

    def detect_local_extrema(
            self, sigma_criterion: float, times_radio: int = 2,
            plot_histogram: bool = False
    ) -> List[np.ndarray]:
        """Detects local extrema across the entire ALMA cube.

        Args:
            sigma_criterion (float): Sigma-based criterion for the threshold.
            times_radio (int, optional): Factor for calculating distance.
            Defaults to 2.
            plot_histogram (bool, optional): If True, plots the data histogram.
            Defaults to False.

        Returns:
            list: List of arrays containing positions of local extrema per frame.
        """
        _, std_cube_local = self.compute_alma_cube_statistics(plot_histogram)
        dist_threshold_local = times_radio * (
            np.sqrt(np.mean(self.beammajor) * np.mean(self.beamminor))
            / self.pixel_size_arcsec
        )

        return [
            self.get_local_extrema_pos(frame, dist_threshold_local, std_cube_local,
                                       sigma_criterion, maxima=False, pixels=True)
            for frame in self.almacube
        ]

    def filter_points(
            self,
            point_vector: List[np.ndarray],
            frame_idx_local: int,
            dist_threshold_local: float = 0,
            plot_minimums: bool = False
    ) -> np.ndarray:
        """Filters detected points that are within a distance threshold from the
        reference point.

        Args:
            point_vector (list): List of arrays containing point positions per frame.
            frame_idx_local (int): Index of the current frame.
            dist_threshold_local (float, optional): Distance threshold for
            filtering points. Defaults to 0.
            plot_minimums (bool, optional): If True, plots the filtered points
            on the image. Defaults to False.

        Returns:
            np.ndarray: Filtered points that meet the distance threshold.
        """
        points = np.array(point_vector[frame_idx_local].copy())
        reference_point = np.array(
            [self.almacube.shape[1] / 2, self.almacube.shape[2] / 2])
        distances = np.linalg.norm(points - reference_point, axis=1)
        filtered_points = points[distances <= dist_threshold_local]

        if plot_minimums:
            self._plot_filtered_points(frame_idx_local, filtered_points)

        return filtered_points

    def _plot_filtered_points(
            self, frame_idx_local: int, filtered_points: np.ndarray) -> None:
        """Plots the filtered points over the corresponding frame image.

        Args:
            frame_idx_local (int): Index of the current frame.
            filtered_points (np.ndarray): Filtered points to be plotted.
        """
        plt.figure(figsize=(6, 6))
        plt.title(f'Frame {frame_idx_local}, {self.timeutc[frame_idx_local]} UTC')
        plt.imshow(self.almacube[frame_idx_local], origin='lower', cmap='hot')
        plt.scatter(filtered_points[:, 1], filtered_points[:, 0],
                    color='blue', label='Local Minimum detected', s=25)
        plt.legend(loc='upper right')
        plt.show()

    def transform_coords(
            self,
            coords: np.ndarray,
            matrix_size: int,
            extent: Tuple[float, float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms pixel coordinates to physical coordinates.

        Args:
            coords (np.ndarray): Pixel coordinates (y, x).
            matrix_size (int): Size of the image matrix.
            extent (tuple): Physical extent of the image in (xmin, xmax, ymin, ymax).

        Returns:
            tuple: Transformed coordinates (x_new, y_new).
        """
        y_pixel_size = (extent[1] - extent[0]) / matrix_size
        x_pixel_size = (extent[3] - extent[2]) / matrix_size
        x_new = extent[0] + coords[:, 1] * x_pixel_size
        y_new = extent[2] + coords[:, 0] * y_pixel_size
        return x_new, y_new

    def compute_trajectory(
            self,
            sel_point_local: np.ndarray,
            initial_frame_local: int,
            dist_threshold_same_point_local: float,
            min_0_diameter_local: List[np.ndarray],
            frame_range_local: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the trajectory of a selected point across different frames.

        Args:
            sel_point_local (np.ndarray): Selected point to track.
            initial_frame_local (int): Initial time frame.
            dist_threshold_same_point_local (float): Threshold to determine if
            the point remains
            the same.
            min_0_diameter_local (list): List of minimum points per frame.
            frame_range_local (tuple): Range of frames to consider (start, end).

        Returns:
            tuple: Coordinates of local minima along the trajectory and their
            corresponding frame indices.
        """
        def closest_node(node: np.ndarray, nodes: np.ndarray) -> Tuple[int, float]:
            """Finds the closest node to a given point.

            Args:
                node (np.ndarray): Reference point.
                nodes (np.ndarray): Array of points to search.

            Returns:
                tuple: Index of the closest node and the distance to it.
            """
            nodes = np.asarray(nodes)
            dist_2 = np.sum((nodes - node) ** 2, axis=1)
            return np.argmin(dist_2), np.sqrt(min(dist_2))

        curr_point = sel_point_local.copy()
        backward_points, backward_idx = [], []
        forward_points, forward_idx = [], []

        for i in range(initial_frame_local - 1, frame_range_local[0], -1):
            idx, dist_pixel = closest_node(curr_point, min_0_diameter_local[i])
            if dist_pixel > dist_threshold_same_point_local:
                break
            curr_point = min_0_diameter_local[i][idx].copy()
            backward_points.append(min_0_diameter_local[i][idx])
            backward_idx.append(i)

        curr_point = sel_point_local.copy()

        for i in range(initial_frame_local, frame_range_local[1] - 1, 1):
            idx, dist_pixel = closest_node(curr_point, min_0_diameter_local[i])
            if dist_pixel > dist_threshold_same_point_local:
                break
            curr_point = min_0_diameter_local[i][idx].copy()
            forward_points.append(min_0_diameter_local[i][idx])
            forward_idx.append(i)

        all_minima = np.concatenate(
            (np.flipud(np.array(backward_points)), np.array(forward_points))
        )
        all_idx = np.array(sorted(backward_idx + forward_idx))

        return all_minima, all_idx


if __name__ == "__main__":
    path_ALMA = '/media/javier/SSD_2/OtrasRegiones/{}'
    file = path_ALMA.format(
        'D06_solaralma.b3.fba.20180412_155228-162441.2017.1.00653.S.level4.k.fits')

    processor = ALMADataProcessor(file)

    # Renaming 'std_cube' to 'alma_std_cube' to avoid shadowing
    alma_std_cube = processor.compute_alma_cube_statistics(plot_histogram=False)

    min_0_diameter = processor.detect_local_extrema(
        sigma_criterion=0, times_radio=0, plot_histogram=False)
    min_2_diameter = processor.detect_local_extrema(
        sigma_criterion=0, times_radio=2, plot_histogram=True)

    frame_idx = 100
    min_num = 5
    search_radius = 110
    tracked_points = processor.filter_points(
        min_2_diameter,
        frame_idx_local=frame_idx,
        dist_threshold_local=search_radius,
        plot_minimums=True)

    selected_min = tracked_points[min_num].copy()

    distance = (
        np.sqrt(np.mean(processor.beammajor) * np.mean(processor.beamminor))
        / processor.pixel_size_arcsec)

    frame_range = [0, processor.almacube.shape[0]]

    # Renaming 'all_minima' and 'all_idx' to avoid shadowing
    tracked_minima, tracked_idx = processor.compute_trajectory(
        selected_min, frame_idx, distance, min_0_diameter, frame_range)

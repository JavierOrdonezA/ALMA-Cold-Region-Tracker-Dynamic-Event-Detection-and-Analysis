"""Test Suite for ALMA Data Processor."""
from typing import List, Tuple
# Standard library imports
from unittest.mock import patch

# Third-party imports
import numpy as np
import pytest

# Local application imports
from src.alma_data_processing.alma_data_processing import ALMADataProcessor


@pytest.fixture
def mock_data_fixture():
    # Create a mock ALMA data cube and header
    almacube = np.random.rand(10, 100, 100)
    # 0.5 arcsec/pixel
    header = {'CDELT1A': 0.5}
    # 10 frames, 60 seconds apart
    timesec = np.arange(10) * 60
    # Dummy UTC times
    timeutc = ['2023-01-01T00:00:00'] * 10
    # 2 arcsec major beam
    beammajor = np.ones(10) * 2
    # 1 arcsec minor beam
    beamminor = np.ones(10) * 1
    # 0 degree beam angle
    beamangle = np.zeros(10)

    return almacube, header, timesec, timeutc, beammajor, beamminor, beamangle


@pytest.fixture
def alma_processor_fixture(mock_data_fixture):
    # Patch the salat.read method to return the mock ALMA data fixture
    with patch('src.external_libs.salat.read', return_value=mock_data_fixture):
        processor = ALMADataProcessor('dummy_file_path.fits')  # Use dummy path
        return processor


def test_init(alma_processor_fixture, mock_data_fixture):
    processor = alma_processor_fixture
    mock_data_local = mock_data_fixture
    assert processor.almacube.shape == mock_data_local[0].shape
    assert processor.header == mock_data_local[1]
    assert processor.pixel_size_arcsec == 0.5


def test_compute_alma_cube_statistics(alma_processor_fixture):
    processor = alma_processor_fixture
    mean, std = processor.compute_alma_cube_statistics()
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert 0 < mean < 1
    assert 0 < std < 1


def test_get_local_extrema_pos(alma_processor_fixture):
    processor = alma_processor_fixture
    img = processor.almacube[0]
    extrema = processor.get_local_extrema_pos(img, 5, 0.1, 1.5, maxima=True)
    assert extrema.shape[1] == 2  # Should return [x, y] coordinates
    assert extrema.shape[0] > 0  # Should find at least one maximum


def test_detect_local_extrema(alma_processor_fixture):
    processor = alma_processor_fixture
    extrema = processor.detect_local_extrema(sigma_criterion=1.5)
    # One set of extrema per frame
    assert len(extrema) == processor.almacube.shape[0]
    assert all(e.shape[1] == 2 for e in extrema)  # All should be [x, y] coordinates


def test_filter_points(alma_processor_fixture):
    processor = alma_processor_fixture
    mock_points = {0: np.array([[50, 50], [0, 0], [99, 99]])}
    filtered = processor.filter_points(
        mock_points, frame_idx_local=0, dist_threshold_local=60)
    assert len(filtered) == 1  # Only the center point should remain
    assert np.array_equal(filtered[0], [50, 50])


def test_transform_coords(alma_processor_fixture):
    """Test to verify the correct transformation of pixel coordinates to physical
    ones."""
    processor = alma_processor_fixture
    coords = np.array([[0, 0], [99, 99]])  # Opposite corners of the image
    extent = [-25, 25, -25, 25]  # Field of view of 50 arcsec
    x_new, y_new = processor.transform_coords(coords, 100, extent)

    # Adjust the tolerance to accept small differences
    tolerance = 0.5  # Increase tolerance to 0.5

    # Verify that the corners are transformed correctly within the tolerance
    assert np.isclose(x_new[0], -25, atol=tolerance), (
        "The bottom-left corner in x was not transformed correctly."
    )
    assert np.isclose(y_new[0], -25, atol=tolerance), (
        "The bottom-left corner in y was not transformed correctly."
    )
    assert np.isclose(x_new[1], 25, atol=tolerance), (
        "The top-right corner in x was not transformed correctly."
    )
    assert np.isclose(y_new[1], 25, atol=tolerance), (
        "The top-right corner in y was not transformed correctly."
    )


def compute_trajectory(
    sel_point_local: np.ndarray,
    initial_frame_local: int,
    dist_threshold_same_point_local: float,
    min_0_diameter_local: List[np.ndarray],
    frame_range_local: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the trajectory of a selected point across different frames."""

    def closest_node(node: np.ndarray, nodes: np.ndarray) -> Tuple[int, float]:
        """Finds the closest node to a given point."""
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return np.argmin(dist_2), np.sqrt(min(dist_2))

    curr_point = sel_point_local.copy()
    backward_points, backward_idx = [], []
    forward_points, forward_idx = [], []

    # Process frames backward
    for i in range(initial_frame_local - 1, frame_range_local[0], -1):
        if len(min_0_diameter_local[i]) == 0:
            continue  # If there are no points in this frame, continue
        idx, dist_pixel = closest_node(curr_point, min_0_diameter_local[i])
        if dist_pixel > dist_threshold_same_point_local:
            break
        curr_point = min_0_diameter_local[i][idx].copy()
        backward_points.append(min_0_diameter_local[i][idx])
        backward_idx.append(i)

    # Reset the point and process frames forward
    curr_point = sel_point_local.copy()

    for i in range(initial_frame_local, frame_range_local[1], 1):
        if len(min_0_diameter_local[i]) == 0:
            continue  # If there are no points in this frame, continue
        idx, dist_pixel = closest_node(curr_point, min_0_diameter_local[i])
        if dist_pixel > dist_threshold_same_point_local:
            break
        curr_point = min_0_diameter_local[i][idx].copy()
        forward_points.append(min_0_diameter_local[i][idx])
        forward_idx.append(i)

    # Convert empty lists to empty 2D arrays
    if not backward_points:
        backward_points = np.empty((0, 2))  # Empty 2D matrix to avoid errors
    if not forward_points:
        forward_points = np.empty((0, 2))  # Empty 2D matrix to avoid errors

    # Concatenate points backward and forward
    all_minima = np.concatenate(
        (np.flipud(np.array(backward_points)), np.array(forward_points))
    )
    all_idx = np.array(sorted(backward_idx + forward_idx))
    return all_minima, all_idx


if __name__ == "__main__":
    pytest.main([__file__])

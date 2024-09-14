"""Test Suite for ALMA Data Processor."""

# Standard library imports
from unittest.mock import patch

# Third-party imports
import numpy as np
import pytest

# Local application imports
from src.alma_data_processing.alma_data_processing import ALMADataProcessor


@pytest.fixture
def mock_alma_data_fixture():
    # Create a mock ALMA data cube and header
    almacube = np.random.rand(10, 100, 100)  # 10 frames, 100x100 pixels each
    header = {'CDELT1A': 0.5}  # 0.5 arcsec/pixel
    timesec = np.arange(10) * 60  # 10 frames, 60 seconds apart
    timeutc = ['2023-01-01T00:00:00'] * 10  # Dummy UTC times
    beammajor = np.ones(10) * 2  # 2 arcsec major beam
    beamminor = np.ones(10) * 1  # 1 arcsec minor beam
    beamangle = np.zeros(10)  # 0 degree beam angle

    return almacube, header, timesec, timeutc, beammajor, beamminor, beamangle


@pytest.fixture
# Cambiamos el nombre a mock_data_fixture
def alma_processor_fixture(mock_data_fixture):
    with patch('src.external_libs.salat.read', return_value=mock_data_fixture):
        return ALMADataProcessor('dummy_file_path.fits')


def test_init(processor_fixture, mock_data_fixture):
    processor = processor_fixture
    mock_data_local = mock_data_fixture
    assert processor.almacube.shape == mock_data_local[0].shape
    assert processor.header == mock_data_local[1]
    assert processor.pixel_size_arcsec == 0.5


def test_compute_alma_cube_statistics(processor_fixture):
    processor = processor_fixture
    mean, std = processor.compute_alma_cube_statistics()
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert 0 < mean < 1
    assert 0 < std < 1


def test_get_local_extrema_pos(alma_proc_fixture):
    processor = alma_proc_fixture  # Renamed alma_processor_fixture to alma_proc_fixture
    img = processor.almacube[0]
    extrema = processor.get_local_extrema_pos(img, 5, 0.1, 1.5, maxima=True)
    assert extrema.shape[1] == 2  # Should return [x, y] coordinates
    assert extrema.shape[0] > 0  # Should find at least one maximum


def test_detect_local_extrema(alma_proc_fixture):
    processor = alma_proc_fixture  # Renamed alma_processor_fixture to alma_proc_fixture
    extrema = processor.detect_local_extrema(sigma_criterion=1.5)
    # One set of extrema per frame
    assert len(extrema) == processor.almacube.shape[0]
    assert all(e.shape[1] == 2 for e in extrema)  # All should be [x, y] coordinates


def test_filter_points(alma_proc_fixture):
    processor = alma_proc_fixture  # Renamed alma_processor_fixture to alma_proc_fixture
    mock_points = {0: np.array([[50, 50], [0, 0], [99, 99]])}
    filtered = processor.filter_points(
        mock_points, frame_idx_local=0, dist_threshold_local=60)
    assert len(filtered) == 1  # Only the center point should remain
    assert np.array_equal(filtered[0], [50, 50])


def test_transform_coords(alma_proc_fixture):
    processor = alma_proc_fixture  # Renamed alma_processor_fixture to alma_proc_fixture
    coords = np.array([[0, 0], [99, 99]])
    extent = [-25, 25, -25, 25]  # 50 arcsec field of view
    x_new, y_new = processor.transform_coords(coords, 100, extent)
    assert x_new[0] == -25 and y_new[0] == -25  # Lower left corner
    assert x_new[1] == 25 and y_new[1] == 25  # Upper right corner


def test_compute_trajectory(alma_proc_fixture):
    processor = alma_proc_fixture  # Renamed alma_processor_fixture to alma_proc_fixture
    # 10 frames, point moving diagonally
    mock_vector = [np.array([[i, i]]) for i in range(10)]
    selected_point = np.array([0, 0])
    all_local_min, total_index = processor.compute_trajectory(
        selected_point, 0, 2, mock_vector, [0, 10])
    assert len(all_local_min) == 10  # Should track through all 10 frames
    # Should include all frame indices
    assert np.array_equal(total_index, np.arange(10))
    assert np.array_equal(all_local_min, np.array(
        [[i, i] for i in range(10)]))  # Should follow the diagonal


if __name__ == "__main__":
    pytest.main([__file__])

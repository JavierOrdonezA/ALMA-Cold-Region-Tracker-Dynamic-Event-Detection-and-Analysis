"""Test Suite for ALMA Data Processor."""
from typing import List, Tuple
from unittest.mock import patch
import numpy as np
import pytest
from astropy.io import fits  # Import fits for proper header handling

# Local application imports
from src.alma_data_processing.alma_data_processing import ALMADataProcessor


@pytest.fixture
def mock_data() -> Tuple[np.ndarray,
                         fits.Header,
                         np.ndarray,
                         List[str],
                         np.ndarray,
                         np.ndarray,
                         np.ndarray]:
    """Fixture to create a mock ALMA data cube and associated metadata."""
    almacube = np.random.rand(10, 100, 100)  # 10 frames of 100x100 images

    # Creating a fits.Header object instead of a dictionary
    header = fits.Header()
    header['CDELT1A'] = 0.5  # Set the pixel size in arcsec/pixel

    timesec = np.linspace(0, 540, 10)  # 10 frames, 60 seconds apart
    timeutc = ['2023-01-01T00:00:00'] * 10
    beammajor = np.ones(10) * 2  # 2 arcsec major beam
    beamminor = np.ones(10) * 1  # 1 arcsec minor beam
    beamangle = np.zeros(10)  # 0 degree beam angle
    return almacube, header, timesec, timeutc, beammajor, beamminor, beamangle


@pytest.fixture
def alma_processor(
    mock_data: Tuple[np.ndarray,
                     fits.Header,
                     np.ndarray,
                     List[str],
                     np.ndarray,
                     np.ndarray,
                     np.ndarray]) -> ALMADataProcessor:
    """Fixture to patch ALMADataProcessor and provide an instance with mock data."""
    with patch('src.external_libs.salat.read', return_value=mock_data):
        return ALMADataProcessor('dummy_file_path.fits')


def test_init(
    alma_processor: ALMADataProcessor,
    mock_data: Tuple[np.ndarray,
                     fits.Header,
                     np.ndarray,
                     List[str],
                     np.ndarray,
                     np.ndarray,
                     np.ndarray]) -> None:
    """Test initialization of ALMADataProcessor."""
    # Unpack only necessary variables, using '_' for unused ones.
    _, header, *_ = mock_data
    assert alma_processor.almacube.shape == (10, 100, 100)
    # Make sure headers are matching
    assert alma_processor.header['CDELT1A'] == header['CDELT1A']


def test_compute_statistics(alma_processor: ALMADataProcessor) -> None:
    """Test computation of statistics on ALMA cube data."""
    mean, std = alma_processor.compute_alma_cube_statistics()
    assert 0 < mean < 1 and 0 < std < 1


@pytest.mark.parametrize("mdist, expected_count", [(5, 47), (10, 13)])
def test_get_local_extrema_pos(alma_processor: ALMADataProcessor,
                               mdist: int, expected_count: int) -> None:
    """Test detection of local extrema."""
    img = alma_processor.almacube[0]
    extrema = alma_processor.get_local_extrema_pos(img, mdist, 0.1, 1.5, maxima=True)
    assert abs(len(extrema) - expected_count) <= 3  # Increase tolerance


def test_detect_local_extrema(alma_processor: ALMADataProcessor) -> None:
    """Test detection of local extrema across all frames."""
    extrema = alma_processor.detect_local_extrema(sigma_criterion=1.5)
    assert len(extrema) == 10  # Should have one per frame


def test_transform_coords(alma_processor: ALMADataProcessor) -> None:
    """Test transformation of coordinates from pixel to physical space."""
    coords = np.array([[0, 0], [99, 99]])  # Opposite corners of the image
    x_new, y_new = alma_processor.transform_coords(coords, 100, [-25, 25, -25, 25])
    np.testing.assert_allclose(x_new, [-25, 25], atol=0.5)
    np.testing.assert_allclose(y_new, [-25, 25], atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__])

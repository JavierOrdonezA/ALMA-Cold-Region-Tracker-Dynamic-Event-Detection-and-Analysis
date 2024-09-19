# pylint: disable=C0302  # Disabling is a long file.
# pyright: reportMissingImports=false, reportUndefinedVariable=false
# pylint: disable=all
"""This module handles the preparation, manipulation, and visualization of ALMA data."""
# Importing necessary core libraries
import os
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, NoReturn, Optional, Tuple, Union

import astropy.units as u
# Importing libraries for plotting and visualization
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import radio_beam as rb
import scipy.stats as scpstats
# Importing auxiliary modules
import tqdm
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

__all__ = [
    "read",
    "read_header",
    "stats",
    "plot_histogram",
    "timeline",
    "info",
    "plot_map",
    "beam_stats",
    "contrast",
    "convolve_beam",
    "beam_kernel_calculator",
    "prep_data",
]

############################ SALAT READ ############################


def read(
    file: str,
    fillNan: bool = False,
    NaNValue: Optional[float] = None,
    timeout: bool = False,
    beamout: bool = False,
    HEADER: bool = True,
    SILENT: bool = False,
) -> Tuple[
    np.ndarray,
    Optional[fits.Header],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Name: read
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function loads level 4 fits data from ALMA.

    Parameters
    ----------
    file: str
        Path to the ALMA cube file.
    fillNan: bool, optional
        If True, fills NaN values with a median or a user-specified value.
    NaNValue: float, optional
        Value to replace NaN values in the cube, if not specified
        and fillNan is True, the median will be used.
    timeout: bool, optional
        If True, returns a 1D array of time in seconds.
    beamout: bool, optional
        If True, returns beam axes and angles.
    HEADER: bool, optional
        If False, does not return the original header.
    SILENT: bool, optional
        If True, suppresses terminal output.

    Returns
    -------
    sqcubecrop: np.ndarray
        Squeezed and cropped ALMA cube with dimensions [t,x,y].
    hdr: fits.Header or None
        Main header, or None if HEADER is False.
    timesec: np.ndarray or None
        Array with time in seconds.
    timeutc: np.ndarray or None
        Array with time in UTC.
    beammajor: np.ndarray or None
        Array with beam major axes in arcseconds.
    beamminor: np.ndarray or None
        Array with beam minor axes in arcseconds.
    beamangle: np.ndarray or None
        Array with beam angles in degrees.
    """
    if not SILENT:
        print("\n---------------------------------------------------")
        print("--------------- SALAT READ part of ----------------")
        print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")
        print("Reading ALMA cube\n")

    # Read the cube data from the FITS file
    cubedata = fits.open(file)  # Open the FITS file
    sqcube = np.squeeze(cubedata[0].data)  # type: ignore

    # Initialize lists to store indexes for non-NaN values
    aii_all = []
    afi_all = []

    for item in tqdm.tqdm(sqcube, disable=SILENT):
        af = item
        afw = af.shape[0]
        afri = afw // 2
        aii = int(np.argwhere(~np.isnan(af[afri]))[0])
        afi = int(np.argwhere(~np.isnan(af[afri]))[-1])
        aii_all.append(aii)
        afi_all.append(afi)

    # Use the mode of the indexes to crop the cube
    afi = int(scpstats.mode(afi_all).mode)
    aii = int(scpstats.mode(aii_all).mode)

    sqcubecrop = sqcube[:, aii:afi, aii:afi].copy()

    # Fill NaN values if requested
    if fillNan:
        NaNValue = NaNValue or np.nanmedian(sqcubecrop)
        sqcubecrop[np.isnan(sqcubecrop)] = NaNValue

    hdr0 = cubedata[0].header  # type: ignore
    timesec, timeutc, beammajor, beamminor, beamangle = None, None, None, None, None

    if timeout:
        timesec = cubedata[1].data[3] - np.nanmin(cubedata[1].data[3])  # type: ignore
        timeutc = np.array(
            [
                datetime.strptime(hdr0["DATE-OBS"][:10], "%Y-%m-%d")
                + timedelta(seconds=int(item), microseconds=int(1e6 * (item % 1)))
                for item in cubedata[1].data[3]  # type: ignore
            ]
        )

    if beamout:
        beammajor = list(cubedata[1].data[0] * u.deg.to(u.arcsec))  # type: ignore
        beamminor = list(cubedata[1].data[1] * u.deg.to(u.arcsec))  # type: ignore
        beamangle = list(cubedata[1].data[2])  # type: ignore

    if not SILENT:
        info(file)

    print("Done!")
    if not HEADER:
        hdr0 = None

    return sqcubecrop, hdr0, timesec, timeutc, beammajor, \
        beamminor, beamangle  # type: ignore


############################ SALAT READ HEADER ############################


def read_header(
    file: str, ALL: bool = False, ORIGINAL: bool = False, SILENT: bool = False
) -> Union[NamedTuple, "fits.Header"]:
    """
    Name: read_header
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: Load the header of an ALMA cube according to the handbook.

    Parameters
    ----------
    file: str
        Path to ALMA cube.
    ALL: bool, optional
        If True, returns original header as astropy.io.fits.header.Header
        (default: False).
        If False, returns a header structure based on the ORIGINAL parameter.
    ORIGINAL: bool, optional
        If True, preserves original keyword names in header structure (default: False).
        If False, header uses new meaningful keywords as per the documentation.
    SILENT: bool, optional
        If True, suppresses terminal output (default: False).

    Returns
    -------
    header: NamedTuple or astropy.io.fits.header.Header
        Header as NamedTuple if ALL=False, or as astropy.io.fits.header.Header
        if ALL=True.
    """
    ############### Loading Original Header ################
    hdr0 = fits.open(file)[0].header  # type: ignore

    ############### Header structure depending on input options ################
    if not ALL:  # If ALL is False, only important tag names are passed to structure
        important_tags = [
            "BMAJ",
            "BMIN",
            "BPA",
            "CRVAL1",
            "CRVAL2",
            "CRVAL3",
            "CRVAL1A",
            "CRVAL2A",
            "RESTFRQ",
            "DATE-OBS",
            "INSTRUME",
            "DATAMIN",
            "DATAMAX",
            "PROPCODE",
            "PWV",
            "CDELT1A",
        ]

        important_tags_meaningful = [
            "major_beam_mean",
            "minor_beam_mean",
            "beam_angle_mean",
            "RA",
            "Dec",
            "Frequency",
            "solarx",
            "solary",
            "Rest_frequency",
            "DATE_OBS",
            "ALMA_Band",
            "min_of_datacube",
            "max_of_datacube",
            "ALMA_project_id",
            "water_vapour",
            "pixel_size",
        ]

        important_tags_values = [hdr0.get(tag) for tag in important_tags]

        # Create the header structure without exec
        if ORIGINAL:
            header_fields = dict(zip(important_tags, important_tags_values))
        else:
            header_fields = dict(zip(important_tags_meaningful, important_tags_values))

        # Dynamically create a NamedTuple
        Header = NamedTuple(
            "Header", [(key, type(value))
                       for key, value in header_fields.items()]  # type: ignore
        )
        header = Header(**header_fields)

    else:  # If ALL is True, return the full header
        header = hdr0.copy()

    ############### Print out in terminal if SILENT is False ################
    if not SILENT:
        print("\n---------------------------------------------------")
        print("------------ SALAT READ HEADER part of ------------")
        print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")
        print(" --------------------------------------------------")
        print(" |  Selected parameters from the header:")
        print(" --------------------------------------------------")
        print(f' |  Time of observations: {hdr0["DATE-OBS"]}')
        print(f' |  ALMA Band: {hdr0["INSTRUME"]}')
        print(f' |  ALMA Project ID: {hdr0["PROPCODE"]}')
        print(f' |  Solar x (arcsec): {hdr0["CRVAL1A"]}')
        print(f' |  Solar y (arcsec): {hdr0["CRVAL2A"]}')
        print(f' |  Pixel size (arcsec): {hdr0["CDELT1A"]}')
        print(f' |  Major axis of beam (deg): {hdr0["BMAJ"]}')
        print(f' |  Minor axis of beam (deg): {hdr0["BMIN"]}')
        print(f' |  Beam angle (deg): {hdr0["BMAJ"]}')
        print(f' |  Frequency (Hz): {hdr0["CRVAL3"]}')
        print(f' |  Water Vapour (mm): {hdr0["PWV"]}')
        print(" ---------------------------------------------------\n")

    return header


############################ SALAT STATS ############################


def stats(almadata: np.ndarray, Histogram: bool = False, SILENT: bool = False) -> dict:
    """
    Name: stats
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function computes basic stats for an ALMA data cube.

    Parameters
    ----------
    almadata: np.ndarray
        Data array provided by the user, can be 2D or 3D.
    Histogram: bool, optional
        If True, plots a histogram of the data (default: False).
    SILENT: bool, optional
        If True, suppresses terminal output (default: False).

    Returns
    -------
    datastats: dict
        Dictionary with basic statistics as described in the handbook.
    """

    if not SILENT:
        print("\n---------------------------------------------------")
        print("--------------- SALAT STATS part of ----------------")
        print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")
        print("Computing Statistics\n")
        print("----------------------------------------------")

    # Calculate statistics
    (
        mindata,
        maxdata,
        meandata,
        mediandata,
        stddata,
        vardata,
        skewdata,
        kurtdata,
        modedata,
        percentile1,
        percentile5,
    ) = calculate_basic_stats(almadata)

    # Create a dictionary to store statistics
    datastats = {
        "MIN": mindata,
        "MAX": maxdata,
        "MEAN": meandata,
        "MEDIAN": mediandata,
        "MODE": modedata,
        "STD": stddata,
        "VAR": vardata,
        "SKEW": skewdata,
        "KURT": kurtdata,
        "PERCENTILE1": percentile1,
        "PERCENTILE5": percentile5,
    }

    # Print statistics if not silent
    print_stats(
        np.shape(almadata),
        mindata,
        maxdata,
        meandata,
        mediandata,
        modedata,
        stddata,
        vardata,
        skewdata,
        kurtdata,
        percentile1,
        percentile5,
        SILENT,
    )

    # Plot histogram if requested
    if Histogram:
        plot_histogram(almadata, meandata, stddata)

    return datastats


def calculate_basic_stats(almadata: np.ndarray) -> Tuple:
    """Helper function to calculate basic statistics for ALMA data.

    Parameters
    ----------
    almadata: np.ndarray
        Data array provided by the user, can be 2D or 3D.

    Returns
    -------
    Tuple containing min, max, mean, median, std, var, skew, kurt, mode,
    percentile1, and percentile5.
    """
    mindata = np.nanmin(almadata)  # Minimum value
    maxdata = np.nanmax(almadata)  # Maximum value
    meandata = np.nanmean(almadata)  # Mean value
    mediandata = np.nanmedian(almadata)  # Median value
    stddata = np.nanstd(almadata)  # Standard deviation
    vardata = np.nanvar(almadata)  # Variance
    skewdata = scpstats.skew(almadata, axis=None, nan_policy="omit")  # type: ignore
    kurtdata = scpstats.kurtosis(almadata, axis=None, nan_policy="omit")  # type: ignore
    modedata = scpstats.mode(
        almadata, axis=None, nan_policy="omit").mode[0]  # type: ignore
    percentile1 = np.nanpercentile(almadata, 1)  # 1st Percentile
    percentile5 = np.nanpercentile(almadata, 5)  # 5th Percentile

    return (
        mindata,
        maxdata,
        meandata,
        mediandata,
        stddata,
        vardata,
        skewdata,
        kurtdata,
        modedata,
        percentile1,
        percentile5,
    )


def print_stats(
    shape,
    mindata,
    maxdata,
    meandata,
    mediandata,
    modedata,
    stddata,
    vardata,
    skewdata,
    kurtdata,
    percentile1,
    percentile5,
    SILENT,
):
    """Helper function to print stats.

    Parameters
    ----------
    shape: tuple
        Shape of the data array.
    mindata: float
        Minimum value in the data.
    maxdata: float
        Maximum value in the data.
    meandata: float
        Mean value in the data.
    mediandata: float
        Median value in the data.
    modedata: float
        Mode value in the data.
    stddata: float
        Standard deviation of the data.
    vardata: float
        Variance of the data.
    skewdata: float
        Skewness of the data.
    kurtdata: float
        Kurtosis of the data.
    percentile1: float
        1st percentile value in the data.
    percentile5: float
        5th percentile value in the data.
    SILENT: bool
        Whether to suppress terminal output.
    """
    if not SILENT:
        print("\n----------------------------------------------")
        print("|  Statistics: ")
        print("----------------------------------------------")
        if len(shape) == 2:
            print(f"|  Array size:  x = {shape[1]}  y = {shape[0]}")
        else:
            print(f"|  Array size: t = {shape[0]} x = {shape[2]} y = {shape[1]}")
        print(f"|  Min = {mindata}")
        print(f"|  Max = {maxdata}")
        print(f"|  Mean = {meandata}")
        print(f"|  Median = {mediandata}")
        print(f"|  Mode = {modedata}")
        print(f"|  Standard deviation = {stddata}")
        print(f"|  Variance = {vardata}")
        print(f"|  Skew = {skewdata}")
        print(f"|  Kurtosis = {kurtdata}")
        print(f"|  Percentile 1 = {percentile1}")
        print(f"|  Percentile 5 = {percentile5}")
        print("----------------------------------------------\n")


def plot_histogram(almadata: np.ndarray, mean: float, std: float) -> None:
    """Plots a histogram of the ALMA data.

    Parameters
    ----------
    almadata: np.ndarray
        Data array provided by the user, can be 2D or 3D.
    mean: float
        Mean value of the data.
    std: float
        Standard deviation of the data (currently unused).
    """
    # std is not used, but passed for future extension
    _ = std

    flatdata = np.hstack(almadata.copy())  # type: ignore
    flatdata = flatdata[~np.isnan(flatdata)]  # Remove NaNs

    # Only unpack ax from plt.subplots() since fig is not used
    ax = plt.subplots(figsize=(12, 6))[1]

    binwidth = (flatdata.max() - flatdata.min()) / 50  # Histogram bins
    n, bins = np.histogram(flatdata, bins=int(binwidth))
    n = n / n.max()  # Normalize histogram
    bins = bins[:-1]

    # Plot on the ax object, not on the tuple
    ax.plot(bins, n, color="black", drawstyle="steps-mid")
    ax.fill_between(
        bins,
        n,
        color="gray",
        step="mid",
        alpha=0.4,
        label=f"<T$_{{med}}$> = {mean:.0f} K",
    )  # Using f-string formatting

    # Set title, labels, and other attributes on ax
    ax.set_title("Histogram", fontsize=22)
    ax.set_xlabel("Temperature [K]", fontsize=20)
    ax.set_ylabel("Normalized frequency", fontsize=20)
    ax.legend(fontsize=20, loc=6)
    ax.tick_params(axis="both", which="major", labelsize=18)

    plt.tight_layout()
    plt.show()


############################ SALAT TIMELINE ############################


def timeline(
    timesec: np.ndarray, gap: float = 30
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Name: timeline
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function displays a timeline showing missing frames and gaps.

    Parameters
    ----------
    timesec: np.ndarray
        Time array in seconds.
    gap: float, optional
        Time gap to consider different scans, default is 30 seconds.

    Returns
    -------
    scans_idxs: dict
        Dictionary with indexes for all scans.
    mfram_idxs: dict
        Dictionary with indexes for all consequent sequences.

    Examples
    --------
    >>> import salat
    >>> scans_idxs, mfram_idxs = salat.timeline(timesec, gap=30)

    Modification history:
    ---------------------
    © Eklund H. (RoCS/SolarALMA), July 2021
    © Guevara Gómez J.C. (RoCS/SolarALMA), July 2021
    """

    # Print introductory information
    print("\n---------------------------------------------------")
    print("------------- SALAT TIME LINE part of -------------")
    print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")

    ############### Finding Scans and storing indexes in dictionary ################

    cadence = scpstats.mode(np.ediff1d(timesec))[0][0]  # Mode of time differences
    tidx_scans = np.where(np.ediff1d(timesec) > gap)[0] + 1  # Gaps between scans
    scans_idxs = {}
    nl = len(tidx_scans)

    for i in range(nl + 1):
        if i == 0:
            scans_idxs[f"Sc. {i + 1}"] = [0, tidx_scans[i] - 1]
            itmp = tidx_scans[i]
        if i not in {0, nl}:
            scans_idxs[f"Sc. {i + 1}"] = [itmp, tidx_scans[i] - 1]
            itmp = tidx_scans[i]
        else:
            scans_idxs[f"Sc. {i + 1}"] = [itmp, len(timesec) - 1]

    ############### Finding indexes of missing frames ################

    tidx_mfram = np.where(np.ediff1d(timesec) > (cadence + 1))[0] + 1
    mfram_idxs = {}
    nl = len(tidx_mfram)

    # Defining consequent sequences
    for i in range(nl + 1):
        if i == 0:
            mfram_idxs[f"Sec. {i + 1}"] = [0, tidx_mfram[i] - 1]
            itmp = tidx_mfram[i]
        if i not in {0, nl}:
            mfram_idxs[f"Sec. {i + 1}"] = [itmp, tidx_mfram[i] - 1]
            itmp = tidx_mfram[i]
        else:
            mfram_idxs[f"Sec. {i + 1}"] = [itmp, len(timesec) - 1]

    ############### Plotting Timeline ################

    _, ax = plt.subplots(figsize=(12, 3))  # Removed the 'fig' variable as it's unused

    # Plot scans
    for key, value in scans_idxs.items():
        ax.plot((timesec[value[0]], timesec[value[-1]]), (1, 1), "k")
        ax.text(timesec[value[0]] + (100 * cadence), 1.02, f"{key}", fontsize=20)

    # Plot missing frames
    for key, value in mfram_idxs.items():
        ax.plot(timesec[value[0]], 1, "|r", ms=15)
        ax.plot(timesec[value[-1]], 1, "|r", ms=15)

    # Set plot parameters
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(r"Observation timeline", fontsize=22)
    ax.set_xlabel(r"Time [s]", fontsize=20)
    plt.tight_layout()
    plt.show()

    return scans_idxs, mfram_idxs


############################ SALAT INFO ############################

def info(file: str) -> NoReturn:  # type: ignore
    """
    Name: info
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function computes basic information for an ALMA FITS data set.

    Parameters
    ----------
    file: str
        Path to ALMA FITS data set.

    Returns
    -------
    Displays the information in the terminal.

    Examples
    --------
    >>> import salat
    >>> salat.info(file)

    Modification history:
    ---------------------
    © Guevara Gómez J.C. (RoCS/SolarALMA), July 2021
    """

    print("---------------------------------------------------")
    print("--------------- SALAT INFO part of ----------------")
    print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")

    ############### Reading Header ################
    hdr0 = read_header(file, ALL=True, SILENT=True)

    # Ensure hdr0 is not None before proceeding
    if hdr0 is None:
        print("Error: Header is missing or could not be read.")

    # Safely access header fields; checking if they exist first
    alma_band = int(hdr0.get('INSTRUME', 'N/A')  # type: ignore
                    [-1]) if 'INSTRUME' in hdr0 else 'N/A'  # type: ignore
    obs_date = hdr0.get(  # type: ignore
        'DATE-OBS', 'Unknown')[:10] if 'DATE-OBS' in hdr0 else 'Unknown'  # type: ignore
    alma_proj = hdr0.get(  # type: ignore
        'PROPCODE', 'Unknown') if 'PROPCODE' in hdr0 else 'Unknown'  # type: ignore
    pix_unit = hdr0.get(  # type: ignore
        'BUNIT', 'Unknown') if 'BUNIT' in hdr0 else 'Unknown'  # type: ignore
    pix_size = hdr0.get(  # type: ignore
        'CDELT1A', 'Unknown') if 'CDELT1A' in hdr0 else 'Unknown'  # type: ignore
    beam_mean = float(hdr0.get('SPATRES', 0)  # type: ignore
                      ) if 'SPATRES' in hdr0 else 'N/A'  # type: ignore
    fov_diam = hdr0.get(  # type: ignore
        'EFFDIAM', 'Unknown') if 'EFFDIAM' in hdr0 else 'Unknown'  # type: ignore
    data_min = hdr0.get(  # type: ignore
        'DATAMIN', 'Unknown') if 'DATAMIN' in hdr0 else 'Unknown'  # type: ignore
    data_max = hdr0.get(  # type: ignore
        'DATAMAX', 'Unknown') if 'DATAMAX' in hdr0 else 'Unknown'  # type: ignore

    ############### Printing Information in Terminal ################
    print("\n----------------------------------------------")
    print("| Data feat.: ")
    print("----------------------------------------------")
    print(f"|  ALMA BAND: {alma_band}")
    print(f"|  Obs. Date: {obs_date}")
    print(f"|  ALMA proj: {alma_proj}")
    print(f"|  Pix. Unit: {pix_unit}")
    print(f"|  Pix. Size: {pix_size} arcsec.")
    print(f"|  Beam mean: {beam_mean} arcsec")
    print(f"|  FOV. diam: {fov_diam}")
    print("----------------------------------------------")
    print("| Data range ")
    print("----------------------------------------------")
    print(f"|  Min = {data_min} Kelvin")
    print(f"|  Max = {data_max} Kelvin")
    print("----------------------------------------------\n")


############################ SALAT PLOT MAP ############################


def plot_map(
    almadata: np.ndarray,
    beam: List[np.ndarray],
    pxsize: float,
    cmap: str = "hot",
    average: bool = False,
    timestp: int = 0,
    savepng: bool = False,
    savejpg: bool = False,
    outputpath: str = "./",
) -> plt.Figure:  # type: ignore
    """
    Name: plot_map
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function generates a map plot centered at (0,0) arcsec.

    Parameters
    ----------
    almadata: np.ndarray
        Data array from the user, can be 2D or 3D.
    beam: list of np.ndarray
        List with np.arrays of beam info [bmaj, bmin, bang].
    pxsize: float
        Pixel size in arcseconds.
    cmap: str, optional
        Matplotlib colormap for imshow, default is "hot".
    average: bool, optional
        If True, it plots the average image (default: False).
    timestp: int, optional
        Index of the timestep to plot, default is 0.
    savepng: bool, optional
        If True, saves the image as PNG (default: False).
    savejpg: bool, optional
        If True, saves the image as JPG (default: False).
    outputpath: str, optional
        Path for saving the image, default is current directory "./".

    Returns
    -------
    fig: plt.Figure
        The Matplotlib figure.

    Examples
    --------
    >>> import salat
    >>> salat.plot_map(almadata, beam, pxsize, cmap='hot', average=False,
                       timestp=0, savepng=False, savejpg=False,
                       outputpath="./")

    Modification history:
    ---------------------
    © Guevara Gómez J.C. (RoCS/SolarALMA), July 2021
    """

    print("---------------------------------------------------")
    print("------------ SALAT PLOT MAP part of ---------------")
    print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")

    ############### Getting Image to Plot ################
    almadatashape = np.shape(almadata)

    if len(almadatashape) == 3:
        if average:
            implot = np.nanmean(almadata, axis=0)
            imylen = almadatashape[1] * pxsize
            imxlen = almadatashape[2] * pxsize
            beamval = np.nanmean(beam, axis=1)
        else:
            implot = almadata[timestp].copy()
            imylen = almadatashape[1] * pxsize
            imxlen = almadatashape[2] * pxsize
            beamval = [beam[0][timestp], beam[1][timestp], beam[2][timestp]]
    else:
        implot = almadata.copy()
        imylen = almadatashape[0] * pxsize
        imxlen = almadatashape[1] * pxsize
        beamval = np.nanmean(beam, axis=1)

    bmaj, bmin, bang = beamval

    ############### Plotting ################

    extplot = [-imylen / 2, imylen / 2, -imxlen / 2, imxlen / 2]  # Plot Extent
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 10))
    im1 = ax.imshow(implot, origin="lower", cmap=cmap, extent=extplot)

    # Adding colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im1, cax=cax, orientation="vertical")

    # Beam artist to add to plot
    ell_beam = matplotlib.patches.Ellipse(  # type: ignore
        ((-imylen / 2) + 5, (-imxlen / 2) + 5),
        bmaj,
        bmin,
        angle=bang + 90,
        fc="k",
        ec="b",
    )

    ax.add_patch(ell_beam)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlabel("arcsec", fontsize=20)
    ax.set_ylabel("arcsec", fontsize=20)
    cb.set_label("Temperature [K]", fontsize=20)
    cb.ax.tick_params(labelsize=18)
    plt.tight_layout()

    ############### Saving ################

    if savepng:
        plt.savefig(f"{outputpath}ALMA_map_timestp{timestp:04d}.png", dpi=150)
    if savejpg:
        plt.savefig(f"{outputpath}ALMA_map_timestp{timestp:04d}.jpg", dpi=150)

    return fig


############################ SALAT BEAM STATS ############################


def beam_stats(
    beammajor: np.ndarray,
    beamminor: np.ndarray,
    beamangle: np.ndarray,
    timesec: np.ndarray,
    plot: bool = False,
) -> None:
    """
    Name: beam_stats
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function computes basic stats for the beam.

    Parameters
    ----------
    beammajor: np.ndarray
        Beam major axis array (from salat_read).
    beamminor: np.ndarray
        Beam minor axis array (from salat_read).
    beamangle: np.ndarray
        Beam angle array (from salat_read).
    timesec: np.ndarray
        Array with time in seconds (from salat_read).
    plot: bool, optional
        If True, plots the beam area over time (default: False).

    Returns
    -------
    None: Prints beam statistics in the terminal and optionally plots the beam area.

    Examples
    --------
    >>> import salat
    >>> salat.beam_stats(beammajor, beamminor, beamangle, timesec=timesec, plot=True)

    Modification history:
    ---------------------
    © Guevara Gómez J.C. (RoCS/SolarALMA), July 2021
    """

    print("\n---------------------------------------------------")
    print("------------ SALAT BEAM STATS part of -------------")
    print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")
    print("Computing Statistics\n")
    print("---------------------------------------------------")

    ############### Computing Stats (NaNs are ignored) ################

    beamarea = (beammajor / 2) * (beamminor / 2) * np.pi  # Beam area in sq. arcsec

    # Beam major axis statistics
    minbmaj, maxbmaj = np.nanmin(beammajor), np.nanmax(beammajor)
    meanbmaj, medianbmaj = np.nanmean(beammajor), np.nanmedian(beammajor)
    stdbmaj = np.nanstd(beammajor)

    # Beam minor axis statistics
    minbmin, maxbmin = np.nanmin(beamminor), np.nanmax(beamminor)
    meanbmin, medianbmin = np.nanmean(beamminor), np.nanmedian(beamminor)
    stdbmin = np.nanstd(beamminor)

    # Beam angle statistics
    minbang, maxbang = np.nanmin(beamangle), np.nanmax(beamangle)
    meanbang, medianbang = np.nanmean(beamangle), np.nanmedian(beamangle)
    stdbang = np.nanstd(beamangle)

    # Beam area statistics
    minbarea, maxbarea = np.nanmin(beamarea), np.nanmax(beamarea)
    meanbarea, medianbarea = np.nanmean(beamarea), np.nanmedian(beamarea)
    stdbarea = np.nanstd(beamarea)

    ############### Printing in Terminal ################
    print("\n----------------------------------------------")
    print("|  Beam Statistics: ")
    print("----------------------------------------------")
    print(
        f"|  Min (major, minor, angle, area) = "
        f"{minbmaj, minbmin, minbang, minbarea}"
    )
    print(
        f"|  Max (major, minor, angle, area) = "
        f"{maxbmaj, maxbmin, maxbang, maxbarea}"
    )
    print(
        f"| Mean (major, minor, angle, area) = "
        f"{meanbmaj, meanbmin, meanbang, meanbarea}"
    )
    print(
        f"| Median (major, minor, angle, area) = "
        f"{medianbmaj, medianbmin, medianbang, medianbarea}"
    )
    print(
        f"| Std Dev (major, minor, angle, area) = "
        f"{stdbmaj, stdbmin, stdbang, stdbarea}"
    )
    print("----------------------------------------------\n")

    ############### Plotting Beam Area ################

    if plot:
        # Removed the 'fig' variable as it's unused
        _, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timesec, beamarea, ".k")
        ax.set_title("Beam Area", fontsize=21)
        ax.set_xlabel("Time [s]", fontsize=18)
        ax.set_ylabel("Area [arcsec$^2$]", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        plt.tight_layout()
        plt.show()


############################ SALAT CONTRAST ############################

def contrast(
    almadata: np.ndarray, timesec: np.ndarray,
    side: int = 5,
        show_best: bool = False) -> np.ndarray:
    """
    Name: contrast
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function calculates RMS contrast and sorts the best frames.

    Parameters
    ----------
    almadata: np.ndarray
        Data array from user, can be 2D or 3D from salat_read.
    timesec: np.ndarray
        Time array in seconds from salat_read.
    side: int, optional
        Number of pixels to exclude from the sides of the field of view
        before calculating the mean intensity and RMS contrast (default: 5).
    show_best: bool, optional
        If True, shows the location of the best frame (highest RMS contrast)
        on the plot (default: False).

    Returns
    -------
    bestframes: np.ndarray
        Indexes of the best frames sorted by highest RMS contrast.
    """
    print("\n---------------------------------------------------")
    print("------------ SALAT CONTRAST part of -------------")
    print("-- Solar Alma Library of Auxiliary Tools (SALAT) --\n")

    ############### Calculate RMS Contrast ################
    meanTbr = np.array([np.nanmean(item)
                       for item in almadata[:, side:-side, side:-side]])
    rmsCont = np.array([np.nanstd(item) / np.nanmean(item)
                       for item in almadata[:, side:-side, side:-side]]) * 100

    bestframes = np.argsort(rmsCont)[::-1]

    ############### Plot Best Frame if Requested ################
    if show_best:
        ax = plt.subplots(ncols=1, nrows=2, sharex=True,
                          figsize=(12, 10))[1]  # Only unpack ax
        ax[0].plot(timesec, meanTbr, "--.k")
        ax[1].plot(timesec, rmsCont, "--.k")
        ax[0].tick_params(axis="both", which="major", labelsize=18)
        ax[1].tick_params(axis="both", which="major", labelsize=18)
        ax[0].set_title(f"Best frame = {bestframes[0]}", fontsize=24)
        ax[1].set_xlabel("Time [sec]", fontsize=20)
        ax[0].set_ylabel("Temperature [K]", fontsize=20)
        ax[1].set_ylabel("% RMS Contrast", fontsize=20)
        ax[0].axvline(x=timesec[bestframes[0]], color="red", linestyle=":")
        ax[1].axvline(x=timesec[bestframes[0]], color="red", linestyle=":")
        plt.tight_layout()
        plt.show()

    return bestframes


############################ SALAT CONVOLVE BEAM ############################


def convolve_beam(
    data: np.ndarray, beam: list[np.ndarray], pxsize: float
) -> np.ndarray:
    """
    Name: convolve_beam
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: Convolve a specified synthetic beam (from an ALMA observation) to
    a user-provided map (e.g., from a simulation or observations with other
    instruments).

    Parameters
    ----------
    data: np.ndarray
        Data array from user, only 2D, could be a Bifrost snapshot.
    beam: list of np.ndarray
        List with np.arrays of beam info [bmaj, bmin, bang].
    pxsize: float
        Pixel size of the data to convolve in arcsec/px.

    Returns
    -------
    data_convolved: np.ndarray
        Data convolved with beam.

    Examples
    --------
    >>> import salat
    >>> filebifrost = path_folder + "bifrost_b3_frame400.fits"
    >>> bifrostdata = fits.open(filebifrost)[0].data
    >>> pxsizebifrost = 0.06  # Pixel size in arcsec/px
    >>> bifrostconv = salat.convolve_beam(bifrostdata,
                                          [beammajor1, beamminor1, beamangle1],
                                          pxsize=pxsizebifrost)

    Modification history:
    ---------------------
    © Guevara Gómez J.C. (RoCS/SolarALMA), July 2021
    """

    print("\n------------------------------------------------------")
    print("------------ SALAT CONVOLVE BEAM part of -------------")
    print("---- Solar Alma Library of Auxiliary Tools (SALAT) ----")
    print("\nFor the input data, NANs are not properly handled.")
    print("Please use fill_nans parameter when loading FITS.\n")
    print("------------------------------------------------------")

    # Compute the beam kernel using the mean beam parameters
    beam_kernel = beam_kernel_calculator(
        np.nanmean(beam[0]), np.nanmean(beam[1]), np.nanmean(  # type: ignore
            beam[2]), pxsize  # type: ignore
    )

    # Convolve data with beam kernel
    data_convolved = ndimage.convolve(data, beam_kernel)

    return data_convolved


def beam_kernel_calculator(
    bmaj_obs: float, bmin_obs: float, bpan_obs: float, pxsz: float
) -> np.ndarray:
    """Calculate the beam kernel array using the observed beam parameters to be used for
    convolving the data.

    Parameters
    ----------
    bmaj_obs: float
        Major axis of the beam in arcseconds.
    bmin_obs: float
        Minor axis of the beam in arcseconds.
    bpan_obs: float
        Beam position angle in degrees.
    pxsz: float
        Pixel size in arcseconds per pixel.

    Returns
    -------
    beam_kernel: np.ndarray
        The beam kernel array for convolution.
    """

    # Create beam using observed parameters
    beam = rb.Beam(bmaj_obs * u.arcsec, bmin_obs * u.arcsec,  # type: ignore
                   bpan_obs * u.deg)  # type: ignore

    # Convert beam to kernel with the given pixel scale
    beam_kernel = np.asarray(beam.as_kernel(pixscale=pxsz * u.arcsec))  # type: ignore

    return beam_kernel


############################ SALAT PREP DATA ############################


def prep_data(file: str, savedir: str = "./") -> str:
    """
    Name: prep_data
        part of -- Solar Alma Library of Auxiliary Tools (SALAT) --

    Purpose: This function prepares a FITS cube to be used in FITS readers
    like CARTA by reducing its dimensions.

    Parameters
    ----------
    file: str
        Original FITS file to be reduced.
    savedir: str, optional
        Output directory for the new FITS file (default: "./").

    Returns
    -------
    savefile: str
        Path to the newly saved FITS file.

    Examples
    --------
    >>> import salat
    >>> salat.prep_data(file)

    Modification history:
    ---------------------
    © Guevara Gómez J.C. (RoCS/SolarALMA), August 2021
    """

    print("\n------------------------------------------------------")
    print("------------ SALAT PREP DATA part of -------------")
    print("---- Solar Alma Library of Auxiliary Tools (SALAT) ----\n")

    ############### Reading and Reducing the Cube ################

    cubedata = fits.open(file)  # Cube data dimensions [t, S, f, x, y]
    imcube = cubedata[0].data[:, 0, 0, :, :].copy()  # type: ignore
    # Generate the output filename
    outfile = os.path.basename(file).replace(".fits", "_modified-dimension.fits")

    ############### Writing the New FITS File ################

    new_hdul = fits.HDUList([fits.PrimaryHDU(data=imcube)])
    output_path = os.path.join(savedir, outfile)
    new_hdul.writeto(output_path, overwrite=True)

    print("Done!")
    return output_path

# Last Modified on 2023-10-13 by Cossas

import os, re, glob

import numpy as np
import pandas as pd

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from photutils import detect_sources, detect_threshold

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord

from shapely.geometry import Point
from shapely.geometry import Polygon as shapely_polygon

import datetime


def fits_reader(filename, wisp=False, debug=False):
    """
    Read a FITS file.

    Args:
        filename (str): The name of the FITS file.
        debug (bool, optional): Print the details for debugging. Defaults to False.
        Note that the detector_number must be specified on the "fourth" suffix of the filename.
    Return:
        Data (dictionary): A dictionary that contains every image in the FITS file.
    """
    data = {}
    if wisp:
        detector_number = filename.split('_')[4][:-5]
    else:
        detector_number = filename.split('_')[3].split('/')[0][3:]

    if debug:
        print(f"Detector reading: {detector_number}")

    data[f'{detector_number}'] = {}
    
    with fits.open(filename, memmap=False) as hdul: 
        for hdu in hdul:
            extname = hdu.header.get('EXTNAME', None)
            if not extname:
                if debug:
                    print("No EXTNAME found for this HDU")
            else:
                if debug:
                    print(f"HDU extension name: {extname}")
            
            # check if the hud object is an image
            if type(hdu) == fits.hdu.image.ImageHDU:
                data[f'{detector_number}'][f'{extname}'] = hdu.data

    hdul.close()
    return data

def image_visualization(data, title=None, auto_color=True, show=True,
                        color_style='jet', save=False, scale_data=None,
                        img_dpi=150, vmin_value=20, vmax_value=97,
                        output_path=None, share_scale=False, wcs=None,
                        sup_title=None):
    """
    Function to visualize given ndarray

    Args:
        data(list): The 2D image.
        title(list of str): The titles of the ndarrays.
        auto_color(bool): If set True, use 'jet' as the colorbar, else, gray scale.
        color_style(str): Any color bar args of plt.
        save(bool): If set True, save the plot into png file.
        output_path(str): Output path for the png files.
        share_scale(bool): If set True, share the same scale for all images.
        wcs(astropy.wcs): World Coordinate System object for projection.

    Returns:
        None
    """
    if not isinstance(data, list):
        data = [data]
    
    if title:
        if not isinstance(title, list):
            title = [title]

    num_images = len(data)
    
    # Determine the subplot grid dimensions
    num_rows = int(np.floor(np.sqrt(num_images)))
    num_cols = int(np.floor(num_images / num_rows))
    
    if num_rows * num_cols < num_images:
        num_cols += 1

    fig = plt.figure(figsize=(12,8), dpi=img_dpi)

    if auto_color:
        for i, d in enumerate(data, start=1):
            ax = fig.add_subplot(num_rows, num_cols, i)
            d = np.copy(d)

            if not share_scale:
                vmin=np.nanpercentile(d.flatten(), vmin_value)
                vmax=np.nanpercentile(d.flatten(), vmax_value)
                im = ax.imshow(d, cmap=color_style, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, fraction=0.046, pad=0.04)

            else:
                if scale_data is not None:
                    vmin=np.nanpercentile(scale_data.flatten(), vmin_value)
                    vmax=np.nanpercentile(scale_data.flatten(), vmax_value)
                else:
                    vmin=np.nanpercentile(data[0].flatten(), vmin_value)
                    vmax=np.nanpercentile(data[0].flatten(), vmax_value)

                im = ax.imshow(d, cmap=color_style, vmin=vmin, vmax=vmax)

            if title:
                ax.set_title(title[i-1])
                ax.title.set_size(24)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Adding colorbar for shared scale
        if share_scale and im:
            # fig.colorbar(im, ax=np.array(fig.axes).flatten()[-1], orientation='vertical', fraction=0.046, pad=0.04)
            cbar = fig.colorbar(im, ax=fig.axes, orientation='horizontal', fraction=0.046, pad=0.04)

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    tick_locations = [0, 0.25, 0.5, 0.75, 1]
    tick_locations = [vmin + a * (vmax-vmin) for a in tick_locations]
    cbar.set_ticks(tick_locations)

    tick_labels = [float('%.3f'%(a)) for a in tick_locations]
    cbar.set_ticklabels(tick_labels)

    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(16)

    if sup_title:
        plt.suptitle(sup_title)

    if show:
        plt.show()

    if save:
        if output_path:
            print(f"Saving output png to: {output_path}")
            plt.savefig(f"{output_path}")
        else:
            print(f"Saving output png to: /mnt/C/JWST/COSMOS/NIRCAM/PNG/")
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(f"/mnt/C/JWST/COSMOS/NIRCAM/PNG/Output_{timestamp}.png")

def image_histogram(data, title=None, bins=np.logspace(-5, 2, 100),
                    share=False, alpha=0.8,
                    x_log=False, y_log=False):
    """
    Function to plot the histogram as the function of flux of given ndarray

    Args:
        data (numpy.array): The 2D image.
        title (list of str): Titles for each image.
        share (boolean): Set to True if you want to overplot each histogram.
        alpha (float): alpha for histogram.
        
    Returns:
        None
    """
    if not isinstance(data, list):
        data = [data]
    
    if title:
        if not isinstance(title, list):
            title = [title]

    num_images = len(data)
    
    # Determine the subplot grid dimensions
    num_rows = int(np.floor(np.sqrt(num_images)))
    num_cols = int(np.floor(num_images / num_rows))

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot()
    
    for i, d in enumerate(data, start=1):
        if not share:
            ax = fig.add_subplot(num_rows, num_cols, i)

        else:
            ax.hist(d.flatten(), bins=bins,
                    alpha=alpha, histtype='step', lw=1.5,
                    label=title[i-1])
            
            if x_log:
                ax.set_xscale('log')
            
            if y_log:
                ax.set_yscale('log')

        ax.legend()

    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def load_fits(path):
    with fits.open(path) as hdul: 
        image = hdul[1].data
        error = hdul[2].data
        wcs = WCS(hdul[1].header)
    hdul.close()
    return image, error, wcs 

def tiered_source_detection(image, sigma_values, snr=3.5):
    """
    Source Extraction method for tired sigma values, SNR is set to 3.5 by default.
    Modified algorithm adapted from CEERS paper.
    The Gaussian kernel size is determined by the 4*sigma_values + 1.

    Args:
        image(2d array): The 2D image contain source information.
        sigma_values(list, float): sigma values threshold for detection.
        snr (float): SNR threshold for detection.
    
    Returns:
        detected_sources(2D array): Masked array containing source information.
    """
    detected_sources = np.zeros_like(image).astype(bool)
    
    for sigma in sigma_values:
        kernel = Gaussian2DKernel(x_stddev=sigma,
                                  x_size=sigma*4 + 1,
                                  y_size=sigma*4 + 1
                                 )
        # modified, reshape the sigma = 25 gaussian into the same shape as sigma = 15 gaussian
        smoothed_image = convolve_fft(image, kernel)
        threshold = detect_threshold(smoothed_image, nsigma=snr)
        sources = detect_sources(smoothed_image, threshold, npixels=5)
        
        if sources:
            detected_sources = np.logical_or(detected_sources, sources.data.astype(bool))

    return detected_sources

def perform_fft(data):
    # Perform FFT
    fft_result = np.fft.fft(data)
    
    return fft_result

def plot_fft(data, label=None, normalized=True, 
             alpha=0.5, xlim=None, ylim=None):

    if not isinstance(data, list):
        data = [data]
    
    fig, ax = plt.subplots(figsize=(16,8))

    for d in range(len(data)):
        # Perform FFT
        result = perform_fft(data[d])
        if normalized :
            magnitude = np.abs(result)/np.max(np.abs(result))
        else:
            magnitude = np.abs(result)
        # Generate frequency bins
        freq = np.fft.fftfreq(len(result))
        # Plot the magnitude spectrum
        if label:
            ax.plot(freq, magnitude, label=label[d], alpha=alpha)
        else:
            ax.plot(freq, magnitude, alpha=alpha)

    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid(True)
    ax.legend(loc=8)
    fig.show()

def gaussian(x, mu, sigma):
    # Define the Gaussian function
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def remove_file_suffix(filename) -> str:
    """
    Remove the file suffix for a given filename.
    The suffix naming rules are:
        Uncalibrated raw input                          uncal
        Corrected ramp data                             ramp
        Corrected countrate image                       rate
        Corrected countrate per integration             rateints
        Background-subtracted image                     bsub
        Per integration background-subtracted image     bsubints
        Calibrated image                                cal
        Calibrated per integration images               calints
        CR-flagged image                                crf
        CR-flagged per integration images               crfints
        Resampled 2D image                              i2d
        Source catalog                                  cat
        Segmentation map                                segm
    """

    suffix = ['uncal', 'ramp', 'rate', 'rateints', 'bsub', 'bsubints', 'cor_wsp', 'cor_cal',
              'cal', 'calints', 'crf', 'crfints', 'i2d', 'cat', 'segm']
    
    for string in suffix:
        pattern = re.compile(rf'_{string}\.fits$')
        new_filename, n = pattern.subn('', filename)
        if n > 0:
            # print(f'Suffix found and removed: {string}')
            # print('New filename:', new_filename)
            break

    return new_filename

def mask_sources(img_data, sigma_values=[5, 2], snr=3, nan=False):
    sources = tiered_source_detection(img_data, sigma_values=sigma_values, snr=snr)
    mask = np.logical_not(sources).astype(float)
    if nan:
        mask = np.where(mask==0, np.nan, mask) 
    masked_image = np.multiply(mask, img_data)

    return masked_image

def record_and_save_data(path, fitsname, corrected_image, pedestal, suffix='cor_cal'):
    if suffix in ['bkg_sub', 'bri_sub']:
        hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_cor_wsp.fits")
        
    elif suffix in ['cor']:
        hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_sbal_0_rampfitstep.fits")
        
    else:
        hdul = fits.open(f"{path}/{remove_file_suffix(fitsname)}_cor_cal.fits")

    if pedestal is not None:
        hdul[1].data = np.array(corrected_image-pedestal)

    else:
        hdul[1].data = np.array(corrected_image)
        
    hdul[1].header['PED_VAL'] = pedestal
    hdul.writeto(f"{path}/{remove_file_suffix(fitsname)}_{suffix}.fits", overwrite=True)
    hdul.close()

def calculate_pedestal(image):
    med = np.nanmedian(image.flatten())
    return med

def sigma_clip_replace_with_median(arr, sigma=3):
    """
    Perform sigma clipping on a 2D array, replacing clipped data with the median,
    and leaving NaN values unchanged.

    :param arr: 2D NumPy array
    :param sigma: Number of standard deviations to use for clipping
    :return: 2D NumPy array with clipped values replaced
    """
    # Making a copy of the array to avoid changing the original data
    arr_copy = np.copy(arr)
    
    # Calculating the median and standard deviation, ignoring NaN values
    median = np.nanmedian(arr_copy)
    std = np.nanstd(arr_copy)
    
    # Identifying outliers
    lower_bound = median - sigma * std
    upper_bound = median + sigma * std
    outliers = np.logical_or(arr_copy < lower_bound, arr_copy > upper_bound)
    
    # Replacing outliers with the median
    arr_copy[outliers] = median
    
    return arr_copy

def read_MJD_times(fitsname):
    return fits.open(fitsname)[1].header['MJD-AVG']

def MJy_Sr_2_AB(F, header): 
    """
    Based on:
        https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints
    """
    zp_AB = -6.10 - 2.5 * np.log10(header['PIXAR_SR'])
    magAB = zp_AB - 2.5 * np.log10(F)
    return magAB

def MJy_Sr_2_mJy(F, header): 
    flux = F * header['PIXAR_SR'] # [MJy]
    flux *= 10**9 # [mJy]
    return flux

def AB_2_mJy(AB_mag): 
    F  = 10**(23-(AB_mag+48.6)/2.5) # [Jy]
    F *= 10**3 # [mJy]
    return F

def mJy_2_AB(F): 
    AB = 2.5*(23-np.log10(F/10**6))-48.6
    return AB

def AB_2_uJy(AB_mag): 
    F  = 10**(23-(AB_mag+48.6)/2.5) # [Jy]
    F *= 10**6 # [uJy]
    return F

def find_ra_dec_columns(df):
    """
    Search for columns in the dataframe that represent Right Ascension (ra) and Declination (dec).
    This function only support catalog from SEX and Photutils.
    Returns a tuple of column names if found, otherwise returns (None, None).
    
    :param df: pandas DataFrame to search
    :return: Tuple containing the names of the ra and dec columns
    """
    ra_column, dec_column = None, None
    
    for column in df.columns:
        if 'ra' in column.lower():
            ra_column = column
        if 'dec' in column.lower():
            dec_column = column
        if 'alpha_j2000' in column.lower():
            ra_column = column
        if 'delta_j2000' in column.lower():
            dec_column = column

        if ra_column is not None and dec_column is not None:
            break
            
    return ra_column, dec_column

def crossmatch(df1, df2, max_sep, ra_dec_only=False, include_unmatched=False, fast=False, verbose=True): 
    """
    Crossmatches two catalogs of celestial coordinates.
    ------------------------------------------------------------------
    Args:
        csv1 (pd.DataFrame): Phot's cat, if not specify.
        csv2 (pd.DataFrame): SJ's cat, if not specify.
        max_sep (float): Maximum separation for matching in arcseconds.
        ra_dec_only (bool): Control the return only contains ra, dec columns.
        include_unmatched (bool): Include un-matched sources from both catalog.
        fast (bool): Use only 50% of the input catalog to run with crossmatching algorithm.
    
    Returns:
        If return_cat == True:
            Return the matched catalog.
        else:
            Return the matched ra, dec columns and separations.
            
        sep (bool): Control whether the separations is returned in the matched catalog.
        include_unmatched (bool): Include un-matched sources in the first catalog.
    """

    # Search for ra, dec columns or similar coordinates in the catalog
    ra_name_1, dec_name_1 = find_ra_dec_columns(df1)
    ra_name_2, dec_name_2 = find_ra_dec_columns(df2)
    
    ra1, dec1 = df1[ra_name_1], df1[dec_name_1]
    ra2, dec2 = df2[ra_name_2], df2[dec_name_2]
    # print(ra1, ra2)
    
    # Convert catalogs to SkyCoord objects
    coords_cat1 = SkyCoord(ra1*u.deg, dec1*u.deg) # type: ignore
    coords_cat2 = SkyCoord(ra2*u.deg, dec2*u.deg) # type: ignore

    # Perform the crossmatching
    if verbose:
        print(f"Start crossmatching......")
    idx_df1, idx_df2, sep2d, _ = coords_cat1.search_around_sky(coords_cat2, max_sep*u.arcsec)
    if verbose:
        print(f"Crossmatching done......")

    
    if include_unmatched:
        # Creating a DataFrame from the matched rows
        matched_rows = [pd.concat([df2.iloc[i], df1.iloc[j]], axis=0) 
                            for i, j in zip(idx_df1, idx_df2)]
        matched_catalog = pd.DataFrame(matched_rows)
        
        # Creating a DataFrame from the un-matched rows
        unmatched_indices_df1 =  np.setdiff1d(np.arange(len(df1)), idx_df2)
        unmatched_catalog_df1 = df1.iloc[unmatched_indices_df1]
        
        unmatched_indices_df2 =  np.setdiff1d(np.arange(len(df2)), idx_df1)
        unmatched_catalog_df2 = df2.iloc[unmatched_indices_df2]
        
        # Create a DataFrame with NaNs for the df2 part
        nan_rows_df1 = pd.DataFrame(np.nan, index=np.arange(len(unmatched_catalog_df1)), columns=df2.columns)
        nan_rows_df2 = pd.DataFrame(np.nan, index=np.arange(len(unmatched_catalog_df2)), columns=df1.columns)

        unmatched_df_1 = pd.concat([unmatched_catalog_df1.reset_index(drop=True), nan_rows_df1], axis=1)
        unmatched_df_2 = pd.concat([unmatched_catalog_df2.reset_index(drop=True), nan_rows_df2], axis=1)

        # Concatenate matched and unmatched DataFrames
        final_df = pd.concat([matched_catalog, unmatched_df_1], ignore_index=True)
        final_df = pd.concat([final_df.reset_index(drop=True), unmatched_df_2.reset_index(drop=True)], ignore_index=True)
        
        return final_df

    else:
        # Create a new DataFrame from df2 rows indicated by idx_df1
        new_df2 = df2.iloc[idx_df1].reset_index(drop=True)

        # Similarly, select rows from df1 using idx_df2
        new_df1 = df1.iloc[idx_df2].reset_index(drop=True)

        # Create a new DataFrame that merges new_df2 and new_df1 side by side
        matched_catalog = pd.concat([new_df2, new_df1], axis=1)

        # Add additional columns, assuming sep2d is a list of separation values
        matched_catalog['RA'] = ra1.iloc[idx_df2].values
        matched_catalog['DEC'] = dec1.iloc[idx_df2].values
        matched_catalog['Alpha_J2000'] = ra2.iloc[idx_df1].values
        matched_catalog['Delta_J2000'] = dec2.iloc[idx_df1].values
        matched_catalog['Separation'] = [sep.arcsec for sep in sep2d]
        
        return matched_catalog
        
def read_catalog_by_obs(obs, _filter):
    return pd.read_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{_filter}/jw01727-o{obs}_t001_NIRCAM-clear-{_filter}-full_i2d.csv", index_col='label')

def ra_dec_plot(max_sep, _filter="F115W", csv1=None, csv2=None, treat_sep=None, save='False', show=True):
    """
    Contour plot for the distribution of RA, Dec from crossmatching 
    two catalogs. Design for NIRCam data.
    -----------------------------------------------------------
    Args:
        max_sep(float): The maximum crossmatching radius.
        _filter(str, opt.): The name of the NIRCam filters.
        csv1(pd.DataFrame, optional):
            If not specified, the function will use from Photoutils catalog in COSMOS-Webb field.
        csv2(pd.DataFrame, optional):
            If not specified, the function will use from SJ's catalog in COSMOS-Webb field.
        treat_sep(list[str], opt.): The desired observation numbers for the crossmatching.
        save(bool/str): Filename to save, default path is `/mnt/C/JWST/COSMOS/NIRCAM/catalog/`.
        
    Returns:
        Separation(float): The average of the separation of matched sources.
        max_ra(float): The angular offset of RA (csv1-csv2).
        max_dec(float): The angular offset of DEC (csv1-csv2).
    """
    
    if csv1 is None:
        csv1 = pd.read_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{_filter}/COSMOS_JWST_{_filter}.csv") # cosmos_main_csv
        
    if csv2 is None:
        csv2 = pd.read_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/SJ/COSMOS_o043-048_F115W.csv")        # SJ's cat
        
    if treat_sep:
        sep_cat = pd.concat([read_catalog_by_obs(i, _filter) for i in treat_sep], ignore_index=True)
        matched = crossmatch(sep_cat, csv2, max_sep)
        
    else:
        matched = crossmatch(csv1, csv2, max_sep, ra_dec_only=True)
    
    first_ra = matched["RA"]
    first_dec = matched["DEC"]
    second_ra = matched["Alpha_J2000"]
    second_dec = matched["Delta_J2000"]
    
    del_ra = (first_ra - second_ra) * np.cos(np.radians(0.5 * (first_dec + second_dec))) * 3600000 # [mas]
    del_dec = (first_dec - second_dec) * 3600000  # [mas]
    
    try:
        # Creating a Gaussian kernel density estimate
        kde = gaussian_kde([del_ra, del_dec])
        x_kde, y_kde = np.mgrid[del_ra.min():del_ra.max():100j, del_dec.min():del_dec.max():100j]
        z_kde = kde(np.vstack([x_kde.flatten(), y_kde.flatten()]))
        max_value_index = np.argmax(z_kde)
        max_indices = np.unravel_index(max_value_index, x_kde.shape)

        max_ra = np.round(x_kde[max_indices], 2)
        max_dec = np.round(y_kde[max_indices], 2)
            
        if show:
            fig = plt.figure(figsize=(8, 8))    
            # Scatter plot on bottom left with larger aspect
            ax_scatter = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
            # ax_scatter.scatter(del_ra, del_dec, marker='x')
            
            ax_scatter.contour(x_kde, y_kde, z_kde.reshape(x_kde.shape), colors='black')
            ax_scatter.contourf(x_kde, y_kde, z_kde.reshape(x_kde.shape), cmap='viridis')
            
            ax_scatter.set_xlabel('$\Delta$RA [mas]', fontsize=20)
            ax_scatter.set_ylabel('$\Delta$Dec [mas]', fontsize=20)
            
            # Lines & indicators for medians
            # ax_scatter.axvline(median_del_ra, ls='--', color='yellow')
            # ax_scatter.axhline(median_del_dec, ls='--', color='yellow')
            # ax_scatter.text(median_del_ra, ax_scatter.get_ylim()[1]-3,
            #                 f' Median RA: {median_del_ra}', 
            #                 color='yellow', 
            #                 verticalalignment='top', 
            #                 horizontalalignment='right', 
            #                 fontsize=12)
            # ax_scatter.text(ax_scatter.get_xlim()[1], median_del_dec, 
            #                 f' Median Dec: {median_del_dec}', 
            #                 color='yellow', 
            #                 verticalalignment='bottom', 
            #                 horizontalalignment='right',
            #                 rotation=90, fontsize=12)
            
            # Lines & indicators for highest points
            ax_scatter.axvline(max_ra, ls='--', color='orange')
            ax_scatter.axhline(max_dec, ls='--', color='orange')
            ax_scatter.text(max_ra, ax_scatter.get_ylim()[0],
                            f' Highest RA: {max_ra} ', 
                            color='orange', 
                            verticalalignment='bottom', 
                            horizontalalignment='right', 
                            fontsize=12)
            ax_scatter.text(ax_scatter.get_xlim()[0]+3,  max_dec, 
                            f' Highest Dec: {max_dec} ', 
                            color='orange', 
                            verticalalignment='bottom', 
                            horizontalalignment='left',
                            rotation=90, fontsize=12)
            
            # Horizontal histogram on bottom right
            ax_histx = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
            ax_histx.hist(del_dec, bins=60, orientation='horizontal')
            # ax_histx.axhline(median_del_dec + sigma_del_dec, ls='-', color='red')
            # ax_histx.axhline(median_del_dec - sigma_del_dec, ls='-', color='red')
            # ax_histx.set_title('Dec')

            # Vertical histogram on top left
            ax_histy = plt.subplot2grid((4, 4), (0, 0), colspan=3)
            ax_histy.hist(del_ra, bins=60)
            # ax_histy.axvline(median_del_ra + sigma_del_ra, ls='-', color='red')
            # ax_histy.axvline(median_del_ra - sigma_del_ra, ls='-', color='red')
            # ax_histy.set_title('RA')
            
            # Hide the tiscks of shared axes
            ax_histx.set_yticklabels([])
            ax_histy.set_xticklabels([])
            
            if treat_sep:
                fig.suptitle("Observation(s): " + " ,".join(treat_sep), fontsize=28)
                
            # Adjusting layout
            ax_histx.tick_params(axis='both', which='major', labelsize=14)
            ax_histy.tick_params(axis='both', which='major', labelsize=14)
            ax_scatter.tick_params(axis='both', which='major', labelsize=14)
            plt.tight_layout()
            # plt.legend()

            if save:
                    plt.savefig("/mnt/C/JWST/COSMOS/NIRCAM/catalog/Obs_" + ' ,'.join(treat_sep) + ".png")
            elif isinstance(save, str):
                    plt.savefig(f"./{save}.png")
            else:
                pass
            
            plt.show()
            
        return np.nanmean(matched['Separation']), max_ra, max_dec
    
    except ValueError:
        print("No matched sources found!")
        
        return False, 0.0, 0.0
        
def overplot(_filter, read=False, find_offset=True, sep=0.15):
    # Creating a figure
    fig, ax = plt.subplots(figsize=(8, 8))

    cosmos_main_csv = pd.read_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{_filter}/COSMOS_JWST_{_filter}.csv")
    SJ_cat = pd.read_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/SJ/COSMOS_o043-048_F115W.csv")
    max_sep = sep  # Maximum separation in arcseconds

    if not read:
        matched = pd.read_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{_filter}/Phot_SJ_matched_F115W.csv")
        
    else:
        #match catalog
        matched = crossmatch(cosmos_main_csv, SJ_cat, max_sep)
        matched.to_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{_filter}/Phot_SJ_matched_F115W.csv")    

    if find_offset:
        fig, ax = plt.subplots(figsize=(8,6), dpi=120)
        for offset_csv in glob.glob(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{_filter}/*o04*full_i2d.csv"):
            obs = offset_csv.split("/")[-1][9:12]
            matched = crossmatch(pd.read_csv(offset_csv), SJ_cat, max_sep)
            ax.hist((matched['RA'] - matched['Ra']) * np.cos(np.radians(matched['DEC'])) * 3600000, histtype='step', label=str(obs))
        plt.legend()
        plt.show()
    
    del_ra = (matched['RA'] - matched['Ra']) * np.cos(np.radians(matched['DEC'])) * 3600000
    del_dec = (matched['DEC'] - matched['Dec']) * 3600000
    
    matched_50 = matched[matched['Separation']*1000<=50]
    matched_250 = matched[(50<matched['Separation']*1000) & (matched['Separation']*1000<=250)]
    matched_500 = matched[(250<matched['Separation']*1000) & (matched['Separation']*1000<=500)]
    # print(len(matched_50), len(matched_250), len(matched_500))
    
    # ax.scatter(matched_50['RA'], matched_50['DEC'], color='red', s=1.2)   
    # ax.scatter(matched_250['RA'], matched_250['DEC'], color='black', s=0.8) 
    # ax.scatter(matched_500['RA'], matched_500['DEC'], color='green', s=1.4)   

    
    # ax.scatter(matched['RA'], matched['DEC'], color='red', s=1.2)   # phot
    # ax.scatter(matched['Ra'], matched['Dec'], color='black', s=0.6) # sj
    
    ax.scatter(cosmos_main_csv['RA'], cosmos_main_csv['DEC'], color='blue', s=0.4, marker='^')    # phot
    ax.scatter(SJ_cat['Ra'], SJ_cat['Dec'], color='black', s=2, marker='x')                       # sj
    
    plt.xlim(min(SJ_cat['Ra']), max(SJ_cat['Ra']))
    plt.ylim(min(SJ_cat['Dec']), max(SJ_cat['Dec']))
             
    # Adjusting layout
    plt.tight_layout()
    plt.legend()
    
def COSMOS_crossmatch(_filter: str, obs_num: list, save_intermediate=False):
    C_2020_fits = fits.open(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/official/COSMOS2020_CLASSIC_R1_v2.2_p3.fits")
    COSMOS_2020 = pd.DataFrame(C_2020_fits[1].data)

    for obs in obs_num:
        JWST_cat = pd.read_csv(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{_filter}/jw01727-o{obs}_t001_NIRCAM-clear-{_filter}-full_i2d.csv")
        _ra , _dec = find_ra_dec_columns(JWST_cat)
        COSMOS2020_Webb = COSMOS_2020[((COSMOS_2020['ALPHA_J2000'] <= max(_ra)) & (COSMOS_2020['ALPHA_J2000'] >= min(_ra))) &
                                      ((COSMOS_2020['DELTA_J2000'] <= max(_dec)) & (COSMOS_2020['DELTA_J2000'] >= min(_dec)))]
        temp_cat = crossmatch(JWST_cat, COSMOS2020_Webb, max_sep=0.5, return_cat=True)
        if isinstance(save_intermediate, str):
            temp_cat.to_csv(save_intermediate + f"_{obs}.csv")

def scatter_plot(data,
                 hline=None, vline=None,
                 color='Black', styles=['-', '-.', '--'],
                 alpha=0.5,
                 size=(8, 4.5),
                 annotate=False, 
                 title=None, 
                 xlabel=None,
                 ylabel=None,
                 save=True, 
                 background_color='white'):
    """
    Args:
        data(list, tuple, list): List of tuples containing the x,y data.
    """
    
    if not isinstance(data, list):
        data = [data]
        
    num_images = len(data)
    
    # Determine the subplot grid dimensions
    num_rows = int(np.floor(np.sqrt(num_images)))
    num_cols = int(np.floor(num_images / num_rows))
    
    if num_rows * num_cols < num_images:
        num_cols += 1

    fig = plt.figure(figsize=size, dpi=120)
    axes =[]
    
    for i, d in enumerate(data, start=1):
        # scatter plot
        ax = fig.add_subplot(num_rows, num_cols, i)
        ax.scatter(data[i-1][0], data[i-1][1], marker='o', s=4, c=color, alpha=alpha)
        
        # Setup the axes labels
        if xlabel:
            ax.set_xlabel(xlabel[i-1], fontsize=20)
        else:
            ax.set_xlabel("Please change afterwards", fontsize=20)
            
        if ylabel:
            ax.set_ylabel(ylabel[i-1], fontsize=20)
        else:
            ax.set_ylabel("Please change afterwards", fontsize=20)
            
        # Vertical/horizontal lines for visual aids
        if isinstance(vline,list):
            for value in vline:
                ax.axvline(value, ls='--', color='black')
                
        if isinstance(hline,list):
            for value in hline:
                ax.axhline(value, ls='--', color='black')
        
        # Annotate the lines, datapoints, etc. on the plot
        ## For vertical texts:
        if annotate:
            # ax.text(median_del_ra, ax_scatter.get_ylim()[1]-3, f' Median RA: {median_del_ra}', 
            #         verticalalignment='top', horizontalalignment='right', fontsize=12)
            pass
            
        ## For Horizontal texts:
        if annotate:
            # ax.text(ax_scatter.get_xlim()[1], median_del_dec, f' Median Dec: {median_del_dec}', 
            #         verticalalignment='bottom', horizontalalignment='right', rotation=90, fontsize=12)
            pass

        if title:
            ax.set_title(title[i-1], fontsize=28)
        else: 
            ax.set_title("Basic scatter plot", fontsize=28)
            
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        axes.append(ax)
    
    # Adjusting layout
    plt.tight_layout()
    # plt.legend()
    
    if isinstance(save, str):
        plt.savefig(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{save}.png", facecolor=background_color)
        
    elif save:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{timestamp}.png", facecolor=background_color)
        
    return fig, axes

def read_header_by_obs(obs, _filter):
    header_path = f"/mnt/C/JWST/COSMOS/NIRCAM/Reduced/{_filter}/fits/jw01727-o{obs}_t001_NIRCAM-clear-{_filter}-full_i2d.fits"
    return fits.open(header_path)[1].header

def add_prefix_columns(df, _filter):
    new_column_names = {col: f"{_filter}_" + col for col in df.columns}
    df.rename(columns=new_column_names, inplace=True)

def find_co_obs():
    obs_115 = set([file.split("/")[-1][9:12] for file in sorted(glob.glob("/mnt/C/JWST/COSMOS/NIRCAM/catalog/F115W/*NIRCAM*.csv"))])
    obs_150 = set([file.split("/")[-1][9:12] for file in sorted(glob.glob("/mnt/C/JWST/COSMOS/NIRCAM/catalog/F150W/*NIRCAM*.csv"))])
    obs_277 = set([file.split("/")[-1][9:12] for file in sorted(glob.glob("/mnt/C/JWST/COSMOS/NIRCAM/catalog/F277W/*NIRCAM*.csv"))])
    obs_444 = set([file.split("/")[-1][9:12] for file in sorted(glob.glob("/mnt/C/JWST/COSMOS/NIRCAM/catalog/F444W/*NIRCAM*.csv"))])
    return sorted(list(obs_115.intersection(obs_150, obs_277, obs_444)))
   
def CC_diagram(obs: list, size=(8, 4.5)):
    """
    """
    limit_mag = {
                    "F115W": 27.45,
                    "F150W": 27.66,
                    "F277W": 28.28,
                    "F444W": 28.17
                }
    
    total_cat = pd.DataFrame()
    
    for obs_number in obs:
        df_115 = read_catalog_by_obs(obs_number, "F115W")
        df_150 = read_catalog_by_obs(obs_number, "F150W")
        df_277 = read_catalog_by_obs(obs_number, "F277W")
        df_444 = read_catalog_by_obs(obs_number, "F444W")

        add_prefix_columns(df_115, "F115W")
        add_prefix_columns(df_150, "F150W")
        add_prefix_columns(df_277, "F277W")
        add_prefix_columns(df_444, "F444W")

        temp1 = crossmatch(df_115, df_150, max_sep=0.2, sep=False, return_cat=True, include_unmatched=False)
        temp2 = crossmatch(temp1, df_277, max_sep=0.2, sep=False, return_cat=True, include_unmatched=False)
        temp3 = crossmatch(temp2, df_444, max_sep=0.2, sep=False, return_cat=True, include_unmatched=False)
        
        AB_115 = MJy_Sr_2_AB(temp3['F115W_kron_flux'], header=read_header_by_obs(obs_number, 'F115W'))
        AB_150 = MJy_Sr_2_AB(temp3['F150W_kron_flux'], header=read_header_by_obs(obs_number, 'F150W'))
        AB_277 = MJy_Sr_2_AB(temp3['F277W_kron_flux'], header=read_header_by_obs(obs_number, 'F277W'))
        AB_444 = MJy_Sr_2_AB(temp3['F444W_kron_flux'], header=read_header_by_obs(obs_number, 'F444W'))
        
        AB_115 = AB_115[AB_115<=limit_mag["F115W"]]
        AB_150 = AB_150[AB_150<=limit_mag["F150W"]]
        AB_277 = AB_277[AB_277<=limit_mag["F277W"]]
        AB_444 = AB_444[AB_444<=limit_mag["F444W"]]
        
        color_115_150 = AB_115 - AB_150
        color_150_277 = AB_150 - AB_277
        color_277_444 = AB_277 - AB_444
        
        total_cat = pd.concat([total_cat,
                                pd.Series(color_115_150, index=['color_115_150']),
                                pd.Series(color_150_277, index=['color_150_277']),
                                pd.Series(color_277_444, index=['color_277_444'])], 
                                axis=0)
            


    fig, axes = scatter_plot([(color_150_277, color_115_150), (color_277_444, color_150_277)],
                                xlabel=["F150W - F277W [ABmag]", "F277W - F444W [ABmag]"],
                                ylabel=["F115W - F150W [ABmag]", "F150W - F277W [ABmag]"],
                                title=['F115W-Dropout', 'F150W-Dropout'],
                                size=size, save=False)
    
    # Harikane's slection
    H_vertices_115 = [(-1.8, 1), (0, 1), (1, 2), (1, 3.5), (-1.8, 3.5)]
    H_vertices_150 = [(-1.8, 1), (0, 1), (1, 2), (1, 3.5), (-1.8, 3.5)]

    H_selection = patches.Polygon(H_vertices_115, closed=True, color='green', alpha=0.45)
    axes[0].add_patch(H_selection)

    H_selection = patches.Polygon(H_vertices_150, closed=True, color='green', alpha=0.45)
    axes[1].add_patch(H_selection)
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/{timestamp}.png")
    
    return fig, axes

def find_dropouts(obs_num: list[str]):
    if not isinstance(obs_num, list):
        obs_num = [obs_num] 
        
    limit_mag = {
                    "F115W": 27.45,
                    "F150W": 27.66,
                    "F277W": 28.28,
                    "F444W": 28.17
                }   
        
    F150W_dropouts_color = []
    F277W_dropouts_color = []
    Candidates = pd.DataFrame()
    
    for obs in obs_num:
        df_115 = read_catalog_by_obs(obs, "F115W")
        df_150 = read_catalog_by_obs(obs, "F150W")
        df_277 = read_catalog_by_obs(obs, "F277W")
        df_444 = read_catalog_by_obs(obs, "F444W")
        
        # Magnitude cut, we don't use sources below 5-sigma limits 
        df_115 = df_115[df_115['kron_AB']<limit_mag["F115W"]]
        df_150 = df_150[df_150['kron_AB']<limit_mag["F150W"]]
        df_277 = df_277[df_277['kron_AB']<limit_mag["F277W"]]
        df_444 = df_444[df_444['kron_AB']<limit_mag["F444W"]]

        add_prefix_columns(df_115, "F115W")
        add_prefix_columns(df_150, "F150W")
        add_prefix_columns(df_277, "F277W")
        add_prefix_columns(df_444, "F444W")

        F115_F150 = crossmatch(df_115, df_150, max_sep=0.2, sep=False, return_cat=True, include_unmatched=True)
        F115 = F115_F150[np.isnan(F115_F150['F150W_RA'])]
        F150 = F115_F150[np.isnan(F115_F150['F115W_RA'])]
        F115_F150 = F115_F150[((~np.isnan(F115_F150['F115W_RA'])) 
                            & (~np.isnan(F115_F150['F150W_RA'])))]

        F115_F150_F277 = crossmatch(F115_F150, df_277, max_sep=0.2, sep=False, return_cat=True, include_unmatched=True)
        F277 = F115_F150_F277[np.isnan(F115_F150_F277['F115W_RA'])
                            & (np.isnan(F115_F150_F277['F150W_RA']))
                            & (~np.isnan(F115_F150_F277['F277W_RA']))]
        F115_F150_F277 = F115_F150_F277[((~np.isnan(F115_F150_F277['F115W_RA']))
                                        & (~np.isnan(F115_F150_F277['F150W_RA']))
                                        & (~np.isnan(F115_F150_F277['F277W_RA'])))]

        F115_F150_F277_F444 = crossmatch(F115_F150_F277, df_444, max_sep=0.2, sep=False, return_cat=True, include_unmatched=True)
        F444 = F115_F150_F277_F444[(np.isnan(F115_F150_F277_F444['F115W_RA']))
                                    & (np.isnan(F115_F150_F277_F444['F150W_RA']))
                                    & (np.isnan(F115_F150_F277_F444['F277W_RA']))
                                    & (~np.isnan(F115_F150_F277_F444['F444W_RA']))]
        
        F115_F150_F277_F444 = F115_F150_F277_F444[((~np.isnan(F115_F150_F277_F444['F115W_RA']))
                                                & (~np.isnan(F115_F150_F277_F444['F444W_RA'])))]
        
        F277 = F277[[col for col in F277.columns if 'F277' in col]]
        F150_F277 = crossmatch(F150.reset_index(drop=True), F277.reset_index(drop=True), max_sep=0.2, sep=False, return_cat=True, include_unmatched=True)
        F150_F277 = F150_F277[((~np.isnan(F150_F277['F150W_RA']))
                                & (~np.isnan(F150_F277['F277W_RA'])))]

        F444 = F444[[col for col in F444.columns if 'F444' in col]]
        F150_F277_F444 = crossmatch(F150_F277.reset_index(drop=True), F444.reset_index(drop=True), max_sep=0.2, sep=False, return_cat=True, include_unmatched=True)
        F444_fin = F150_F277_F444[(np.isnan(F150_F277_F444['F277W_RA'])) & (~np.isnan(F150_F277_F444['F444W_RA']))]
        F150_F277_F444 = F150_F277_F444[(~np.isnan(F150_F277_F444['F277W_RA'])) & (~np.isnan(F150_F277_F444['F444W_RA']))]
        
        Candidates = pd.concat([Candidates, F150_F277_F444], ignore_index=True)
        
        _150 = F150_F277_F444['F150W_kron_AB']
        _277 = F150_F277_F444['F277W_kron_AB']
        _444 = F150_F277_F444['F444W_kron_AB']
                    
        color_115_150_ = limit_mag["F115W"] - _150
        color_150_277_ = _150 - _277
        color_277_444_ = _277 - _444
        
        F150W_dropouts_color.append((color_115_150_, color_150_277_))
        F277W_dropouts_color.append((color_150_277_, color_277_444_))
    
    return F150W_dropouts_color, F277W_dropouts_color, Candidates

def find_obs_from_coords(ra, dec, _filter):
    obs = None
    import warnings
    warnings.filterwarnings("ignore", module="astropy")
    for file in sorted(glob.glob(f'/mnt/C/JWST/COSMOS/NIRCAM/Reduced/{_filter}/fits/jw01727*full_i2d.fits')):
        hd = fits.getheader(file, 1)
        # Image Boundary
        Img_Bound = [(0, 0), 
                    (0, hd['NAXIS2']), 
                    (hd['NAXIS1'], hd['NAXIS2']),
                    (hd['NAXIS1'], 0)]
        poly = shapely_polygon(Img_Bound)

        y, x = WCS(hd).wcs_world2pix(ra, dec, 0)

        if Point(x, y).within(poly):
            obs = file.split("/")[-1][9:12]
            print(f"Coordinate found in {obs}.")
            break
    return obs

def NIR_cutout(df):
    radius = 6 # [arcsec]
    filters = ["F115W", "F150W", "F277W", "F444W"]
    images = {}
    WCSs = {}
    pix_sizes = {}
    headers = {}
    limit_mag = [
                  str(np.round(AB_2_uJy(27.45), 3)), # "F115W"
                  str(np.round(AB_2_uJy(27.66), 3)), # "F150W"
                  str(np.round(AB_2_uJy(28.28), 3)), # "F277W"
                  str(np.round(AB_2_uJy(28.17), 3)), # "F444W"                
                  ]
    
    obs = find_obs_from_coords(df['F444W_RA'], df['F444W_DEC'], "F444W")
    
    if obs:
        print(obs)
        # Prepare the super title
        title_parts = []
        
        for f in range(len(filters)):
            image_path = f'/mnt/C/JWST/COSMOS/NIRCAM/Reduced/{filters[f]}/fits/jw01727-o{obs}_t001_NIRCAM-clear-{filters[f]}-full_i2d.fits'
            images[filters[f]] = fits.open(image_path)[1].data
            headers[filters[f]] = fits.open(image_path)[1].header
            WCSs[filters[f]] = WCS(headers[filters[f]])
            pix_sizes[filters[f]] = np.sqrt(headers[filters[f]] ['PIXAR_A2'])
            
        fig = plt.figure(figsize=(22, 11)) 
        

        for f in range(len(filters)):
            size = radius/pix_sizes[filters[f]]
            ax = fig.add_subplot(1, 4, f+1, projection=WCSs[filters[f]])
            img_data = images[filters[f]]
            
            if f != 3:
                _, ra_offset, dec_offset = ra_dec_plot(
                                    max_sep = 0.125, 
                                    csv1 = pd.read_csv(f"./{filters[f]}/jw01727-o{obs}_t001_NIRCAM-clear-{filters[f]}-full_i2d.csv"),
                                    csv2 = pd.read_csv(f"./F444W/jw01727-o{obs}_t001_NIRCAM-clear-F444W-full_i2d.csv"),
                                    save = f'o{obs}_offset_to_F444W_temp',
                                    show = False)
            else:
                ra_offset, dec_offset = 0.0, 0.0
            
            ra_offset  /= 3600000 
            dec_offset /= 3600000
            # print(ra_offset, dec_offset)
            # print(df[f'{filters[f]}_RA'])
            # print(img_data)
            if df[f'{filters[f]}_RA'].isna().any().any():
                # print("No annotation")
                x, y = WCSs[filters[f]].wcs_world2pix(df['F444W_RA'] - ra_offset, 
                                                      df['F444W_DEC'] - dec_offset, 0)
                
            else:
                x, y = WCSs[filters[f]].wcs_world2pix(df[f'{filters[f]}_RA'] - ra_offset, 
                                                      df[f'{filters[f]}_DEC'] - dec_offset, 0)
            print(x, y) 
            # print(img_data[int(y-size):int(y+size), 
                        #    int(x-size):int(x+size)])
            
            ax.imshow(  img_data[int(y-size):int(y+size), 
                                 int(x-size):int(x+size)], 
                        vmax=np.nanpercentile(img_data.flatten(), 99), 
                        vmin=np.nanpercentile(img_data.flatten(), 10)
                        )
            # ax.scatter(size, size, s=96, marker='o', edgecolors='red', facecolor='None')
            
            ax.plot([size, size], [size+0.5/pix_sizes[filters[f]], size+1/pix_sizes[filters[f]]], color='red', lw=3)
            ax.plot([size+0.5/pix_sizes[filters[f]], size+1/pix_sizes[filters[f]]], [size, size], color='red', lw=3)
            # ax.grid(color='white', ls='--', lw=2)
            ax.set_title(filters[f], fontsize=30)
            # ax.set_xlabel('RA', fontsize=24)
            # ax.set_ylabel('Dec', fontsize=24)
            
            # Hiding x and y axis ticks
            ax.coords[0].set_ticks_visible(False) 
            ax.coords[0].set_ticklabel_visible(False) 
            ax.coords[1].set_ticks_visible(False)
            ax.coords[1].set_ticklabel_visible(False) 
            
            # Flux as x lable
            if df[f'{filters[f]}_kron_flux'].isna().any().any():
               flux_info = f"▼{limit_mag[f]}$\mu$Jy"
            
            else:
                brightness = AB_2_uJy(df[f'{filters[f]}_kron_AB'])
                flux_info = f"{np.round(brightness.values[0], 3)}$\mu$Jy"
                
            ax.text(0.5, -0.1, flux_info, va='center', ha='center', fontsize=24,
                        transform=ax.transAxes)

    
    Id = f"{np.round(df[f'F150W_RA'].values[0], 6)}_{np.round(df[f'F150W_DEC'].values[0], 6)}"
    # Setting the super title
    plt.suptitle(Id, fontsize=36)
    plt.savefig(f"/mnt/C/JWST/COSMOS/NIRCAM/PNG/Dropout_Cutouts/Dropout_{Id}.png", 
                facecolor="#EEEBE1")
    
def flux_error_to_mag_error(flux, flux_error):
    """
    Convert flux error from mJy to AB magnitude error.
    
    Parameters:
    flux (pd.Series-like): The flux series in mJy.
    flux_error (pd.Series-like): The flux error series in mJy.
    
    Returns:
    float: The AB magnitude error.
    """
    # Check if the length of the incoming two series is the same
    if len(flux) != len(flux_error):
        raise ValueError("flux and flux_error must have the same length!")
    
    mag_error = 1.0857 * (flux_error / flux)
        
    return mag_error

def mag_error_to_flux_error(mag, mag_error):
    """
    Convert AB magnitude error to flux error in mJy.
    
    Parameters:
    mag (float): The magnitude in AB magnitudes.
    mag_error (float): The magnitude error in AB magnitudes.
    
    Returns:
    float: The flux error in mJy.
    """
    # Check if the length of the incoming two series is the same
    # if len(mag) != len(mag_error):
    #     raise ValueError("flux and flux_error must have the same length!")
    
    conversion_factor = np.log(10)/2.5
    flux = AB_2_mJy(mag)
    flux_error = flux * conversion_factor * mag_error
    return flux_error

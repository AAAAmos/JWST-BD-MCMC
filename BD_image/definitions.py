import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json, glob

from astropy.io import fits
from astropy.io import ascii
from astropy.wcs import WCS

from reproject import reproject_interp, reproject_adaptive

from shapely.geometry import Point
from shapely.geometry import Polygon as shapely_polygon

from utility import crossmatch, add_prefix_columns

def find_xy(table, count=5):
    """
    Find the x,y coordinates of the sources in the table.
    The default number of sources is 5.
    
    Args:
    -----   
    table: astropy.table.Table
        The table of the sources.
    count: int
        The number of sources to return.
        Set to -1 to return all sources.
    
    Returns:
    --------
    coordinates: list[tuple(int, int)]
        A list of the x,y coordinates of the sources.
    """

    # get the top five rows of the x,y coordinate in the brightest source table
    coordinates = table[:count]["X_IMAGE", "Y_IMAGE"]
    # convert the table to a list
    coordinates = list(coordinates.to_pandas().values)
    # convert the data type of the elements into int tuples
    coordinates = [tuple(map(int, i)) for i in coordinates]
    
    return coordinates
        
def plot_seg(coordinates, 
             main_catalog, 
             size, 
             filter, 
             resolution, 
             area
             ):
    """
    Plot the image around the sources in the coordinates.

    Args:
    -----
    coordinates: list[tuple(int, int)]
        A list of the x,y coordinates of the sources.
    main_catalog: astropy.table.Table
        The main catalog containing the nearby sources.
    size: int
        The size of the cutout.
    filter: str
        The filter of the image. (F115W, F150W, F277W, F444W)
    resolution: str
        The resolution of the image. (30mas or 60 mas)
    area: str
        The area number of the image. (A1-A10)
    aperture: bool
        Whether to plot the aperture file.
    """
    
    # Read the SCI-only file
    sci_fits = fits.open(f"./{resolution}/mosaic_nircam_{filter.lower()}_COSMOS-Web_{resolution}_{area}_v0_5_SCI.fits")
    sci_data = sci_fits[1].data
    
    shape = sci_data.shape

    for coordinate in coordinates:
        image_center = sci_data[coordinate[1]-size:coordinate[1]+size, coordinate[0]-size:coordinate[0]+size]
        
        # get the 80 and 20 percentile of center_seg
        low = np.nanpercentile(image_center.flatten(), 20)
        high = np.nanpercentile(image_center.flatten(), 99.5)
        # print(low, high)

        # plot the center part of the SEG_check.fits
        plt.imshow(image_center, cmap="gray", 
                    vmin=low,
                    vmax=high)
        
        # find the nearby sources in the center part of the image
        nearby_sources = main_catalog[(main_catalog["X_IMAGE"] > coordinate[0]-size) 
                                    & (main_catalog["Y_IMAGE"] > coordinate[1]-size) 
                                    & (main_catalog["X_IMAGE"] < coordinate[0]+size) 
                                    & (main_catalog["Y_IMAGE"] < coordinate[1]+size)]
        
        # plot hollow indicators on the position of the nearby sources
        plt.scatter(nearby_sources["X_IMAGE"]-coordinate[0]+size, 
                    nearby_sources["Y_IMAGE"]-coordinate[1]+size, 
                    s=40, marker="o",
                    facecolors='none', 
                    edgecolors='r')
        
        # set the dpi of image as 150
        plt.gcf().set_dpi(150)
        plt.show()

def make_sex_conf(_filter, 
                  resolution,
                  area, 
                  index=1,
                  conv_files=None, 
                  deblend_thres=None,
                  deblend_min=None,
                  thresholds=None, 
                  mag_zeros=None, 
                  fwhm=None,
                  aper_size=None):
    """
    Create a configuration file for SExtractor.
    
    Args:
    -----
    _filter: str
        Filter name.
    resolution: str
        Resolution of the image.
    area: str
        Area of the image. (A1-A10)
    index: int
        The index of the folder.
        Default value is 1.
    conv_files: dict
        The specified convolution files for each filter. 
        Default file is "gauss_2.0_3x3.conv".
    deblend_thres: dict
        The specified deblend threshold for each filter.
        Default value is "48".
    deblend_min: dict
        The specified deblend minimum for each filter.
        Default value is "0.05".
    thresholds: dict
        The specified threshold for each filter.
        Default value is "1.5".
    mag_zeros: dict
        The specified mag_zero for each filter.
        Default value is read from the header of the SCI file.
    fwhm: dict
        The specified FWHM for each filter.
        Default value is "0.040" for F115W, "0.050" for F150W, "0.092" for F277W, "0.145" for F444W.
    
    Returns:
    --------
    str:
        The path to the configuration file.
    """

    # If the input conv_files is empty, use the default values
    if not conv_files:
        conv_files = {
            "F115W": "gauss_2.0_3x3.conv",
            "F150W": "gauss_2.0_3x3.conv",
            "F277W": "gauss_2.0_3x3.conv",
            "F444W": "gauss_2.0_3x3.conv",
            "F770W": "gauss_3.0_5x5.conv"
        }

    # If the input deblend_thres is empty, use the default values
    if not deblend_thres:
        deblend_thres = {
            "F115W": "48",
            "F150W": "48",
            "F277W": "48",
            "F444W": "48",
            "F770W": "48"
        }

    # If the input deblend_min is empty, use the default values
    if not deblend_min:
        deblend_min = {
            "F115W": "0.05",
            "F150W": "0.05",
            "F277W": "0.05",
            "F444W": "0.05",
            "F770W": "0.05"
        }

    # If the input thresholds is empty, use the default values
    if not thresholds:
        thresholds = {
            "F115W": "1.5",
            "F150W": "1.5",
            "F277W": "1.5",
            "F444W": "1.5",
            "F770W": "1.5"
        }

    # If the input mag_zeros is empty, read the mag_zero from the header of the SCI file
    if not mag_zeros:
        sci_file = f"mosaic_nircam_{_filter.lower()}_COSMOS-Web_{resolution}_{area}_v0_5_SCI.fits"
        # Read the header for the 
        with fits.open(f"./{resolution}/{sci_file}")as hdul:
            header = hdul[1].header
        mag_zeros = {
            "F115W": -6.10 - 2.5 * np.log10(header['PIXAR_SR']),
            "F150W": -6.10 - 2.5 * np.log10(header['PIXAR_SR']),
            "F277W": -6.10 - 2.5 * np.log10(header['PIXAR_SR']),
            "F444W": -6.10 - 2.5 * np.log10(header['PIXAR_SR']),    
        }

    # If the input aper_size is empty, use the default values in pixel
    if not aper_size:
        aper_size = {"F115W": "7.0, 11.0, 15.0, 20.0, 25.0",
                     "F150W": "7.0, 11.0, 15.0, 20.0, 25.0",
                     "F277W": "7.0, 11.0, 15.0, 20.0, 25.0",
                     "F444W": "7.0, 11.0, 15.0, 20.0, 25.0"}
        
    # FWHM values in arcsec
    fwhm = {
        "F115W": "0.040",
        "F150W": "0.050",
        "F277W": "0.092",
        "F444W": "0.145",
    }

    ## Create folder that named as the day of the run
    ## If the folder already exists, try to create a new folder with incrementing number
    import datetime, os
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # folder for configuration files
    i = 1
    while os.path.exists(f"./configures/{today}_{i}"):
        i += 1
    # use specified index for the folder
    if index:
        today = f"{today}_{index}"
    else:
        today = f"{today}_{i}"
        
    try:
        os.makedirs(f"./configures/{today}")
    except FileExistsError:
        pass
    try:
        os.makedirs(f"./catalogs/raw/{today}")
    except FileExistsError:
        pass
    try:
        os.makedirs(f"./catalogs/updated/{today}")
    except FileExistsError:
        pass
    
    # Re-write the configuration file from default 
    print("Writing configuration file for SExtractor......")
    with open("default.conf") as f:
        lines = f.read()
        # replace the filter, resolution and area in the file
        lines = lines.replace("%CAT_PATH%", f"./catalogs/raw/{today}/%Filter%_%Res%_%Area%.cat")
        lines = lines.replace("%Filter%", _filter)
        lines = lines.replace("%Res%", resolution)
        lines = lines.replace("%Area%", area)
        
        # replace the aperture size
        lines = lines.replace("%APER_SIZE%", aper_size[_filter])
        # replace the threshold value
        lines = lines.replace("%THRESHOLD%", thresholds[_filter])
        # replace the convolution file
        lines = lines.replace("%CONV_FILE%", conv_files[_filter])
        # replace the deblend threshold value
        lines = lines.replace("%DEBLEND_THRES%", deblend_thres[_filter])
        # replace the deblend min value
        lines = lines.replace("%DEBLEND_MIN%", deblend_min[_filter])
        # replace the mag_zero value
        lines = lines.replace("%MAG_ZERO%", str(mag_zeros[_filter]))
        # replace the FWHM value
        lines = lines.replace("%FWHM%", fwhm[_filter])
    
    # Write the configuration to file
    with open(f"./configures/{today}/{_filter}_{resolution}_{area}.conf", "w") as file:
        file.write(lines)
        
    return f"./configures/{today}/{_filter}_{resolution}_{area}.conf"

def make_sex_table(_filter, resolution, area, conf_file, verbose="DEFAULT"):
    """
    Run SExtractor with specified configuration file.
    
    Args:
    -----
    _filter: str
        Filter name.
    resolution: str
        Resolution of the image.
    area: str
        Area of the image. (A1-A10)
    conf_file: str
        The path to the configuration file name.
    verbose: str
        The level of verbosity. 
        Default value is "DEFAULT".
        Set to "MUTE" to mute the output.
        
    Returns:
    --------
    astropy.table.Table:
        The table of the SEx result.
    
    table_path: str
        The path to the table.
    """
    import os

    # First check if the SCI-only file is present. If not, create it
    sci_file = f"mosaic_nircam_{_filter.lower()}_COSMOS-Web_{resolution}_{area}_v0_5_SCI.fits"
    gz_file = f"mosaic_nircam_{_filter.lower()}_COSMOS-Web_{resolution}_{area}_v0_5_i2d.fits.gz"
    if not os.path.exists(f"./{resolution}/{sci_file}"):
        print(f"Creating SCI-only file for {_filter.lower()} {resolution} {area}......")
        try:
            with fits.open(f"./gz_files/{gz_file}") as hdul:
                new_hdul = hdul[:2]
                new_hdul.writeto(f"./{resolution}/{sci_file}", overwrite=True)
        except FileNotFoundError:
            print(f"File {gz_file} not found. Trying to read .fits file......")
            with fits.open(f"./{resolution}/{gz_file[:-3]}") as hdul:
                new_hdul = hdul[:2]
                new_hdul.writeto(f"./{resolution}/{sci_file}", overwrite=True)

    # Print the parameters of the configuration file
    with open(conf_file) as f:
        lines = f.readlines()
        
    if not verbose=="MUTE":
        print(lines[15].split("#")[0])# DETECT_MINAREA   15 
        print(lines[19].split("#")[0])# DETECT_THRESH    %THRESHOLD%    
        print(lines[23].split("#")[0])# FILTER_NAME      %CONV_FILE%    
        print(lines[26].split("#")[0])# DEBLEND_NTHRESH  %DEBLEND_THRES%
        print(lines[27].split("#")[0])# DEBLEND_MINCONT  %DEBLEND_MIN%  
        print(lines[52].split("#")[0])# PHOT_APERTURES   15             
        print(lines[63].split("#")[0])# MAG_ZEROPOINT    %MAG_ZERO%     
        print(lines[71].split("#")[0])# SEEING_FWHM      %FWHM%     
        print(lines[78].split("#")[0])# BACK_SIZE        64   

    # Run SExtractor    
    print("Running SExtractor......")
    os.system(f"sex ./{resolution}/{sci_file} -c {conf_file}")
    print(f"Source Extraction Done for {_filter} {resolution} {area}...")
    #                                                   V: strip the god damn EOL
    table_path = lines[6].split("#")[0].split(" ")[-1][:-1]
    table = ascii.read(table_path,
                        guess=False, 
                        format="sextractor", 
                        fast_reader=True)
    
    return table, table_path
    
def update_sex_table(table, header):
    """
    Convert the flux values to AB magnitudes and update the table.
    Remove the flux columns and replace them with the AB magnitudes.
    Correct the RA and DEC values using header information.
    Drop rows that contain NaN values.
    
    Args:
    -----
        table (astropy.table.Table): 
            The table to be updated.
        header (astropy.io.fits.header.Header): 
            The header of the fits file.
    
    Returns:
    --------
        astropy.table.Table: 
            The updated table.
    """
    from utility import MJy_Sr_2_mJy, flux_error_to_mag_error, MJy_Sr_2_AB
    print("Updating the table......")
    # Convert the FLUX AUTO to AB magnitudes
    AB_mag = MJy_Sr_2_AB(table["FLUX_AUTO"], header=header)
    table["FLUX_AUTO"] = MJy_Sr_2_mJy(table["FLUX_AUTO"], header=header)
    table["FLUXERR_AUTO"] = MJy_Sr_2_mJy(table["FLUXERR_AUTO"], header=header)
    table["FLUXERR_AUTO"] = flux_error_to_mag_error(table["FLUX_AUTO"], table["FLUXERR_AUTO"])
    table["FLUX_AUTO"] = AB_mag
    
    # Convert the FLUX PETRO to AB magnitudes
    AB_mag = MJy_Sr_2_AB(table["FLUX_PETRO"], header=header)
    table["FLUX_PETRO"] = MJy_Sr_2_mJy(table["FLUX_PETRO"], header=header)
    table["FLUXERR_PETRO"] = MJy_Sr_2_mJy(table["FLUXERR_PETRO"], header=header)
    table["FLUXERR_PETRO"] = flux_error_to_mag_error(table["FLUX_PETRO"], table["FLUXERR_PETRO"])
    table["FLUX_PETRO"] = AB_mag
    
    # Convert the FLUX ISO to AB magnitudes
    AB_mag = MJy_Sr_2_AB(table["FLUX_ISO"], header=header)
    table["FLUX_ISO"] = MJy_Sr_2_mJy(table["FLUX_ISO"], header=header)
    table["FLUXERR_ISO"] = MJy_Sr_2_mJy(table["FLUXERR_ISO"], header=header)
    table["FLUXERR_ISO"] = flux_error_to_mag_error(table["FLUX_ISO"], table["FLUXERR_ISO"])
    table["FLUX_ISO"] = AB_mag
    
    # Convert the FLUX APER to AB magnitudes
    AB_mag = MJy_Sr_2_AB(table["FLUX_APER"], header=header)
    table["FLUX_APER"] = MJy_Sr_2_mJy(table["FLUX_APER"], header=header)
    table["FLUXERR_APER"] = MJy_Sr_2_mJy(table["FLUXERR_APER"], header=header)
    table["FLUXERR_APER"] = flux_error_to_mag_error(table["FLUX_APER"], table["FLUXERR_APER"])
    table["FLUX_APER"] = AB_mag
    
    # Convert the FLUX BEST to AB magnitudes
    AB_mag = MJy_Sr_2_AB(table["FLUX_BEST"], header=header)
    table["FLUX_BEST"] = MJy_Sr_2_mJy(table["FLUX_BEST"], header=header)
    table["FLUXERR_BEST"] = MJy_Sr_2_mJy(table["FLUXERR_BEST"], header=header)
    table["FLUXERR_BEST"] = flux_error_to_mag_error(table["FLUX_BEST"], table["FLUXERR_BEST"])
    table["FLUX_BEST"] = AB_mag
    
    # Replace every MAG & MAGERR columns with the AB magnitudes from FLUX columns
    table["MAG_PETRO"] = table["FLUX_PETRO"]
    table["MAGERR_PETRO"] = table["FLUXERR_PETRO"]

    table["MAG_ISO"] = table["FLUX_ISO"]
    table["MAGERR_ISO"] = table["FLUXERR_ISO"]

    table["MAG_APER"] = table["FLUX_APER"]
    table["MAGERR_APER"] = table["FLUXERR_APER"]

    table["MAG_BEST"] = table["FLUX_BEST"]
    table["MAGERR_BEST"] = table["FLUXERR_BEST"]

    table["MAG_AUTO"] = table["FLUX_AUTO"]
    table["MAGERR_AUTO"] = table["FLUXERR_AUTO"]

    # Remove the FLUX columns
    table.remove_columns(["FLUX_AUTO", "FLUXERR_AUTO", 
                                "FLUX_PETRO", "FLUXERR_PETRO", 
                                "FLUX_ISO", "FLUXERR_ISO", 
                                "FLUX_APER", "FLUXERR_APER", 
                                "FLUX_BEST", "FLUXERR_BEST"])
    
    # Re-calculate the RA and DEC values
    wcs = WCS(header)
    ra, dec = wcs.wcs_pix2world(table["X_IMAGE"].astype(float), 
                                table["Y_IMAGE"].astype(float), 0)
    table["ALPHA_J2000"] = ra
    table["DELTA_J2000"] = dec
    
    # drop rows that contains NaN values
    mask = ~np.isnan(table["MAG_AUTO"])
    updated_table = table[mask]
    
    return updated_table

def calculate_offsets(resolution, area, date, index):
    """
    Calculate the offsets of the sources in the input catalog respect to the F444W image.
    
    Args:
    -----
        resolution: str
            The resolution of the image.
        area: str
            The area of the image.
        date: str
            The date of the run of SExtractor.
        index: int
            The index of the output catalog.    
        
    Returns:
    --------
        tuple(float, float):
            The offsets in ra and dec.
    """
    from utility import ra_dec_plot

    all_offsets = {
        "F115W": (0.0, 0.0),
        "F150W": (0.0, 0.0),
        "F277W": (0.0, 0.0),
        "F444W": (0.0, 0.0)
    }

    # Calculate the offsets
    filters = ["F115W", "F150W", "F277W", "F444W"]
    
    df_444 = ascii.read(f"./catalogs/updated/{date}_{index}/F444W_{resolution}_{area}.cat",
                        guess=False, 
                        format="commented_header", 
                        fast_reader=True).to_pandas()
    
    for f in range(len(filters)-1):
        _, ra_offset, dec_offset = ra_dec_plot(max_sep = 0.125, 
                                                csv1 = ascii.read(f"./catalogs/updated/{date}_{index}/{filters[f]}_{resolution}_{area}.cat", 
                                                                guess=False, 
                                                                format="commented_header", 
                                                                fast_reader=True).to_pandas(),
                                                csv2 = df_444,
                                                save = f'False',
                                                show = False,
                                                )
        ra_offset  /= 3600000 
        dec_offset /= 3600000
        all_offsets[filters[f]] = (ra_offset, dec_offset)
        
    with open(f"./offsets/{resolution}_{area}.json", "w") as f:
        json.dump(all_offsets, f)

def calculate_JWST_HST_offsets(resolution, date, index):
    """
    Calculate the offsets of the sources in the input catalog respect to the F444W image.
    
    Args:
    -----
        resolution: str
            The resolution of the image.
        date: str
            The date of the run of SExtractor.
        index: int
            The index of the output catalog.    
        
    Returns:
    --------
        tuple(float, float):
            The offsets in ra and dec.
    """
    from utility import ra_dec_plot

    # areas = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A9", "A10"]
    areas = ["A1", "A4"]
    all_offsets = {}
    COSMOS_2020_cat = pd.read_csv(f"./catalogs/COSMOS_2020_cut.csv")
    
    for area in areas:
        with open(f"./HST_mapping/{area}_mapping.json", "r") as f:
            mapping = json.load(f)
            
        df_444 = ascii.read(f"./catalogs/updated/{date}_{index}/F444W_{resolution}_{area}.cat",
                        guess=False, 
                        format="commented_header", 
                        fast_reader=True).to_pandas()
        
        for filename in mapping[area]:
            # Calculate the offsets
            print(f"Calculating offsets between {filename} & {area}......")
            _, ra_offset, dec_offset = ra_dec_plot(max_sep = 0.125, 
                                                    csv1 = COSMOS_2020_cat,
                                                    csv2 = df_444,
                                                    save = False,
                                                    show = True,
                                                    )
            if _:
                ra_offset  /= 3600000 
                dec_offset /= 3600000
                all_offsets[filename] = (ra_offset, dec_offset)
            else:
                all_offsets[filename] = (0.0, 0.0)
        
    with open(f"./offsets/HST_JWST_F444W.json", "w") as f:
        json.dump(all_offsets, f)

def COSMOS_cutout(total_cat, resolution="30mas", area="A1", size=150, kernel='hann'):
    # Load the JWST data needed for plotting
    images = {}
    headers = {}
    WCSs = {}
    pix_sizes = {}
    filters = ["F115W", "F150W", "F277W", "F444W"]

    for f in range(len(filters)):
        with fits.open(f"./{resolution}/mosaic_nircam_{filters[f].lower()}_COSMOS-Web_{resolution}_{area}_v0_5_SCI.fits") as hdul:
            images[filters[f]] = hdul[1].data
            headers[filters[f]] = hdul[1].header
            WCSs[filters[f]] = WCS(hdul[1].header)
            pix_sizes[filters[f]] = np.sqrt(hdul[1].header['PIXAR_A2'])
            
    # Read the archived HST-JWST offset
    HST_offsets = json.load(open(f"./offsets/HST_JWST_F444W.json"))
    all_offsets = json.load(open(f"./offsets/{resolution}_{area}.json"))
    
    # Iterate over the input dataframe 
    for i, row in total_cat.iterrows():
        fig = plt.figure(figsize=(22, 11))
        
        # Plot HST F814W data as first image
        try:
            HST_file = find_HST_data((row['RA'], row['DEC']), area)
            HST_ra_offset, HST_dec_offset = HST_offsets[HST_file.split('/')[-1]]
            
            with fits.open(HST_file) as f:
                HST_image = f[0].data
                # The code is attempting to access the header of the first element in the list `f` and
                # assign it to the variable `HST_header`.
                HST_header = f[0].header
                
            HST_WCS = WCS(HST_header)
            HST_pix_size = 0.03 # [arcsec]
            x_hst, y_hst = HST_WCS.all_world2pix(row[f'RA'], row[f'DEC'], 0)
            
            x_hst -= HST_ra_offset*3600/HST_pix_size
            y_hst -= HST_dec_offset*3600/HST_pix_size
            
            print("Reprojecting HST image......")
            scale = 1.414
            
            y_start = max(0, int(y_hst - size*scale))
            y_end = min(HST_image.shape[0], int(y_hst + size*scale))
            x_start = max(0, int(x_hst - size*scale))
            x_end = min(HST_image.shape[1], int(x_hst + size*scale))

            cut_data_shape = (int(2*size*scale), int(2*size*scale))
            HST_img_center = np.full(cut_data_shape, np.nan)

            y_offset = max(0, int(size*scale - y_hst))
            x_offset = max(0, int(size*scale - x_hst))

            HST_img_center = HST_image[y_offset:y_offset+(y_end-y_start), x_offset:x_offset+(x_end-x_start)] = HST_image[y_start:y_end, x_start:x_end]
            
            HST_header["NAXIS1"] = HST_img_center.shape[1]
            HST_header["NAXIS2"] = HST_img_center.shape[0]
            headers["F444W"]["NAXIS1"] = HST_img_center.shape[1]
            headers["F444W"]["NAXIS2"] = HST_img_center.shape[0]
            
            HST_WCS = WCS(HST_header)
            JWST_WCS = WCS(headers["F444W"])
            # Measure the rotation of the JWST WCS
            theta = -np.arctan(JWST_WCS.wcs.pc[1, 0]/JWST_WCS.wcs.pc[0, 0])
            
            # Re-define the input and output WCS by changing the rotation angle
            input_wcs = HST_WCS.deepcopy()
            input_wcs.wcs.crpix = HST_img_center.shape[1]/2, HST_img_center.shape[0]/2
            input_wcs.wcs.cdelt = HST_pix_size/3600, HST_pix_size/3600
            
            output_wcs = input_wcs.deepcopy()
            output_wcs.wcs.crpix = HST_img_center.shape[1]/2, HST_img_center.shape[0]/2
            output_wcs.wcs.pc = [[-np.cos(theta), 
                                np.sin(theta)], 
                                [np.sin(theta), 
                                np.cos(theta)]]
            output_wcs.wcs.cdelt = HST_pix_size/3600/scale, HST_pix_size/3600/scale

            # Define the shape of the output image
            shape_out = (HST_img_center.shape[1], HST_img_center.shape[0])
            # print(shape_out, HST_img_center.shape, cut_data_shape)
            
            # Reproject the HST image on newly defined WCS  
            HST_img_center = reproject_adaptive((HST_img_center, input_wcs),
                                                output_wcs, shape_out=shape_out,
                                                kernel=kernel,
                                                return_footprint=False)
            
            ax = fig.add_subplot(2, 5, 1, projection=output_wcs)
            ax.imshow(HST_img_center,
                        cmap='viridis',
                        vmax=np.nanpercentile(HST_img_center, 99),
                        vmin=np.nanpercentile(HST_img_center, 10)
                        )
            
            # Plot center indicator of the center of the image
            ax.plot([size*scale, size*scale], [size*scale+0.5/HST_pix_size, size*scale+1/HST_pix_size], color='red', lw=3)
            ax.plot([size*scale+0.5/HST_pix_size, size*scale+1/HST_pix_size], [size*scale, size*scale], color='red', lw=3)
            
            # mag = round(row[f'{filters[f]}_MAG_AUTO'], 3)
            # mag_err = round(row[f'{filters[f]}_MAGERR_AUTO'], 3)
            ax.set_title("F814W", fontsize=30)

            # Hiding x and y axis ticks on regular axis
            ax.coords[0].set_ticks_visible(False)
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[1].set_ticks_visible(False)
            ax.coords[1].set_ticklabel_visible(False)
            
        except Exception as e:
            print(str(e))
            pass
        
        # set the super title
        fig.suptitle(f"ID: {i} {round(row['RA'], 5)} {round(row['DEC'], 5)}", fontsize=30)
        
        # plot data from JWST
        for f in range(len(filters)):
            ax = fig.add_subplot(2, 5, f+2, projection=WCSs[filters[f]])
            img_data = images[filters[f]]
            # print(img_data.shape)
            
            # Calibrate the coordinate to F444W image
            ra_offset, dec_offset = all_offsets[filters[f]]
            # print(ra_offset, dec_offset)
            x_single, y_single = WCSs[filters[f]].all_world2pix(row[f'RA'], row[f'DEC'], 0)
            x_single = x_single - ra_offset*3600/pix_sizes[filters[f]]
            y_single = y_single - dec_offset*3600/pix_sizes[filters[f]]
            # print(x_single, y_single)
            
            # Plot the image
            y_start = max(0, int(y_single - size))
            y_end = min(img_data.shape[0], int(y_single + size))
            x_start = max(0, int(x_single - size))
            x_end = min(img_data.shape[1], int(x_single + size))

            cut_data_shape = (2*size, 2*size)
            img_center = np.full(cut_data_shape, np.nan)

            y_offset = max(0, int(size - y_single))
            x_offset = max(0, int(size - x_single))

            img_center[y_offset:y_offset+(y_end-y_start), x_offset:x_offset+(x_end-x_start)] = img_data[y_start:y_end, x_start:x_end]
            
            ax.imshow(img_center, 
                    vmax=np.nanpercentile(img_center.flatten(), 99), 
                    vmin=np.nanpercentile(img_center.flatten(), 10)
                    )

            # Plot indicator of the center of the image
            ax.plot([size, size], [size+0.5/pix_sizes[filters[f]], size+1/pix_sizes[filters[f]]], color='red', lw=3)
            ax.plot([size+0.5/pix_sizes[filters[f]], size+1/pix_sizes[filters[f]]], [size, size], color='red', lw=3)

            mag = round(row[f'{filters[f]}_MAG_AUTO'], 3)
            mag_err = round(row[f'{filters[f]}_MAGERR_AUTO'], 3)
            ax.set_title(f"{filters[f]} \n {mag}$\pm${mag_err}" , fontsize=30)
            
            # Hiding x and y axis ticks
            ax.coords[0].set_ticks_visible(False) 
            ax.coords[0].set_ticklabel_visible(False) 
            ax.coords[1].set_ticks_visible(False)
            ax.coords[1].set_ticklabel_visible(False)
            
            # ax.set_xlabel(f"", fontsize=26)
                        
            # # Plot the Radius vs Magnitude plot
            # ax = fig.add_subplot(2, 5, f+7)
            # ax.scatter(target[f'{filters[f]}_PETRO_RADIUS'] * target[f'{filters[f]}_A_IMAGE'], 
            # target[f'{filters[f]}_MAG_PETRO'], s=50, label=f'{filters[f]}',
            # alpha=0.7, edgecolors='none', c='gray')

            # ax.scatter(row[f'{filters[f]}_PETRO_RADIUS'] * row[f'{filters[f]}_A_IMAGE'],
            #             row[f'{filters[f]}_MAG_PETRO'], s=40, label=f'{filters[f]}',
            #             alpha=1, edgecolors='none', c='red')
            
            # # axes labels & title & ticks size
            # ax.set_xlabel('PETRO_RADIUS [pix]', fontsize=26)
            # if f+7 == 7:
            #     ax.set_ylabel('MAG_PETRO [AB Mag]', fontsize=26)
            # ax.set_title(f'{filters[f]}', fontsize=30)
            # ax.tick_params(axis='both', which='major', labelsize=20)
            # ax.invert_yaxis()
            # ax.set_xlim(0, 125)
            # ax.set_ylim(28.5, 21)
            
        plt.savefig(f"./cutouts/{area}_{resolution}_{i}.png")
        
def plot_ellipse(isolist):
    """
    Plot the ellipse fitting results.
    
    Args:
    -----
        isolist: photutils.isophote.IsophoteList
            The list of the ellipse fitting results.
    """
    import matplotlib.pyplot as plt
    
    
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    plt.subplot(2, 2, 1)
    plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
                fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('Ellipticity')

    plt.subplot(2, 2, 2)
    plt.errorbar(isolist.sma, isolist.pa / np.pi * 180.0,
                yerr=isolist.pa_err / np.pi * 80.0, fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('PA (deg)')

    plt.subplot(2, 2, 3)
    plt.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('x0')

    plt.subplot(2, 2, 4)
    plt.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('y0')
    
def aper_photometry(a:float,
                    b:float,
                    pos_angle:float,
                    image:np.array,
                    radii:list[float],
                    show=False):
    """
    Perform aperture photometry on the center of the input image.
    Aperture sizes are given in the unit of pixel.

    Args:
    -----
        a: float
            The semi-major axis of the aperture. (in pixel)
        b: float
            The semi-minor axis of the aperture. (in pixel)
        pos_angle: float
            The position angle of the aperture. (in degree)
        image: np.array
            The image to be analyzed.
        radii: List[float]
            The radii of the aperture and annulus.
        show: bool
            Whether to show the aperture and annulus on the image.
    
    Returns:
    --------
        table: pd.DataFrame
            The results of the photoutil aperture photometry.
    """
    from photutils.aperture import EllipticalAperture, EllipticalAnnulus
    from photutils.aperture import aperture_photometry, ApertureStats
    
    center = (image.shape[0]/2, image.shape[1]/2)
    
    # Define the aperture and annulus
    apertures = [EllipticalAperture(center, scale*a, scale*b, pos_angle* np.pi / 180.0) for scale in radii]
    annuli = [EllipticalAnnulus(center, a_in=scale*a+3, a_out=scale*a+10, b_in=scale*b+3 ,b_out=scale*b+10, 
                                theta=pos_angle* np.pi / 180.0) for scale in radii]
    
    # Perform the aperture photometry that include the error
    error = np.sqrt(image)
    error = np.where(np.isnan(error), 0.0, error)
    phot_table = aperture_photometry(image, apertures, error=error)
    bkg_table = aperture_photometry(image, annuli, error=error)
    
    # Calculate the background subtracted flux
    for i in range(len(radii)):
        bkg_area = annuli[i].area_overlap(image)
        phot_table[f"aperture_sum_{i}"] = phot_table[f"aperture_sum_{i}"] - bkg_table[f"aperture_sum_{i}"] / bkg_area
        phot_table[f"aperture_sum_err_{i}"] = np.sqrt(phot_table[f"aperture_sum_err_{i}"]**2 + (bkg_table[f"aperture_sum_{i}"]/bkg_area)**2)
        
    print(phot_table.to_pandas())     
          
    # Plot the aperture and annulus and data
    if show:
        fig = plt.figure(figsize=(22, 11))
        ax = fig.add_subplot(1, 2, 1)
        # for aper in apertures:
        #     aper.plot(color='white', lw=2)
        # for annulus in annuli:
        #     annulus.plot(color="red", lw=2)
        ax.imshow(image, origin='lower')
        
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(error, origin='lower')
    
    table = phot_table.to_pandas()
    return table

def plot_ellipse_iso(image, model_image, isolist):
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(image, origin='lower')
    ax1.set_title('Data')

    residual = image - model_image

    ax2.imshow(model_image, origin='lower')
    ax2.set_title('Ellipse Model')

    ax3.imshow(residual, origin='lower')
    ax3.set_title('Residual')

def map_HST_data(area):
    """
    Find the mapping of HST data on the certain COSMOS-Webb area.
    
    Args:
    -----
        area: str
            The area of the COSMOS-Webb image.
    
    Returns:
    --------
        None
    """
    from shapely.geometry import Polygon as shapely_polygon
    
    mapping = {}
    mapping[area] = []
    
    # Read COSMOS-Webb area data
    with fits.open(f"./30mas/mosaic_nircam_f444w_COSMOS-Web_30mas_{area}_v0_5_SCI.fits") as hdul:
        header = hdul[1].header
    
    webb_wcs = WCS(header)
    webb_shape = (header["NAXIS1"], header["NAXIS2"])
    
    # Find four corners of the JWST image
    webb_ra_c1, webb_dec_c1 = webb_wcs.wcs_pix2world(0, 0, 0)
    webb_ra_c2, webb_dec_c2 = webb_wcs.wcs_pix2world(webb_shape[0], 0, 0)
    webb_ra_c3, webb_dec_c3 = webb_wcs.wcs_pix2world(0, webb_shape[1], 0)
    webb_ra_c4, webb_dec_c4 = webb_wcs.wcs_pix2world(webb_shape[0], webb_shape[1], 0)
    
    # construct a rectangle using the coordinates
    webb_polygon = shapely_polygon([(webb_ra_c1, webb_dec_c1), 
                                    (webb_ra_c2, webb_dec_c2), 
                                    (webb_ra_c4, webb_dec_c4), 
                                    (webb_ra_c3, webb_dec_c3)])

    # Find the HST data within the COSMOS-Webb area
    HST_dir = "/mnt/D/JWST_data/COSMOS-Webb/others/HST/"
    for file in glob.glob(f"{HST_dir}*.fits"):
        print(f"Checking HST data: {file.split('/')[-1]}")
        with fits.open(file, memmap=False) as hdul:
            header = hdul[0].header
        
        shape = (header["NAXIS1"], header["NAXIS2"])
        wcs = WCS(header)
        
        # Find four corners of the HST image
        ra_c1, dec_c1 = wcs.wcs_pix2world(0, 0, 0)
        ra_c2, dec_c2 = wcs.wcs_pix2world(shape[0], 0, 0)
        ra_c3, dec_c3 = wcs.wcs_pix2world(0, shape[1], 0)
        ra_c4, dec_c4 = wcs.wcs_pix2world(shape[0], shape[1], 0)

        # construct a rectangle using the coordinates
        hst_polygon = shapely_polygon([(ra_c1, dec_c1), (ra_c2, dec_c2), (ra_c4, dec_c4), (ra_c3, dec_c3)])
        
        # Check if the rectangles are overlapping
        if webb_polygon.intersects(hst_polygon):
            # print(f"HST data: {file.split('/')[-1]} is within COSMOS {area}...")
            # # plot the two polygons 
            # fig, ax = plt.subplots()
            # ax.plot(*webb_polygon.exterior.xy)
            # ax.plot(*hst_polygon.exterior.xy)
            # plt.show()
            
            mapping[area].append(file.split('/')[-1])
            
    with open(f"./HST_mapping/{area}_mapping.json", "w") as f:
        json.dump(mapping, f)
    
def find_HST_data(coordinate, area):
    """
    Find the HST image that contains the coordinate.
    
    Args:
    -----
    coordinate: tuple (float, float)
        The ra and dec of the target in the unit of degree.
    area: str
        The area of the image.(A1-A10)
        
    Returns:
    --------
    file_path: str
        The path to the HST image that contains the target.
    """
    # search images from mapping
    print(f"Search {area} in HST image......")
    mapping = json.load(open(f"./HST_mapping/{area}_mapping.json", "r"))
    HST_dir = "/mnt/D/JWST_data/COSMOS-Webb/others/HST"
    
    print(f"Finding source in which tiles......")
    # for file in glob.glob(f"{HST_dir}/*sci*.fits"):
    for file in mapping[area]:
        with fits.open(f"./others/HST/{file}") as hdul:
            header = hdul[0].header
        
        shape = (header["NAXIS1"], header["NAXIS2"])
        wcs = WCS(header)
        
        # Find four corners of the HST image
        ra_c1, dec_c1 = wcs.wcs_pix2world(0, 0, 0)
        ra_c2, dec_c2 = wcs.wcs_pix2world(shape[0], 0, 0)
        ra_c3, dec_c3 = wcs.wcs_pix2world(0, shape[1], 0)
        ra_c4, dec_c4 = wcs.wcs_pix2world(shape[0], shape[1], 0)

        # construct a rectangle using the coordinates
        hst_polygon = shapely_polygon([(ra_c1, dec_c1), (ra_c2, dec_c2), (ra_c4, dec_c4), (ra_c3, dec_c3)])
        
        if Point(coordinate[0], coordinate[1]).within(hst_polygon):
            return f"{HST_dir}/{file}"
        
def make_colors(resolution, area, date, index):
    """
    Construct the colors with 4 filters with the given resolution and area.
    
    Args:
    -----
    resolution: str
        The resolution of the image
    area: str
        The area of the image
    date: str
        The date on the run of SEx
    index: int
        The index of the output sex catalog
        
    Returns:
    --------
    color_115_150: np.array
        The color of F115W - F150W.
    color_150_277: np.array
        The color of F150W - F277W.
    color_277_444: np.array
        The color of F277W - F444W.
    total_cat: pd.DataFrame
        The total catalog of the 4 filters.
    """

    limit_mag = {
        "F115W": 27.45,
        "F150W": 27.66,
        "F277W": 28.28,
        "F444W": 28.17
    }

    df_115 = ascii.read(f"./catalogs/updated/{date}_{index}/F115W_{resolution}_{area}.cat", 
                        guess=False, 
                        format="commented_header", 
                        fast_reader=True).to_pandas()
    df_150 = ascii.read(f"./catalogs/updated/{date}_{index}/F150W_{resolution}_{area}.cat", 
                        guess=False, 
                        format="commented_header", 
                        fast_reader=True).to_pandas()
    df_277 = ascii.read(f"./catalogs/updated/{date}_{index}/F277W_{resolution}_{area}.cat", 
                        guess=False, 
                        format="commented_header", 
                        fast_reader=True).to_pandas()
    df_444 = ascii.read(f"./catalogs/updated/{date}_{index}/F444W_{resolution}_{area}.cat", 
                        guess=False, 
                        format="commented_header", 
                        fast_reader=True).to_pandas()

    # Add filter prefix to the column names
    add_prefix_columns(df_115, "F115W")
    add_prefix_columns(df_150, "F150W")
    add_prefix_columns(df_277, "F277W")
    add_prefix_columns(df_444, "F444W")

    # Crossmatch for 4 filters
    temp1 = crossmatch(df_115, df_150, max_sep=0.125, include_unmatched=False, verbose=False)
    temp2 = crossmatch(temp1, df_277, max_sep=0.125, include_unmatched=False, verbose=False)
    temp3 = crossmatch(temp2, df_444, max_sep=0.125, include_unmatched=False, verbose=False)

    # Extract the AB_MAG_AUTO values
    AB_AUTO_115 = temp3["F115W_MAG_AUTO"]
    AB_AUTO_150 = temp3["F150W_MAG_AUTO"]
    AB_AUTO_277 = temp3["F277W_MAG_AUTO"]
    AB_AUTO_444 = temp3["F444W_MAG_AUTO"]

    # Drop sources that is below limiting magnitude
    # df_115 = df_115[df_115["MAG_AUTO"]<=limit_mag["F115W"]]
    # df_150 = df_150[df_150<=limit_mag["F150W"]]
    # df_277 = df_277[df_277<=limit_mag["F277W"]]
    # df_444 = df_444[df_444<=limit_mag["F444W"]]

    # Construct Colors
    color_115_150 = AB_AUTO_115 - AB_AUTO_150
    color_150_277 = AB_AUTO_150 - AB_AUTO_277
    color_277_444 = AB_AUTO_277 - AB_AUTO_444
    
    total_cat = temp3
    
    return color_115_150, color_150_277, color_277_444, total_cat

def mag_comparison(_filter, resolution, area):
    """
    Make a plot of the comparison between the F150W MAG_AUTO and UVISTA_H_MAG_AUTO.
    The plot will show the median of each bin of F150W MAG_AUTO with its standard error.
    The plot will also show the 1:1 line.
    
    Args:
    -----
        _filter: str 
            The filter name
        resolution: str 
            The resolution of the image
        area: str
            The area of the image
    Returns:
    --------
        None
    """
    
    # Read the output catalog
    sex_table = ascii.read(f"./catalogs/{_filter}_{resolution}_{area}_updated.cat", 
                        guess=False, 
                        format="commented_header", 
                        fast_reader=True)

    # Read the image shape
    shape = fits.getdata(f"./{resolution}/mosaic_nircam_{_filter.lower()}_COSMOS-Web_{resolution}_{area}_v0_5_SCI.fits").shape

    # Cut the table by the boudaries of the image shape
    sorted_table = sex_table[(sex_table["X_IMAGE"] > 200) 
                    & (sex_table["Y_IMAGE"] > 200) 
                    & (sex_table["X_IMAGE"] < shape[1]-200) 
                    & (sex_table["Y_IMAGE"] < shape[0]-200)]

    # sort the table by the flux values (Increasing)
    sorted_table.sort("MAG_AUTO", reverse=False)

    # Read COSMOS2020 catalog
    C_2020_fits = fits.open(f"/mnt/C/JWST/COSMOS/NIRCAM/catalog/official/COSMOS2020_CLASSIC_R1_v2.2_p3.fits")
    COSMOS_2020 = pd.DataFrame(C_2020_fits[1].data)
    
    # cut the table into smaller pieces based on the ra and dec values
    ra_min, ra_max = sorted_table["ALPHA_J2000"].min(), sorted_table["ALPHA_J2000"].max() 
    dec_min, dec_max = sorted_table["DELTA_J2000"].min(), sorted_table["DELTA_J2000"].max()
    CUT_2020 = COSMOS_2020[(COSMOS_2020["ALPHA_J2000"] > ra_min) 
                        & (COSMOS_2020["ALPHA_J2000"] < ra_max) 
                        & (COSMOS_2020["DELTA_J2000"] > dec_min) 
                        & (COSMOS_2020["DELTA_J2000"] < dec_max)]
    
    # Perform crossmatch
    matched_sources = crossmatch(CUT_2020.reset_index(), 
                             sorted_table.to_pandas(index="NUMBER"), 
                             max_sep=0.075, 
                             return_cat=True, 
                             sep=True, 
                             include_unmatched=False)

    # overplot the median of each bin of UVISTA_H_MAG_AUTO with its standard error
    all_median, all_std = [], []
    bins = np.arange(18, 32, 0.5)
    for i in bins:
        bin_sources = matched_sources[matched_sources["MAG_AUTO"].between(i, i+0.5)]
        median = bin_sources["UVISTA_H_MAG_AUTO"].median()
        std = bin_sources["UVISTA_H_MAG_AUTO"].std()
        all_median.append(median)
        all_std.append(std)
        
    # plt median and error bars
    all_median = np.array(all_median)
    all_std = np.array(all_std)
    plt.plot(bins, all_median, color="blue", zorder=5, label=f"Median")
    plt.plot(bins, all_median + all_std, color="red", linestyle="--", zorder=4, label=f"1 $\sigma$ error")
    plt.plot(bins, all_median - all_std, color="red", linestyle="--", zorder=4)

    # plot 1:1 line
    plt.plot(np.arange(10, 35), np.arange(10, 35), color="black", label="1:1 line", zorder=3)

    # plot the F150W MAG_AUTO vs UVISTA_H_MAG_AUTO with error bars
    plt.errorbar(matched_sources["UVISTA_H_MAG_AUTO"], matched_sources["MAG_AUTO"],
                xerr=0, yerr=0, 
                fmt="o", markersize=2, alpha=0.2, zorder=1,
                )

    plt.xlabel("F150W MAG AUTO", fontsize=26)
    plt.ylabel("UVISTA H MAG AUTO", fontsize=26)
    plt.title(f"{_filter} {resolution} {area}", fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # limit the axis to avoid scientific notation
    plt.xlim(18, 32)
    plt.ylim(18, 32)

    # vertical lines
    plt.axvline(27, color="violet", linestyle="--", alpha=0.7)

    plt.legend()
    plt.show()
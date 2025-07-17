"""
Structure
---------

1. Aurora Kesseli's functions. Many of these include tweaks by me (MJF)
2. My functions
3. Functions from Aurora and myself relating to using model templates instead
of real templates

"""

import os
import numpy as np
from astropy.io import fits
from PyAstronomy import pyasl
from scipy.signal.windows import boxcar
from scipy.signal import convolve, peak_widths, argrelextrema, find_peaks
import astropy.convolution as astroconv
from ldtk import LDPSetCreator, BoxcarFilter


### 1. ###
## Aurora Kesseli's functions
# I note where I have tweaked them denoted by "MJF - ..."
# Docstrings mostly written by MJF

def interpOntoEvenGrid(wave, flux):
    """
    Need evenly spaced wavelength values for the rotational broadening kernel to work
    """

    #see what the average spacing is
    medSpacing = np.mean(np.diff(wave)) # MJF - changed median to mean bc median was 0, resulting in errors
    
    #Make the new wavelength list
    # MJF - changed this to linspace to preserve the bounds and the shape
    newWave = np.linspace(wave[0], wave[-1], round(1 + ( (wave[-1]-wave[0]) / medSpacing) ))

    #interpolate the flux and variance to the same values
    # MJF - changed this to np.interp and removed fill_value='extrapolate' 
    newFlux = np.interp(newWave, wave, flux)
    
    return newWave, newFlux

def makeArtificialSlowRots(newWave, newFlux, coef):
    """
    Apply the rotational broadening kernel to the evenly spaced flux and wavelength values.
    Returns a list of fluxes that match newWave and are artificially broadened for values from
    2.0 to 65 km/s every 4 km/s (chose a spacing of 4km/s since that is similar to IGRINS resolution)

    Parameters
    ----------
    newWave, newFlux : array-like
        The wavelength and flux arrays. The "new" prefix means these are intended to be the wavelength and flux
        arrays of the evenly-spaced grid made with interpOntoEvenGrid
    coef : float
        The limb-darkening coefficient.

    Returns
    -------
    velocities : np.ndarray
        The array of velocity values in km/s
    fluxList : list
        List containing the broadened flux array at each velocity
    """

    #array of velocity values for the rotational broadening routine
    velocities = np.arange(2.0, 65.0, 4.0) 
    # velocities = np.arange(2.0, 105.0, 4.0)
    fluxList = []
    #loop through the velocities and add the rotationally broadened flux to a list
    for i in range(len(velocities)):
        flux = pyasl.rotBroad(newWave, newFlux, coef, velocities[i])
        fluxList.append(flux)
    
    return velocities, fluxList

def padCrossCorrelation(wavelength, flux):
    """
    Pad spectrum of the template by wraping the spectrum

    Used in crossCorrelate. Cross-correlation works by shifting a signal
    (in this case a spectrum) relative to another signal and comparing the two.
    Padding the spectrum (essentially appending copies of the spectrum onto 
    the end of itself) allows this shifting to happen without going past
    the edge of the bounds.
    """
    
    #see what the average spacing is
    avgSpacing = np.mean(np.diff(wavelength))
    #see what the length of the wavelengths array is
    lenWave = len(wavelength)
    #figure out the lowest wavelength value to have 3 wrapped template spectra
    lowWave = wavelength[0] - (avgSpacing*lenWave)
    highWave = wavelength[-1] + (avgSpacing*(lenWave+1))
    #make a new wavelength grid
    # MJF - change this to linspace to preserve the bounds and the shape
    newWave = np.linspace(lowWave, highWave, round(1 + ( (highWave-lowWave) / avgSpacing) ))
    #append 3 of the flux grids together
    newFlux = np.concatenate((flux, flux, flux))

    return newWave, newFlux

def crossCorrelate(wavelength, flux, waveTemplate, fluxTemplate):
    """
    Cross-correlate the target flux with the template flux
    (or the "fast" rotating flux with the "slow" rotating flux if
    comparing an artificially-broadened template to the original template).

    Returns
    -------
    [0] : np.ndarray
        An array of radial velocity values
    [1] : np.ndarray
        An array of cross-correlation function (CCF) values
    
    Essentially, these are the x and y values, respectively, of a CCF plot
    """

    #need the flux to be evenly spaced in log wavelength units
    logWave = np.log10(wavelength)
    logWaveTemp = np.log10(waveTemplate)
    
    #want my rv grid to be spaced every 0.1 km/s
    # MJF - change this to linspace to preserve the bounds and the shape
    newLogWave = np.linspace(
        min(logWave),
        max(logWave),
        round(1 + ( (max(logWave)-min(logWave)) / (0.1*np.log10(np.e)/(2.998*10**5))) )
    )

    #interpolate the fluxes onto that grid
    # MJF - changed this to np.interp and removed fill_value='extrapolate' 
    newFlux = np.interp(newLogWave, logWave, flux)
    newFluxTemp = np.interp(newLogWave, logWaveTemp, fluxTemplate)

    #pad the template but only need ~150 km/s
    paddedWaveTemp, paddedFluxTemp = padCrossCorrelation(newLogWave, newFluxTemp)
    paddedWaveTemp = paddedWaveTemp[len(newLogWave)-2000:-(len(newLogWave)-2000)]
    paddedFluxTemp = paddedFluxTemp[len(newLogWave)-2000:-(len(newLogWave)-2000)]

    #cross correlate
    cc = np.correlate(paddedFluxTemp, newFlux, mode='valid')

    #find the corresponding radial velocity grid
    rvGrid = []
    for i in range(len(cc)):
        rv = (10**(newLogWave[-1])/10**(paddedWaveTemp[len(newLogWave)+i-1]) -1 )*2.998*10**5
        rvGrid.append(rv)

    sort_idx = np.argsort(rvGrid) # MJF - so that rv is in the correct (ascending) order

    return np.array(rvGrid)[sort_idx], cc[sort_idx]

def measureFWHM(RV, CC, clip_ends=False):
    """
    Measure the FWHM of the cross correlation function.
    This function should be used for high SNR (~100) spectra only since it does not 
    fit a gaussian or curve to the cc function, and noisy peaks in the cc function 
    can significantly change the FWHM.

    MJF - one of the updated functions; it's completely different from the original
    (in form but not in function)
    """

    maxima = argrelextrema(CC, np.greater)[0]
    
    if clip_ends:
        if len(maxima) > 2:
            maxima = maxima[1:-1] # cut off the ends because these sometimes cause trouble

    widths, heights, lefts, rights = peak_widths(CC, maxima, rel_height=0.5)

    mask = np.argmax(CC[maxima])
        
    spl = np.interp([lefts[mask], rights[mask]], np.arange(len(RV)), RV)
    
    FWHM = np.diff(spl)[0]

    return FWHM, int(maxima[mask]), spl


### 2. ###
## MJF functions

def get_spectrum_orders(filepath, orders, nsub=None):
    """
    Takes a filepath and returns a dictionary with wavelength and
    flux arrays for each order specified.

    The keyword argument "nsub" is seldom used. It allows us to break up
    a spectrum order into subsecions (given by the integer nsub).

    The cuts for specific orders are taken from Aurora Kesseli's code.

    Example
    -------
    ```
    >>> d = get_spectrum_orders(<path-to-fits-file>, [4, 5])
    >>> d
    {
        'wave4' : <np.ndarray>,
        'flux4' : <np.ndarray>,
        'wave5' : <np.ndarray>,
        'flux5' : <np.ndarray>
    }
    ```
    """
    
    order_dict = dict()
    
    with fits.open(filepath) as hdul:
    
        for n in orders:
            
            wave = hdul['WAVELENGTH'].data[n]
            flux = hdul['SPEC_DIVIDE_A0V'].data[n]
            
            if n == 4:
                clipwave = wave[200:500]
                clipflux = flux[200:500]
                
            elif n == 5:
                clipwave = wave[120:1000]
                clipflux = flux[120:1000]
                
            else:
                clipwave = wave[120:-70]
                clipflux = flux[120:-70]
                
            normflux = clipflux / np.nanmedian(clipflux)
            
            # try a boxcar smoothing if 2 pixels
            smoothflux = convolve(normflux, boxcar(M=3))
            norm_smoothflux = smoothflux[2:-2] / np.nanmedian(smoothflux) # changed np.mean to np.nanmedian; changes flux values slightly from Aurora's code
            
            finalwave = clipwave[1:-1]
            finalflux = norm_smoothflux
            
            
            if nsub is not None:
                
                incr = len(finalwave) // nsub
                
                for i in range(nsub):
                    
                    subwave = finalwave[incr*(i):incr*(i+1)]
                    subflux = finalflux[incr*(i):incr*(i+1)]
                    
                    order_dict.update({
                        f'wave{n}.{i+1:02}' : subwave,
                        f'flux{n}.{i+1:02}' : subflux,
                        })
            
            else:
            
                order_dict.update({
                    f'wave{n}' : finalwave,
                    f'flux{n}' : finalflux
                    })
        
    
    return order_dict

def interp_vsini(FWHM, artbroadFWHM_list, velocities):
    """
    Interpolate (linearly) a vsini value given the FWHM of the target (or "fast")
    CCF, the grid of FWHMs of artificially-broadened CCFs, and the list of
    vsini values corresponding to the latter.

    Returns
    -------

    vsiniFWHM : float
        The vsini value in km/s
    """
    
    vsiniFWHM = np.interp(FWHM, artbroadFWHM_list, velocities)
    
    return vsiniFWHM

### 3. ###
## Functions for model templates

def convolveToR(wave, flux, R):
    """
    Aurora Kesseli's function.

    Convolves the model spectrum (wave, flux) to the target spectrograph
    resolution (R).

    Returns
    -------
    convData : np.ndarray
        The convolved model flux (wavelength remains the same).
    """
    
    mid_index = int(len(wave)/2)
    deltaWave = np.mean(wave)/ R
    left = wave[mid_index-1]
    right = wave[mid_index]

    # MJF - Changed this to '-i' instead of '-1' because the spacing and precision can make mid_index-1 and 
    # mid_index point to the same value
    i = 1
    while abs(right-left) == 0:
        i += 1
        left = wave[mid_index-i]

    fwhm = deltaWave / ((wave[mid_index] - wave[mid_index-i]) / i)
    std = fwhm / ( 2.0 * np.sqrt( 2.0 * np.log(2.0) ) ) #convert FWHM to a standard deviation
    g = astroconv.Gaussian1DKernel(stddev=std)
    #2. convolve the flux with that gaussian
    flux = np.asarray(flux)
    convData = astroconv.convolve(flux, g)

    return convData

def prepare_model(
    path,
    targetwave,
    targetflux,
    wave_correction=None,
    peak_height=None,
):
    """
    MJF function

    1. Takes a path where the model spectrum is located
    2. Finds the median spectral resolution of the target spectrum from
    widths of the prominent absorption lines
    3. Convolves the model spectrum to that median R
    4. Returns: model wavelength and flux, locations of the absorption lines,
    and the median R.
    """

    if wave_correction is None:
        wave_correction = 1e-4

    if peak_height is None:
        peak_height = 0.9

    slow = np.loadtxt(path)

    slowwave = slow[:, 0] * wave_correction
    slowflux = slow[:, 1]

    mask = np.where((slowwave >= targetwave.min()) & (slowwave <= targetwave.max()))

    if peak_height is None:
        h = peak_height
    else:
        h = -peak_height
    
    peaks, properties = find_peaks(-targetflux, height=h)

    width_results = peak_widths(-targetflux, peaks=peaks, rel_height=0.5)
    
    spl = np.interp([*width_results[2:]], np.arange(len(targetwave)), targetwave)

    Rlist = [(targetwave[peak] / (right - left)) for (peak, (left, right)) in zip(peaks, spl.T)]

    newslowflux = convolveToR(slowwave[mask], slowflux[mask], np.median(Rlist))

    return slowwave[mask], newslowflux / np.median(newslowflux), peaks, np.median(Rlist)

def limbdark(wavemin, wavemax, teff_tup=None, m=None, r=None, logg=None, z_tup=None, model='vis-lowres'):
    """
    MJF function

    Calculates the limb-darkening coefficient using LDTk (github.com/hpparvi/ldtk).

    Parameters
    ----------
    wave{min,max} : float
        The lower,upper extremes of the wavelength array.
    teff_tup : tuple of floats; optional
        The effective temperature (value, error) tuple.
    m, r : float; optional
        The mass and radius, respectively in solar units. If both are passed,
        logg is not needed. Otherwise, logg must be passed.
    logg : float; optional
        Log-surface gravity in log(cm/s**2). If passed, mass and radius are not
        necessary. Otherwise, mass and radius are required to calculate logg.
    z_tup : tuple of floats; optional
        Metallicity (value, error) tuple. If not passed, the default is
        (0, 0.1).
    model : str; optional
        The dataset used for calculating the limb-darkening coefficient.
        See the LDTk documentation for more options. The default is 'vis-lowres'.

    Returns
    -------
    coeff[0][0] : float
        The limb-darkening coefficient
    
    Note
    ----
    Sometimes the backend calculation gets confused (possibly from using the 
    low resolution model) and returns a clearly wrong value. Usually running the 
    function again returns a more realistic value.
    """
    
    filter_ = BoxcarFilter("filter", wavemin*1000, wavemax*1000) # um -> nm
    
    if logg is None:
        # these parameters are for the template
        logg = np.log10(100 * 6.67e-11 * m * 1.989e30 / r**2 / 6.957e8**2) # [cm/s^2]
        e_logg = 0.1
    
    elif isinstance(logg, (list, tuple)):
        logg, e_logg = logg
    
    elif isinstance(logg, float):
        e_logg = 0.1

    if z_tup is None:
        z_tup = (0, 0.1)
    
    sc = LDPSetCreator(teff=teff_tup, logg=(logg, e_logg), z=z_tup, filters=[filter_], dataset=model)
    
    ps = sc.create_profiles()
    
    coeff, _ = ps.coeffs_ln()
    
    return coeff[0][0]
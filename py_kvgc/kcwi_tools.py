import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import cosmology
import reproject
import pyregion
import os
import warnings
import pdb
from matplotlib import pyplot as plt


def subcube(hdu,wave,writefn='',box=[-1,-1,-1,-1],pixel_wave=False,pixel_box=True):
    """
    Trim a big cube down to a smaller one.

    Parameters
    ----------
    hdu: HDU object or str
        The input data cube in header-data unit or a string specifying its
        path.

    wave: array_like with 2 elements
        The lower and higher boundaries in the wavelength direction.

    writefn: str, optional
        The file name of the output HDU.

    box: array_like with 4 elements, optional
        Coordinates of the lower-left and upper-right corners of the sub-cube.

    pixel_wave: bool, optional
        Using pixel coordinate in the wavelength direction? Default: False.

    pixel_box: bool, optional
        Using pixel coordinate in the spatial directions? Default: True.

    Returns
    -------
        newhdu: HDU object
            The extracted sub-cube.

    """

    if type(hdu)==type(''):
        tmp=fits.open(hdu)
        hdu=tmp[0]

    shape=hdu.data.shape
    wcs0=wcs.WCS(hdu.header)
    shape0=hdu.data.shape
    wave0=wcs0.wcs_pix2world(np.zeros(shape0[0]),np.zeros(shape0[0]),np.arange(shape0[0]),0)
    wave0=wave0[2]*1e10

    newhdu=hdu.copy()


    if pixel_wave==False:
        qwave=(wave0 >= wave[0]) & (wave0 < wave[1])
    else:
        qwave=(np.arange(shape0[2]) >= wave[0]) & (np.arange(shape0[2]) < wave[1])

    if np.sum(qwave)==0:
        return -1

    newwave=wave0[qwave]
    newhdu.data=newhdu.data[qwave,:,:]

    if np.sum(newhdu.data)==0:
        return -1

    newhdu.header['NAXIS3']=newhdu.data.shape[0]
    newhdu.header['CRPIX3']=1
    newhdu.header['CRVAL3']=newwave[0]

    if box[0]!=-1:
        ra0=wcs0.wcs_pix2world(np.arange(shape0[2]),np.zeros(shape0[2]),np.zeros(shape0[2]),0)
        ra0=ra0[0]
        dec0=wcs0.wcs_pix2world(np.zeros(shape0[1]),np.arange(shape0[1]),np.zeros(shape0[1]),0)
        dec0=dec0[1]

        if pixel_box==False:
            # real RA DEC from WCS

            qra=(ra0 <= box[0]) & (ra0 > box[2])
            qdec=(dec0 >= box[1]) & (dec0 < box[3])
        else:
            # pixel RA DEC
            qra=(np.arange(shape[2]) >= box[0]) & (np.arange(shape[2]) < box[2])
            qdec=(np.arange(shape[1]) >= box[1]) & (np.arange(shape[1])< box[3])

        newra=ra0[qra]
        newdec=dec0[qdec]
        newhdu.data=newhdu.data[:,qdec,:]
        newhdu.data=newhdu.data[:,:,qra]

        newhdu.header['NAXIS1']=newhdu.data.shape[2]
        newhdu.header['NAXIS2']=newhdu.data.shape[1]
        newhdu.header['CRPIX1']=1
        newhdu.header['CRPIX2']=1
        newhdu.header['CRVAL1']=newra[0]
        newhdu.header['CRVAL2']=newdec[0]



    if writefn!='':
        newhdu.writeto(writefn,overwrite=True)

    return newhdu


def collapse_header(hdr):
    """
    Quick wrapper to collapse a 3-D header into a 2-D one.
    Removes the wavelength axis from the datacube.

    Parameters
    ----------
    hdr: header

    Returns
    -------
    hdr_img: collapsed header

    """

    hdr_img=hdr.copy()
    hdr_img['NAXIS']=2
    del hdr_img['NAXIS3']
    del hdr_img['CD3_3']
    del hdr_img['CTYPE3']
    del hdr_img['CUNIT3']
    del hdr_img['CNAME3']
    del hdr_img['CRVAL3']
    del hdr_img['CRPIX3']

    return hdr_img



def collapse(hdu,wavebin=[-1.,-1.],usepix=False,var=False,weight=False,usemean=False,usesum=False,writefn='',ignore_blank=False):
    """
    Collapse the cube into a 2-D image whitelight/narrowband image.

    Parameters
    ----------
    hdu: HDU object or string of the file name

    wavebin: array_like (n*2 elements), optional
        The range of which the cube is collapsed into.
        Can be split in n seperate ranges. The final image will be collapsed into one.

    usepix: bool, optional
        Use pixel indices for wavebin?

    var: bool, optional
        variance cube?

    weight: bool, optional
        Output image in weight, instead of variance.

    usemean: bool, optional
        Using mean instead of median.

    ignore_blank: bool, optional
        Ignore blank images without writing files.
    """

    # default wavebin
    tab_grating=np.array(['BL','BM'])
    tab_wave=np.array([500,300])

    if type(hdu)==type(''):
        ofn=hdu
        tmp=fits.open(hdu)
        hdu=tmp[0]
    else:
        ofn=''


    if type(wavebin)==type([]) or type(wavebin)==type(()):
        wavebin=np.array(wavebin)

    if weight==True:
        var=True

    if len(wavebin.shape)==1:
        wavebin=np.array([wavebin])


    # get cube parameters
    wcs0=wcs.WCS(hdu.header)
    shape0=hdu.data.shape
    wave0=wcs0.wcs_pix2world(np.zeros(shape0[0]),np.zeros(shape0[0]),np.arange(shape0[0]),0)
    wave0=wave0[2]*1e10
    cwave=hdu.header['BCWAVE']

    # get pixel indices
    if wavebin[0,0]==-1:
        grat=hdu.header['BGRATNAM']
        qg=(tab_grating==grat)

        if np.sum(qg)==0:
            wrange=[np.max([(cwave-500),3500]),np.min([(cwave+500),5500])]
        else:
            wrange=[np.max([(cwave-tab_wave[qg][0]),3500]),
                            np.min([(cwave+tab_wave[qg][0]),5500])]
            qwave=(wave0>wrange[0]) & (wave0<wrange[1])
    else:
        if usepix==False:
            qwave=np.zeros(wave0.shape[0],dtype=bool)
            for i in range(wavebin.shape[0]):
                qwave_tmp=(wave0>wavebin[i,0]) & (wave0<wavebin[i,1])
                qwave=np.bitwise_or(qwave_tmp,qwave)

        else:
            qwave=np.zeros(wave0.shape[0],dtype=bool)
            windex=np.arange(wave0.shape[0])
            for i in range(wavebin.shape[0]):
                qwave_tmp=(windex>wavebin[i,0]) & (windex<wavebin[i,1])
                qwave=np.bitwise_or(qwave_tmp,qwave)

    # collapse
    if np.sum(qwave)==0:
        if ignore_blank:
            return -1
        else:
            img=np.zeros(hdu.shape[1:3])

    else:
        cube_0=hdu.data.copy()
        cube_0[cube_0==0]=np.nan
        if var==False:
            if usemean:
                img=np.nanmean(cube_0[qwave,:,:],axis=0)
            elif usesum:
                img=np.nansum(cube_0[qwave,:,:],axis=0)
            else:
                img=np.nanmedian(cube_0[qwave,:,:],axis=0)
        else:
            if usemean==False:
                img=np.nanmean(cube_0[qwave,:,:],axis=0)/np.sum(np.isfinite(cube_0[qwave,:,:]),axis=0)
            elif usesum:
                img=np.nansum(cube_0[qwave,:,:],axis=0)
            else:
                img=np.nanmedian(cube_0[qwave,:,:],axis=0)/np.sum(np.isfinite(cube_0[qwave,:,:]),axis=0)

    # convert to weight
    if weight==True:
        img=np.nan_to_num(1./img)


    hdr_img=collapse_header(hdu.header)
    if weight==True:
        hdr_img['BUNIT']='weight'

    hdu_img=fits.PrimaryHDU(img,header=hdr_img)

    if writefn!='':
        hdu_img.writeto(writefn,overwrite=True)

    return hdu_img


import numpy as np


def iter_polyfit(x,y,deg,max_iter=5,nsig=2.5):
    """

    """
    
    y1=y.copy()
    x1=x.copy()

    index=(np.isfinite(y1) & (y1!=0))
    if np.sum(index)==0:
        poly_fit=np.poly1d([0])
        return poly_fit
    y1=y1[index]
    x1=x1[index]

    for i in range(max_iter):
        if deg>=1:
            param=np.polyfit(x1,y1,deg)
            poly_fit=np.poly1d(param)
            y_fit=poly_fit(x1)

            residual=y1-y_fit

            rms=np.sqrt(np.mean(residual**2))
            index=(residual < nsig*rms)
            
            if np.sum(~index)==0:
                break
            else:
                x1=x1[index]
                y1=y1[index]

        else:
            med=np.median(y1)
            poly_fit=np.poly1d([np.mean(y1)])
            residual=y1-med

            rms=np.sqrt(np.mean(residual**2))
            index=(residual < nsig*rms)

            if np.sum(~index)==0:
                break
            else:
                x1=x1[index]
                y1=y1[index]
    
    return poly_fit


def cont_sub(hdu,wrange,writefn='',fit_order=1,w_center=None,w_vel=False,auto_reduce=True):
    """
    Conduct continuum-subtraction to cubes.

    Parameters
    ----------
    hdu: HDU object or str
        The input data cube in header-data unit or a string specifying its
        path.

    wrange: N*2 array
        The range of wavelength in Angstrom for the polynomial fit.

    writefn: str, optional
        The file name of the output HDU.

    fit_order: int, optional
        Order of the polynomial fit.

    w_center: float, optional
        If w_vel==True, this specifies the center of the wavelength.

    w_vel: bool, optional
        Use velocity in km/s for the wrange bins, intead of Angstrom.

    auto_reduce: bool, optional
        Automatically reduce the fitting order to 0, if all valid wavelength bins are on
        one side of the central wavelength.

    """

    if w_vel==True and w_center is None:
        print('[Error] Central wavelength required for w_vel=True.')
        return -1

    if type(hdu)==type(''):
        ofn=hdu
        tmp=fits.open(hdu)
        hdu=tmp[0]
    else:
        ofn=''

    data_new=hdu.data.copy()
    data_new=np.nan_to_num(data_new)

    sz=hdu.shape
    wcs0=wcs.WCS(hdu.header)
    wave0=wcs0.wcs_pix2world(np.zeros(sz[0]),np.zeros(sz[0]),np.arange(sz[0]),0)
    wave0=wave0[2]*1e10
    if not (w_center is None):
        v0=(wave0-w_center)/w_center*3e5


    # Get wavelength bins
    if type(wrange)==type([]):
        wrange=np.array(wrange)

    if len(wrange.shape)==1:
        wrange=np.array([wrange])

    sz_wr=wrange.shape

    index0=np.zeros(sz[0],dtype=bool)
    for i in range(sz_wr[0]):
        if w_vel==True:
            index=((v0>=np.min(wrange[i,:])) & (v0<np.max(wrange[i,:])))
        else:
            index=((wave0>np.min(wrange[i,:])) & (wave0<np.max(wrange[i,:])))

        index0=np.bitwise_or(index0,index)



    # Fitting
    cube_cont=np.zeros(data_new.shape)
    for i in range(sz[2]):
        for j in range(sz[1]):
            fit_order_tmp=fit_order
            if not (w_center is None):
                index_r=((wave0[index0] > w_center) & (data_new[index0,j,i]!=0))
                index_l=((wave0[index0] < w_center) & (data_new[index0,j,i]!=0))

                if (np.sum(index_r)==0) or (np.sum(index_r)==0):
                    if auto_reduce==True:
                        fit_order_tmp=0


            fit_func=kcwi_stats.iter_polyfit(wave0[index0],hdu.data[index0,j,i],fit_order_tmp)
            dp=fit_func(wave0)
            cube_cont[:,j,i]=dp

    # subtract
    index_num=(data_new!=0)
    data_new[index_num]=data_new[index_num]-cube_cont[index_num]

    # write file
    hdr_new=hdu.header.copy()
    hdr_new['CSUB']=1
    hdu_new=fits.PrimaryHDU(data_new,header=hdr_new)

    if writefn!='':
        hdu_new.writeto(writefn,overwrite=True)

    return hdu_new


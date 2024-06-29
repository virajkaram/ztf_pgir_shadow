import sys
import numpy as np
import pandas as pd
import healpy as hp
import astropy.units as u

from matplotlib.patches import Polygon
from astropy.coordinates import SkyCoord

"""
Utility functions for vizualizing ZTF fields

Most of this code comes from the ztfquery package 
and has been modified by Zach Vanderbosch (Caltech)
"""

# corners of each CCD
_CCD_COORDS  = pd.read_csv("ztf_ccd_layout.tbl")

FIELD_SOURCE = "ZTF_Fields.txt"
FIELD_DATAFRAME = pd.read_csv(FIELD_SOURCE, index_col="ID", delim_whitespace=True)
FIELDSNAMES = FIELD_DATAFRAME.index.values


def get_field_centroid(fieldid, system="radec"):
    """ """
    if system in ["radec", "RADec","RA,Dec", "ra,dec"]:
        syst = ["RA", "Dec"]
    elif system.lower() in ["gal","galactic"]:
        syst = ["Gal Long","Gal Lat"]
    elif system.lower() in ["ecl","ecliptic"]:
        syst = ["Ecl Long","Ecl Lat"]
    else:
        raise ValueError("unknown coordinate system %s select among: [radec / galactic / ecliptic]"%system)
    fieldid = np.atleast_1d(fieldid)
    radec = np.asarray(FIELD_DATAFRAME[np.in1d(FIELD_DATAFRAME.index, fieldid)][syst].values)
    
    return radec


def get_corners(ra_field, dec_field, inclccd=False, ccd=None, steps=5, squeeze=True, inrad=False):
    """ """
    from ztfquery.utils.tools import rot_xz_sph, _DEG2RA
    
    if not inclccd:
        upper_left_corner = _CCD_COORDS.max()
        lower_right_corner = _CCD_COORDS.min()
    elif ccd is None:
        upper_left_corner = _CCD_COORDS.groupby("CCD").max()
        lower_right_corner = _CCD_COORDS.groupby("CCD").min()
    else:
        upper_left_corner = _CCD_COORDS.groupby("CCD").max().loc[ccd]
        lower_right_corner = _CCD_COORDS.groupby("CCD").min().loc[ccd]
        
    ewmin = -np.atleast_1d(upper_left_corner["EW"])
    nsmax = np.atleast_1d(upper_left_corner["NS"])
    ewmax = -np.atleast_1d(lower_right_corner["EW"])
    nsmin = np.atleast_1d(lower_right_corner["NS"])

    ra1  = (np.linspace(ewmax, ewmin, steps)/np.cos(nsmax*_DEG2RA)).T
    dec1 = (np.ones((steps,1))*nsmax).T
    #
    dec2  = np.linspace(nsmax,nsmin, steps).T
    ra2   = ewmin[:,None]/np.cos(dec2*_DEG2RA)
    #
    ra3 = (np.linspace(ewmin,ewmax, steps)/np.cos(nsmin*_DEG2RA)).T
    dec3 = (np.ones((steps,1))*nsmin).T
    #
    dec4  = np.linspace(nsmin,nsmax, steps).T
    ra4 = ewmax[:,None]/np.cos(dec4*_DEG2RA)

    ra_bd = np.concatenate((ra1, ra2, ra3, ra4  ), axis=1)  
    dec_bd = np.concatenate((dec1, dec2, dec3,dec4 ), axis=1)
    
    ra,dec = rot_xz_sph(np.moveaxis(ra_bd,0,1), np.moveaxis(dec_bd,0,1), np.moveaxis(np.atleast_3d(dec_field),0,1))
    ra += np.moveaxis(np.atleast_3d(ra_field),0,1)

    if inrad:
        ra *= _DEG2RA
        dec *= _DEG2RA
        
    radec = np.moveaxis([ra,dec],(0,1,2,3),(3,0,2,1))
    return radec if not squeeze else np.squeeze(radec)



def get_field_vertices(fieldid=None, system='equatorial',inclccd=False, 
                       ccd=None, asdict=False, aspolygon=False, squeeze=True):
    """ """
    if fieldid is None:
        fieldid = FIELDSNAMES
        
    if inclccd and ccd is None:
        ccd = np.arange(1,17)

    # - Actual calculation
    rafields, decfields  = get_field_centroid( np.asarray(np.atleast_1d(fieldid), dtype="int") ).T
    fields_verts = get_corners(rafields, decfields, inclccd=inclccd, ccd=ccd,
                               inrad=False, squeeze=True)

    # Convert to Galactic coordinates in radians (wrapped at 180-degrees)
    vert_ra = fields_verts[:,0]
    vert_dec = fields_verts[:,1]
    c = SkyCoord(ra=vert_ra*u.deg, dec=vert_dec*u.deg, frame='icrs')
    if system == 'galactic':
        vert_longitude = c.galactic.l.wrap_at(180 * u.deg).radian
        vert_latitude = c.galactic.b.radian
    elif system == 'equatorial':
        vert_longitude = c.ra.wrap_at(180 * u.deg).radian
        vert_latitude = c.dec.radian
    else:
        print(f'Coordinate System {system} not available')
        sys.exit(1)
    fields_verts[:,0] = vert_longitude
    fields_verts[:,1] = vert_latitude
    fields_countours = fields_verts

    return fields_countours if not squeeze else np.squeeze(fields_countours)



def get_grid_field(which,lowlim,upplim):
    """ """
    if which in ["main","Main","primary"]:
        return FIELDSNAMES[(FIELDSNAMES<upplim) & (FIELDSNAMES>lowlim)]
    if which in ["aux","secondary", "auxiliary"]:
       return FIELDSNAMES[FIELDSNAMES>999]
    if which in ["all","*","both"]:
        return FIELDSNAMES
        
    raise ValueError(f"Cannot parse which field grid you want {which}")



def add_fields(ax, fields, system, inclccd=False, ccd=None, show_text=False,
    facecolor="#333333", edgecolor="None", linestyle='-', alpha=0.5, lw=0.5, 
    hpy=False, fs=6, fc='k', fa=0.5):
        """ """
        fields_verts = get_field_vertices(fields, system=system)


        # Check whether vertices cross over the pi/-pi boundary
        if max(fields_verts[:,0]) - min(fields_verts[:,0]) > np.pi:

            # Get vertices with negative longitude
            negative_verts = np.copy(fields_verts)
            xvals = negative_verts[:,0]
            Nneg = len(xvals[xvals<0])
            xvals[xvals>0] = -np.pi
            negative_verts[:,0] = xvals

            # Get vertices with positive longitudes
            positive_verts = np.copy(fields_verts)
            xvals = positive_verts[:,0]
            Npos = len(xvals[xvals>0])
            xvals[xvals<0] = np.pi
            positive_verts[:,0] = xvals

            # Special case for fields that cover the poles (bit of a kluge)
            if not inclccd and fields in [626,878,879]:
                fields_verts = fields_verts[fields_verts[:, 1].argsort()]
                fields_verts = np.unique(fields_verts,axis=0)
                fields_verts = np.append(np.asarray([[-np.pi,np.pi/2]]), fields_verts, axis=0)
                fields_verts = np.append(fields_verts, np.asarray([[np.pi,np.pi/2]]), axis=0)
                pg1 = Polygon(fields_verts, facecolor=facecolor, edgecolor=edgecolor,alpha=alpha, lw=lw)
                if hpy:
                    verts = pg1.get_xy()
                    vert_phi = verts[:,0]
                    vert_theta = -verts[:,1] + np.pi/2
                    hp.projplot(vert_theta, vert_phi, c=edgecolor,lw=lw,alpha=alpha)
                else:
                    ax.add_patch(pg1)
                    if show_text:
                        ax.text(np.mean(fields_verts[:,0]), np.mean(fields_verts[:,1]), str(fields),
                                ha='center',va='center',fontsize=fs,color=fc, alpha=fa)
            else:
                pg1 = Polygon(negative_verts, facecolor=facecolor, edgecolor=edgecolor,alpha=alpha, lw=lw, ls=linestyle)
                pg2 = Polygon(positive_verts, facecolor=facecolor, edgecolor=edgecolor,alpha=alpha, lw=lw, ls=linestyle)
                if hpy:
                    verts1 = pg1.get_xy()
                    verts2 = pg2.get_xy()
                    vert_phi1 = verts1[:,0]
                    vert_phi2 = verts2[:,0]
                    vert_theta1 = -verts1[:,1] + np.pi/2
                    vert_theta2 = -verts2[:,1] + np.pi/2
                    hp.projplot(vert_theta1, vert_phi1, c=edgecolor,lw=lw,alpha=alpha,ls=linestyle)
                    hp.projplot(vert_theta2, vert_phi2, c=edgecolor,lw=lw,alpha=alpha,ls=linestyle)
                    if show_text:
                        if Npos > Nneg:
                            hp.projtext(np.mean(vert_theta2), np.mean(vert_phi2), str(fields),
                                ha='center',va='center',fontsize=fs,color=fc,alpha=fa)
                        else:
                            hp.projtext(np.mean(vert_theta1), np.mean(vert_phi1), str(fields),
                                ha='center',va='center',fontsize=fs,color=fc,alpha=fa)
                else:
                    ax.add_patch(pg1)
                    ax.add_patch(pg2)

                    if show_text:
                        if Npos > Nneg:
                            ax.text(np.mean(positive_verts[:,0]), np.mean(positive_verts[:,1]), str(fields),
                                    ha='center',va='center',fontsize=fs,color=fc,alpha=fa)
                        else:
                            ax.text(np.mean(negative_verts[:,0]), np.mean(negative_verts[:,1]), str(fields),
                                    ha='center',va='center',fontsize=fs,color=fc,alpha=fa)

        else:
            pg = Polygon(fields_verts, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=lw, ls=linestyle)
            if hpy:
                verts = pg.get_xy()
                vert_phi = verts[:,0]
                vert_theta = -verts[:,1] + np.pi/2
                hp.projplot(vert_theta, vert_phi, c=edgecolor,lw=lw,alpha=alpha,ls=linestyle)
                if show_text:
                    hp.projtext(np.mean(vert_theta), np.mean(vert_phi), str(fields),
                            ha='center',va='center',fontsize=fs,color=fc,alpha=fa)
            else:
                ax.add_patch(pg)
                if show_text:
                    ax.text(np.mean(fields_verts[:,0]), np.mean(fields_verts[:,1]), str(fields),
                            ha='center',va='center',fontsize=fs,color=fc,alpha=fa)


def show_fields(ax, fields, system, colors, edgecolor='w', linestyle='-', show_text=False, lw=0.5, 
        alpha=1.0, colorbar=True, clabel=None, cfontsize=None, bins="auto", hpy=False, 
        fontsize=6, fontcolor='k', fontalpha=0.5, **kwargs):
    """ fields could be a list of field or a dictionary with single values """
            
    for v,c in zip(fields, colors):
        add_fields(ax, v, system, facecolor=c,edgecolor=edgecolor, 
            linestyle=linestyle,alpha=alpha, show_text=show_text, hpy=hpy, lw=lw,
            fs=fontsize, fc=fontcolor, fa=fontalpha)

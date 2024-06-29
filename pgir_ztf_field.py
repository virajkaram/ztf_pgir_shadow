from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from ztf_fields import show_fields
import argparse


def get_pgir_fields_from_ztf_field(ztf_field_id, seplimit=5 * u.deg, plot=False):
    ztf_fields = ascii.read('data/ZTF_Fields.txt')
    pgir_fields = ascii.read('data/gattini_fields.csv')
    ztf_field = ztf_fields[ztf_fields['ID'] == ztf_field_id]

    pgir_field_centers = SkyCoord(ra=pgir_fields['RA'], dec=pgir_fields['Dec'],
                                  unit=(u.deg, u.deg))
    ztf_field_center = SkyCoord(ra=ztf_field['RA'], dec=ztf_field['Dec'],
                                unit=(u.deg, u.deg))
    id1, id2, d2d, d3d = ztf_field_center.search_around_sky(pgir_field_centers,
                                                            seplimit=seplimit)
    pgir_matched_fields = pgir_fields[id1]
    pgir_matched_fields_centers = pgir_field_centers[id1]
    if plot:
        pgir_fields_phi = pgir_matched_fields_centers.ra.radian
        pgir_fields_theta = -(pgir_matched_fields_centers.dec.radian - np.pi / 2)

        NSIDE = 32
        NPIX = hp.nside2npix(NSIDE)
        m = np.zeros(NPIX)
        hp.mollview(m)
        hp.graticule()

        ax = plt.gca()
        show_fields(
            ax, [ztf_field_id], 'equatorial', 'black', edgecolor='mistyrose',
            lw=1, linestyle='-', fontsize=6, fontcolor='w', fontalpha=1,
            alpha=1, zorder=2, hpy=True, show_text=False)

        hp.projscatter(pgir_fields_theta, pgir_fields_phi, c='k', lw=1.5, alpha=1,
                       zorder=1)
        # save figure
        plt.savefig('pgir_fields.pdf', bbox_inches='tight')
    return pgir_matched_fields, pgir_matched_fields_centers, ztf_field, ztf_field_center


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ztf_field_id', type=int,
                        help='ZTF field ID')
    parser.add_argument('-separation', type=float,
                        help='Separation limit in degrees', default=5)
    args = parser.parse_args()
    (pgir_matched_fields, pgir_matched_fields_centers, ztf_field,
     ztf_field_center) = get_pgir_fields_from_ztf_field(
        args.ztf_field_id, seplimit=args.separation * u.deg, plot=True)
    print('##########################################')
    print('PGIR field details')
    print("fields =", [int(x['ID']) for x in pgir_matched_fields])
    print("ra =", [float(x['RA']) for x in pgir_matched_fields])
    print("dec =", [float(x['Dec']) for x in pgir_matched_fields])

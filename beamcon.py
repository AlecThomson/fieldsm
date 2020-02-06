#!/usr/bin/env python
import numpy as np
from scipy import signal as sig
from astropy.io import fits
from radio_beam import Beam
from astropy import units as u
import au2

#############################################
#### ADAPTED FROM SCRIPT BY T. VERNSTROM ####
#############################################


def getbeam(datadict, beamfolder, beamlog, bmaj=None, bmin=None, bpa=None, verbose=False):
    """Get beam info
    """
    if verbose:
        print(f'Getting beam data from {beamfolder}/{beamlog}')
    if beamfolder is not None and beamlog is not None:
        beams = np.genfromtxt(f"{beamfolder}/{beamlog}", names=True)
        # nchan=beams.shape[0]
        colnames = ['Channel', 'BMAJarcsec', 'BMINarcsec', 'BPAdeg']

        bmajs = beams['BMAJarcsec'].copy()
        bmins = beams['BMINarcsec'].copy()
        bpas = beams['BPAdeg'].copy()
        bpasr = np.radians(bpas)

        bmaj_mx = bmajs.max()
        bmin_mx = bmins.max()

        old_beam = Beam(
            bmaj_mx*u.arcsec,
            bmin_mx*u.arcsec,
            bpas*u.deg
        )
        if verbose:
            print(f'Current beam is {old_beam}')
        if bmaj is None:
            bmaj = bmaj_mx
        if bmin is None:
            bmin = bmaj_mx
    else:
        print('No beamlog file given!')
        if bmaj is None or bmin is None:
            raise Exception('Please supply BMAJ and BMIN')

    if bpa is None:
        bpa = 0

    final_beam = [bmaj, bmin, bpa]

    conbm = au2.gaussianDeconvolve(
        final_beam[0], final_beam[1], final_beam[2], bmajs[0], bmins[0], bpas[0])
    inputbm = [bmajs[0], bmins[0], bpas[0]]
    fac, amp, outbmaj, outbmin, outbpa = au2.gauss_factor(
        conbm, beamOrig=inputbm, dx1=datadict['dx'], dy1=datadict['dy'])

    conbeams = [conbm[0], conbm[1], bpas[2]]
    sfactors = fac
    return conbeams, final_beam, sfactors


def getimdata(cubenm, verbose=False):
    """Get fits image data
    """
    if verbose:
        print(f'Getting image data from {cubenm}')
    with fits.open(cubenm, memmap=True, mode='denywrite') as hdu:

        dxas = hdu[0].header['CDELT1']*(-3600.)
        dyas = hdu[0].header['CDELT2']*(3600.)

        nx, ny = hdu[0].data[0, 0, :,
                             :].shape[0], hdu[0].data[0, 0, :, :].shape[1]

        datadict = {
            'image': hdu[0].data[0, 0, :, :],
            'header': hdu[0].header,
            'nx': nx,
            'ny': ny,
            'dx': dxas,
            'dy': dxas
        }
    return datadict


def smooth(datadict, verbose=False):
    """Do the smoothing
    """
    # using Beams package
    final_bm = Beam(
        datadict["final_beam"][0]*u.arcsec,
        datadict["final_beam"][1]*u.arcsec,
        datadict["final_beam"][2]*u.deg
    )
    if verbose:
        print(f'Smoothing so beam is {final_bm}')
    pix_scale = datadict['dy'] * u.arcsec

    con_bm = Beam(
        datadict["conbeams"][0]*u.arcssec,
        datadict["conbeams"][1]*u.arcsec,
        datadict["conbeams"][2]*u.deg
    )
    gauss_kern = con_bm.as_kernel(pix_scale)

    conbm1 = gauss_kern.array/gauss_kern.array.max()

    newim = sig.fftconvolve(datadict['image'], conbm1, mode='same')
    newim = newim*datadict["sfactors"]
    return final_bm, newim


def savefile(datadict, filename, verbose=False):
    """Save file to disk
    """
    if verbose:
        print(f'Saving to {filename}')
    header = datadict['header']
    beam = datadict['newbeam']
    header['BMIN'] = beam.minor.to(u.arcsec)
    header['BMAJ'] = beam.minor.to(u.arcsec)
    header['BPA'] = beam.pa.to(u.deg)
    fits.writeto(filename, datadict['newim'], header=header, overwrite=True)


def main(args, verbose=False):
    """Main script
    """
    if args.outfile is None:
        outfile = args.infile.replace('.fits', '.sm.fits')

    beamfolder = args.beamfolder
    if beamfolder is not None:
        if beamfolder[-1] == '/':
            beamfolder = beamfolder[:-1]

    datadict = getimdata(args.infile)

    conbeams, final_beam, sfactors = getbeam(
        datadict,
        beamfolder,
        args.bmlognm,
        verbose=verbose
    )

    datadict.update(
        {
            "conbeams": conbeams,
            "final_beam": final_beam,
            "sfactors": sfactors
        }
    )

    final_bm, newim = smooth(datadict, verbose=verbose)

    datadict.update(
        {
            "newimage": newim,
            "newbeam": final_bm
        }
    )

    savefile(datadict, outfile, verbose=verbose)
    if verbose:
        print('Done!')


def cli():
    """Command-line interface
    """
    import argparse
    #import schwimmbad

    # Help string to be shown using the -h option
    descStr = """
    Smooth a field to a common resolution.

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        'infile',
        metavar='infile',
        type=str,
        help='Input beam FITS image to smooth.')

    parser.add_argument(
        'outfile',
        metavar='outfile',
        type=str,
        default=None,
        help='Output name of smoothed FITS image [infile.sm.fits].')

    parser.add_argument(
        '-f',
        '--beamfolder',
        dest='beamfolder',
        type=str,
        help='Directory containing beamlog file.')

    parser.add_argument(
        '-l',
        '--beamlog',
        dest='bmlognm',
        type=str,
        help='Name of beamlog file.')

    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="verbose output [False].")

    parser.add_argument(
        "--bmaj",
        dest="bmaj",
        type=float,
        default=None,
        help="BMAJ to convolve to [Max BMAJ from beamlog].")

    parser.add_argument(
        "--bmin",
        dest="bmin",
        type=float,
        default=None,
        help="BMIN to convolve to [Max BMAJ from beamlog].")

    parser.add_argument(
        "--bpa",
        dest="bpa",
        type=float,
        default=None,
        help="BPA to convolve to [0].")

    #group = parser.add_mutually_exclusive_group()

    # group.add_argument("--ncores", dest="n_cores", default=1,
    #                   type=int, help="Number of processes (uses multiprocessing).")
    # group.add_argument("--mpi", dest="mpi", default=False,
    #                   action="store_true", help="Run with MPI.")
    #
    args = parser.parse_args()
    #pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    verbose = args.verbose

    main(args, verbose=verbose)


if __name__ == "__main__":
    cli()

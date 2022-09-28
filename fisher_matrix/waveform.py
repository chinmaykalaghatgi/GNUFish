from __future__ import division
from collections import defaultdict
import numpy as np
import sys

import lal
import lalsimulation as lalsim
import utils as ut

__author__ = "Chinmay Kalaghatgi <chinmay.kalaghatgi@ligo.org>, Soumen Roy <soumen.roy@ligo.org>"

# Maps required between intrinsic parameters to parameres used by _waveform_base
m1m2_maps={
    ('mchirp','q'): ut.mchirp_q_to_mass1_mass2,
    ('mchirp','eta'): ut.mchirp_eta_to_mass1_mass2,
    ('mtotal','q'): ut.mtotal_q_to_mass1_mass2,
    ('mtotal','eta'): ut.mtotal_eta_to_mass1_mass2,
    ('mass1', 'mass2'): ut.check_mass1_mass2
}


# Waveform generator
class Waveform(object):
    """
    This class is specific to generate non-eccentric frequency domain
    Binary Black Hole waveforms.

    Parameters
    ----------
    phi0 : float
        Phase at coalescence.
    fref : float
        Reference frequency. Unit: Hz.
    flow : float
        Lower cutoff frequency. Unit: Hz.
    fhigh : float
        Higher cutoff frequency. Unit: Hz.
    df : float
        Frequency sampling. Unit: Hz.
    approximant : String of a frequency domain waveform present in lalsimulation.
        String; for eg: 'IMRPhenomXPHM'
    intrinsic_dict : Dictionary of the intrinsic parameters.
    For now; only specific combinations of intrinsic parameters are accepted.

    The first two parameters of the dictionary have to be the mass parameters with the
    last 7 being the individual spins / inclination parameters.

    Acceptable formats for masses;
    {'mchirp, q'}; {'mchirp', 'eta'}; {'mtotal', 'q'}; {'mtotal', 'eta'}; {'mass1', 'mass2'}

    Acceptable formats for spins: (Make sure they are in this specific order)
    cartesian - {'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', 'inc'}
    spherical - {'chi1', 'theta1', 'phi1', 'chi2', 'theta2', 'phi2', 'inc'}
    precessing format - {'chi1', 'chi2', 'phiJL', 'tilt1', 'tilt2', 'phi12', 'thetaJN'}
    """
    def __init__(self, phi0, fRef, flow, fhigh, df, intrinsic_dict, approximant):

        # Set initial parameters
        self._phiRef = phi0
        self._fRef = fRef
        self._fLow = flow
        self._fHigh = fhigh
        self._deltaF = df
        self._approx = lalsim.__dict__[approximant]

        # Set frequency series
        self._ind_min = int(self._fLow/self._deltaF)
        self._ind_max = int(self._fHigh/self._deltaF + 1)
        self._frequencies = np.linspace(flow, fhigh, self._ind_max-self._ind_min)

        # Set a constant distance of 1. as we would want to rescale it later anyways  ¯\_(ツ)_/¯
        self._distance = intrinsic_dict["distance"]

        self._convert_intrinsic_params(intrinsic_dict)
        self.hp, self.hc = self._generate_waveform()

    def _generate_waveform(self):

        hp, hc = self._waveform_base(self.mass1, self.mass2, self.spin1x, self.spin1y, self.spin1z,
                                    self.spin2x, self.spin2y, self.spin2z, self.inc)
        return hp.data.data[self._ind_min:self._ind_max], hc.data.data[self._ind_min:self._ind_max]

    def _convert_intrinsic_params(self, intrinsic_dict):
        # Here, for intrinsic_dict; the first two parameters have to be the mass parameters
        # Convert from the parameters given in the intrinsic parameters dictionary to mass1, mass2
        all_keys            = tuple(intrinsic_dict.keys())
        mass_params         = all_keys[:2]
        conversion_function = m1m2_maps[mass_params]
        self.mass1, self.mass2        = conversion_function(intrinsic_dict[mass_params[0]], intrinsic_dict[mass_params[1]])

        # Will change this later when including the different spin parameterizations (for eg; spherical to cartesian, phi_jl etc)
        if all(a in tuple(intrinsic_dict.keys()) for a in ['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z']):
            self.spin1x, self.spin1y, self.spin1z = intrinsic_dict['spin1x'], intrinsic_dict['spin1y'], intrinsic_dict['spin1z']
            self.spin2x, self.spin2y,self. spin2z = intrinsic_dict['spin2x'], intrinsic_dict['spin2y'], intrinsic_dict['spin2z']
            self.inc                              = intrinsic_dict['inc']
        elif all(a in tuple(intrinsic_dict.keys()) for a in ['chi1', 'theta1', 'phi1', 'chi2', 'theta2', 'phi2']):
            self.spin1x, self.spin1y, self.spin1z = ut.sph2cart(intrinsic_dict['chi1'], intrinsic_dict['theta1'], intrinsic_dict['phi1'])
            self.spin2x, self.spin2y, self.spin2z = ut.sph2cart(intrinsic_dict['chi2'], intrinsic_dict['theta2'], intrinsic_dict['phi2'])
            self.inc                              = intrinsic_dict['inc']
        elif all(a in tuple(intrinsic_dict.keys()) for a in ['thetaJN', 'phiJL', 'tilt1', 'tilt2', 'chi1', 'chi2']):
            self.inc, self.spin1x, self.spin1y, self.spin1z, self.spin2x, self.spin2y, self.spin2z = lalsim.SimInspiralTransformPrecessingNewInitialConditions(
            intrinsic_dict['thetaJN'], intrinsic_dict['phiJL'], intrinsic_dict['tilt1'], intrinsic_dict['tilt2'], intrinsic_dict['phi12'], intrinsic_dict['chi1'],
            intrinsic_dict['chi2'], self.mass1*lal.MSUN_SI, self.mass2*lal.MSUN_SI, self._fRef, self._phiRef)
        else:
            print(intrinsic_dict)
            raise NameError("Wrong parameter dictionary passed!")



    def _waveform_base(self, mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, inc):


        FD_args = {
        'm1' : mass1*lal.MSUN_SI,
        'm2' : mass2*lal.MSUN_SI,
        'S1x' : spin1x,
        'S1y' : spin1y,
        'S1z' : spin1z,
        'S2x' : spin2x,
        'S2y' : spin2y,
        'S2z' : spin2z,
        'distance' : self._distance*1e6*lal.PC_SI,
        'inclination' : inc,
        'phiRef' : self._phiRef,
        'deltaF' : self._deltaF,
        'f_min' : self._fLow,
        'f_max' : self._fHigh,
        'f_ref' : self._fRef,
        'longAscNodes': 0,
        'eccentricity': 0,
        'meanPerAno': 0,
        'LALpars': None,
        'approximant':self._approx
        }

        if lalsim.SimInspiralImplementedFDApproximants(self._approx):
            hp_lal, hc_lal  = lalsim.SimInspiralChooseFDWaveform(**FD_args)
        else:
            FD_args['LALparams'] = FD_args.pop('LALpars')
            hp_lal, hc_lal  = lalsim.SimInspiralFD(**FD_args)

        return hp_lal,  hc_lal

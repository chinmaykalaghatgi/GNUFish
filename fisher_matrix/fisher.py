from __future__ import division
from collections import defaultdict
import numpy as np
import sys

import lal
import lalsimulation as lalsim
import utils as ut
import waveform as wfm
import bilby

__author__ = "Chinmay Kalaghatgi <chinmay.kalaghatgi@ligo.org>, Justin Janquart <justin.janquart@ligo.org>"


def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.

    a -- square NumPy array, such that a_ij = 0 or a_ji = 0,
    for i != j.
    """
    return a + a.T - np.diag(a.diagonal())



##################################################################################################
##################################################################################################

m1m2_maps={
    ('mchirp','q'): ut.mchirp_q_to_mass1_mass2,
    ('mchirp','eta'): ut.mchirp_eta_to_mass1_mass2,
    ('mtotal','q'): ut.mtotal_q_to_mass1_mass2,
    ('mtotal','eta'): ut.mtotal_eta_to_mass1_mass2,
    ('mass1', 'mass2'): ut.check_mass1_mass2
}

class FisherMatrix_SingleIFO(object):
    """
    This class computes the numerical metric using first order
    finite difference method. Metric for BBH system is determined
    assuming equal aligned spin binary (spin1z = spin2z) and for
    NSBH system, the neutron star are assumed to be non-spinning.

    Ref: Sec. II of https://arxiv.org/pdf/1711.08743.pdf

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
    ASD : function for any ASD - should be in Unit (Hz)
        Amplitude spectral density of detector noise. Unit: sqrt(Hz).
    delta_param : float, optional
        Step size to calculate the partial derivative. generally set to 0.0005
    approximant : str
        Approximant of the waveform model.
    param_limits:
        This should be a dictionary of the upper and lower limits for every param.
        For eg: param_limits = {
        'mtotal' : [10, 100],
        'q'      : [1., 8.],
        'chi1'   : [0., 1.],
        'theta1' : [0., np.pi],
        'phi1'   : [0., 2*np.pi],
        'chi2'   : [0., 1.],
        'theta2' : [0., np.pi],
        'phi2'   : [0., 2*np.pi],
        'inc'    : [0, 2*np.pi]
        }
    """
    def __init__(self, parameters_dict, vary_dict, param_limits, ifo, approximant, delta=5e-6, psd_file = None, asd_file=None):
        self._phi0 = parameters_dict['phi0']
        self._fref = parameters_dict['fref']
        self._flow = parameters_dict['flow']


        self._fhigh = parameters_dict['fhigh']

        self._psi = parameters_dict['psi']
        self._df = parameters_dict['df']
        self._tgps = parameters_dict['tgps']

        self._ifo = bilby.gw.detector.InterferometerList([ifo])[0]

        # add the correct PSD if needed
        if psd_file is not None:
            self._ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file = psd_file)
        if asd_file is not None:
            self._ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file = asd_file)

        self._ind_min = int(self._flow/self._df)
        self._ind_max = int(self._fhigh/self._df + 1)
        self._freqSeries = np.linspace(self._flow, self._fhigh, self._ind_max-self._ind_min)

        self._PSD = self._ifo.power_spectral_density.get_power_spectral_density_array(self._freqSeries)
        #self._delta_param = delta_param
        self._approximant = approximant

        self.param_limits = param_limits
        self.int_dim = len(vary_dict)
        self.delta = delta


    def _get_inner_product(self, h1, h2):
        from scipy.integrate import simpson
        numerator = (h1*h2.conj() + h2*h1.conj())
        den       = self._PSD
        inner_product = 2*simpson(numerator/den, self._freqSeries, dx=self._df)

        return inner_product

    def _get_snr(self, param_dic):
        h1 = self._get_waveform(param_dic)
        snr = self._get_inner_product(h1, h1)
        return np.sqrt(snr)

    def _get_waveform(self, param_dic):
        intrinsic_dict = param_dic.copy()
        hp, hc = wfm.Waveform(param_dic["phi0"], self._fref, self._flow, self._fhigh, self._df, intrinsic_dict, self._approximant)._generate_waveform()
        Fp = self._ifo.antenna_response(param_dic['ra'], param_dic['dec'], param_dic["tgps"], param_dic["psi"], 'plus')
        Fc = self._ifo.antenna_response(param_dic['ra'], param_dic['dec'], param_dic["tgps"], param_dic["psi"], 'cross')


        rel_time_delay = self._ifo.time_delay_from_geocenter(param_dic['ra'], param_dic['dec'], param_dic["tgps"])
        tcoal = param_dic["tcoal"]

        h1 = (Fp*hp+Fc*hc)*np.exp(-2*1j*self._freqSeries*np.pi*(rel_time_delay+tcoal))

        return h1

    def _delta_param_val(self, param_dic, param):
        delta_param = self.delta

        return delta_param

    def _get_derivative_intrinsic(self, param_dic, param):

        param_limits = self.param_limits


        # Check if param is within the limit and use either central, forward or backward difference method.
        param_current = param_dic[param]

        delta_param = self._delta_param_val(param_dic, param)


        if self.param_limits[param][0]<param_current-delta_param and param_current+delta_param<self.param_limits[param][1]:
        # Use centeral finite difference if param is between the two limits

            h_base                   = self._get_waveform(param_dic)
            amp_base                 = np.abs(h_base)
            phase_base               = np.unwrap(np.angle(h_base))

            params_plus          = param_dic.copy()
            params_plus[param]   = params_plus[param]+delta_param
            h_plus               = self._get_waveform(params_plus)
            amp_plus             = np.abs(h_plus)
            phase_plus           = np.unwrap(np.angle(h_plus))

            params_minus           = param_dic.copy()
            params_minus[param]    = params_minus[param]-delta_param
            h_minus                = self._get_waveform(params_minus)
            amp_minus             = np.abs(h_minus)
            phase_minus           = np.unwrap(np.angle(h_minus))

            damp        = (amp_plus - amp_minus)/(2*delta_param)
            dphase      = (phase_plus - phase_minus)/(2*delta_param)
            dh_dlambda  = damp*np.exp(1j*phase_base) + 1j*amp_base*np.exp(1j*phase_base)*dphase

        elif param_current-delta_param<=self.param_limits[param][0]:
        # Use forward difference method if the param is close to the lower limit
            hBase                    = self._get_waveform(param_dic)
            amp_base                 = np.abs(hBase)
            phase_base               = np.unwrap(np.angle(hBase))

            params_plus          = param_dic.copy()
            params_plus[param]   = params_plus[param]+delta_param
            h_plus               = self._get_waveform(params_plus)
            amp_plus             = np.abs(h_plus)
            phase_plus           = np.unwrap(np.angle(h_plus))

            damp        = (amp_plus - amp_base)/(delta_param)
            dphase      = (phase_plus - phase_base)/(delta_param)

            dh_dlambda           = damp*np.exp(1j*phase_base) + 1j*amp_base*np.exp(1j*phase_base)*dphase

        elif self.param_limits[param][1]<=param_current+delta_param:
        # Use Backward difference method if the param is close to the upper limit
            hBase                = self._get_waveform(param_dic)
            amp_base                 = np.abs(hBase)
            phase_base               = np.unwrap(np.angle(hBase))

            params_minus           = param_dic.copy()
            params_minus[param]    = params_minus[param]-delta_param
            h_minus                = self._get_waveform(params_minus)
            amp_minus             = np.abs(h_minus)
            phase_minus           = np.unwrap(np.angle(h_minus))

            damp        = (amp_base - amp_minus)/(delta_param)
            dphase      = (phase_base - phase_minus)/(delta_param)
            dh_dlambda  = damp*np.exp(1j*phase_base) + 1j*amp_base*np.exp(1j*phase_base)*dphase

        # dh_dlambda for dh/dtc : Here, tgps technically enters the detector response functions
        # but in reality we measure tc by time-shifts etc. Hence, have to be 2*pi*f*tc's derivative

        if param=="tcoal":
            hBase      = self._get_waveform(param_dic)
            dh_dlambda     = -hBase * 2* np.pi * self._freqSeries*1j

        return dh_dlambda


    def _fisher_matrix(self, intrinsic_dict, vary_dict):
        # Params list : The initial 8 parameters have to be mass + spin parameters, with the first two being the mass
        # and next six being the spins due to how the waveform module transforms the parameters to lal parameters

        fisher = np.zeros((self.int_dim,self.int_dim))

        params_dict = intrinsic_dict.copy()

        # Check if set of intrinsic params and metric params have overlap.
        if len(set(params_dict).intersection(vary_dict))!=self.int_dim:
            raise ValueError("Intrinsic parameters and metric parameters are NOT the same!!!")

        # Check whether parameters being varied are within the parameter limits - for forward - backward - central difference methods
        for key in vary_dict:

            if self.param_limits[key][0]<=params_dict[key]<=self.param_limits[key][1]:
                continue
            else:
                raise ValueError(" Parameters in the intrinsic_dict are not within given limits")


        #param_keys = tuple(params_dict.keys())
        param_keys  = vary_dict.copy()


        # Populate the matrix elements
        for i in np.arange(0,self.int_dim):
            for j in np.arange(0, self.int_dim):

                if i<=j:
                    #print("At fisher element i,j = ", i, j)
                    dh_dlambdai = self._get_derivative_intrinsic(params_dict, param_keys[i])
                    dh_dlambdaj = self._get_derivative_intrinsic(params_dict, param_keys[j])

                    # Fisher value
                    fisher[i,j]  = self._get_inner_product(dh_dlambdai, dh_dlambdaj)
                else:
                    continue

        fisher = symmetrize(fisher)
        return fisher



##################################################################################################
##################################################################################################

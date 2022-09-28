from __future__ import division
from collections import defaultdict
import numpy as np
import sys

import lal
import lalsimulation as lalsim


# All required / needed mass parameter conversions. These are copied from ./lalinspiral/python/sbank/tau0tau3.py
# q ones added.

def sph2cart(r,theta,phi):
    """
    Utiltiy function to convert r,theta,phi to cartesian co-ordinates.
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z


def cart2sph(x,y,z):
    """
    Utility function to convert cartesian coords to r,theta,phi.
    """
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z/r)
    phi = np.fmod(2*np.pi + np.arctan2(y,x), 2*np.pi)

    return r,theta,phi

def check_mass1_mass2(mass1, mass2):
    if mass1>=mass2:
        return mass1, mass2
    else:
        return mass2, mass1

def mchirp_q_to_mass1_mass2(m_chirp,q):
    # Here assume that q>1
    eta = q/(1+q)**2
    return mchirp_eta_to_mass1_mass2(m_chirp, eta)

def mchirp_eta_to_mass1_mass2(m_chirp, eta):
    M = m_chirp / (eta**(3./5.))
    return mtotal_eta_to_mass1_mass2(M, eta)

def mtotal_q_to_mass1_mass2(m_total,q):
    # Here assume that q>1
    eta = q/(1+q)**2
    return mtotal_eta_to_mass1_mass2(m_total, eta)

def mtotal_eta_to_mass1_mass2(m_total, eta):
    mass1 = 0.5 * m_total * (1.0 + (1.0 - 4.0 * eta)**0.5)
    mass2 = 0.5 * m_total * (1.0 - (1.0 - 4.0 * eta)**0.5)
    return mass1,mass2

def mass1_mass2_to_mtotal_eta(mass1, mass2):
    """
    Compute total mass and symmetric mass ration from
    individual masses.

    Parameters
    ----------
    mass1 : float
        Mass of the first object. Unit : Solar mass
    mass2 : float
        Mass of the second object. Unit : Solar mass

    Returns
    -------
    mtotal : float
        Total mass of the system. Unit : Solar mass
    eta : float
        Symmetric mass ratio of the system.
    """
    mtotal = mass1 + mass2
    eta = mass1*mass2/(mtotal*mtotal)
    return mtotal, eta


def mchirpm1_to_m2(mc, m1, tol=1e-6):
    # solve cubic for m2
    p = -mc**5/m1**3
    q = p*m1
    rts = numpy.roots([1, 0, p, q])

    # remove complex and negative roots
    rts = [r.real for r in rts if abs(r.imag) < tol and r.real > 0]

    if len(rts) == 0:
        m2 = float('nan')
    elif len(rts) == 1:
        m2 = rts[0]
    else:
        raise ValueError("No unique real solution for m2 found for mchirp=%f and m1=%f"%(mc, m1))

    return m2

def m1m2_to_mratio(m1,m2):
    return m1/m2

def m1m2_to_m1(m1,m2):
    return m1

def m1m2_to_m2(m1,m2):
    return m2

def m1m2_to_mtotal(m1,m2):
    return m1+m2

def m1m2_to_mchirp(m1,m2):
    return (m1 * m1 * m1 * m2 * m2 * m2 / (m1 + m2))**0.2

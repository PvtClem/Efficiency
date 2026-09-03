"""
26.09.2025
This program calculates the exposure (km^2 day sr) of GP300 using the ADC trace. 

The input data is the output of judge_trigger_with_ADCtrace_like_experiment.py, which implements a realistic trigger algorithm considering a coincidence of detectors in a specific time window.
The output dataset of judge_trigger_with_ADCtrace_like_experiment.py can be found in the directory ./out_judge_trigger_with_ADCtrace_like_experiment.

The weighting scheme of the simulation data is determined so that the weighted CR spectrum follows the one measured by the Pierre Auger Observatory; see calculate_weighting_factor_energy_PAO_spectrum in utils.py.
"""


import sys
import numpy as np
import glob as gb
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy.integrate import trapezoid as trap
from scipy.interpolate import PchipInterpolator
from utils import *

### Basic parameters ###
lemin, lemax = 17., 20.01
emin_eV, emax_eV = 10 ** lemin, 10 ** lemax
znmin, znmax = 65.-0.01, 88.+0.01
czmin, czmax = np.cos(znmax*np.pi/180.), np.cos(znmin*np.pi/180.)
azmin, azmax = 0, 360
S_geo = 180 # km^2
T_obs = 1   # day


### Start main routine
data = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
for file in gb.glob('../out_judge_trigger_with_ADCtrace_like_experiment/sim_*'):
    data = np.concatenate([data, np.loadtxt(file)], axis=0)
data = np.delete(data, 0, axis=0)

### Extract observables from the input ###
deg2rad = np.pi/180.
ev_nb = data[:,1].astype(int)
ptlid = data[:,2].astype(int)
le_eV = np.log10(data[:,3]*1.e9) # GeV -> eV
zn = data[:,4]
cz = np.cos(zn*deg2rad)
az = data[:,5]
core_x = data[:,6] * 1.e-3 # m -> km
core_y = data[:,7] * 1.e-3 # m -> km
core_z = data[:,8] * 1.e-3 # m -> km
trig_flag = data[:,9]
print("data.shape:", data.shape)

nbin_le, nbin_cz, nbin_az = 30, 10, 12
#nbin_le, nbin_cz, nbin_az = 30, 10, 10
#nbin_le, nbin_cz, nbin_az = 30, 10, 30
binw_le = (lemax - lemin) / nbin_le
binw_cz = (czmax - czmin) / nbin_cz
binw_az = (azmax - azmin) / nbin_az
azmin, azmax = azmin - binw_az/2, azmax - binw_az/2
x_le = (np.array(range(nbin_le)) + 0.5) * binw_le + lemin
x_cz = (np.array(range(nbin_cz)) + 0.5) * binw_cz + czmin
x_az = (np.array(range(nbin_az)) + 0.5) * binw_az + azmin
mean_le = np.zeros((nbin_le))

Nall   = np.zeros((nbin_le, nbin_cz, nbin_az))
Nall_w = np.zeros((nbin_le, nbin_cz, nbin_az))
Nana   = np.zeros((nbin_le, nbin_cz, nbin_az))
Nana_w = np.zeros((nbin_le, nbin_cz, nbin_az))
Nana_w_cos = np.zeros((nbin_le, nbin_cz, nbin_az))

norm_coeff_cz = 0
norm_coeff_e  = 0


for n in range(data.shape[0]):

    #if ptlid[n] != 1: continue # only consider proton (ptlid == 1)
    #if ptlid[n] != 56: continue # only consider iron (ptlid == 56)
    
    bin_le = (int) ((le_eV[n] - lemin) / binw_le)
    bin_cz = (int) ((cz[n]    - czmin) / binw_cz)
    if (az[n] >= azmax): az[n] -= 360
    bin_az = (int) ((az[n]    - azmin) / binw_az)
    weight_e = calculate_weighting_factor_energy_PAO_spectrum(10 ** le_eV[n], emin_eV, emax_eV)
    norm_coeff_cz += np.sin(zn[n]*deg2rad)
    norm_coeff_e  += weight_e
    
    if trig_flag[n]:
        Nana[bin_le,bin_cz,bin_az] += 1
        Nana_w[bin_le,bin_cz,bin_az] += np.sin(zn[n]*deg2rad) * weight_e
        Nana_w_cos[bin_le,bin_cz,bin_az] += np.sin(zn[n]*deg2rad) * weight_e * cz[n]

    Nall[bin_le,bin_cz,bin_az] += 1
    Nall_w[bin_le,bin_cz,bin_az] += np.sin(zn[n]*deg2rad) * weight_e
    
# Normalization w.r.t. the cos(theta) space
norm_coeff_cz = data.shape[0] / norm_coeff_cz
norm_coeff_e  = data.shape[0] / norm_coeff_e
Nall_w *= norm_coeff_cz * norm_coeff_e
Nana_w *= norm_coeff_cz * norm_coeff_e

### check
print("data.shape[0]:", data.shape[0])
#print("np.sum(Nana):", np.sum(Nana))
#print("np.sum(Nana_w):", np.sum(Nana_w))
print("np.sum(Nall):", np.sum(Nall))
print("np.sum(Nall_w):", np.sum(Nall_w))


'''
Exposure calculation
Several type of exposures are calculated
1. exp_cz_az: Exposure in the (cos(Zenith), Azimuth) space (integrated along energy)
2. exp_le: Exposure as a function of log10(Energy) (integrated over the sky)
3. exp_cz: Exposure as a function of cos(Zenith) (integrated over azimuth & energy)
   It is used to check that the simulated zenith angle region is enough to calculate
   the GP300 exposure for the whole sky
'''
### Calculation of the solid angle of each sky bin (cz~cz+d_cz, az~az+d_az)
omega_tot = 2. * np.pi * (czmax-czmin)
d_omega = deg2rad * binw_az * binw_cz
omega_cz = 2. * np.pi * binw_cz
omega_az = omega_tot / nbin_az
print("omega_tot:", omega_tot, "sr, expected:", 2.*np.pi*(czmax-czmin), "sr")
#print("d_omega:",  d_omega,  "sr")
#print("omega_cz:", omega_cz, "sr")

### 1. Exposure & # of CR events / day as a function of Energy & cos(Zenith)
###    The error is calculated as s.t.d. of the binomial distribution
exp_le_cz = S_geo * omega_cz * T_obs * np.sum(Nana_w_cos, axis=2) / np.sum(Nall_w, axis=2)
J_int_le, Nevent_le_cz = np.zeros(nbin_le), np.zeros((nbin_le, nbin_cz))
yr2day = 365
for n in range(nbin_le):
    J_int_le[n], err = integrate.quad(calculate_PAO_spectrum, 10 ** (x_le[n]-binw_le/2),
                                                              10 ** (x_le[n]+binw_le/2))
    Nevent_le_cz[n,:] = exp_le_cz[n,:] * (J_int_le[n]/yr2day)
print("np.sum(Nevent_le_cz):", np.sum(Nevent_le_cz))

### 2. Exposure & # of CR events / day as a function of Energy
exp_le = S_geo * omega_tot * T_obs * \
         np.sum(np.sum(Nana_w_cos, axis=1), axis=1) / np.sum(np.sum(Nall_w, axis=1), axis=1)
exp_le_err = exp_le * \
             np.sqrt( 1. / np.sum(np.sum(Nana, axis=1), axis=1) - \
                      1. / np.sum(np.sum(Nall, axis=1), axis=1) )
Nevent_le, Nevent_le_err = (J_int_le/yr2day)*exp_le, (J_int_le/yr2day)*exp_le_err
print("np.sum(Nevent_le):", np.sum(Nevent_le))

### 3. Exposure & # of CR events / day as a function of cos(Zenith)
exp_cz = S_geo * omega_cz * T_obs * \
         np.sum(np.sum(Nana_w_cos, axis=0), axis=1) / np.sum(np.sum(Nall_w, axis=0), axis=1)
exp_cz_err = exp_cz * \
             np.sqrt( 1. / np.sum(np.sum(Nana, axis=0), axis=1) - \
                      1. / np.sum(np.sum(Nall, axis=0), axis=1) )
J_int, err = integrate.quad(calculate_PAO_spectrum, emin_eV, emax_eV)
Nevent_cz, Nevent_cz_err = (J_int/yr2day)*exp_cz, (J_int/yr2day)*exp_cz_err
print("np.sum(Nevent_cz):", np.sum(Nevent_cz))
#print("exp_cz:", exp_cz)
#print("Nevent_cz:", Nevent_cz)


### Plot figures to check the weighting scheme
params = {
    "legend.fontsize": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 23,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    #"figure.figsize": (10, 8),
    "axes.grid": False,
}
plt.rcParams.update(params)
fig = plt.figure(figsize=(30, 20))

ax = fig.add_subplot(2,2,1)
ax.set_yscale('log')
ax.plot(x_le, exp_le, c='black')
ax.errorbar(x_le, exp_le, yerr = exp_le_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'${\rm log}_{10}$(Energy [eV])')
ax.set_ylabel('One-day exposure (km$^2$ day sr)')
ax.set_xlim(16.9, 20.0)
ax.set_ylim(8.e-2,1.5e2)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.set_xticks([17, 17.5, 18, 18.5, 19.0, 19.5, 20.0])
ax.minorticks_on()
#ax.legend()
#ax.text(17.5, 2, 'GRAND preliminary', color=(1, 0, 0, 0.5), fontsize='60', rotation=30)
#ax.text(17.5, 1, 'GRAND preliminary', color=(0, 0, 1, 0.4), fontsize='40', rotation=0)

ax = fig.add_subplot(2,2,2)
ax.set_yscale('log')
ax.plot(x_le, Nevent_le, c='black')
ax.errorbar(x_le, Nevent_le, yerr = Nevent_le_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'${\rm log}_{10}$(Energy [eV])')
ax.set_ylabel('Number of CR events / day / bin')
ax.set_xlim(16.9, 20.0)
ax.set_ylim(1.e-5,5.e1)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.set_xticks([17, 17.5, 18, 18.5, 19.0, 19.5, 20.0])
ax.minorticks_on()
#ax.legend()
#ax.text(17.5, 5.e-3, 'GRAND preliminary', color=(1, 0, 0, 0.5), fontsize='60', rotation=30)
#ax.text(17.5, 1.e-3, 'GRAND preliminary', color=(0, 0, 1, 0.4), fontsize='40', rotation=0)

ax = fig.add_subplot(2,2,3)
ax.set_yscale('log')
ax.plot(x_cz, exp_cz, c='black')
ax.errorbar(x_cz, exp_cz, yerr = exp_cz_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'cos(Zenith [deg.])')
ax.set_ylabel('Exposure (km$^2$ day sr) / bin')
ax.set_xlim(0, 0.5)
ax.set_ylim(1.e-3,3)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.minorticks_on()

ax = fig.add_subplot(2,2,4)
ax.set_yscale('log')
ax.plot(x_cz, Nevent_cz, c='black')
ax.errorbar(x_cz, Nevent_cz, yerr = Nevent_cz_err, capsize=5, fmt='o', markersize=7, ecolor='black', markeredgecolor = "black", color='black')
ax.set_title('')
ax.set_xlabel(r'cos(Zenith [deg.])')
ax.set_ylabel('Exposure (km$^2$ day sr) / bin')
ax.set_xlim(0, 0.5)
ax.set_ylim(1.e-2,3.e1)
ax.tick_params(which='both', direction='in')
ax.tick_params(which='major', direction='in', length=9, width=1)
ax.tick_params(which='minor', direction='in', length=5, width=1)
ax.minorticks_on()

plt.savefig('calculate_exposure.pdf')

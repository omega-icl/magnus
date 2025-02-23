#!/usr/bin/env python
# coding: utf-8

# # Parameter Estimation in a Dynamic Model Describing Microalgae Growth and Lipid Biosynthesis

# Consider the following model by [del Rio-Chanona _et al_ (2018)](https://doi.org/10.1002/bit.26483) describing the growth and lipid biosynthesis in _N. oceanica_, a microalga with a naturally high cellular oil content, in bubble column photobioreactor:
# $$\begin{align}
# \dot{X}(t) =\ & \left[\bar{\mu}(I_0(t),X(t))\left(1-\frac{k_q}{q(t)}\right) - k_d \right] X(t)\\
# \dot{N}(t) =\ & - \rho_m \frac{N(t)}{k_N+N(t)} X(t)\\
# \dot{q}(t) =\ & \rho_m \frac{N(t)}{k_N+N(t)} - \bar{\mu}(I_0(t),X(t))\left(1-\frac{k_q}{q(t)}\right) q(t)\\
# \dot{f}(t) =\ & - \gamma_N \rho_m \frac{N(t)}{k_N+N(t)} + \bar{\mu}(I_0(t),X(t))\left(1-\frac{k_q}{q(t)}\right) \left(\gamma_q q(t)- \gamma_f f(t)\right)\\
# \text{with:}\ \ \bar{\mu}(I_0,X) \coloneqq\ & \frac{1}{H} \int_{0}^{H} \mu(I_0,X,z) dz\\ \approx\ & \frac{1}{2K} \left[ \mu(I_0,X,0) + 2 \sum_{k=1}^{K-1} \mu(I_0,X,{\textstyle\frac{k}{K}H}) + \mu(I_0,X,H) \right]\\
#                 \mu(I_0,X,z) \coloneqq\ & \mu_m \frac{I(I_0,X,z)}{k_I+I(I_0,X,z)+I(I_0,X,z)^2/k'_I}\\
#                 I(I_0,X,z) \coloneqq\ & I_0 \exp\left(-(\epsilon_0+\epsilon_X X)z\right)
# \end{align}$$
# where $N$ (mg L$^{-1}$) is the culture nitrate concentration; $X$ (g L$^{-1}$), the biomass concentration; $q$ (mg g$^{-1}$), the nitrogen quota; and $f$ [mg g$^{-1}$], the FAME quota.
# 
# The bubble column is illuminated on one side only and assimilated to a column with square cross-section of width $H=4.4$ cm. Guesses and ranges for the model parameters are reported in the table below.
# 
# Parameter | Guess | Units | Range |   | Parameter | Guess | Units | Range
# :-------- | ----: | :---- | :---- | - | :-------- | ----: | :---- | :----
# $\mu_m$      | $0.36$  | $\rm h^{-1}$                  | $[0.3,1]$  || $\gamma_q$   | $6.69$   | $\rm g\,mg^{-1}$ |  $[4,15]$
# $k_d$        | $0.00$  | $\rm h^{-1}$                  | $[0,0.5]$  || $\gamma_N$   | $7.53$   | $\rm L\,mg^{-1}$ |  $[1,10]$
# $k_q$        | $19.6$  | $\rm mg\,g^{-1}$              | $[10,100]$ || $\gamma_f$   | $0.001$  | $\rm g\,mg^{-1}$ |  $[0.0002,0.004]$
# $\rho_m$     | $2.69$  | $\rm mg\,g^{-1}\,h^{-1}$      | $[1,10]$   || $\tau_q$     | $0.138$  | $\rm g\,mg^{-1}$ |  $[0.05,0.2]$
# $k_N$        | $0.80$  | $\rm mg\,L^{-1}$              | $[0.2,2]$  || $\delta$     | $9.90$   | $-$              |  $[5,20]$
# $k_I$        | $91.2$  | $\rm \mu mol\,m^{-2}\,s^{-1}$ | $[50,200]$ || $Y_0$        | $-0.456$ | $-$              |  $[-2,2]$
# $k'_I$       | $100.0$ | $\rm \mu mol\,m^{-2}\,s^{-1}$ | $[80,250]$ || $\epsilon_0$ | $0.00$   | $\rm m^{-1}$ |  $[0,50]$
# $\epsilon_X$ | $196.4$ | $\rm L\,kg^{-1}\,m^{-1}$      | $[50,300]$ | $\qquad\qquad$ 
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import pymc
import cronos
import canon
from magnus import ParEst


# We start by defining the DAG for the dynamic model:

# In[2]:


IVPDAG = pymc.FFGraph()

# States
X  = pymc.FFVar(IVPDAG, "X") # biomass concentration [mg/L]
N  = pymc.FFVar(IVPDAG, "N") # nitrate concentration [mg/L]
q  = pymc.FFVar(IVPDAG, "q") # internal nitrogen quota [mg/g]
f  = pymc.FFVar(IVPDAG, "f") # internal FAME quota [mg/g]

# Controls
I0 = pymc.FFVar(IVPDAG, "I0") # incident light intensity [µmol/m2/s]

# Initial concentrations
X0 = pymc.FFVar(IVPDAG, "X0")
N0 = pymc.FFVar(IVPDAG, "N0")
q0 = pymc.FFVar(IVPDAG, "q0")
f0 = pymc.FFVar(IVPDAG, "f0")

# Parameters
mum  = pymc.FFVar(IVPDAG, "mum")   # maximal growth rate [/h]
kd   = pymc.FFVar(IVPDAG, "kd")    # decay rate [/h]
rhom = pymc.FFVar(IVPDAG, "rhom")  #  maximal nitrate internalisation rate [mg/g/h]
kq   = pymc.FFVar(IVPDAG, "kq")    # minimal internal nitrogen quota [mg/g]
kN   = pymc.FFVar(IVPDAG, "kN")    # nitrate half-saturation constant [mg/L]
kI1  = pymc.FFVar(IVPDAG, "kI1")   # light half-saturation constant [µmol/m2/s]
kI2  = pymc.FFVar(IVPDAG, "kI2")   # light inhibition constant [µmol/m2/s]
gq   = pymc.FFVar(IVPDAG, "gq")    # yield of nitrate internalisation for FAME synthesis [g/mg]
gN   = pymc.FFVar(IVPDAG, "gN")    # yield of internal nitrogen quota for FAME synthesis [L/mg]
gf   = pymc.FFVar(IVPDAG, "gf")    # yield of internal FAME quota for FAME synthesis [g/mg]
e0   = pymc.FFVar(IVPDAG, "e0")    # basal light attenuation rate [/m]
eX   = pymc.FFVar(IVPDAG, "eX")    # biomass-dependent light attenuation rate [m2/kg = 1e6 L/g/m]
Y0   = pymc.FFVar(IVPDAG, "Y0")    # basal chlorophyll fluorescence Y(II) level [-]
d    = pymc.FFVar(IVPDAG, "dq")    # nitrogen-dependent chlorophyll fluorescence Y(II) reference [-]
tq   = pymc.FFVar(IVPDAG, "tq")    # nitrogen-dependent chlorophyll fluorescence Y(II) rate [g/mg]

# Constants
H = 0.044 # light depth [m]


# We instantiate `ODESLV` and populate it with the dynamic model expressions:

# In[3]:


IVP = cronos.ODESLV()

IVP.set_dag( IVPDAG )
IVP.set_parameter( [mum, kd, rhom, kq, kN, kI1, kI2, gq, gN, gf, e0, eX, Y0, d, tq] )
IVP.set_constant( [I0, X0, N0, q0, f0] )
IVP.set_state( [X, N, q, f] )

timegrid = np.linspace( 0, 252, 22 ).tolist()
timegrid.insert( 1, 1e-3 ) # extra timepoint to account for initial measurements
print( timegrid )
IVP.set_time( timegrid ) # measurement times every 12 h

K = 20
for k in range(K+1):
    z   = k / K * H
    Iz  = I0 * pymc.exp( -z * ( e0 + eX * X ) )
    muz = mum * Iz / ( kI1 + Iz + Iz**2 / kI2 )

    if k == 0:
        mu = muz
    elif k == K:
        mu += muz
    else:
        mu += 2 * muz
mu /= 2*K
#mu = mum * I0 / ( kI1 + I0 + I0**2 / kI2 )
#print( mu.str() )

rho = rhom * N / ( kN + N )

dX  = ( mu * ( 1 - kq / q ) - kd ) * X
dN  = - rho * X
dq  = rho - mu * ( 1 - kq / q ) * q
df  = - gN * rho + mu * ( 1 - kq / q ) * ( gq * q - gf * f )
YII = Y0 + pymc.exp( tq * q ) / ( pymc.exp( tq * q ) + d )

IVP.set_differential( [dX, dN, dq, df] )
IVP.set_initial( [ X0, N0, q0, f0] )

Y = [ X, N, q, f, YII ]
F = []
for k in range(len(timegrid)):
  F.append( [ pymc.FFVar(0) for i in range(k*len(Y)) ] + Y + [ pymc.FFVar(0) for i in range((len(timegrid)-k-2)*len(Y)) ] )
IVP.set_function( F )
#print( IVP.eqn_function )

IVP.setup()


# We solve the dynamic model with initial parameter guesses and record trajectories for display:

# In[4]:


IVP.options.DISPLEVEL = 1 # displays numerical integration results
IVP.options.RESRECORD = 50 # record 50 points along time horizon
IVP.solve_state( [0.36, 0.0, 2.69, 19.6, 0.8, 91.2, 100, 6.69, 7.53, 0.001, 0.0, 196.4, -0.456, 9.90, 0.136],
                 [80, 0.18, 35, 80, 120] )
#IVP.solve_state( [0.443, 0.0, 2.850, 19.91, 1.64, 111.8, 250, 6.58, 7.35, 0.0002, 0.0, 260.0, -0.459, 11.05, 0.146],
#                 [80, 0.18, 35, 80, 120] )


# In[5]:


# Gather simulation results for Experiment 1
Pred_Exp1 = []
for rec in IVP.results_state:
    [recY] = IVPDAG.eval( [YII], [q, Y0, d, tq], [rec.x[2], -0.456, 9.90, 0.136] )
    Pred_Exp1.append( [rec.t] + rec.x + [recY] )
Pred_Exp1 = np.array( Pred_Exp1 ) # convert list into numpy array


# In[6]:


# Experimental dataset 1
Tmeas_Exp1 = np.linspace( 0, 252, 22 )
Xmeas_Exp1 = [ 0.178333333333333, 0.361666666666667, 0.526666666666667, 0.693333333333333, 0.858333333333333, 1.005, 1.14666666666667, 1.28, 1.38166666666667, 1.48333333333333, 1.56333333333333, 1.665, 1.73666666666667, 1.85666666666667, 1.92833333333333, 2.01, 2.11833333333333, 2.205, 2.29166666666667, 2.415, 2.515, 2.635 ]
Xstd_Exp1  = [ 0.0132916013582513, 0.0116904519445001, 0.0163299316185545, 0.0233809038890002, 0.0348807492274273, 0.0459347363114235, 0.0377712412645741, 0.0675277720645366, 0.033115957885386, 0.0539135109844153, 0.0508592829940284, 0.0288097205817759, 0.0163299316185545, 0.053166405433005, 0.0865832932306611, 0.0629285308902092, 0.0801040989379862, 0.0880340843082949, 0.143166569654604, 0.11273863579093, 0.113973681172453, 0.123409886151799 ]
Nmeas_Exp1 = [ 35.0166666666667, 26.35, 13.0833333333333, 0.366666666666667 ]#, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
Nstd_Exp1  = [ 0.783368793523629, 0.568330889535314, 1.02257844034904, 0.136626010212795 ] #, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]
qmeas_Exp1 = [ 79.85, 66.15, 72.6, 73.15, 58.25, 49.7, 44, 39.45, 36, 31.8, 32.1, 29.4, 28.3, 27.45, 25.85, 25.35, 24.2, 23.45, 22.35, 22.25, 21.7, 21.15 ]
qstd_Exp1  = [ 3.32340187157678, 2.33345237791561, 0.848528137423858, 1.90918830920368, 0.777817459305205, 1.5556349186104, 1.5556349186104, 1.20208152801713, 0.848528137423858, 1.41421356237309, 1.4142135623731, 0.14142135623731, 1.27279220613578, 1.20208152801713, 0.919238815542511, 0.919238815542511, 1.13137084989848, 1.06066017177982, 0.353553390593272, 1.06066017177982, 0.707106781186548, 1.34350288425444 ]
fmeas_Exp1 = [ 120.310094932449, 186.974203436981, 154.586561820694, 139.225664382881, 168.941168514762, 214.93311515847, 260.038778399742, 298.426581520616, 328.060190155951, 347.432746376905, 372.944306996692, 376.294037160035, 397.480426557128, 407.158291504756, 416.992060948065, 418.780596921408, 434.808432059336, 428.557248744812, 448.621729384578, 446.639578285446, 446.865230358034, 449.124283061455 ]
fstd_Exp1  = [ 5.18659127581244, 5.3913211602626, 0.942201962375798, 9.68060010713876, 4.59883435611163, 5.17932269635727, 5.69311350957571, 8.38465999407327, 12.9684736352478, 7.55740219017954, 13.099330721354, 8.79595058120159, 14.2178596157322, 5.19632230316718, 13.2373056891963, 11.3836830193887, 8.69612815856261, 7.93505803439914, 18.2289597003499, 12.8767750328968, 11.4447261725692, 8.90649551508841 ]
Ymeas_Exp1 = [ 0.561, 0.546, 0.5725, 0.54325, 0.53, 0.51575, 0.4925, 0.479, 0.454, 0.43425, 0.4255, 0.39575, 0.38375, 0.36625, 0.35, 0.3345, 0.313, 0.308, 0.2855, 0.2765, 0.2625, 0.23175 ]
Ystd_Exp1  = [ 0.00282842712474611, 0.0014142135623731, 0.00129099444873581, 0.0104363148029689, 0.00627162924074227, 0.00960468635614928, 0.0138924439894498, 0.00828653526310404, 0.00836660026534076, 0.0158823801742686, 0.0104083299973307, 0.0291132844820596, 0.0214067746286077, 0.0199394918022836, 0.0255994791613684, 0.0218097837372741, 0.00898146239020499, 0.0124899959967968, 0.0113871272350258, 0.0114455231422596, 0.0228108161479008, 0.0133759734848222 ]


# In[7]:


fig, axes = plt.subplots(2, 3, figsize=(10, 7))

axes[0,0].plot( Pred_Exp1[:,0], Pred_Exp1[:,1], color="blue" )
axes[0,0].errorbar(Tmeas_Exp1, Xmeas_Exp1, yerr=Xstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,0].set(xlabel="$t$ (h)")
axes[0,0].set(ylabel="$X$ (g L$^{-1}$)")

axes[0,1].plot( Pred_Exp1[:,0], Pred_Exp1[:,2], color="blue" )
axes[0,1].errorbar(Tmeas_Exp1[:len(Nmeas_Exp1)], Nmeas_Exp1, yerr=Nstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,1].set(xlabel="$t$ (h)")
axes[0,1].set(ylabel="$N$ (mg L$^{-1}$)")

axes[1,0].plot( Pred_Exp1[:,0], Pred_Exp1[:,3], color="blue" )
axes[1,0].errorbar(Tmeas_Exp1, qmeas_Exp1, yerr=qstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,0].set(xlabel="$t$ (h)")
axes[1,0].set(ylabel="$q$ (mg g$^{-1}$)")

axes[1,1].plot( Pred_Exp1[:,0], Pred_Exp1[:,4], color="blue" )
axes[1,1].errorbar(Tmeas_Exp1, fmeas_Exp1, yerr=fstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,1].set(xlabel="$t$ (h)")
axes[1,1].set(ylabel="$f$ (mg g$^{-1}$)")

axes[0,2].plot( Pred_Exp1[:,0], Pred_Exp1[:,5], color="blue" )
axes[0,2].errorbar(Tmeas_Exp1, Ymeas_Exp1, yerr=Ystd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,2].set(xlabel="$t$ (h)")
axes[0,2].set(ylabel="YII ($-$)")

[ fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data() ]
fig.tight_layout()


# In[8]:


IVP.options.DISPLEVEL = 1 # displays numerical integration results
IVP.options.RESRECORD = 50 # record 50 points along time horizon
IVP.solve_state( [0.36, 0.0, 2.69, 19.6, 0.8, 91.2, 100, 6.69, 7.53, 0.001, 0.0, 196.4, -0.456, 9.90, 0.136],
                 [160, 0.17, 24.6, 79, 112] )
#IVP.solve_state( [0.443, 0.0, 2.850, 19.91, 1.64, 111.8, 250, 6.58, 7.35, 0.0002, 0.0, 260.0, -0.459, 11.05, 0.146],
#                 [160, 0.17, 24.6, 79, 112] )


# In[9]:


# Gather simulation results for Experiment 2
Pred_Exp2 = []
for rec in IVP.results_state:
    [recY] = IVPDAG.eval( [YII], [q, Y0, d, tq], [rec.x[2], -0.456, 9.90, 0.136] )
    Pred_Exp2.append( [rec.t] + rec.x + [recY] )
Pred_Exp2 = np.array( Pred_Exp2 ) # convert list into numpy array


# In[10]:


# Experimental dataset 2
Tmeas_Exp2 = np.linspace( 0, 252, 22 )
Xmeas_Exp2 = [ 0.168, 0.412, 0.6234, 0.9114, 1.121, 1.25, 1.376, 1.491, 1.547, 1.616, 1.6466, 1.73, 1.744, 1.752, 1.716, 1.754, 1.746, 1.796, 1.86, 1.868, 1.912, 1.906 ]
Xstd_Exp2  = [ 0.0109544511501033, 0.0131576973669408, 0.078662570514827, 0.0486805916151396, 0.0489131884055824, 0.0651920240520265, 0.0477493455452532, 0.0288097205817759, 0.0399374510954317, 0.0456070170039655, 0.0571646744064899, 0.0346410161513776, 0.0433589667773576, 0.0884307638777367, 0.159154013458662, 0.170967833231868, 0.202188031297601, 0.23880954754783, 0.239269722280111, 0.275535841588712, 0.345427271650633, 0.35711342735887 ]
Nmeas_Exp2 = [ 24.56, 16.44, 1.46 ] #, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
Nstd_Exp2  = [ 0.654217089351845, 0.952890339965727, 0.792464510246358 ] #, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]
qmeas_Exp2 = [ 79.4, 54.7, 57.2, 44.25, 35.2, 30.8, 28.15, 25.45, 26, 24.45, 23.15, 22.45, 22.4, 21.95, 22.5, 22.45, 22.45, 22.35, 22.1, 22.65, 23.05, 23.35 ]
qstd_Exp2  = [ 2.26274169979695, 2.82842712474619, 0.565685424949238, 1.62634559672906, 0.565685424949238, 0.707106781186548, 1.06066017177982, 0.0707106781186532, 1.5556349186104, 0.0707106781186564, 0.0707106781186532, 0.0707106781186532, 0.14142135623731, 0.212132034355963, 1.13137084989848, 0.919238815542511, 2.1920310216783, 2.33345237791561, 2.54558441227157, 2.89913780286485, 3.04055915910216, 3.74766594028872 ]
fmeas_Exp2 = [ 111.541419555231, 265.097777707633, 258.388703449306, 288.601296621711, 335.754543631457, 386.8327041946, 414.59026056901, 439.507223502483, 432.844087468803, 446.664778453333, 449.893233709044, 449.08012514686, 448.115811128386, 451.062200660261, 447.850247629155, 449.383167348635, 437.771893179336, 432.920001998863, 431.965099784869, 423.909181037424, 419.672561698015, 410.466366110378 ]
fstd_Exp2  = [ 7.01025545078043, 0.655484470985941, 5.19140785492464, 13.1526708129503, 3.65650860011034, 5.38552447401581, 3.36976396296463, 11.9645601509965, 1.55513007881774, 0.944056660954174, 7.31483963056877, 5.43718473787734, 7.90628227387481, 0.769130602993408, 10.4490818167955, 20.2549610837514, 19.674661511446, 18.6758933431357, 17.0328029457106, 14.5343388053181, 24.9095089630175, 15.6070595409698 ]
Ymeas_Exp2 = [ 0.555, 0.53425, 0.5575, 0.50325, 0.49725, 0.45525, 0.41425, 0.38025, 0.3295, 0.27975, 0.2525, 0.22675, 0.21275, 0.20575, 0.1995, 0.19325, 0.193, 0.18825, 0.186, 0.17825, 0.17725, 0.175 ]
Ystd_Exp2  = [ 0.00707106781186548, 0.00125830573921179, 0.00519615242270664, 0.0172892066523212, 0.0109048918686371, 0.00704154339142588, 0.0253558014400387, 0.0365365114189446, 0.0478365271872167, 0.0528795801798764, 0.0410649891432269, 0.0310201547384923, 0.0162762608318577, 0.0192591969372211, 0.0122338328690017, 0.0164595463687997, 0.0147648230602334, 0.0185898717944297, 0.0298105126870818, 0.0321597574617719, 0.0398612175763194, 0.0449073119510248 ]


# In[11]:


fig, axes = plt.subplots(2, 3, figsize=(10, 7))

axes[0,0].plot( Pred_Exp2[:,0], Pred_Exp2[:,1], color="blue" )
axes[0,0].errorbar(Tmeas_Exp2, Xmeas_Exp2, yerr=Xstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,0].set(xlabel="$t$ (h)")
axes[0,0].set(ylabel="$X$ (g L$^{-1}$)")

axes[0,1].plot( Pred_Exp2[:,0], Pred_Exp2[:,2], color="blue" )
axes[0,1].errorbar(Tmeas_Exp2[:len(Nmeas_Exp2)], Nmeas_Exp2, yerr=Nstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,1].set(xlabel="$t$ (h)")
axes[0,1].set(ylabel="$N$ (mg L$^{-1}$)")

axes[1,0].plot( Pred_Exp2[:,0], Pred_Exp2[:,3], color="blue" )
axes[1,0].errorbar(Tmeas_Exp2, qmeas_Exp2, yerr=qstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,0].set(xlabel="$t$ (h)")
axes[1,0].set(ylabel="$q$ (mg g$^{-1}$)")

axes[1,1].plot( Pred_Exp2[:,0], Pred_Exp2[:,4], color="blue" )
axes[1,1].errorbar(Tmeas_Exp2, fmeas_Exp2, yerr=fstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,1].set(xlabel="$t$ (h)")
axes[1,1].set(ylabel="$f$ (mg g$^{-1}$)")

axes[0,2].plot( Pred_Exp2[:,0], Pred_Exp2[:,5], color="blue" )
axes[0,2].errorbar(Tmeas_Exp2, Ymeas_Exp2, yerr=Ystd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,2].set(xlabel="$t$ (h)")
axes[0,2].set(ylabel="YII ($-$)")

[ fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data() ]
fig.tight_layout()


# In[12]:


OpIVP = cronos.FFODE()
IVP.options.DISPLEVEL = 0 # turn off display during numerical simulation
IVP.options.RESRECORD = 0 # turn off trajectory record
IVP.options.ATOL, IVP.options.ATOLS, IVP.options.RTOL, IVP.options.RTOLS = 1e-10, 1e-10, 1e-9, 1e-9

# Experiment 1
I0_Exp1 = pymc.FFVar(IVPDAG, "I0_Exp1") # incident light intensity [µmol/m2/s]
X0_Exp1 = pymc.FFVar(IVPDAG, "X0_Exp1")
N0_Exp1 = pymc.FFVar(IVPDAG, "N0_Exp1")
q0_Exp1 = pymc.FFVar(IVPDAG, "q0_Exp1")
f0_Exp1 = pymc.FFVar(IVPDAG, "f0_Exp1")
Y_Exp1 = OpIVP( [mum, kd, rhom, kq, kN, kI1, kI2, gq, gN, gf, e0, eX, Y0, d, tq], [I0_Exp1, X0_Exp1, N0_Exp1, q0_Exp1, f0_Exp1], IVP )
IVPDAG.output( Y_Exp1 )

# Experiment 2
I0_Exp2 = pymc.FFVar(IVPDAG, "I0_Exp2") # incident light intensity [µmol/m2/s]
X0_Exp2 = pymc.FFVar(IVPDAG, "X0_Exp2")
N0_Exp2 = pymc.FFVar(IVPDAG, "N0_Exp2")
q0_Exp2 = pymc.FFVar(IVPDAG, "q0_Exp2")
f0_Exp2 = pymc.FFVar(IVPDAG, "f0_Exp2")
Y_Exp2 = OpIVP( [mum, kd, rhom, kq, kN, kI1, kI2, gq, gN, gf, e0, eX, Y0, d, tq], [I0_Exp2, X0_Exp2, N0_Exp2, q0_Exp2, f0_Exp2], IVP )
IVPDAG.output( Y_Exp2 )


# In[24]:


# Instantiate parameter estimation solver
PE = ParEst()

PE.options.DISPLEVEL = 1
PE.options.NLPSLV.DISPLEVEL   = 0;
PE.options.NLPSLV.GRADCHECK   = 0;
PE.options.NLPSLV.MAXTHREAD   = 0;
PE.options.NLPSLV.GRADMETH    = PE.options.NLPSLV.FSYM
PE.options.NLPSLV.FCTPREC     = 1e-6;
PE.options.NLPSLV.GRADLSEARCH = False;

PE.set_dag( IVPDAG )
PE.add_model( Y_Exp1, [I0_Exp1, X0_Exp1, N0_Exp1, q0_Exp1, f0_Exp1] )
PE.add_model( Y_Exp2, [I0_Exp2, X0_Exp2, N0_Exp2, q0_Exp2, f0_Exp2] )
PE.set_parameter( [mum, kd,  rhom, kq,    kN,  kI1,   kI2,   gq,   gN,   gf,     e0,   eX,    Y0,   d,    tq   ],
                   [0.3, 0.0,  1.0,  10.0, 0.2,  50.0,  80.0,  4.0,  1.0, 0.0002,  0.0,  50.0, -2.0,  5.0, 0.05 ],
                   [1.0, 0.5, 10.0, 100.0, 2.0, 200.0, 250.0, 15.0, 10.0, 0.0040, 50.0, 300.0,  2.0, 20.0, 2.0  ] )


# In[25]:


IVPDAG.eval( Y_Exp1,
             [mum, kd, rhom, kq, kN, kI1, kI2, gq, gN, gf, e0, eX, Y0, d, tq, I0_Exp1, X0_Exp1, N0_Exp1, q0_Exp1, f0_Exp1],
             [0.36, 0.0, 2.69, 19.6, 0.8, 91.2, 100, 6.69, 7.53, 0.001, 0.0, 196.4, -0.456, 9.90, 0.136, 80, 0.18, 35, 80, 120] )
IVPDAG.eval( Y_Exp2,
             [mum, kd, rhom, kq, kN, kI1, kI2, gq, gN, gf, e0, eX, Y0, d, tq, I0_Exp2, X0_Exp2, N0_Exp2, q0_Exp2, f0_Exp2],
             [0.36, 0.0, 2.69, 19.6, 0.8, 91.2, 100, 6.69, 7.53, 0.001, 0.0, 196.4, -0.456, 9.90, 0.136, 160, 0.17, 24.6, 79, 112] )


# In[26]:


# Set experimental data
Exp1 = ParEst.Experiment( [80, 0.18, 35, 80, 120] )

Exp1_out = {}
for k in range(len(Xmeas_Exp1)):
    Exp1_out[len(Y)*k]   = ParEst.Record( [Xmeas_Exp1[k]], 1e-2 ) #Xstd_Exp1[k]**2 )
for k in range(len(Nmeas_Exp1)):
    Exp1_out[len(Y)*k+1] = ParEst.Record( [Nmeas_Exp1[k]], 1e0 ) #Nstd_Exp1[k]**2 )
for k in range(len(qmeas_Exp1)):
    Exp1_out[len(Y)*k+2] = ParEst.Record( [qmeas_Exp1[k]], 4e0 ) #qstd_Exp1[k]**2 )
for k in range(len(fmeas_Exp1)):
    Exp1_out[len(Y)*k+3] = ParEst.Record( [fmeas_Exp1[k]], 1e2 ) #fstd_Exp1[k]**2 )
for k in range(len(Ymeas_Exp1)):
    Exp1_out[len(Y)*k+4] = ParEst.Record( [Ymeas_Exp1[k]], 4e-4 ) #Ystd_Exp1[k]**2 )
#print(Exp1_out)

Exp1.output = Exp1_out
#print(Exp1)

Exp2 = ParEst.Experiment( [160, 0.17, 24.6, 79, 112] )

Exp2_out = {}
for k in range(len(Xmeas_Exp2)):
    Exp2_out[len(Y)*k]   = ParEst.Record( [Xmeas_Exp2[k]], 1e-2 ) #Xstd_Exp2[k]**2 )
for k in range(len(Nmeas_Exp2)):
    Exp2_out[len(Y)*k+1] = ParEst.Record( [Nmeas_Exp2[k]], 1e0 ) #Nstd_Exp2[k]**2 )
for k in range(len(qmeas_Exp2)):
    Exp2_out[len(Y)*k+2] = ParEst.Record( [qmeas_Exp2[k]], 4e0 ) #qstd_Exp2[k]**2 )
for k in range(len(fmeas_Exp2)):
    Exp2_out[len(Y)*k+3] = ParEst.Record( [fmeas_Exp2[k]], 1e2 ) #fstd_Exp2[k]**2 )
for k in range(len(Ymeas_Exp2)):
    Exp2_out[len(Y)*k+4] = ParEst.Record( [Ymeas_Exp2[k]], 4e-4 ) #Ystd_Exp2[k]**2 )
#print(Exp2_out)

Exp2.output = Exp2_out
#print(Exp2)

Data = [ Exp1, Exp2 ]
PE.set_data( Data );


# In[27]:


PE.setup()
#PE.mle_solve( [0.36, 0.0, 2.69, 19.6, 0.8, 91.2, 100, 6.69, 7.53, 0.001, 0.0, 196.4, -0.456, 9.90, 0.136] )
#PE.mle_solve( [0.443, 0.0, 2.850, 19.91, 1.64, 111.8, 250, 6.58, 7.35, 0.0002, 0.0, 260.0, -0.459, 11.05, 0.146] )
PE.mle_solve( 16 )
#print( PE.mle )


# In[32]:


IVP.options.RESRECORD = 50 # record 50 points along time horizon
IVP.solve_state( PE.mle.x, [80, 0.18, 35, 80, 120] )

# Gather MLE results for Experiment 2
MLE_Exp1 = []
for rec in IVP.results_state:
    [recY] = IVPDAG.eval( [YII], [q, Y0, d, tq], [rec.x[2], -0.456, 9.90, 0.136] )
    MLE_Exp1.append( [rec.t] + rec.x + [recY] )
MLE_Exp1 = np.array( MLE_Exp1 ) # convert list into numpy array


# In[33]:


IVP.options.RESRECORD = 50 # record 50 points along time horizon
IVP.solve_state( PE.mle.x, [160, 0.17, 24.6, 79, 112] )

# Gather MLE results for Experiment 2
MLE_Exp2 = []
for rec in IVP.results_state:
    [recY] = IVPDAG.eval( [YII], [q, Y0, d, tq], [rec.x[2], -0.456, 9.90, 0.136] )
    MLE_Exp2.append( [rec.t] + rec.x + [recY] )
MLE_Exp2 = np.array( MLE_Exp2 ) # convert list into numpy array


# In[34]:


fig, axes = plt.subplots(2, 3, figsize=(10, 7))

axes[0,0].plot( Pred_Exp1[:,0], Pred_Exp1[:,1], color="blue" )
axes[0,0].plot( MLE_Exp1[:,0], MLE_Exp1[:,1], color="magenta" )
axes[0,0].errorbar(Tmeas_Exp1, Xmeas_Exp1, yerr=Xstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,0].set(xlabel="$t$ (h)")
axes[0,0].set(ylabel="$X$ (g L$^{-1}$)")

axes[0,1].plot( Pred_Exp1[:,0], Pred_Exp1[:,2], color="blue" )
axes[0,1].plot( MLE_Exp1[:,0], MLE_Exp1[:,2], color="magenta" )
axes[0,1].errorbar(Tmeas_Exp1[:len(Nmeas_Exp1)], Nmeas_Exp1, yerr=Nstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,1].set(xlabel="$t$ (h)")
axes[0,1].set(ylabel="$N$ (mg L$^{-1}$)")

axes[1,0].plot( Pred_Exp1[:,0], Pred_Exp1[:,3], color="blue" )
axes[1,0].plot( MLE_Exp1[:,0], MLE_Exp1[:,3], color="magenta" )
axes[1,0].errorbar(Tmeas_Exp1, qmeas_Exp1, yerr=qstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,0].set(xlabel="$t$ (h)")
axes[1,0].set(ylabel="$q$ (mg g$^{-1}$)")

axes[1,1].plot( Pred_Exp1[:,0], Pred_Exp1[:,4], color="blue" )
axes[1,1].plot( MLE_Exp1[:,0], MLE_Exp1[:,4], color="magenta" )
axes[1,1].errorbar(Tmeas_Exp1, fmeas_Exp1, yerr=fstd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,1].set(xlabel="$t$ (h)")
axes[1,1].set(ylabel="$f$ (mg g$^{-1}$)")

axes[0,2].plot( Pred_Exp1[:,0], Pred_Exp1[:,5], color="blue" )
axes[0,2].plot( MLE_Exp1[:,0], MLE_Exp1[:,5], color="magenta" )
axes[0,2].errorbar(Tmeas_Exp1, Ymeas_Exp1, yerr=Ystd_Exp1, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,2].set(xlabel="$t$ (h)")
axes[0,2].set(ylabel="$YII$ ($-$)")

[ fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data() ]
fig.tight_layout()


# In[35]:


fig, axes = plt.subplots(2, 3, figsize=(10, 7))

axes[0,0].plot( Pred_Exp2[:,0], Pred_Exp2[:,1], color="blue" )
axes[0,0].plot( MLE_Exp2[:,0], MLE_Exp2[:,1], color="magenta" )
axes[0,0].errorbar(Tmeas_Exp2, Xmeas_Exp2, yerr=Xstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,0].set(xlabel="$t$ (h)")
axes[0,0].set(ylabel="$X$ (g L$^{-1}$)")

axes[0,1].plot( Pred_Exp2[:,0], Pred_Exp2[:,2], color="blue" )
axes[0,1].plot( MLE_Exp2[:,0], MLE_Exp2[:,2], color="magenta" )
axes[0,1].errorbar(Tmeas_Exp2[:len(Nmeas_Exp2)], Nmeas_Exp2, yerr=Nstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,1].set(xlabel="$t$ (h)")
axes[0,1].set(ylabel="$N$ (mg L$^{-1}$)")

axes[1,0].plot( Pred_Exp2[:,0], Pred_Exp2[:,3], color="blue" )
axes[1,0].plot( MLE_Exp2[:,0], MLE_Exp2[:,3], color="magenta" )
axes[1,0].errorbar(Tmeas_Exp2, qmeas_Exp2, yerr=qstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,0].set(xlabel="$t$ (h)")
axes[1,0].set(ylabel="$q$ (mg g$^{-1}$)")

axes[1,1].plot( Pred_Exp2[:,0], Pred_Exp2[:,4], color="blue" )
axes[1,1].plot( MLE_Exp2[:,0], MLE_Exp2[:,4], color="magenta" )
axes[1,1].errorbar(Tmeas_Exp2, fmeas_Exp2, yerr=fstd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[1,1].set(xlabel="$t$ (h)")
axes[1,1].set(ylabel="$f$ (mg g$^{-1}$)")

axes[0,2].plot( Pred_Exp2[:,0], Pred_Exp2[:,5], color="blue" )
axes[0,2].plot( MLE_Exp2[:,0], MLE_Exp2[:,5], color="magenta" )
axes[0,2].errorbar(Tmeas_Exp2, Ymeas_Exp2, yerr=Ystd_Exp2, marker='o', ms=4, c='r', ls='', capsize=2, label="data")
axes[0,2].set(xlabel="$t$ (h)")
axes[0,2].set(ylabel="$YII$ ($-$)")

[ fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data() ]
fig.tight_layout()


# In[36]:


chi2 = PE.chi2_test( 0.95 )
#print( chi2 )


# In[37]:


print( PE.mle )


# In[23]:


cov1  = PE.cov_linearized()
#print( cov1)

cint95 = PE.conf_interval( cov1, 0.95, "T" )
#print( cint95 )


# In[ ]:





import pymc
import cronos
import canon
import magnus
#import numpy as np

from magnus import ParEst

# DAG and model
DAG = pymc.FFGraph()
DAG.options.MAXTHREAD = 1

Tmin = pymc.FFVar( DAG, "Tmin" )
Tmax = pymc.FFVar( DAG, "Tmax" )
b    = pymc.FFVar( DAG, "b" )
c    = pymc.FFVar( DAG, "c" )

T    = pymc.FFVar( DAG, "T" )
Y    = pymc.pow( b * ( T - Tmin ) * ( 1 - pymc.exp( c * ( T - Tmax ) ) ), 2 )

# Experimental data - artificially split into two submodels for testing
Data = [
  ParEst.Experiment( [294], { 0: ParEst.Record( [0.25], 0.01 ) }, 0 ),
  ParEst.Experiment( [296], { 0: ParEst.Record( [0.56], 0.01 ) }, 0 ),
  ParEst.Experiment( [298], { 0: ParEst.Record( [0.61], 0.01 ) }, 0 ),
  ParEst.Experiment( [300], { 0: ParEst.Record( [0.79], 0.01 ) }, 0 ),
  ParEst.Experiment( [302], { 0: ParEst.Record( [0.94], 0.01 ) }, 0 ),
  ParEst.Experiment( [304], { 0: ParEst.Record( [1.04], 0.01 ) }, 0 ),
  ParEst.Experiment( [306], { 0: ParEst.Record( [1.16], 0.01 ) }, 0 ),
  ParEst.Experiment( [308], { 0: ParEst.Record( [1.23], 0.01 ) }, 1 ),
  ParEst.Experiment( [310], { 0: ParEst.Record( [1.36], 0.01 ) }, 1 ),
  ParEst.Experiment( [312], { 0: ParEst.Record( [1.32], 0.01 ) }, 1 ),
  ParEst.Experiment( [314], { 0: ParEst.Record( [1.36], 0.01 ) }, 1 ),
  ParEst.Experiment( [316], { 0: ParEst.Record( [1.34], 0.01 ) }, 1 ),
  ParEst.Experiment( [318], { 0: ParEst.Record( [0.96], 0.01 ) }, 1 ),
  ParEst.Experiment( [319], { 0: ParEst.Record( [0.83], 0.01 ) }, 1 ),
  ParEst.Experiment( [320], { 0: ParEst.Record( [0.16], 0.01 ) }, 1 )
]

# Parameter guess and bounds
P0  = [ 275, 320, 0.03, 0.3  ]
PLB = [ 255, 315, 0.01, 0.05 ]
PUB = [ 290, 330, 0.1,  1.0  ]

# Parameter estimation solver
PE = ParEst()
PE.options.DISPLEVEL = 1
PE.options.NLPSLV.DISPLEVEL   = 0;
PE.options.NLPSLV.GRADCHECK   = 0;
PE.options.NLPSLV.MAXTHREAD   = 0;
PE.options.NLPSLV.GRADMETH    = PE.options.NLPSLV.FSYM 

PE.set_dag( DAG )
PE.add_model( [Y], [T], 0 ) # submodel 0
PE.add_model( [Y], [T], 1 ) # submodel 1
PE.set_parameter( [Tmin,Tmax,b,c], PLB, PUB )
PE.set_data( Data )

PE.setup()
PE.mle_solve( 20 ) #P0 )
print( PE.mle )

chi2 = PE.chi2_test( 0.95 )
print( chi2 )

cov  = PE.cov_bootstrap( 100 )
print( cov )
print( PE.crsam )

cov  = PE.cov_linearized()
print( cov )

cint95 = PE.conf_interval( cov, 0.95, "T" )
print( cint95 )

cell95_Tmin_Tmax = PE.conf_ellipsoid( cov, 0, 1, 0.95, "F", 50 )
print( cell95_Tmin_Tmax )


import pymc
from magnus import NSFeas

# Create DAG
DAG = pymc.FFGraph()
DAG.options.MAXTHREAD = 1

d = DAG.add_vars( 2, "d" )
p = DAG.add_var( "p" )
y = p*d[0]**2 + d[1]
print( "y = ", y.str() )

# Instantiate nested sampler
NS = NSFeas()

NS.options.DISPLEVEL = 1
NS.options.FEASCRIT  = NS.options.VAR #CVAR
NS.options.FEASTHRES = 0.05
NS.options.NUMLIVE   = 500
NS.options.NUMPROP   = 16
NS.options.MAXITER   = 0

NS.set_dag( DAG )
NS.set_constraint( [ y-0.75, 0.20-y ] )
NS.set_control( d, [ -1, -1 ], [ 1, 1 ] )

# Nominal case
NS.set_parameter( [p], [1] );
NS.setup()
NS.sample()

print( NS.live_points )

# Probabilistic case
import numpy as np
np.random.seed( 0 )
psam = []
[ psam.append( [ np.random.normal( 1, np.sqrt(3) ) ] ) for i in range(0,100) ]
#print( psam )

NS.set_parameter( [p], psam );
NS.setup()
NS.sample()

print( NS.live_points )

# Custom model passed from python
def model( x ):
  d = x[0:2]
  p = x[2]
  return [ p*d[0]**2 + d[1] ]

OpY = pymc.FFCustom()
OpY.set_D_eval( model )
y = OpY( d+[p], 0 )
print( "y = ", y.str() )

NS.set_constraint( [ y-0.75, 0.20-y ] )
NS.setup()
NS.sample()

print( NS.live_points )

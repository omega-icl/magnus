import pymc
from magnus import ExpDes

DAG = pymc.FFGraph()
P = DAG.add_vars(2,"P")
X = DAG.add_vars(1,"X")
C = DAG.add_vars(1,"C")
Y = P[0] * pymc.exp( P[1] * X[0] ) + C[0]

# Sampled parameters - uniform Sobol' sampling
NPSAM = 64
Plb  = [ 1., -10.]
Pub  = [10.,   0.]

# Experimental control space
NXSAM = 50;
Xlb = [0.]
Xub = [0.5]

ED = ExpDes()

# Set solver options
##ED.options.CRITERION = ED.DOPT
#ED.options.RISK      = ED.options.NEUTRAL
#ED.options.CVARTHRES = 0.25
##ED.options.UNCREDUC  = 1e-3
#ED.options.DISPLEVEL = 1
#ED.options.MINLPSLV.DISPLEVEL = 1
#ED.options.MINLPSLV.MAXITER   = 100
#ED.options.MINLPSLV.NLPSLV.GRADCHECK = 0
#ED.options.MINLPSLV.NLPSLV.DISPLEVEL = 0
#ED.options.MINLPSLV.MIPSLV.DISPLEVEL = 0
#ED.options.MINLPSLV.NLPSLV.GRADMETH  = ED.options.MINLPSLV.NLPSLV.FAD;
#ED.options.NLPSLV.OPTIMTOL    = 1e-5
#ED.options.NLPSLV.MAXITER     = 250
#ED.options.NLPSLV.DISPLEVEL   = 1
#ED.options.NLPSLV.GRADCHECK   = 0
#ED.options.NLPSLV.GRADMETH    = ED.options.NLPSLV.FAD
#ED.options.NLPSLV.GRADLSEARCH = 0
#ED.options.NLPSLV.FCTPREC     = 1e-7

ED.set_dag( DAG )
ED.set_model( [Y] )#, [0.1] )
ED.set_control( X, Xlb, Xub )
ED.set_constant( C, [0.] )
ED.set_parameter( P, ED.uniform_sample( NPSAM, Plb, Pub ) )

# Run solver
ED.setup()
ED.sample_support( NXSAM )
ED.combined_solve( 5 );


# In[ ]:


#ED.setup( 0 )


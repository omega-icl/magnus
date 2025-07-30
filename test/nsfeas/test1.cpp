//#undef MC__FFFEAS_DEBUG
//#define MAGNUS__NSFEAS_SETUP_DEBUG
//#define MAGNUS__NSFEAS_SAMPLE_DEBUG

#include "nsfeas.hpp"

// Example 2 in Paulen et al (2020, https://doi.org/10.1016/j.ifacol.2020.12.555)
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define model

  mc::FFGraph DAG;  // DAG describing the model

  const unsigned NP = 2;       // Number of model parameters
  std::vector<mc::FFVar> P(NP);  // Parameters
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );
  std::vector<double> nomP{ 1., 1. };

  const unsigned NU = 1;       // Number of inputs
  std::vector<mc::FFVar> U(NU);  // Inputs
  for( unsigned int i=0; i<NU; i++ ) U[i].set( &DAG );

  const unsigned NY = 1;       // Number of outputs
  std::vector<mc::FFVar> Y(NY);  // Constraints
  Y[0] = P[0]*exp(P[1]*U[0]);

  const unsigned NT = 10;       // Number of records
  std::vector<double> dU( NT ), dY( NT );
  for( unsigned int k=0; k<NT; k++ ){
    dU[k] = 1e-1*k;
    DAG.eval( NY, &Y[0], &dY[k], NU, &U[0], &dU[k], NP, &P[0], &nomP[0] ); // nominal noisefree measurements at P=[1 1] 
  }
  
  const unsigned NG = 2*NT;       // Number of constraints
  const double errY = 1e0; // symmetric error bound
  std::vector<mc::FFVar> G(NG);  // Constraints
  for( unsigned int k=0; k<NT; k++ ){
    G[2*k]   =   P[0]*exp(P[1]*dU[k]) - dY[k] - errY;
    G[2*k+1] = - P[0]*exp(P[1]*dU[k]) + dY[k] - errY;
  }
  
  /////////////////////////////////////////////////////////////////////////
  // Define sampler

  // Experimental control space
  std::vector<double> PLB( { -1e1, -1e1 } );
  std::vector<double> PUB( {  1e1,  1e1 } );

  mc::NSFEAS NS;
  NS.options.FEASCRIT  = mc::NSFEAS::Options::VAR;//CVAR;
  NS.options.FEASTHRES = 0.0;
  NS.options.NUMLIVE   = 500;
  NS.options.NUMPROP   = 16;
  NS.options.MAXITER   = 0;

  NS.set_dag( DAG );
  NS.set_constraint( G );
  NS.set_control( P, PLB, PUB );
  //NS.set_parameter( P, PSAM );

  NS.setup();
  NS.sample();

  NS.stats.display();
  
  for( auto const& [lkh,pcon] : NS.live_points() ){
    std::cout << std::scientific << std::setprecision(5)
              << std::setw(15) << lkh;
    for( unsigned i=0; i<NP; ++i )
      std::cout << std::setw(15) << pcon[i];
    std::cout << std::endl;
  }
  
  return 0;
}

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
    //std::cout << k << ": " << dY[k];
    //dY[k] = std::round(dY[k]*1e2)*1e-2;
    //std::cout << "  " << dY[k] << std::endl;
  }

  const unsigned NG = NT;        // Number of constraints
  std::vector<mc::FFVar> G(NG);  // Non-negativitiy constraints
  for( unsigned int k=0; k<NT; k++ )
    G[k] = - P[0]*exp(P[1]*dU[k]);

  mc::FFVar LL = 0.; // Likelihood function
  using mc::sqr;
  for( unsigned int k=0; k<NT; k++ )
    LL += sqr( P[0]*exp(P[1]*dU[k]) - dY[k] );
  LL *= -0.5;

  /////////////////////////////////////////////////////////////////////////
  // Define sampler

  // Experimental control space
  std::vector<double> PLB( { -1e1, -1e1 } );
  std::vector<double> PUB( {  1e1,  1e1 } );
//  std::vector<double> PLB( { 0e0, 0e0 } );
//  std::vector<double> PUB( { 3e0, 3e0 } );

  mc::NSFEAS NS;
  NS.options.LKHCRIT  = mc::NSFEAS::Options::VAR;//CVAR;
  NS.options.LKHTHRES = 0.00;
  NS.options.LKHTOL   = 0.05;
  NS.options.NUMLIVE  = 512;
  NS.options.NUMPROP  = 32;
  NS.options.MAXITER  = 500;

  NS.set_dag( DAG );
  NS.set_constraint( G );
  NS.set_loglikelihood( LL );
  NS.set_control( P, PLB, PUB );

  NS.setup();
  NS.sample();

  NS.stats.display();
  
  for( auto const& [lkh,pcon] : NS.live_points() ){
    std::cout << std::scientific << std::setprecision(5)
              << std::setw(15) << lkh;
    for( unsigned i=0; i<NP; ++i )
      std::cout << std::setw(15) << std::get<0>(pcon)[i];
    std::cout << std::endl;
  }
  
  return 0;
}

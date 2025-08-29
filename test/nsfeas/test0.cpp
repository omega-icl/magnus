#define MC__FFFEAS_CHECK
//#undef MC__FFFEAS_DEBUG
//#define MAGNUS__NSFEAS_SETUP_DEBUG
//#define MAGNUS__NSFEAS_SAMPLE_DEBUG

#include "nsfeas.hpp"

// Illustrative exampe in Kusumo et al (2020, https://doi.org/10.1021/acs.iecr.9b05006)
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define model

  mc::FFGraph DAG;  // DAG describing the model

  const unsigned NX = 2;       // Number of controls
  std::vector<mc::FFVar> X = DAG.add_vars( NX, "d" );  // Controls

  const unsigned NP = 1;       // Number of uncertain parameters
  std::vector<mc::FFVar> P = DAG.add_vars( NP, "p" );  // Parameters

  const unsigned NC = 2;       // Number of constants
  std::vector<mc::FFVar> C = DAG.add_vars( NC, "c" );  // Constants

  const unsigned NG = 2;       // Number of constraints
  const double GL = 0.20, GU = 0.75;
  std::vector<mc::FFVar> G{ C[0] - ( P[0]*X[0]*X[0] + X[1] ),
                            ( P[0]*X[0]*X[0] + X[1] ) - C[1] };

  /////////////////////////////////////////////////////////////////////////
  // Define sampler

  // Sampled parameters - normally distributed
  unsigned const NSAM = 100;
  //arma::vec vecPSAM = arma::randn( NSAM, arma::distr_param( 0, 1 ) );
  arma::vec vecPSAM = arma::randn( NSAM, arma::distr_param( 1., std::sqrt(0.3) ) );
  std::list<std::vector<double>> PSAM;
  for( unsigned i=0; i<NSAM; ++i ) PSAM.push_back( { vecPSAM[i] } );

  // Experimental control space
  std::vector<double> XLB( { -1e0, -1e0 } );
  std::vector<double> XUB( {  1e0,  1e0 } );

  mc::NSFEAS NS;
  NS.options.FEASCRIT  = mc::NSFEAS::Options::VAR;//CVAR;
  NS.options.FEASTHRES = 0.1;
  NS.options.NUMLIVE   = 500;
  NS.options.NUMPROP   = 16;
  NS.options.MAXITER   = 0;

  NS.set_dag( DAG );
  NS.set_constraint( G );
  NS.set_constant( C, { GL, GU } );
  NS.set_control( X, XLB, XUB );
  NS.set_parameter( P, PSAM );//std::vector<double>({ 1. }) );

  NS.setup();
  NS.sample();// { GL, GU } );

  NS.stats.display();

  for( auto const& [feas,point] : NS.live_points() ){
    std::cout << std::scientific << std::setprecision(5)
              << std::setw(15) << feas
              << std::setw(15) << std::get<1>(point);
    for( unsigned i=0; i<NX; ++i )
      std::cout << std::setw(15) << std::get<0>(point)[i];
    std::cout << std::endl;
  }
 
  return 0;
}

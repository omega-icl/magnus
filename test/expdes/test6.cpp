//#define MAGNUS__EXPDES_SETUP_DEBUG
#define MAGNUS__EXPDES_SHOW_APPORTION
#define MC__FFDOECRIT_CHECK
#define MC__FFGRADDOECRIT_CHECK
#define MC__FFDOEEFF_CHECK
#define MC__FFGRADDOEEFF_CHECK
#define MC__FFBREFF_CHECK
#define MC__FFGRADBREFF_CHECK

#include "expdes.hpp"

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define model

  mc::FFGraph DAG;  // DAG describing the model

  const unsigned NX = 2;       // Number of experimental controls
  std::vector<mc::FFVar> X(NX);  // Controls
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );

  const unsigned NP = 3;       // Number of estimated parameters
  std::vector<mc::FFVar> P(NP);  // Parameters
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );

  const unsigned NY = 2;       // Number of measured outputs
  std::vector<mc::FFVar> Y(NY);  // Outputs
  Y[0] = P[0] + P[1] * X[0] + P[2] * X[1];
  Y[1] = P[0] + P[1] * exp( X[0] ) + P[2] * exp( X[1] );

  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  // Sampled parameters - uniform Sobol' sampling
  unsigned const NSAM = 128;
  std::vector<double> PLB( { 3e-1, 3e-1, 3e-1 } );
  std::vector<double> PUB( { 7e-1, 7e-1, 7e-1 } );

  // Experimental control space
  std::vector<double> XLB( { -1e0, -1e0 } );
  std::vector<double> XUB( {  1e0,  1e0 } );

  // Output variance
  std::vector<double> YVAR( { 1e-2, 1e-2 } );

  mc::EXPDES DOE;
  DOE.options.CRITERION = mc::EXPDES::ODIST;//DOPT;//BRISK;//
  DOE.options.RISK      = mc::EXPDES::Options::NEUTRAL;//AVERSE;//NEUTRAL;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 1;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-8;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.MIPSLV.INTEGRALITYFOCUS = 0;
  DOE.options.MINLPSLV.MIPSLV.INTFEASTOL = 1e-9;
  DOE.options.MINLPSLV.MIPSLV.OUTPUTFILE = "";//"test0b.lp";
  DOE.options.MINLPSLV.NLPSLV.GRADMETH  = DOE.options.MINLPSLV.NLPSLV.FSYM;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 1;
  DOE.options.NLPSLV.GRADMETH = DOE.options.NLPSLV.FSYM;//FAD;

  DOE.set_dag( DAG );
  DOE.set_model( Y, YVAR );
  DOE.set_controls( X, XLB, XUB );
  DOE.set_parameters( P, DOE.uniform_sample( NSAM, PLB, PUB ) );

  DOE.setup();
  DOE.sample_supports( 256 );//512 );
/*
  size_t i=0;
  for( auto const& vout : DOE.output_samples().front() )
    std::cout << i++ << "  " << vout.t();
*/

  //DOE.combined_solve( 5 );
  DOE.effort_solve( 6 );//, false );
  DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 5, DOE.efforts() );
  //DOE.file_export( "test0b" );
  auto campaign = DOE.campaign();

  DOE.options.CRITERION = mc::EXPDES::ODIST;
  DOE.setup();
  DOE.evaluate_design( campaign, "ODIST" );

  DOE.options.CRITERION = mc::EXPDES::DOPT;
  DOE.options.RISK      = mc::EXPDES::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-NEUTRAL" );

  DOE.options.CRITERION = mc::EXPDES::DOPT;
  DOE.options.RISK      = mc::EXPDES::Options::AVERSE;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-AVERSE" );

  DOE.options.CRITERION = mc::EXPDES::BRISK;
  DOE.setup();
  DOE.evaluate_design( campaign, "BROPT" );

  return 0;
}

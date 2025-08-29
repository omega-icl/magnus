#define MAGNUS__EXPDES_SETUP_DEBUG
#define MAGNUS__EXPDES_SHOW_APPORTION
#define MC__FFFIMCRIT_CHECK
#define MC__FFGRADFIMCRIT_CHECK
#define MC__FFDOEEFF_CHECK
#define MC__FFGRADDOEEFF_CHECK
#define MC__FFBREFF_CHECK
#define MC__FFGRADBREFF_CHECK
#define MC__FFBRCRIT_CHECK
#define MC__FFGRADBRCRIT_CHECK

#include "expdes.hpp"

// ILLUSTRATIVE EXAMPLE IN KUSUMO ET AL (2023) REACTION CHEMISTRY & ENGINEERING
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

  const unsigned NP = 6;       // Number of estimated parameters
  std::vector<mc::FFVar> P(NP);  // Parameters
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );

  const unsigned NY = 1;       // Number of measured outputs
  std::vector<mc::FFVar> Y(NY);  // Outputs
  Y[0] = P[0] + P[1]*X[0] + P[2]*X[1] + P[3]*X[0]*X[1] + P[4]*X[0]*X[0] + P[5]*X[1]*X[1];

  const unsigned NC = 2;       // Number of constants
  std::vector<mc::FFVar> C(NC);  // constants
  for( unsigned int i=0; i<NC; i++ ) C[i].set( &DAG );

  const unsigned NG = 2;       // Number of output constraints
  std::vector<mc::FFVar> G(NG);  // Outputs
  G[0] = C[0] - Y[0]; // lower bound
  G[1] = Y[0] - C[1]; // upper bound

  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  // Sampled parameters - gaussian Sobol' sampling
  unsigned const NSAM = 128;
  std::vector<double> PM( { 2e0, 1e0, 1e0, 1e0, 2e0, 2e0 } );
  std::vector<double> PV( 6, 0.05 );

  // Experimental control space
  std::vector<double> XLB( { -1e0, -1e0 } );
  std::vector<double> XUB( {  1e0,  1e0 } );

  // Output variance
  std::vector<double> YVAR( { 1e0 } );

  mc::EXPDES DOE;
  DOE.options.CRITERION = mc::EXPDES::DOPT;//BRISK;//
  DOE.options.RISK      = mc::EXPDES::Options::NEUTRAL;//AVERSE;//
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.MAXITER = 200;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-8;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 0;
  DOE.options.NLPSLV.GRADMETH = DOE.options.NLPSLV.FAD;//FSYM;

  DOE.set_dag( DAG );
  DOE.set_model( Y, YVAR );
  DOE.set_constraint( G );
  DOE.set_control( X, XLB, XUB );
  DOE.set_constant( C, { 1.85, 3. } );
  DOE.set_parameter( P, DOE.gaussian_sample( NSAM, PM, PV, true ) ); //false ) );
  //DOE.set_parameter( P, PM );

  //std::list<std::pair<double,std::vector<double>>> prior_campaign
  //{
  //  { 1, { 1e-1 } }
  //};
  //DOE.add_prior_campaign( prior_campaign );
/*
  std::map<size_t,double> const& effort
  {
//** EFFORT-BASED EXACT DESIGN: -1.10102e+01
//   SUPPORT #30: 1.00 x [ -3.09440e-02 3.50111e-01 ]
    { 30, 1 },
//   SUPPORT #51: 1.00 x [ 3.76960e-01 -5.40916e-01 ]
    { 51, 1 },
//   SUPPORT #52: 1.00 x [ -3.12500e-01 4.37500e-01 ]
    { 52, 1 },
//   SUPPORT #53: 1.00 x [ 4.23744e-01 -2.29443e-01 ]
    { 53, 1 },
//   SUPPORT #55: 1.00 x [ 3.18310e-01 -5.05739e-01 ]
    { 55, 1 },
//   SUPPORT #56: 1.00 x [ 1.31578e-01 5.87126e-02 ]
    { 56, 1 },
//   SUPPORT #60: 1.00 x [ -6.78601e-01 2.66991e-01 ]
    { 60, 1 },
//   SUPPORT #62: 1.00 x [ -3.81223e-01 3.01655e-01 ]
    { 62, 1 }
  };
*/
  DOE.setup();
  DOE.sample_support( 128 );
  for( auto ui : DOE.control_sample() )
    std::cout << arma::rowvec( ui.data(), ui.size(), false );
  //return 0;
  //DOE.gradient_solve( effort );
  //DOE.combined_solve( 8, false ); // continuous design
  //auto CNTEFF = DOE.efforts();
  //DOE.combined_solve( 8, true, CNTEFF ); // exact design
  DOE.combined_solve( 8 );
  //DOE.effort_solve( 8 );//, false );
  //return 0;
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 5, DOE.efforts() );
  //DOE.file_export( "test0" );

  auto campaign = DOE.campaign();

/*
  std::list<std::pair<double,std::vector<double>>> campaign // ** EFFORT-BASED EXACT DESIGN: 2.58167e+01
  {
//** EFFORT-BASED EXACT DESIGN: -1.10102e+01
//   SUPPORT #30: 1.00 x [ -3.09440e-02 3.50111e-01 ]
    { 1, { -3.09440e-02, 3.50111e-01 } },
//   SUPPORT #51: 1.00 x [ 3.76960e-01 -5.40916e-01 ]
    { 1, { 3.76960e-01, -5.40916e-01 } },
//   SUPPORT #52: 1.00 x [ -3.12500e-01 4.37500e-01 ]
    { 1, { -3.12500e-01, 4.37500e-01 } },
//   SUPPORT #53: 1.00 x [ 4.23744e-01 -2.29443e-01 ]
    { 1, { 4.23744e-01, -2.29443e-01 } },
//   SUPPORT #55: 1.00 x [ 3.18310e-01 -5.05739e-01 ]
    { 1, { 3.18310e-01, -5.05739e-01 } },
//   SUPPORT #56: 1.00 x [ 1.31578e-01 5.87126e-02 ]
    { 1, { 1.31578e-01, 5.87126e-02 } },
//   SUPPORT #60: 1.00 x [ -6.78601e-01 2.66991e-01 ]
    { 1, { -6.78601e-01, 2.66991e-01 } },
//   SUPPORT #62: 1.00 x [ -3.81223e-01 3.01655e-01 ]
    { 1, { -3.81223e-01, 3.01655e-01 } }
  };
*/
  DOE.options.CRITERION = mc::EXPDES::DOPT;
  DOE.options.RISK      = mc::EXPDES::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT-NEUTRAL" );

//  DOE.options.CRITERION = mc::EXPDES::DOPT;
//  DOE.options.RISK      = mc::EXPDES::Options::AVERSE;
//  DOE.setup();
//  DOE.evaluate_design( campaign, "DOPT-AVERSE" );

//  DOE.options.CRITERION = mc::EXPDES::BRISK;
//  DOE.setup();
//  DOE.evaluate_design( campaign, "BROPT" );
  
  return 0;
}

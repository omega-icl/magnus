#define SAVE_RESULTS		// <- Whether to save bounds to file
#define MAGNUS__EXPDES_SHOW_APPORTION

#include "expdes.hpp"

////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{
  /////////////////////////////////////////////////////////////////////////
  // Define model

  mc::FFGraph DAG;  // DAG describing the IVP-ODE

  const unsigned NS = 5;  // Time stages
  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( unsigned k=0; k<NS; k++ ) tk[k+1] = tk[k] + 2e0; // [hour]

  const unsigned NP = 4;  // Number of estimated parameters
  const unsigned NU = 2;  // Number of experimental controls
  const unsigned NX = 2;  // Number of states
  const unsigned NY = 2;  // Number of outputs

  std::vector<mc::FFVar> U( NU );  // Controls
  for( unsigned int i=0; i<NU; i++ ) U[i].set( &DAG );
  U[0].set("u1");
  U[1].set("u2");
  mc::FFVar& u1 = U[0];
  mc::FFVar& u2 = U[1];

  std::vector<mc::FFVar> P( NP );  // Parameters
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );
  P[0].set("p1");
  P[1].set("p2");
  P[2].set("p3");
  P[3].set("p4");
  mc::FFVar& p1 = P[0]; // max growth rate
  mc::FFVar& p2 = P[1]; // half-saturation constant
  mc::FFVar& p3 = P[2]; // proudct yield
  mc::FFVar& p4 = P[3]; // respiration rate

  std::vector<mc::FFVar> X( NX );  // States & state sensitivities
  for( unsigned int i=0; i<NX; i++ ) X[i].set( &DAG );
  X[0].set("y1");
  X[1].set("y2");
  mc::FFVar& y1  = X[0]; // biomass
  mc::FFVar& y2  = X[1]; // substrate

  std::vector<mc::FFVar> RHS( NX );  // Right-hand side function
  mc::FFVar r = p1 * y2 / ( p2 + y2 );
  RHS[0]   = ( r - u1 - p4 ) * y1;
  RHS[1] = -r * y1 / p3 + u1 * ( u2 - y2 );
  
  std::vector<mc::FFVar> IC( NX );   // Initial value function
  IC[0] = 7e0;
  IC[1] = 1e-1;

  std::vector<std::map<size_t,mc::FFVar>> FCT(NS+1);  // State functions
  for( unsigned s=0; s<NS; s++ ) FCT[1+s] = { { NY*s, y1 }, { NY*s+1, y2 } };

  //std::vector<std::vector<mc::FFVar>> FCT( NS+1, std::vector<mc::FFVar>( NY*(NS+1), 0. ) );  // State functions
  //for( unsigned i=0; i<NS; i++ ){
  //  FCT[i+1][NY*i]   = y1;
  //  FCT[i+1][NY*i+1] = y2;
  //}

  mc::ODESLVS_CVODES IVP;
  IVP.options.INTMETH   = mc::BASE_CVODES::Options::MSBDF;//MSADAMS;//
  IVP.options.NLINSOL   = mc::BASE_CVODES::Options::NEWTON;//FIXEDPOINT;//
  IVP.options.LINSOL    = mc::BASE_CVODES::Options::DIAG;//DENSE;//
  IVP.options.FSACORR   = mc::BASE_CVODES::Options::STAGGERED;//STAGGERED1;//SIMULTANEOUS;
  IVP.options.NMAX      = 2000;
  IVP.options.DISPLAY   = 0;
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-11;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-10;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS  = 1;
  IVP.options.ASACHKPT  = 2000;
#if defined( SAVE_RESULTS )
  IVP.options.RESRECORD = 100;
#endif

  IVP.set_dag( &DAG );
  IVP.set_time( tk );
  IVP.set_state( X );
  IVP.set_constant( P );
  IVP.set_parameter( U );
  IVP.set_differential( RHS );
  IVP.set_initial( IC );
  IVP.set_function( FCT );
  IVP.setup();

  mc::FFODE OpODE;
  std::vector<mc::FFVar> Y(NY*NS);
  for( unsigned int j=0; j<NY*NS; j++ ) Y[j] = OpODE( j, NU, U.data(), NP, P.data(), &IVP );//, mc::FFODE::SHALLOW );
  //std::cout << DAG;

  /////////////////////////////////////////////////////////////////////////
  // Simulate model
/*
  // Nominal control and model parameters
  std::vector<double> dU{ 5.000000e-02, 8.915310e+00 };
  std::vector<double> dP{ 1e-01, 1e-01, 1e-01, 1e-01 };
  std::vector<double> dY( NY*NS );
  DAG.eval( NY*NS, Y.data(), dY.data(), NU, U.data(), dU.data(), NP, P.data(), dP.data() );
  for( unsigned i=0, k=0; i<NS; i++ )
    for( unsigned j=0; j<NY; j++, k++ )
      std::cout << "Y[" << i << "][" << j << "] = " << dY[k] << std::endl;
*/
  /////////////////////////////////////////////////////////////////////////
  // Define EXPDES

  // Nominal parameter values
  std::vector<double> PNOM{ 0.31, 0.18, 0.55, 0.05 };

  // Experimental control space
  std::vector<double> ULB{ 5e-2, 5e0 }, UUB{ 2e-1, 35e0 };

  // Output variance
  std::vector<double> YVAR( NY*NS, 4e-2 );

  mc::EXPDES DOE;
  DOE.options.CRITERION = mc::EXPDES::DOPT;//ODIST;
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK  = 0;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL  = 0;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL   = 1e-5;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL  = 0;
  DOE.options.MINLPSLV.MIPSLV.INTFEASTOL = 1e-9;
  DOE.options.NLPSLV.DISPLEVEL   = 1;
  DOE.options.NLPSLV.GRADCHECK   = 0;
  DOE.options.NLPSLV.GRADMETH    = DOE.options.NLPSLV.FSYM;//FAD;
  DOE.options.NLPSLV.GRADLSEARCH = 1;
  DOE.options.NLPSLV.FCTPREC     = 1e-7;
  DOE.options.NLPSLV.OPTIMTOL    = 1e-6;
  DOE.set_dag( DAG );
  DOE.set_model( Y, YVAR );
  DOE.set_control( U, ULB, UUB );
  DOE.set_parameter( P, PNOM );
  DOE.setup();
  DOE.sample_support( 128 );
  DOE.combined_solve( 4 );
//  DOE.effort_solve( 4 );
  //DOE.gradient_solve( DOE.effort(), true );
  //DOE.effort_solve( 4, DOE.effort() );
//  DOE.file_export( "test2" );
  auto campaign = DOE.campaign();

  DOE.options.CRITERION = mc::EXPDES::ODIST;
  DOE.setup();
  DOE.evaluate_design( campaign, "ODIST" );

  DOE.options.CRITERION = mc::EXPDES::DOPT;
  DOE.options.RISK      = mc::EXPDES::Options::NEUTRAL;
  DOE.setup();
  DOE.evaluate_design( campaign, "DOPT" );

  return 0;
}

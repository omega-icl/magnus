#include "parest.hpp"

// The problem of calibrating a dynamic model to describe dissolved oxygen concentration in a flask after inoculating biomass
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{

  mc::FFGraph DAG;  // DAG describing the model
  DAG.options.MAXTHREAD = 1;

  /////////////////////////////////////////////////////////////////////////
  // Define model

  const unsigned NS = 40;  // Time stages
  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( unsigned k=0; k<NS; k++ ) tk[k+1] = tk[k] + 1e1/(NS); // [day]

  const unsigned NP = 3;  // Number of estimated parameters
  const unsigned NC = 4;  // Number of model constants
  const unsigned NX = 3;  // Number of states
  const unsigned NY = 1;  // Number of outputs

  std::vector<mc::FFVar> P( NP );  // Parameters
  mc::FFVar& mu   = P[0]; P[0].set( &DAG, "mu"   ); // max growth rate
  mc::FFVar& beta = P[1]; P[1].set( &DAG, "beta" ); // decay rate
  mc::FFVar& B0   = P[2]; P[2].set( &DAG, "X(0)" ); // initial biomass

  std::vector<mc::FFVar> C( NC );  // Constants
  mc::FFVar& f  = C[0]; C[0].set( &DAG, "f"  ); // biodegradable fraction
  mc::FFVar& y  = C[1]; C[1].set( &DAG, "Y"  ); // yield
  mc::FFVar& S0 = C[2]; C[2].set( &DAG, "S0" ); // initial substrate
  mc::FFVar& D0 = C[3]; C[3].set( &DAG, "D0" ); // initial dissolved oxygen

  std::vector<mc::FFVar> X( NX );  // States & state sensitivities
  mc::FFVar& B = X[0]; X[0].set( &DAG, "X"); // biomass concentration
  mc::FFVar& S = X[1]; X[1].set( &DAG, "S"); // substrate concentration
  mc::FFVar& D = X[2]; X[2].set( &DAG, "D"); // dissolved oxygen concentration

  std::vector<mc::FFVar> RHS( NX );  // Right-hand side function
  RHS[0] = mu * S * B - beta * B;
  RHS[1] = - mu / y * S * B + f * beta * B;
  RHS[2] = - (1-y)/y * mu * S * B;

  std::vector<mc::FFVar> IC( NX );   // Initial value function
  IC[0] = B0;
  IC[1] = S0;
  IC[2] = D0;

  std::vector<std::map<size_t,mc::FFVar>> FCT(NS+1);  // State functions
  for( unsigned s=0; s<=NS; s++ ) FCT[s] = { { s, D } }; // 1 record in time stage s is for output s

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
  IVP.set_constant( C );
  IVP.set_parameter( P );
  IVP.set_differential( RHS );
  IVP.set_initial( IC );
  IVP.set_function( FCT );
  IVP.setup();

  mc::FFODE OpODE;
  std::vector<mc::FFVar> Y(NS+1);
  for( unsigned int j=0; j<NS+1; j++ ) Y[j] = OpODE( j, NP, P.data(), NC, C.data(), &IVP );
  //std::cout << DAG;

  /////////////////////////////////////////////////////////////////////////
  // Define experiment

  std::vector<mc::PAREST::Experiment> Data
  { { {}, { {  0, { { 10.0 }, 1e-2 } },
            {  1, { { 9.80 }, 1e-2 } },
            {  2, { { 9.70 }, 1e-2 } },
            {  3, { { 9.30 }, 1e-2 } },
            {  4, { { 8.80 }, 1e-2 } },
            {  5, { { 8.55 }, 1e-2 } },
            {  6, { { 8.50 }, 1e-2 } },
            {  7, { { 8.35 }, 1e-2 } },
            {  8, { { 8.40 }, 1e-2 } },
            {  9, { { 8.30 }, 1e-2 } },
            { 10, { { 8.20 }, 1e-2 } },
            { 11, { { 8.20 }, 1e-2 } },
            { 12, { { 8.05 }, 1e-2 } },
            { 13, { { 8.10 }, 1e-2 } },
            { 14, { { 8.00 }, 1e-2 } },
            { 15, { { 7.90 }, 1e-2 } },
            { 16, { { 8.00 }, 1e-2 } },
            { 17, { { 7.90 }, 1e-2 } },
            { 18, { { 7.85 }, 1e-2 } },
            { 19, { { 7.80 }, 1e-2 } },
            { 20, { { 7.80 }, 1e-2 } },
            { 21, { { 7.75 }, 1e-2 } },
            { 22, { { 7.70 }, 1e-2 } },
            { 23, { { 7.55 }, 1e-2 } },
            { 24, { { 7.60 }, 1e-2 } },
            { 25, { { 7.60 }, 1e-2 } },
            { 26, { { 7.50 }, 1e-2 } },
            { 27, { { 7.55 }, 1e-2 } },
            { 28, { { 7.50 }, 1e-2 } },
            { 29, { { 7.45 }, 1e-2 } },
            { 30, { { 7.50 }, 1e-2 } },
            { 31, { { 7.40 }, 1e-2 } },
            { 32, { { 7.35 }, 1e-2 } },
            { 33, { { 7.30 }, 1e-2 } },
            { 34, { { 7.35 }, 1e-2 } },
            { 35, { { 7.30 }, 1e-2 } },
            { 36, { { 7.30 }, 1e-2 } },
            { 37, { { 7.25 }, 1e-2 } },
            { 38, { { 7.30 }, 1e-2 } },
            { 39, { { 7.20 }, 1e-2 } },
            { 40, { { 7.20 }, 1e-2 } } } } };

  /////////////////////////////////////////////////////////////////////////
  // Perform MLE calculation

  std::vector<double> C0{ 0.9e0, 0.67e0, 4e0, 10e0 };
  std::vector<double> P0{ 1e0, 4e-1, 1e-3 };
  std::vector<double> PLB{ 0e0, 0e0, 1e-4 };
  std::vector<double> PUB{ 4e0, 1e0, 1e0  };

  // Parameter estimation solver
  mc::PAREST PE;
  PE.options.DISPLEVEL = 1;
  PE.options.NLPSLV.GRADLSEARCH = 0;
  PE.options.NLPSLV.FCTPREC     = 1e-7;
  PE.options.NLPSLV.DISPLEVEL   = 0;
  PE.options.NLPSLV.GRADCHECK   = 0;
  PE.options.NLPSLV.MAXTHREAD   = 0;
  //PE.options.NLPSLV.GRADMETH = PE.options.NLPSLV.FSYM;//FAD;

  PE.set_dag( DAG );
  PE.add_model( Y );
  PE.set_constant( C );
  PE.set_parameter( P, PLB, PUB );
  PE.set_data( Data );

  PE.setup();
  PE.mle_solve( P0, C0 );
  //PE.mle_solve( 10, C0 );
  auto MLEOPT   = PE.mle();
  auto CHI2TEST = PE.chi2_test( 0.95 );
  auto BCOV     = PE.cov_bootstrap( 200 );
  auto LCOV     = PE.cov_linearized();
  auto CINTT    = PE.conf_interval( LCOV, 0.95, "T" );
  auto CELLF    = PE.conf_ellipsoid( LCOV, 0, 1, 0.95, "F" );

  return 0;
}


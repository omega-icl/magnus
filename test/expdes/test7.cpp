#undef MAGNUS__EXPDES_SHOW_APPORTION
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
  DAG.options.MAXTHREAD = 0; // Use all available CPU cores for DAG evaluation
  
  // Time stages
  size_t const NS = 1;
  std::vector<double> tk( NS+1 );
  tk[0] = 0.;
  for( size_t k=0; k<NS; k++ ) tk[k+1] = tk[k] + 1e3;

  // Experimental controls
  const unsigned NU = 6;
  std::vector<mc::FFVar> U(NU);
  for( unsigned int i=0; i<NU; i++ ) U[i].set( &DAG );
  mc::FFVar& CA0 = U[0];
  mc::FFVar& CA0_CD0 = U[1];
  mc::FFVar& q0 = U[2];
  mc::FFVar& V1 = U[3];
  mc::FFVar& V2 = U[4];
  mc::FFVar& T0 = U[5];

  // Model parameters (fixed)
  const unsigned NP = 8;
  std::vector<mc::FFVar> P(NP);
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );
  mc::FFVar& k0_1 = P[0];
  mc::FFVar& k0_2 = P[1];
  mc::FFVar& Ea_1 = P[2];
  mc::FFVar& Ea_2 = P[3];
  mc::FFVar& DH1 = P[4];
  mc::FFVar& DH2 = P[5];
  mc::FFVar& cp = P[6];
  mc::FFVar& rho = P[7];

  // Model states
  const unsigned NX = 10;
  std::vector<mc::FFVar> X( NX );
  for( size_t i=0; i<NX; i++ ) X[i].set( &DAG );
  mc::FFVar& CA1 = X[0];
  mc::FFVar& CB1 = X[1];
  mc::FFVar& CC1 = X[2];
  mc::FFVar& CD1 = X[3];
  mc::FFVar& T1  = X[4];
  mc::FFVar& CA2 = X[5];
  mc::FFVar& CB2 = X[6];
  mc::FFVar& CC2 = X[7];
  mc::FFVar& CD2 = X[8];
  mc::FFVar& T2  = X[9];

  // Fixed inlet concentrations
  mc::FFVar CB0( 0.0 );
  mc::FFVar CC0( 0.0 );

  // Right-hand side function
  std::vector<std::vector<mc::FFVar>> RHS( NS, std::vector<mc::FFVar>(NX) );
  for( size_t i=0; i<NS; i++ ){
    mc::FFVar CD0 = CA0 / CA0_CD0;

    // Reactor 1
    mc::FFVar k11 = k0_1 * exp ( - Ea_1 / ( T1 + 273.15 ) );
    mc::FFVar k21 = k0_2 * exp ( - Ea_2 / ( T1 + 273.15 ) );
    mc::FFVar R11 = k11 * CA1 * CD1;
    mc::FFVar R21 = k21 * CB1;
    RHS[i][0] = q0 * ( CA0 - CA1 ) + V1 * ( -R11 );
    RHS[i][1] = q0 * ( CB0 - CB1 ) + V1 * ( R11 - R21 );
    RHS[i][2] = q0 * ( CC0 - CC1 ) + V1 * ( R21 );
    RHS[i][3] = q0 * ( CD0 - CD1 ) + V1 * ( -R11 );
    RHS[i][4] = q0 * rho * cp * ( T0 - T1 ) + V1 * ( -R11 * DH1 - R21 * DH2 );

    // Reactor 2
    mc::FFVar k12 = k0_1 * exp ( - Ea_1 / ( T2 + 273.15 ) );
    mc::FFVar k22 = k0_2 * exp ( - Ea_2 / ( T2 + 273.15 ) );
    mc::FFVar R12 = k12 * CA2 * CD2;
    mc::FFVar R22 = k22 * CB2;
    RHS[i][5] = q0 * ( CA1 - CA2 ) + V2 * ( -R12 );
    RHS[i][6] = q0 * ( CB1 - CB2 ) + V2 * ( R12 - R22 );
    RHS[i][7] = q0 * ( CC1 - CC2 ) + V2 * ( R22 );
    RHS[i][8] = q0 * ( CD1 - CD2 ) + V2 * ( -R12 );
    RHS[i][9] = q0 * rho * cp * ( T1 - T2 ) + V2 * ( -R12 * DH1 - R22 * DH2 );
  }

  // Initial value function
  std::vector<mc::FFVar> IC( NX );
  IC[0] = CA0;
  IC[1] = CB0;
  IC[2] = CC0;
  IC[3] = CA0 / CA0_CD0;
  IC[4] = T0;
  IC[5] = CA0;
  IC[6] = CB0;
  IC[7] = CC0;
  IC[8] = CA0 / CA0_CD0;
  IC[9] = T0;

  // State functions
  const unsigned NY = 3;
  std::vector<std::vector<mc::FFVar>> FCT( NS, std::vector<mc::FFVar>( NY*NS, 0. ) );  // State functions
  for( unsigned i=0; i<NS; i++ ){
    FCT[i][NY*i]   = CA2 / ( CA2 + CB2 + CC2 + CD2 );
    FCT[i][NY*i+1] = CC2 / ( CA2 + CB2 + CC2 + CD2 );
    FCT[i][NY*i+2] = T2;
  }

  // Define IVP
  mc::ODESLVS_CVODES IVP;
  IVP.options.INTMETH   = mc::BASE_CVODES::Options::MSBDF;//MSADAMS;//
  IVP.options.NLINSOL   = mc::BASE_CVODES::Options::NEWTON;//FIXEDPOINT;//
  IVP.options.LINSOL    = mc::BASE_CVODES::Options::DIAG;//DENSE;//
  IVP.options.FSACORR   = mc::BASE_CVODES::Options::STAGGERED;//STAGGERED1;//SIMULTANEOUS;
  IVP.options.NMAX      = 2000;
  IVP.options.DISPLAY   = 0;
  IVP.options.ATOL      = IVP.options.ATOLB     = IVP.options.ATOLS  = 1e-9;
  IVP.options.RTOL      = IVP.options.RTOLB     = IVP.options.RTOLS  = 1e-9;
  IVP.options.FSAERR    = IVP.options.QERR      = IVP.options.QERRS     = 1;
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

  // #2 	0.800 	0.800 	0.036 	1.743 	1.826 	22.0 	0.0018 	0.020 	67.9
  std::vector<double> dU{ 0.8, 0.8, 0.036, 1.743, 1.826, 22.0 };
  // #1 	0.800 	0.800 	0.025 	0.729 	2.000 	22.0 	0.0020 	0.020 	67.9
  //std::vector<double> dU{ 0.8, 0.8, 0.025, 0.729, 2.0, 22.0 };
  std::vector<double> dP{ 2800, 12, 2995, 4427, -80, 0, 1.7, 0.8 };
  //IVP.solve_state( dU, dP );
  
  //std::ofstream direcSTA;
  //direcSTA.open( "test7_STA.dat", std::ios_base::out );
  //IVP.record( direcSTA );

  //return 0;

  mc::FFODE OpODE;
  std::vector<mc::FFVar> Y(NY*NS);
  for( unsigned int j=0; j<NY*NS; j++ ) Y[j] = OpODE( j, NU, U.data(), NP, P.data(), &IVP );//, mc::FFODE::SHALLOW );
  std::cout << DAG;

  /////////////////////////////////////////////////////////////////////////
  // Simulate model

  // Nominal control and model parameters
  std::vector<double> dY( NY*NS );
  DAG.eval( Y, dY, U, dU, P, dP );
  for( unsigned i=0, k=0; i<NS; i++ )
    for( unsigned j=0; j<NY; j++, k++ )
      std::cout << "Y[" << i << "][" << j << "] = " << dY[k] << std::endl;

  //return 0;

  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  size_t const NEXP = 8;
  
  // Experimental control space
  size_t NSAM = 256;//5096;
  std::vector<double> ULB( { 0.8, 0.8, 0.0083, 0.5, 0.5, 22 } );
  std::vector<double> UUB( { 1.1, 0.835, 0.08, 2, 2, 35 } );

  mc::EXPDES DOE;
  DOE.options.CRITERION = mc::EXPDES::ODIST;
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
  DOE.options.NLPSLV.GRADCHECK = 0;
  DOE.options.NLPSLV.GRADMETH = DOE.options.NLPSLV.FSYM;//FAD;

  DOE.set_dag( DAG );
  DOE.set_model( Y ); // YVAR
  DOE.set_controls( U, ULB, UUB );
  DOE.set_parameters( P, dP );

  DOE.setup();
  DOE.sample_supports( NSAM );
  //DOE.file_export( "test7_5096" );
  //return 0;

  // identify min/max output range for rescaling
  double const* pY = DOE.output_samples().front()[0].memptr();
  std::vector<double> YLB( pY, pY+NS*NY ), YUB = YLB;
  for( size_t i=1; i<DOE.output_samples().front().size(); ++i ){
    pY = DOE.output_samples().front()[i].memptr();
    for( size_t k=0; k<NS*NY; ++k ){
      if( pY[k] < YLB[k] ) YLB[k] = pY[k];
      if( pY[k] > YUB[k] ) YUB[k] = pY[k];
    }
  }
  std::vector<double> YVAR( NS*NY );
  for( size_t k=0; k<NS*NY; ++k ){
    YVAR[k] = mc::sqr( YUB[k]-YLB[k] );
    std::cout << "  " << std::scientific << std::setprecision(4) << std::setw(10) << YLB[k]
              << "  " << std::scientific << std::setprecision(4) << std::setw(10) << YUB[k]
              << "  " << std::scientific << std::setprecision(4) << std::setw(10) << YVAR[k]
              << std::endl;
  }
  DOE.set_model( Y, YVAR ); // to give output variables about the same weight
  DOE.setup();
  
  //DOE.sample_supports( NSAM );
  //DOE.combined_solve( NEXP );
  DOE.effort_solve( NEXP );//, false );
  DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 5, DOE.efforts() );
  //DOE.file_export( "test7" );

  //auto campaign = DOE.campaign();
  //DOE.options.CRITERION = mc::EXPDES::ODIST;
  //DOE.setup();
  //DOE.evaluate_design( campaign, "ODIST" );

  return 0;
}

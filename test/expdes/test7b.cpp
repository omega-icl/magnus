#undef MC__NLPSLV_SNOPT_DEBUG
#undef MAGNUS__EXPDES_SHOW_APPORTION
#undef MAGNUS__EXPDES_EVAL_DEBUG
#define MC__FFODISTEFF_CHECK
#define MC__FFGRADODISTEFF_CHECK
#define MC__FFODISTCRIT_CHECK
#define MC__FFGRADODISTCRIT_CHECK
#undef MC__FFODISTCRIT_DEBUG


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
  mc::FFVar& CA0 = U[0];     CA0.set("CA0");
  mc::FFVar& CA0_CD0 = U[1]; CA0_CD0.set("CA0_CD0");
  mc::FFVar& q0 = U[2];      q0.set("q0");
  mc::FFVar& V1 = U[3];      V1.set("vol1");
  mc::FFVar& V2 = U[4];      V2.set("vol2");
  mc::FFVar& T0 = U[5];      T0.set("T0");

  // Model parameters (fixed)
  const unsigned NP = 8;
  std::vector<mc::FFVar> P(NP);
  for( unsigned int i=0; i<NP; i++ ) P[i].set( &DAG );
  mc::FFVar& k0_1 = P[0]; k0_1.set("k0_1");
  mc::FFVar& k0_2 = P[1]; k0_1.set("k0_2");
  mc::FFVar& Ea_1 = P[2]; k0_1.set("Ea_1");
  mc::FFVar& Ea_2 = P[3]; k0_1.set("Ea_2");
  mc::FFVar& DH1 = P[4];  DH1.set("DH1");
  mc::FFVar& DH2 = P[5];  DH2.set("DH2");
  mc::FFVar& cp = P[6];   cp.set("cp");
  mc::FFVar& rho = P[7];  rho.set("rho");

  // Model states
  const unsigned NX = 10;
  std::vector<mc::FFVar> X( NX );
  for( size_t i=0; i<NX; i++ ) X[i].set( &DAG );
  mc::FFVar& CA1 = X[0]; CA1.set("CA1");
  mc::FFVar& CB1 = X[1]; CB1.set("CB1");
  mc::FFVar& CC1 = X[2]; CC1.set("CC1");
  mc::FFVar& CD1 = X[3]; CD1.set("CD1");
  mc::FFVar& T1  = X[4]; T1.set("T1");
  mc::FFVar& CA2 = X[5]; CA2.set("CA2");
  mc::FFVar& CB2 = X[6]; CB2.set("CB2");
  mc::FFVar& CC2 = X[7]; CC2.set("CC2");
  mc::FFVar& CD2 = X[8]; CD2.set("CD2");
  mc::FFVar& T2  = X[9]; T2.set("T2");

  // Fixed inlet concentrations
  mc::FFVar CB0( 0.0 );
  mc::FFVar CC0( 0.0 );

  // Right-hand side function
  std::vector<std::vector<mc::FFVar>> RHS( NS, std::vector<mc::FFVar>(NX) );
  for( size_t i=0; i<NS; i++ ){
    mc::FFVar CD0 = CA0 / CA0_CD0; CD0.set("CD0");

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
  std::vector<std::map<size_t,mc::FFVar>> FCT(NS+1);  // State functions
  FCT[NS] = { { 0, CA2 / ( CA2 + CB2 + CC2 + CD2 ) },
              { 1, CC2 / ( CA2 + CB2 + CC2 + CD2 ) },
              { 2, T2 } };

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

  std::vector<double> dU{ 9.03125e-01, 8.25156e-01, 5.98344e-02, 1.10938e+00, 5.46875e-01, 3.13438e+01 };
//  std::vector<double> dU{ 8.18750e-01, 8.32812e-01, 4.86313e-02, 9.68750e-01, 1.53125e+00, 2.44375e+01 };
//  std::vector<double> dU{ 9.94087e-01, 8.32946e-01, 5.43341e-02, 1.20523e+00, 1.19121e+00, 2.85314e+01 };
//  std::vector<double> dU{ 9.96875e-01, 8.22969e-01, 1.05406e-02, 1.01562e+00, 1.01562e+00, 2.40312e+01 };

  // #2 	0.800 	0.800 	0.036 	1.743 	1.826 	22.0 	0.0018 	0.020 	67.9
//  std::vector<double> dU{ 0.8, 0.8, 0.036, 1.743, 1.826, 22.0 };
  // #1 	0.800 	0.800 	0.025 	0.729 	2.000 	22.0 	0.0020 	0.020 	67.9
  //std::vector<double> dU{ 0.8, 0.8, 0.025, 0.729, 2.0, 22.0 };
  std::vector<double> dP{ 2800, 12, 2995, 4427, -80, 0, 1.7, 0.8 };
  //IVP.solve_state( dU, dP );

  //std::ofstream direcSTA;
  //direcSTA.open( "test7_STA.dat", std::ios_base::out );
  //IVP.record( direcSTA );

  //return 0;

  mc::FFODE OpODE;
  std::vector<mc::FFVar> Y(NY);
  for( unsigned int j=0; j<NY; j++ ) Y[j] = OpODE( j, NU, U.data(), NP, P.data(), &IVP );//, mc::FFODE::SHALLOW );
  //std::cout << DAG;

  // Model constraints
  const unsigned NG = 3;
  std::vector<mc::FFVar> G(NG);
  G[0] = Y[0] - 2e-2;//1;
  G[1] = Y[1] - 2e-3;//2;
  G[2] = Y[2] - 85;
//  const unsigned NG = 1;
//  std::vector<mc::FFVar> G(NG);
//  G[0] = Y[2] - 85;

  /////////////////////////////////////////////////////////////////////////
  // Simulate model

  // Nominal control and model parameters
  std::vector<double> dY( NY );
  DAG.eval( Y, dY, U, dU, P, dP );
  for( unsigned j=0; j<NY; j++ )
    std::cout << "Y[" << j << "] = " << dY[j] << std::endl;

  std::vector<double> dG( NG );
  DAG.eval( G, dG, U, dU, P, dP );
  for( unsigned j=0; j<NG; j++ )
    std::cout << "G[" << j << "] = " << dG[j] << std::endl;

  //return 0;

  /////////////////////////////////////////////////////////////////////////
  // Define MBDOE

  size_t const NEXP = 8;

  mc::EXPDES DOE;
  DOE.options.CRITERION = mc::EXPDES::ODIST;
  DOE.options.RISK      = mc::EXPDES::Options::NEUTRAL;//AVERSE;
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 0;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-8;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.MIPSLV.INTEGRALITYFOCUS = 0;
  DOE.options.MINLPSLV.MIPSLV.INTFEASTOL = 1e-9;
  DOE.options.MINLPSLV.MIPSLV.OUTPUTFILE = "";//"test0b.lp";
  DOE.options.MINLPSLV.NLPSLV.GRADMETH  = DOE.options.MINLPSLV.NLPSLV.FSYM;
  DOE.options.NLPSLV.DISPLEVEL   = 1;
  DOE.options.NLPSLV.GRADCHECK   = 0;
  DOE.options.NLPSLV.GRADLSEARCH = 1;
  DOE.options.NLPSLV.FCTPREC     = 1e-7;
  DOE.options.NLPSLV.GRADMETH = DOE.options.NLPSLV.FSYM;//FAD;

  DOE.set_dag( DAG );
  DOE.set_model( Y ); // YVAR
  
  // Experimental control space
  std::vector<double> ULB( { 0.8, 0.8, 0.0083, 0.5, 0.5, 22 } );
  std::vector<double> UUB( { 1.1, 0.835, 0.08, 2, 2, 35 } );
  DOE.set_control( U, ULB, UUB );

  // Parametric uncertainty
  size_t const NPSAM = 64;//128;
  std::vector<double> PLB( NP ), PUB( NP );
  for( size_t i=0; i<NP; ++i ){
    //PLB[i] = PUB[i] = dP[i];
    PLB[i] = dP[i]*0.95e0;  PUB[i] = dP[i]*1.05e0;
  }
  DOE.set_parameter( P, dP );
  //DOE.set_parameter( P, DOE.uniform_sample( NPSAM, PLB, PUB ) );

//  DOE.set_constraint( G );
//  DOE.options.FEASPROP = 128;
//  DOE.setup();
//  DOE.sample_support( 8192 );
//  DOE.file_export( "test7b_8192" );
//  //return 0;

//  // identify min/max output range for rescaling
//  double const* pY = DOE.output_sample().front()[0].memptr();
//  std::vector<double> YLB( pY, pY+NY ), YUB = YLB;
//  for( size_t i=1; i<DOE.output_sample().front().size(); ++i ){
//    pY = DOE.output_sample().front()[i].memptr();
//    for( size_t k=0; k<NY; ++k ){
//      if( pY[k] < YLB[k] ) YLB[k] = pY[k];
//      if( pY[k] > YUB[k] ) YUB[k] = pY[k];
//    }
//  }
//  std::vector<double> YVAR( NY );
//  for( size_t k=0; k<NY; ++k ){
//    YVAR[k] = mc::sqr( YUB[k]-YLB[k] );
//    std::cout << "  " << std::scientific << std::setprecision(4) << std::setw(10) << YLB[k]
//              << "  " << std::scientific << std::setprecision(4) << std::setw(10) << YUB[k]
//              << "  " << std::scientific << std::setprecision(4) << std::setw(10) << YVAR[k]
//              << std::endl;
//  }
//  return 0;
  std::vector<double> YVAR{ 1e-5, 1e-07, 1e2 };
  //std::vector<double> YVAR{ 2.7342e-02, 5.9986e-04, 1.5588e+03};
  
  DOE.set_model( Y, YVAR ); // to give output variables about the same weight
  DOE.set_constraint( G );
  DOE.setup();

  size_t NUSAM = 64;
  DOE.sample_support( NUSAM );
  //DOE.combined_solve( NEXP );
  DOE.effort_solve( NEXP );//, false );
  DOE.gradient_solve( DOE.effort(), std::vector<double>(), true );
  //DOE.effort_solve( 5, true, DOE.effort() );
  DOE.file_export( "test7b_N="+std::to_string(NEXP)+"_"+std::to_string(NUSAM) );
  DOE.stats.display();
  
  auto campaign = DOE.campaign();
  DOE.options.CRITERION = mc::EXPDES::ODIST;
  //DOE.setup();
  DOE.evaluate_design( campaign, "ODIST" );
 
  return 0;
}

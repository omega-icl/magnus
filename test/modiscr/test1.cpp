#define MAGNUS__MODISCR_SETUP_DEBUG
#define MAGNUS__EXPDES_SHOW_APPORTION
#define MC__FFBREFF_CHECK
#define MC__FFGRADBREFF_CHECK
#undef MC__NLPSLV_SNOPT_DEBUG_CALLBACK
#undef MC__FFBREFF_DEBUG

#include "modiscr.hpp"

// The problem of finding an optimal design to discriminate between a cubic polynomial model
// and a linear model was considered by Dette & Titoff (https://doi.org/10.1214/08-AOS635)
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{

  mc::FFGraph DAG;  // DAG describing the model

  size_t const NX = 1;       // Number of experimental controls
  std::vector<mc::FFVar> X(NX);  // Controls
  X[0].set( &DAG, "T" );

  size_t const NY = 1;       // Number of measured outputs

  /////////////////////////////////////////////////////////////////////////
  // Define model #1: Linear

  size_t const NP1 = 2;  // Number of model parameters
  std::vector<mc::FFVar> P1(NP1);  // Parameters
  P1[0].set( &DAG, "a0" );
  P1[1].set( &DAG, "a1" );

  std::vector<mc::FFVar> Y1(NY);  // Outputs
  Y1[0] = P1[0] + P1[1] * X[0];

  /////////////////////////////////////////////////////////////////////////
  // Define model #2: Quadratic

  size_t const NP2 = 4;  // Number of estimated parameters
  std::vector<mc::FFVar> P2(NP2);  // Parameters
  P2[0].set( &DAG, "a0" );
  P2[1].set( &DAG, "a1" );
  P2[2].set( &DAG, "a2" );
  P2[3].set( &DAG, "a3" );

  std::vector<mc::FFVar> Y2(NY);  // Outputs
  Y2[0] = P2[0] + P2[1]*X[0] + P2[2]*pow(X[0],2) + P2[3]*pow(X[0],3);

  /////////////////////////////////////////////////////////////////////////
  // Perform Initial Experiment

  std::list<std::pair<double,std::vector<double>>> prior_campaign
  {
    { 1, { -1e0  } },
    //{ 1, { -7.5e-1 } },
    { 1, { -5e-1 } },
    //{ 1, { -2.5e-1 } },
    { 1, { 0e0  } },
    //{ 1, { 2.5e-1 } },
    { 1, { 5e-1 } },
    //{ 1, { 7.5e-1 } },
    { 1, { 1e0  } }
  };

  std::vector<double> YVAR( { 1e-2 } ); // Output variance
  std::list<std::pair<std::vector<double>,std::map<size_t,std::vector<double>>>> prior_meas;

  std::vector<double> DY2(NY);
  std::vector<double> DP2NOM = { 1e0, 1e0, 0e0, 1e0 };
  std::cout << std::scientific << std::setprecision(5) << std::right;
  std::cout << "SYNTHETIC DATA:" << std::endl;
  size_t er = 0;
  for( auto const& [e,DX] : prior_campaign ){
    for( size_t r=0; r<std::round(e); ++r, ++er ){
      for( size_t k=0; k<NX; ++k )
        std::cout << std::setw(12) << DX[k];
      DAG.eval( Y2, DY2, X, DX, P2, DP2NOM );
      std::cout << " | ";
      for( size_t k=0; k<NY; ++k ){
        prior_meas.push_back( { DX, { { k, { DY2[k] } } } } );
        std::cout << std::setw(12) << DY2[k];
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  //return 0;

  /////////////////////////////////////////////////////////////////////////
  // Boostrapped MLE calculations for Models #1 & #2

  // List of parameter samples from joint confidence region
  std::list<std::vector<double>> PSAM;

  // Local optimization solver
#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT PE1;
#else
  mc::NLPSLV_IPOPT PE1;
#endif
  PE1.options.DISPLEVEL = 0;
  PE1.options.MAXITER   = 40;
  PE1.options.FEASTOL   = 1e-6;
  PE1.options.OPTIMTOL  = 1e-6;
  PE1.options.GRADMETH  = mc::NLPSLV_SNOPT::Options::FSYM;
  PE1.options.GRADCHECK = false;
  PE1.options.MAXTHREAD = 0;

  std::vector<double> DP1 = { 0e0, 1e0 };
  PE1.set_dag( &DAG );
  PE1.set_var( P1 );

#ifdef MC__USE_SNOPT
  mc::NLPSLV_SNOPT PE2;
#else
  mc::NLPSLV_IPOPT PE2;
#endif
  PE2.options = PE1.options;

  std::vector<double> DP2 = { 0e0, 0e0, 0e0, 1e0 };
  PE2.set_dag( &DAG );
  PE2.set_var( P2 );

  mc::FFMLE OpMLE;

  size_t const NSAM = 128;//512;
  for( size_t isam=0; isam<NSAM; ++isam ){

    // Add measurement noise
    auto prior_meas_noise = prior_meas;
    arma::vec YM( NY, arma::fill::zeros );
    arma::mat YC( NY, NY, arma::fill::zeros ); YC.diag() = arma::vec( YVAR );
    for( auto& [DX,MAPY] : prior_meas_noise ){
      arma::mat YNOISE = arma::mvnrnd( YM, YC );
      for( auto& [k,VECYk] : MAPY )
        for( size_t k=0; k<NY; ++k )
          VECYk[0] += YNOISE(k);
    }
 
    mc::FFVar FMLE1 = OpMLE( P1.data(), &DAG, &P1, &X, &Y1, &YVAR, &prior_meas_noise );
    PE1.set_obj( mc::BASE_OPT::MIN, FMLE1 );
    PE1.setup();
    PE1.solve( DP1.data() );
    //std::cout << "PARAMETER ESTIMATION SOLUTION:\n" << PE1.solution();
    PSAM.push_back( PE1.solution().x );

    mc::FFVar FMLE2 = OpMLE( P2.data(), &DAG, &P2, &X, &Y2, &YVAR, &prior_meas_noise );
    PE2.set_obj( mc::BASE_OPT::MIN, FMLE2 );
    PE2.setup();
    PE2.solve( DP2.data() );
    //std::cout << "PARAMETER ESTIMATION SOLUTION:\n" << PE2.solution();
    PSAM.back().insert( PSAM.back().end(), PE2.solution().x.cbegin(), PE2.solution().x.cend() );
    //PSAM.back().insert( PSAM.back().end(), DP2NOM.cbegin(), DP2NOM.cend() );

    std::cout << std::scientific << std::setprecision(5);
    std::cout << isam << "  ";
    auto DP = arma::vec(PSAM.back());
    DP.t().raw_print( std::cout );
  }

  /////////////////////////////////////////////////////////////////////////
  // Define discrimination problem

  // Joint model parameters
  std::vector<mc::FFVar> P = P1;
  P.insert( P.end(), P2.cbegin(), P2.cend() );

  // Experimental control space
  std::vector<double> XLB( { -1e0 } );
  std::vector<double> XUB( {  1e0 } );

  // Discrimination solver
  mc::MODISCR DOE;
  DOE.options.CRITERION = mc::MODISCR::BRISK;
  DOE.options.RISK      = mc::MODISCR::Options::AVERSE;//NEUTRAL;//
  DOE.options.CVARTHRES = 0.95;
  DOE.options.DISPLEVEL = 1;
  DOE.options.MINLPSLV.DISPLEVEL = 1;
  DOE.options.MINLPSLV.NLPSLV.GRADCHECK = 1;
  DOE.options.MINLPSLV.NLPSLV.OPTIMTOL  = 1e-8;
  DOE.options.MINLPSLV.NLPSLV.DISPLEVEL = 0;
  DOE.options.MINLPSLV.MIPSLV.DISPLEVEL = 0;
  DOE.options.NLPSLV.DISPLEVEL = 1;
  DOE.options.NLPSLV.GRADCHECK = 1;
  //DOE.options.NLPSLV.GRADMETH = DOE.options.NLPSLV.FSYM;//FAD;

  DOE.set_dag( DAG );
  DOE.set_models( { Y1, Y2 }, YVAR );
  DOE.set_controls( X, XLB, XUB );
  DOE.set_parameters( P, PSAM );

  //DOE.add_prior_campaign( prior_campaign );

  DOE.setup();
  DOE.sample_supports( 64 );
  //DOE.file_export( "test0" );
  //DOE.combined_solve( 5, false ); // continuous design
  //auto CNTEFF = DOE.efforts();
  //DOE.combined_solve( 5, true, CNTEFF ); // exact design
  DOE.combined_solve( 8 );
  //DOE.effort_solve( 5 );//, false );
  //DOE.gradient_solve( DOE.efforts(), true );
  //DOE.effort_solve( 5, DOE.efforts() );
  //DOE.file_export( "test0" );
  auto campaign = DOE.campaign();

  DOE.options.CRITERION = mc::MODISCR::BRISK;
  DOE.setup();
  DOE.evaluate_design( campaign, "BRISK" );
  
  return 0;
}

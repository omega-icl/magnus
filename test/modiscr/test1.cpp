#define MAGNUS__MODISCR_SETUP_DEBUG
#define MAGNUS__EXPDES_SHOW_APPORTION
#define MC__FFBREFF_CHECK
#define MC__FFGRADBREFF_CHECK
#undef MC__NLPSLV_SNOPT_DEBUG_CALLBACK
#undef MC__FFBREFF_DEBUG

#include "parest.hpp"
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
  // Perform in-silico experiment from model #2

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
  std::vector<mc::PAREST::Experiment> prior_meas;

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
        prior_meas.push_back(
          mc::PAREST::Experiment( DX, { { k, mc::PAREST::Record( { DY2[k] }, YVAR[0] ) } } )
        );
        std::cout << std::setw(12) << DY2[k];
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  //return 0;

  /////////////////////////////////////////////////////////////////////////
  // Boostrapping for models #1 & #2 around in-silico experiment

  size_t const NSAM = 128;//512;
  
  // Parameter estimation in model #1
  mc::PAREST PE1;
  PE1.options.RNGSEED = 123;
  PE1.options.NLPSLV.DISPLEVEL = 0;
  PE1.options.NLPSLV.MAXITER   = 40;
  PE1.options.NLPSLV.FEASTOL   = 1e-6;
  PE1.options.NLPSLV.OPTIMTOL  = 1e-6;
  PE1.options.NLPSLV.GRADMETH  = mc::NLPSLV_SNOPT::Options::FSYM;
  PE1.options.NLPSLV.GRADCHECK = false;
  PE1.options.NLPSLV.MAXTHREAD = 0;

  PE1.set_dag( DAG );
  PE1.add_model( Y1, X );
  PE1.set_data( prior_meas );
  PE1.set_parameter( P1 );
  PE1.setup();

  std::vector<double> DP1 = { 0e0, 1e0 };
  PE1.mle_solve( DP1 );
  PE1.cov_bootstrap( prior_meas, NSAM );

  // Parameter estimation in model #2
  mc::PAREST PE2;
  PE2.options = PE1.options;

  PE2.set_dag( DAG );
  PE2.add_model( Y2, X );
  PE2.set_data( prior_meas );
  PE2.set_parameter( P2 );
  PE2.setup();

  std::vector<double> DP2 = { 0e0, 0e0, 0e0, 1e0 };
  PE2.mle_solve( DP2 );
  PE2.cov_bootstrap( prior_meas, NSAM );

  // List of parameter samples from joint confidence region
  std::list<std::vector<double>> PSAM;
  arma::mat const& CRSAM1 = PE1.crsam().t();
  arma::mat const& CRSAM2 = PE2.crsam().t();
  for( size_t isam=0; isam<NSAM; ++isam ){
    std::vector<double> rec;
    rec.insert( rec.end(), CRSAM1.colptr(isam), CRSAM1.colptr(isam)+NP1 );
    rec.insert( rec.end(), CRSAM2.colptr(isam), CRSAM2.colptr(isam)+NP2 );
    PSAM.push_back( rec );

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
  DOE.options.CVARTHRES = 0.05;
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
  DOE.set_model( { Y1, Y2 }, YVAR );
  DOE.set_control( X, XLB, XUB );
  DOE.set_parameter( P, PSAM );

  //DOE.add_prior_campaign( prior_campaign );

  DOE.setup();
  DOE.sample_support( 64 );
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

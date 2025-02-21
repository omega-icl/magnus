#define MAGNUS__MODISCR_SETUP_DEBUG
#define MAGNUS__EXPDES_SHOW_APPORTION
#define MC__FFBREFF_CHECK
#define MC__FFGRADBREFF_CHECK
#undef MC__NLPSLV_SNOPT_DEBUG_CALLBACK
#undef MC__FFBREFF_DEBUG
#undef MC__VEVAL_DEBUG


#include "parest.hpp"

// The problem of finding an optimal design to discriminate between a cubic polynomial model
// and a linear model was considered by Dette & Titoff (https://doi.org/10.1214/08-AOS635)
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{

  mc::FFGraph DAG;  // DAG describing the model
  DAG.options.MAXTHREAD = 1;

  size_t const NX = 1;       // Number of experimental controls
  std::vector<mc::FFVar> X(NX);  // Controls
  X[0].set( &DAG, "X" );

  size_t const NY = 1;       // Number of measured outputs

  /////////////////////////////////////////////////////////////////////////
  // Define model

  size_t const NP = 4;  // Number of estimated parameters
  std::vector<mc::FFVar> P(NP);  // Parameters
  P[0].set( &DAG, "a0" );
  P[1].set( &DAG, "a1" );
  P[2].set( &DAG, "a2" );
  P[3].set( &DAG, "a3" );

  std::vector<mc::FFVar> Y(NY);  // Outputs
  Y[0] = P[0] + P[1]*X[0] + P[2]*pow(X[0],2) + P[3]*pow(X[0],3);

  /////////////////////////////////////////////////////////////////////////
  // Perform simulated experiment

  std::list<std::pair<double,std::vector<double>>> Campaign
  {
    { 1, { -1e0  } },
    { 1, { -9e-1 } },
    { 1, { -8e-1 } },
    { 1, { -7e-1 } },
    { 1, { -6e-1 } },
    { 1, { -5e-1 } },
    { 1, { -4e-1 } },
    { 1, { -3e-1 } },
    { 1, { -2e-1 } },
    { 1, { -1e-1 } },
    { 1, {  0e0  } },
    { 1, {  1e-1 } },
    { 1, {  2e-1 } },
    { 1, {  3e-1 } },
    { 1, {  4e-1 } },
    { 1, {  5e-1 } },
    { 1, {  6e-1 } },
    { 1, {  7e-1 } },
    { 1, {  8e-1 } },
    { 1, {  9e-1 } },
    { 1, {  1e0  } }
  };

  std::vector<double> DPNOM = { 1e0, 1e0, 0e0, 1e0 };
  std::vector<double> YVAR( { 1e-2 } ); // Output variance
  std::vector<mc::PAREST::Experiment> Data;

  std::vector<double> DY(NY);
  arma::vec YM( NY, arma::fill::zeros );
  arma::mat YC( NY, NY, arma::fill::zeros ); YC.diag() = arma::vec( YVAR );

  std::cout << std::scientific << std::setprecision(5) << std::right;
  std::cout << "SYNTHETIC DATA:" << std::endl;
  size_t er = 0;
  for( auto const& [e,DX] : Campaign ){
    DAG.eval( Y, DY, X, DX, P, DPNOM );
    mc::PAREST::Experiment exp( DX );
    for( size_t k=0; k<NX; ++k )
      std::cout << std::setw(12) << DX[k];
    for( size_t r=0; r<std::round(e); ++r, ++er ){
      std::cout << " | ";
      arma::mat YNOISE = arma::mvnrnd( YM, YC );
      for( size_t k=0; k<NY; ++k ){
        if( exp.output.count( k ) )
          exp.output[k].measurement.push_back( DY[k] + YNOISE(k) );
        else
          exp.output[k].measurement = { DY[k] + YNOISE(k) };
        std::cout << std::setw(12) << DY[k] + YNOISE(k);
      }
      std::cout << std::endl;
    }
    for( size_t k=0; k<NY; ++k )
      exp.output[k].variance = YVAR[k];
    Data.push_back( exp );
  }
  std::cout << std::endl;

  //return 0;

  /////////////////////////////////////////////////////////////////////////
  // Perform MLE calculation

  std::vector<double> PLB = { -1e1, -1e1, 1e-1, -1e1 };
  //std::vector<double> PLB = { -1e1, -1e1, -1e1, -1e1 };
  std::vector<double> PUB = {  1e1,  1e1,  1e1,  1e1 };

  // Parameter estimation solver
  mc::PAREST PE;
  PE.options.NLPSLV.DISPLEVEL = 0;
  PE.options.NLPSLV.GRADCHECK = 0;
  PE.options.NLPSLV.MAXTHREAD = 0;
  //PE.options.NLPSLV.GRADMETH = PE.options.NLPSLV.FSYM;//FAD;

  PE.set_dag( DAG );
  PE.add_model( Y, X );
  PE.set_data( Data );
  PE.set_parameter( P, PLB, PUB );
  PE.add_constraint( P[3], mc::BASE_OPT::EQ, P[1] );

  PE.setup();
  PE.mle_solve( DPNOM );
  //PE.mle_solve( 10 );
  auto MLEOPT   = PE.mle();
  auto CHI2TEST = PE.chi2_test( 0.95 );
  auto BCOV     = PE.cov_bootstrap( 100 );
  auto LCOV     = PE.cov_linearized();
  auto CINTT    = PE.conf_interval( LCOV, 0.95, "T" );
  auto CELLF    = PE.conf_ellipsoid( LCOV, 0, 1, 0.95, "F" );

  return 0;
}


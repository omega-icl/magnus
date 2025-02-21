#include "parest.hpp"

// The problem of calibrating a temperature-dependent growth rate model
// based on Lobry et al (1991, Binary 3:86, http://pbil.univ-lyon1.fr/members/lobry/articles/Binary_1991_3_86.pdf)
////////////////////////////////////////////////////////////////////////
int main()
////////////////////////////////////////////////////////////////////////
{

  mc::FFGraph DAG;  // DAG describing the model
  DAG.options.MAXTHREAD = 1;

  /////////////////////////////////////////////////////////////////////////
  // Define model: Ratkowsky

  const unsigned NP = 4;  // Number of estimated parameters
  const unsigned NU = 1;  // Number of experimental controls
  const unsigned NY = 1;  // Number of outputs

  std::vector<mc::FFVar> P( NP );  // Parameters
  mc::FFVar& Tmin = P[0]; P[0].set( &DAG, "Tmin" );
  mc::FFVar& Tmax = P[1]; P[1].set( &DAG, "Tmax" );
  mc::FFVar& b    = P[2]; P[2].set( &DAG, "b" );
  mc::FFVar& c    = P[3]; P[3].set( &DAG, "c" );

  std::vector<mc::FFVar> U( NU );  // Parameters
  mc::FFVar& T = U[0]; U[0].set( &DAG, "T" );

  std::vector<mc::FFVar> Y( NY );  // Outputs
  using mc::sqr;
  //Y[0] = max( sqr( b * ( T - Tmin ) * ( 1. - exp( c * ( T - Tmax ) ) ) ), 0. );
  Y[0] = sqr( b * ( T - Tmin ) * ( 1. - exp( c * ( T - Tmax ) ) ) );

  /////////////////////////////////////////////////////////////////////////
  // Define experiment

  std::vector<mc::PAREST::Experiment> Data
  {
    { {294}, { { 0, { {0.25}, 0.01 } } } },
    { {296}, { { 0, { {0.56}, 0.01 } } } },
    { {298}, { { 0, { {0.61}, 0.01 } } } },
    { {300}, { { 0, { {0.79}, 0.01 } } } },
    { {302}, { { 0, { {0.94}, 0.01 } } } },
    { {304}, { { 0, { {1.04}, 0.01 } } } },
    { {306}, { { 0, { {1.16}, 0.01 } } } },
    { {308}, { { 0, { {1.23}, 0.01 } } } },
    { {310}, { { 0, { {1.36}, 0.01 } } } },
    { {312}, { { 0, { {1.32}, 0.01 } } } },
    { {314}, { { 0, { {1.36}, 0.01 } } } },
    { {316}, { { 0, { {1.34}, 0.01 } } } },
    { {318}, { { 0, { {0.96}, 0.01 } } } },
    { {319}, { { 0, { {0.83}, 0.01 } } } },
    { {320}, { { 0, { {0.16}, 0.01 } } } },
  };

  /////////////////////////////////////////////////////////////////////////
  // Perform MLE calculation

  std::vector<double> P0 { 275, 320, 0.03, 0.3  };
  std::vector<double> PLB{ 255, 315, 0.01, 0.05 };
  std::vector<double> PUB{ 290, 330, 0.1,  1    };

  // Parameter estimation solver
  mc::PAREST PE;
  PE.options.DISPLEVEL = 1;
  PE.options.NLPSLV.DISPLEVEL   = 0;
  PE.options.NLPSLV.GRADCHECK   = 0;
  PE.options.NLPSLV.MAXTHREAD   = 0;
  //PE.options.NLPSLV.GRADMETH = PE.options.NLPSLV.FSYM;//FAD;

  PE.set_dag( DAG );
  PE.add_model( Y, U );
  PE.set_parameter( P, PLB, PUB );
  PE.set_data( Data );

  PE.setup();
  PE.mle_solve( P0 );
  //PE.mle_solve( 10 );
  auto MLEOPT   = PE.mle();
  auto CHI2TEST = PE.chi2_test( 0.95 );
  auto BCOV     = PE.cov_bootstrap( 100 );
  auto LCOV     = PE.cov_linearized();
  auto CINTT    = PE.conf_interval( LCOV, 0.95, "T" );
  auto CINTZ    = PE.conf_interval( LCOV, 0.95, "Z" );
  auto CELLF    = PE.conf_ellipsoid( LCOV, 0, 1, 0.95, "F" );

  return 0;
}


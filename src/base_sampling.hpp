// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MAGNUS__BASE_SAMPLING_HPP
#define MAGNUS__BASE_SAMPLING_HPP

#include <assert.h>
#include <iterator>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions/normal.hpp>

#include <armadillo>

namespace mc
{
//! @brief C++ base class for sampling techniques
////////////////////////////////////////////////////////////////////////
//! mc::BASE_SAMPLING is a C++ base class for sampling techniques
////////////////////////////////////////////////////////////////////////
class BASE_SAMPLING
{
public:

  //! @brief Class constructor
  BASE_SAMPLING()
    {}

  //! @brief Class destructor
  virtual ~BASE_SAMPLING()
    {}

  //! @brief Set uniform sample within bounds
  static std::list<std::vector<double>> uniform_sample
    ( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB,
      bool const sobol=true );

  //! @brief Set uniform sample within bounds
  template <typename T>
  static void uniform_sample
    ( T& TSAM, size_t NSAM, std::vector<double> const& LB,
      std::vector<double> const& UB, bool const sobol=true );

  //! @brief Set Gaussian sample with given mean and variance
  static std::list<std::vector<double>> gaussian_sample
    ( size_t NSAM, std::vector<double> const& M, std::vector<double> const& V,
      bool const sobol=true );

  //! @brief Set Gaussian sample with given mean and variance
  static std::list<std::vector<double>> gaussian_sample
    ( size_t NSAM, std::vector<double> const& M, double const& V,
      bool const sobol=true );

  //! @brief Set Gaussian sample with given mean and variance
  template <typename T>
  static void gaussian_sample
    ( T& TSAM, size_t NSAM, std::vector<double> const& M, std::vector<double> const& V,
      bool const sobol=true );

  //! @brief Compute the log of the sum of exponentials of log-likelihood map
  template <typename T>
  static double log_sum_exp
    ( std::multimap<double,T> const& points )
    {
      double maxlkh = points.rbegin()->first, sum = 0;
      for( auto const& [lkh,dum] : points )
        sum += std::exp( lkh - maxlkh );
      return maxlkh + std::log( sum );
    }

  //! @brief Compute the log of the sum of exponentials of two log-likelihood values
  static double log_sum_exp
    ( double const& logZ1, double const& logZ2 )
    {
      double maxlogZ = ( logZ1 > logZ2 ? logZ1 : logZ2 );
      //std::cout << std::scientific << std::setprecision(10);
      //std::cout << "log(Z1) = " << logZ1 << "  log(Z2) = " << logZ2 << std::endl;
      //std::cout << "log(Z1+Z2) = " << std::log( std::exp( logZ1 ) + std::exp( logZ2 ) )
      //          << " = " << maxlogZ + std::log( std::exp( logZ1 - maxlogZ ) + std::exp( logZ2 - maxlogZ ) ) << std::endl;
      return maxlogZ + std::log( std::exp( logZ1 - maxlogZ ) + std::exp( logZ2 - maxlogZ ) );
    }

private:

  //! @brief Private methods to block default compiler methods
  BASE_SAMPLING( BASE_SAMPLING const& ) = delete;
  BASE_SAMPLING& operator=( BASE_SAMPLING const& ) = delete;
};

inline std::list<std::vector<double>>
BASE_SAMPLING::uniform_sample
( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB,
  bool const sobol )
{
  std::list<std::vector<double>> LSAM;
  uniform_sample( LSAM, NSAM, LB, UB, sobol );
  return LSAM;
}

template <typename T>
inline void
BASE_SAMPLING::uniform_sample
( T& TSAM, size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB,
  bool const sobol )
{
  assert( NSAM && LB.size() && LB.size() == UB.size() );
  size_t NDIM = LB.size();

  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( NDIM );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( 0 );

  TSAM.clear();
  for( size_t s=0; s<NSAM; ++s ){
    TSAM.push_back( std::vector<double>( NDIM ) );
    for( size_t k=0; k<NDIM; k++ ){
      double const rnd = ( sobol? gen(): arma::randu() );
      TSAM.back()[k] = LB[k] + ( UB[k] - LB[k] ) * rnd;
    }
  }
}

inline std::list<std::vector<double>>
BASE_SAMPLING::gaussian_sample
( size_t NSAM, std::vector<double> const& M, double const& V,
  bool const sobol )
{
  std::list<std::vector<double>> LSAM;
  gaussian_sample( LSAM, NSAM, M, std::vector<double>( M.size(), V ), sobol );
  return LSAM;
}

inline std::list<std::vector<double>>
BASE_SAMPLING::gaussian_sample
( size_t NSAM, std::vector<double> const& M, std::vector<double> const& V,
  bool const sobol )
{
  std::list<std::vector<double>> LSAM;
  gaussian_sample( LSAM, NSAM, M, V, sobol );
  return LSAM;
}

template <typename T>
inline void
BASE_SAMPLING::gaussian_sample
( T& TSAM, size_t NSAM, std::vector<double> const& M, std::vector<double> const& V,
  bool const sobol )
{
  assert( NSAM && M.size() && M.size() == V.size() );
  size_t NDIM = M.size();
  TSAM.clear();

  arma::mat mX;

  if( sobol ){
    using boost::math::normal;
    using boost::math::quantile;
    typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
    typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
    sobol64 eng( NDIM );
    qrgen gen( eng, boost::uniform_01<double>() );
    gen.engine().seed( 0 );
    mX.resize( NDIM, NSAM );
    for( size_t s=0; s<NSAM; ++s )
      for( size_t d=0; d<NDIM; ++d )
        mX(d,s) = quantile( normal( M[d], std::sqrt(V[d]) ), gen() );
  }

  else{
    arma::vec vM( const_cast<double*>(M.data()), M.size(), false );
    arma::vec vV( const_cast<double*>(V.data()), V.size(), false );
    arma::mat mV = diagmat( vV );
    if( !mvnrnd( mX, vM, mV, NSAM ) )
      return;
  }
    
  for( size_t s=0; s<NSAM; ++s ){
    TSAM.push_back( std::vector<double>( mX.colptr(s), mX.colptr(s)+NDIM ) );
    std::cout << arma::rowvec( mX.colptr(s), NDIM, false );
  }
}

} // end namescape mc

#endif


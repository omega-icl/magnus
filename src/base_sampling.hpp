// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MAGNUS__BASE_SAMPLING_HPP
#define MAGNUS__BASE_SAMPLING_HPP

#include <assert.h>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
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
    ( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB );

  //! @brief Set uniform sample within bounds
  template <typename T>
  static void uniform_sample
    ( T& TSAM, size_t NSAM, std::vector<double> const& LB,
      std::vector<double> const& UB );

private:

  //! @brief Private methods to block default compiler methods
  BASE_SAMPLING( BASE_SAMPLING const& ) = delete;
  BASE_SAMPLING& operator=( BASE_SAMPLING const& ) = delete;
};

inline std::list<std::vector<double>>
BASE_SAMPLING::uniform_sample
( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB )
{
  std::list<std::vector<double>> LSAM;
  uniform_sample( LSAM, NSAM, LB, UB );
  return LSAM;
}

template <typename T>
inline void
BASE_SAMPLING::uniform_sample
( T& TSAM, size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB )
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
    for( size_t k=0; k<NDIM; k++ )
      TSAM.back()[k] = LB[k] + ( UB[k] - LB[k] ) * gen();
  }
}

} // end namescape mc

#endif


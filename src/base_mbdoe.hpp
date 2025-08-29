// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MAGNUS__BASE_MBDOE_HPP
#define MAGNUS__BASE_MBDOE_HPP

#undef  MAGNUS__DEBUG__BASE_MBDOE

#include <assert.h>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <armadillo>

#include "ffunc.hpp"
#include "base_mbfa.hpp"

namespace mc
{
//! @brief C++ base class for defining of model-based design of experiment problems
////////////////////////////////////////////////////////////////////////
//! mc::BASE_MBDOE is a C++ base class for defining the controls,
//! parameters and outputs participating in model-based design of
//! experiment (MBDoE) problems
////////////////////////////////////////////////////////////////////////
class BASE_MBDOE
: public virtual BASE_MBFA
{
protected:

  using BASE_MBFA::set_parameter;


  //! @brief Size of model candidates
  size_t _nm;

  //! @brief Size of model output
  size_t _ny;

  //! @brief Size of prior experiments
  size_t _ne0;

  //! @brief vector of model outputs
  std::vector<std::vector<FFVar>> _vOUT;

  //! @brief vector of model output variances
  std::vector<double> _vOUTVAR;

  //! @brief vector of model weights
  std::vector<double> _vOUTWEI;

  //! @brief matrix of model parameter scaling factors
  arma::mat _mPARSCA;

  //! @brief vector of prior experimental control values
  std::vector<std::vector<double>> _vCONAP;

  //! @brief vector of prior experimental effort values
  std::vector<double> _vEFFAP;

public:

  //! @brief Enumeration type for optimality criterion
  enum TYPE{
    AOPT=0,  //!< A-optimality
    DOPT,    //!< D-optimality
    EOPT,    //!< E-optimality
    BRISK,   //!< Bayes risk
    ODIST    //!< Output spread
  };

  //! @brief Class constructor
  BASE_MBDOE()
    : BASE_MBFA(),
      _nm(0), _ny(0), _ne0(0)
    {}

  //! @brief Class destructor
  virtual ~BASE_MBDOE()
    {}

  //! @brief Get size of model candidates
  size_t nm
    ()
    const
    { return _nm; }

  //! @brief Get size of model outputs
  size_t ny
    ()
    const
    { return _ny; }

  //! @brief Set model outputs for single model
  void set_model
    ( std::vector<FFVar> const& Y,
      std::vector<double> const& varY=std::vector<double>() )
    {
      assert( !Y.empty() );
      _nm = 1;
      _ny = Y.size();
      _vOUT.assign( { Y } );
      _vOUTWEI.assign( { 1. } );
      _vOUTVAR = varY;
    }

  //! @brief Set model outputs for multiple candidate models
  void set_model
    ( std::list<std::vector<FFVar>> const& l_Y,
      std::vector<double> const& varY=std::vector<double>() )
    {
      assert( !l_Y.empty() && !l_Y.front().empty() );
      _nm = l_Y.size();
      _ny = l_Y.front().size();

      _vOUT.clear();
      _vOUT.reserve( _nm );
      _vOUTWEI.clear();
      _vOUTWEI.reserve( _nm );
      for( auto const& Ym : l_Y ){
        assert( Ym.size() == _ny );
        _vOUT.push_back( Ym );
        _vOUTWEI.push_back( 1./(double)_nm );
      }

      _vOUTVAR = varY;
    }

  //! @brief Set model outputs for multiple candidate models
  void set_model
    ( std::list<std::pair<std::vector<FFVar>,double>> const& l_Y,
      std::vector<double> const& varY=std::vector<double>() )
    {
      assert( !l_Y.empty() && !l_Y.front().first.empty() );
      _nm = l_Y.size();
      _ny = l_Y.front().first.size();

      _vOUT.clear();
      _vOUT.reserve( _nm );
      double Wtot = 0.;
      for( auto const& [Ym,Wm] : l_Y ){
        assert( Ym.size() == _ny && Wm >= 0. );
        _vOUT.push_back( Ym );
        Wtot += Wm;
      }

      _vOUTWEI.clear();
      _vOUTWEI.reserve( _nm );
      for( auto const& [Ym,Wm] : l_Y )
        _vOUTWEI.push_back( Wm/Wtot );

      _vOUTVAR = varY;
    }

  //! @brief Set nominal model parameters and parameter scaling (matrix format)
  void set_parameter
    ( std::vector<FFVar> const& P, std::vector<double> const& valP,
      arma::mat const& scaP )
    {
      BASE_MBFA::set_parameter( P, valP );
      _mPARSCA = scaP;
    }

  //! @brief Set nominal model parameters and parameter scaling (vector format)
  void set_parameter
    ( std::vector<FFVar> const& P, std::vector<double> const& valP,
      std::vector<double> const& scaP=std::vector<double>() )
    {
      BASE_MBFA::set_parameter( P, valP );

      assert( scaP.empty() || scaP.size() == _np );
      _mPARSCA.reset();
      if( !scaP.empty() )
        _mPARSCA = arma::diagmat( arma::vec( scaP ) );
    }

  //! @brief Set list of model parameter scenarios and parameter scaling (matrix format)
  void set_parameter
    ( std::vector<FFVar> const& P, std::list<std::vector<double>> const& l_valP,
      arma::mat const& scaP )
    {
      BASE_MBFA::set_parameter( P, l_valP );
      _mPARSCA = scaP;
    }

  //! @brief Set list of model parameter scenaros and parameter scaling (vector format)
  void set_parameter
    ( std::vector<FFVar> const& P, std::list<std::vector<double>> const& l_valP,
      std::vector<double> const& scaP=std::vector<double>() )
    {
      BASE_MBFA::set_parameter( P, l_valP );

      assert( scaP.empty() || scaP.size() == _np );
      _mPARSCA.reset();
      if( !scaP.empty() )
        _mPARSCA = arma::diagmat( arma::vec( scaP ) );
    }

  //! @brief Set list of model parameters and parameter scaling
  void set_parameter
    ( std::vector<FFVar> const& P, std::list<std::pair<std::vector<double>,double>> const& l_valP,
      arma::mat const& scaP )
    {
      BASE_MBFA::set_parameter( P, l_valP );
      _mPARSCA = scaP;
    }

  //! @brief Set list of model parameters
  void set_parameter
    ( std::vector<FFVar> const& P, std::list<std::pair<std::vector<double>,double>> const& l_valP,
      std::vector<double> const& scaP=std::vector<double>() )
    {
      BASE_MBFA::set_parameter( P, l_valP );

      assert( scaP.empty() || scaP.size() == _np );
      _mPARSCA.reset();
      if( !scaP.empty() )
        _mPARSCA = arma::diagmat( arma::vec( scaP ) );
    }

  //! @brief Retreive parameter scaling
  arma::mat parameter_scaling
    ()
    const
    { 
      return _mPARSCA;
    }

  //! @brief Append prior experimental campaign
  void add_prior_campaign
    ( std::list<std::pair<double,std::vector<double>>> const& C )
    {
      for( auto const& [E0,C0] : C ){
        _vCONAP.push_back( C0 );
        _vEFFAP.push_back( E0 );
      }
      _ne0 = _vCONAP.size();
    }

  //! @brief Set prior experimental campaign
  void set_prior_campaign
    ( std::list<std::pair<double,std::vector<double>>> const& C )
    {
      reset_prior_campaign();
      add_prior_campaign( C );
    }

  //! @brief Set prior experimental campaign
  void reset_prior_campaign
    ()
    {
      _vCONAP.clear();
      _vEFFAP.clear();
      _ne0 = 0;
    }

  //! @brief Retreive prior experimental campaign
  std::list<std::pair<double,std::vector<double>>> prior_campaign
    ()
    const
    {
      assert( _vCONAP.size() == _vEFFAP.size() );
      std::list<std::pair<double,std::vector<double>>> C;
      auto iteff = _vEFFAP.cbegin();
      auto itsup = _vCONAP.cbegin();
      for( ; iteff != _vEFFAP.cend(); ++iteff, ++itsup )
        C.push_back( { *iteff, *itsup } );
      return C;
    }

  //! @brief Round fractional experimental efforts to nearest integer
  static void effort_rounding
    ( unsigned const n, unsigned const* typ, double* val );

  //! @brief Apportion fractional experimental efforts
  static void effort_apportion
    ( unsigned const n, unsigned const* typ, double* val );

private:

  //! @brief Private methods to block default compiler methods
  BASE_MBDOE
    ( BASE_MBDOE const& )
    = delete;
  BASE_MBDOE& operator=
    ( BASE_MBDOE const& )
    = delete;
  void set_loglikelihood
    ( FFVar const& LL )
    = delete;
  void reset_loglikelihood
    ()
    = delete;
  std::vector<FFVar> const& var_loglikelihood
    ()
    const
    = delete;
};

inline void
BASE_MBDOE::effort_apportion
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

#ifdef MAGNUS__EXPDES_SHOW_APPORTION
  std::cout << "Initial efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    if( val[i] > TOLZERO ) std::cout << "X0[" << i << "]: " << val[i] << std::endl;
  }
#endif

  double sum = 0.;
  unsigned supp = 0;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    sum += val[i];
    if( val[i] >= TOLZERO ) supp++;
  }
  sum = std::round( sum );

  std::vector<double> intval( n );
  unsigned iround = 0;
  for( double ratio=1.; ratio>0.1 && iround<100; ++iround ){
    double intsum = 0.;
    for( unsigned i=0; i<n; ++i ){
      if( !typ[i] ) continue;
      intval[i] = (val[i]<TOLZERO? 0: (supp<=sum? std::ceil( ratio*val[i] ): std::round( ratio*val[i] )));
      intsum += intval[i];
    }
    if( sum < intsum )
      ratio *= 0.9;
    else if( sum > intsum )
      ratio /= 0.95;
    else
      break;
  }

#ifdef MAGNUS__EXPDES_SHOW_APPORTION
  std::cout << "Apportioned efforts:" << std::endl;
#endif
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
#ifdef MAGNUS__EXPDES_SHOW_APPORTION
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
#endif
  }
}

inline void
BASE_MBDOE::effort_rounding
( unsigned const n, unsigned const* typ, double* val )
{
  double const TOLZERO = 1e-10;
  //double const TOLINT  = 1e-5;

#ifdef MAGNUS__EXPDES_SHOW_APPORTION
  std::cout << "Initial efforts:" << std::endl;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    if( val[i] > TOLZERO ) std::cout << "X0[" << i << "]: " << val[i] << std::endl;
  }
#endif

  double sum = 0.;
  unsigned supp = 0;
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    sum += val[i];
    if( val[i] >= TOLZERO ) supp++;
  }
  sum = std::round( sum );

  std::vector<double> intval( n );
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    intval[i] = std::ceil( (1.-supp/(2*sum)) * val[i] );
  }

  for( ; ; ){
    //std::cout << "Intermediate efforts:" << std::endl;
    double intsum = 0.;
    for( unsigned i=0; i<n; ++i ){
      if( !typ[i] ) continue;
      intsum += intval[i];
      //if( val[i] > TOLZERO ) std::cout << "X1[" << i << "]: " << intval[i] << std::endl;
    }
    if( std::fabs( sum - intsum ) < TOLZERO ) break;
    if( sum > intsum ){
      int imin = -1;
      double effmin = 1.;
      for( unsigned i=0; i<n; ++i ){
        if( !typ[i] || val[i] < TOLZERO ) continue;
        if( intval[i]/val[i] < effmin ){
          imin = i;
          effmin = intval[i]/val[i];
        }
      }
      assert( imin >= 0 );
      intval[imin] += 1;
    }
    else{
      int imax = -1;
      double effmax = 1.;
      for( unsigned i=0; i<n; ++i ){
        if( !typ[i] || val[i] < TOLZERO ) continue;
        if( intval[i]/val[i] > effmax ){
          imax   = i;
          effmax = intval[i]/val[i];
        }
      }
      assert( imax >= 0 );
      intval[imax] -= 1;
    }
  }
  
#ifdef MAGNUS__EXPDES_SHOW_APPORTION
  std::cout << "Rounded efforts:" << std::endl;
#endif
  for( unsigned i=0; i<n; ++i ){
    if( !typ[i] ) continue;
    val[i] = intval[i];
#ifdef MAGNUS__EXPDES_SHOW_APPORTION
    if( val[i] > TOLZERO ) std::cout << "X[" << i << "]: " << val[i] << std::endl;
#endif
  }
}

} // end namescape mc

#endif


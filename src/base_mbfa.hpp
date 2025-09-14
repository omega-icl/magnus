// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MAGNUS__BASE_MBFA_HPP
#define MAGNUS__BASE_MBFA_HPP

#undef  MAGNUS__DEBUG__BASE_MBFA

#include <assert.h>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <armadillo>

#include "ffunc.hpp"
#include "base_opt.hpp"
#include "base_sampling.hpp"

namespace mc
{
//! @brief C++ base class for defining of model-based feasibility analysis problems
////////////////////////////////////////////////////////////////////////
//! mc::BASE_MBFA is a C++ base class for defining the controls,
//! parameters and constraints in model-based feasibility analysis
//! problems
////////////////////////////////////////////////////////////////////////
class BASE_MBFA
: public virtual BASE_OPT,
  public virtual BASE_SAMPLING
{
protected:

  //! @brief pointer to DAG of equation
  FFGraph* _dag;

  //! @brief Size of model constraints
  size_t _ng;

  //! @brief Size of model likelihood
  size_t _nl;

  //! @brief Size of model parameter
  size_t _np;

  //! @brief Size of model constants
  size_t _nc;

  //! @brief Size of experimental control
  size_t _nu;

  //! @brief vector of model constraints
  std::vector<FFVar> _vCTR;

  //! @brief vector of model likelihood
  std::vector<FFVar> _vLKH;

  //! @brief vector of model constants
  std::vector<FFVar> _vCST;

  //! @brief vector of model constant values
  std::vector<double> _vCSTVAL;

  //! @brief vector of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief vector of model parameter values
  std::vector<std::vector<double>> _vPARVAL;

  //! @brief vector of model parameter weights
  std::vector<double> _vPARWEI;

  //! @brief vector of experimental controls
  std::vector<FFVar> _vCON;

  //! @brief vector of experimental control lower bounds
  std::vector<double> _vCONLB;

  //! @brief vector of experimental control upper bounds
  std::vector<double> _vCONUB;

public:

  //! @brief Class constructor
  BASE_MBFA()
    : _dag(nullptr),
      _ng(0), _nl(0), _np(0), _nc(0), _nu(0)
    {}

  //! @brief Class destructor
  virtual ~BASE_MBFA()
    {}

  //! @brief Get pointer to DAG
  FFGraph const& dag()
    const
    { return *_dag; }

  //! @brief Set DAG
  void set_dag
    ( FFGraph& dag )
    { _dag = &dag; }

  //! @brief Set pointer to DAG
  void set_dag
    ( FFGraph* dag )
    { _dag = dag; }

  //! @brief Get size of model constraints
  size_t ng
    ()
    const
    { return _ng; }

  //! @brief Get size of model constants
  size_t nc
    ()
    const
    { return _nc; }

  //! @brief Get size of experimental controls
  size_t nu
    ()
    const
    { return _nu; }

  //! @brief Get size of model parameters
  size_t np
    ()
    const
    { return _np; }

  //! @brief Set model constants and (optionally) values
  void set_constant
    ( std::vector<FFVar> const& C, std::vector<double> const& valC=std::vector<double>() )
    {
      assert( valC.empty() || valC.size() == C.size() );
      _nc   = C.size();
      _vCST = C;
      _vCSTVAL = valC;
    }

  //! @brief Reset model constants
  void reset_constant
    ()
    {
      _nc = 0;
      _vCST.clear();
      _vCSTVAL.clear();
    }

  //! @brief Get model constants
  std::vector<FFVar> const& var_constant
    ()
    const
    { return _vCST; }

  //! @brief Set model parameters and nominal values
  void set_parameter
    ( std::vector<FFVar> const& P, std::vector<double> const& valP )
    {
      assert( !P.empty() && valP.size() == P.size() );
      _np   = P.size();
      _vPAR = P;
      _vPARVAL.clear();
      _vPARVAL.push_back( valP );
      _vPARWEI.assign( 1, 1. );
    }

  //! @brief Set model parameters and list of scenarios (equal-weighting)
  void set_parameter
    ( std::vector<FFVar> const& P, std::list<std::vector<double>> const& l_valP )
    {
      assert( !P.empty() && !l_valP.empty() );
      _np   = P.size();
      _vPAR = P;
      _vPARVAL.clear();
      for( auto const& valP : l_valP ){
        assert( valP.size() == _np );
        _vPARVAL.push_back( valP );
      }
      _vPARWEI.assign( _vPARVAL.size(), 1/(double)l_valP.size() ); // equal frequencies
    }

  //! @brief Set model parameters and list of scenarios with weights
  void set_parameter
    ( std::vector<FFVar> const& P, std::list<std::pair<std::vector<double>,double>> const& l_valP )
    {
      assert( !P.empty() && !l_valP.empty() );
      _np   = P.size();
      _vPAR = P;

      _vPARWEI.clear();
      double prTot = 0.;
      for( auto const& [valP,prP] : l_valP ){
        assert( prP > 0 );
        prTot += prP;
      }

      _vPARVAL.clear();
      for( auto const& [valP,prP] : l_valP ){
        assert( valP.size() == _np );
        _vPARVAL.push_back( valP );
        _vPARWEI.push_back( prP/prTot );
      }
    }

  //! @brief Get model parameters
  std::vector<FFVar> const& var_parameter
    ()
    const
    { return _vPAR; }

  //! @brief Set model controls
  void set_control
    ( std::vector<FFVar> const& C, std::vector<double> const& CLB, std::vector<double> const& CUB )
    {
      assert( !C.empty() && CLB.size() == C.size() && CUB.size() == C.size() );
      _nu     = C.size();
      _vCON   = C;
      _vCONLB = CLB;
      _vCONUB = CUB;
    }

  //! @brief Get model controls
  std::vector<FFVar> const& var_control
    ()
    const
    { return _vCON; }

  //! @brief Set vector of model-based constraints
  void set_constraint
    ( std::vector<FFVar> const& G )
    {
      assert( !G.empty() );
      _ng = G.size();
      _vCTR = G;
    }

  //! @brief Add vector of model-based constraints
  void add_constraint
    ( std::vector<FFVar> const& G )
    {
      if( G.empty() ) return;
      _ng += G.size();
      _vCTR.insert( _vCTR.end(), G.cbegin(), G.cend() );
    }

  //! @brief Set single model-based constraint
  void set_constraint
    ( FFVar const& lhs, t_CTR const type=LE, FFVar const& rhs=FFVar(0.) )
    {
      assert( type != EQ );
      _ng = 1;
      if( type == LE ) _vCTR = { lhs - rhs };
      else             _vCTR = { rhs - lhs };
    }

  //! @brief Add single model-based constraint
  void add_constraint
    ( FFVar const& lhs, t_CTR const type=LE, FFVar const& rhs=FFVar(0.) )
    {
      assert( type != EQ );
      _ng++;
      if( type == LE ) _vCTR.push_back( lhs - rhs );
      else             _vCTR.push_back( rhs - lhs );
    }

  //! @brief Reset vector of model-based constraints
  void reset_constraint
    ()
    {
      _ng = 0;
      _vCTR.clear();
    }

  //! @brief Get model constraints
  std::vector<FFVar> const& var_constraint
    ()
    const
    { return _vCTR; }

  //! @brief Set likelihood criterion
  void set_loglikelihood
    ( FFVar const& LL )
    {
      _nl = 1;
      _vLKH = { LL };
    }

  //! @brief Reset likelihood criterion
  void reset_loglikelihood
    ()
    {
      _nl = 0;
      _vLKH.clear();
    }

  //! @brief Get likelihood
  std::vector<FFVar> const& var_loglikelihood
    ()
    const
    { return _vLKH; }

private:

  //! @brief Private methods to block default compiler methods
  BASE_MBFA( BASE_MBFA const& ) = delete;
  BASE_MBFA& operator=( BASE_MBFA const& ) = delete;
};

} // end namescape mc

#endif


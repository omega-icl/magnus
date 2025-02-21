// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MAGNUS__BASE_PAREST_HPP
#define MAGNUS__BASE_PAREST_HPP

#undef  MAGNUS__DEBUG__BASE_PAREST

#include <assert.h>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <armadillo>

#include "ffunc.hpp"

namespace mc
{
//! @brief C++ base class for defining of parameter estimation problems
////////////////////////////////////////////////////////////////////////
//! mc::BASE_PAREST is a C++ base class for defining the controls,
//! parameters and outputs participating in parameter estimation
//! problems
////////////////////////////////////////////////////////////////////////
class BASE_PAREST
: public virtual BASE_OPT
{
public:

  //! @brief Experiment storage
  struct Record
  {
    //! @brief Default constructor
    Record
      ()
      {}

    //! @brief Constructor
    Record
      ( std::vector<double> const& measurement_, double const& variance_=0. )
      : measurement( measurement_ ), variance( variance_ )
      {}

    //! @brief Assignment operator
    Record& operator=
      ( Record const& rec )
      {
        measurement = rec.measurement;
        variance    = rec.variance;
        return *this;
      }

    //! @brief Vector of measurement replicates
    std::vector<double> measurement;

    //! @brief Measurement variance
    double variance;
  };

  //! @brief Experiment storage
  struct Experiment
  {
    //! @brief Default constructor
    Experiment
      ()
      {}

    //! @brief Constructor
    Experiment
      ( std::vector<double> const& control_,
        std::map<size_t,Record> const& output_=std::map<size_t,Record>(), 
        size_t const index_=0 )
      : output( output_ ), control( control_ ), index( index_ )
      {}

    //! @brief Assignment operator
    Experiment& operator=
      ( Experiment const& exp )
      {
        output   = exp.output;
        control  = exp.control;
        index    = exp.index;
        return *this;
      }

    //! @brief Map of output measurements
    std::map<size_t,Record> output;

    //! @brief Vector of control values
    std::vector<double> control;

    //! @brief Model index
    size_t index;
  };

protected:

  //! @brief Pointer to DAG of equation
  FFGraph* _dag;

  //! @brief Number of models
  size_t _nm;

  //! @brief Size of model outputs
  std::vector<size_t> _ny;

  //! @brief Size of experimental controls
  std::vector<size_t> _nu;

  //! @brief Size of model parameters
  size_t _np;

  //! @brief Size of model constants
  size_t _nc;

  //! @brief Size of cost regularizations
  size_t _nr;

  //! @brief Size of model constraints
  size_t _ng;

  //! @brief Number of data points
  size_t _nd;

  //! @brief vector of model outputs
  std::vector<std::vector<FFVar>> _vOUT;

  //! @brief vector of experimental controls
  std::vector<std::vector<FFVar>> _vCON;

  //! @brief vector of model constants
  std::vector<FFVar> _vCST;

  //! @brief vector of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief vector of model parameter lower bounds
  std::vector<double> _vPARLB;

  //! @brief vector of model parameter upper bounds
  std::vector<double> _vPARUB;

  //! @brief matrix of model parameter scaling factors
  arma::mat _mPARSCA;

  //! @brief vector of experimental data: 
  std::vector<std::vector<Experiment>> _vDAT;

  //! @brief vector of cost regularization terms
  std::vector<FFVar> _vREG;

  //! @brief constraints (constraint lhs, type, constraint rhs)
  std::tuple< std::vector<FFVar>, std::vector<t_CTR>, std::vector<FFVar> > _vCTR;

  //! @brief Add to experimental data
  static void _add_data
    ( std::vector<std::vector<Experiment>>& DAT, Experiment const& EXP )
    {
      assert( EXP.output.size() );
      if( EXP.index >= DAT.size() )
        DAT.resize( EXP.index+1 );
      DAT[EXP.index].push_back( EXP );
    }

public:

  //! @brief Class constructor
  BASE_PAREST()
    : _dag(nullptr), _nm(0), _np(0), _nc(0), _nr(0), _ng(0), _nd(0)
    {}

  //! @brief Class destructor
  virtual ~BASE_PAREST()
    {}

  //! @brief Get pointer to DAG
  FFGraph const& dag()
    const
    { return *_dag; }

  //! @brief Set pointer to DAG
  void set_dag
    ( FFGraph& dag )
    { _dag = &dag; }

  //! @brief Get number of models
  size_t nm
    ()
    const
    { return _nm; }

  //! @brief Get size of model outputs
  size_t ny
    ( size_t m=0 )
    const
    { return m<_ny.size()? _ny[m]: 0; }

  //! @brief Get size of model controls
  size_t nu
    ( size_t m=0 )
    const
    { return m<_nu.size()? _nu[m]: 0; }

  //! @brief Get size of model constants
  size_t nc
    ()
    const
    { return _nc; }

  //! @brief Get size of model parameters
  size_t np
    ()
    const
    { return _np; }

  //! @brief Get size of cost regularizations
  size_t nr
    ()
    const
    { return _nr; }

  //! @brief Get size of model constraints
  size_t ng
    ()
    const
    { return _ng; }

  //! @brief Get total number of experiments
  size_t nd
    ()
    const
    { return _nd; }

  //! @brief Reset model
  void reset_model
    ()
    {
      _vOUT.clear();
      _vCON.clear();
      _ny.clear();
      _nu.clear();
    }

  //! @brief Set model
  void add_model
    ( std::vector<FFVar> const& Y, std::vector<FFVar> const& U=std::vector<FFVar>(),
      size_t const m=0 )
    {
      assert( !Y.empty() );
      if( m >= _nm ){
        _vOUT.resize( m+1 );
        _vCON.resize( m+1 );
        _ny.resize( m+1, 0 );
        _nu.resize( m+1, 0 );
        _nm = _vOUT.size();
      }
      _ny[m]   = Y.size();
      _nu[m]   = U.size();
      _vOUT[m] = Y;
      _vCON[m] = U;
    }

  //! @brief Get model outputs
  std::vector<FFVar> const& var_output
    ( size_t const m=0 )
    const
    { assert( m<_ny.size() ); return _vOUT[m]; }

  //! @brief Get experimental controls
  std::vector<FFVar> const& var_control
    ( size_t const m=0 )
    const
    { assert( m<_nu.size() ); return _vCON[m]; }

  //! @brief Set model constants
  void set_constant
    ( std::vector<FFVar> const& C )
    {
      _nc   = C.size();
      _vCST = C;
    }
    
  //! @brief Get model constants
  std::vector<FFVar> const& var_constant
    ()
    const
    { return _vCST; }

  //! @brief Set nominal model parameters and bounds
  void set_parameter
    ( std::vector<FFVar> const& P,
      std::vector<double> const& PLB,
      std::vector<double> const& PUB,
      arma::mat const& scaP )
    {
      set_parameter( P, PLB, PUB );
      if( scaP.n_rows == P.size() && scaP.n_cols == P.size() ) _mPARSCA = scaP;
    }

  //! @brief Set model parameters and bounds
  void set_parameter
    ( std::vector<FFVar> const& P,
      std::vector<double> const& PLB=std::vector<double>(),
      std::vector<double> const& PUB=std::vector<double>(),
      std::vector<double> const& scaP=std::vector<double>() )
    {
      assert( !P.empty() && (!PLB.size() || PLB.size() == P.size())
                         && (!PUB.size() || PUB.size() == P.size()) );
      _np   = P.size();
      _vPAR = P;
      _vPARLB = PLB;
      _vPARUB = PUB;

      assert( scaP.empty() || scaP.size() == _np );
      _mPARSCA.reset();
      if( !scaP.empty() )
        _mPARSCA = arma::diagmat( arma::vec( scaP ) );
    }

  //! @brief Get model parameters
  std::vector<FFVar> const& var_parameter
    ()
    const
    { return _vPAR; }

  //! @brief Get parameter scaling
  arma::mat scaling_parameter
    ()
    const
    { 
      return _mPARSCA;
    }

  //! @brief Reset experimental data
  void reset_data
    ()
    {
      _vDAT.clear();
      _nd = 0;
    }

  //! @brief Add to experimental data
  void add_data
    ( Experiment const& EXP )
    {
      _add_data( _vDAT, EXP );
      for( auto const& [ k, RECk ] : EXP.output )
        _nd += RECk.measurement.size();
    }

  //! @brief Add experimental data
  void add_data
    ( std::vector<Experiment> const& DAT )
    {
      for( auto const& EXP : DAT )
        add_data( EXP );
    }

  //! @brief Set experimental data
  void set_data
    ( std::vector<Experiment> const& DAT )
    {
      reset_data();
      add_data( DAT );
    }

  //! @brief Get experimental data
  std::vector<std::vector<Experiment>> const& get_data
    ()
    const
    { return _vDAT; }

  //! @brief Add regularisation term in estimation objective
  void add_regularization
    ( FFVar const& R )
    {
      _vREG.push_back( R );
      _nr = _vREG.size();
    }

  //! @brief Reset regularisation term in estimation objective
  void reset_regularization
    ()
    {
      _vREG.clear();
    }

  //! @brief Get constraints
  std::tuple< std::vector<FFVar>, std::vector<t_CTR>, std::vector<FFVar> > const& get_constraint
    ()
    const
    { return _vCTR; }

  //! @brief Add constraint
  void add_constraint
    ( FFVar const& lhs, t_CTR const type, FFVar const& rhs=FFVar(0.) )
    {
      std::get<0>(_vCTR).push_back( lhs  );
      std::get<1>(_vCTR).push_back( type );
      std::get<2>(_vCTR).push_back( rhs  );
      _ng = std::get<0>(_vCTR).size();
    }

  //! @brief Reset constraints
  void reset_constraint
    ()
    { std::get<0>(_vCTR).clear(); std::get<1>(_vCTR).clear(); std::get<2>(_vCTR).clear(); }

  //! @brief Sobol sampling within bounds
  static std::list<std::vector<double>> sobol_sample
    ( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB );

private:

  //! @brief Private methods to block default compiler methods
  BASE_PAREST( BASE_PAREST const& ) = delete;
  BASE_PAREST& operator=( BASE_PAREST const& ) = delete;
};

inline std::list<std::vector<double>>
BASE_PAREST::sobol_sample
( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB )
{
  assert( NSAM && LB.size() && LB.size() == UB.size() );
  size_t NDIM = LB.size();

  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( NDIM );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( 0 );

  std::list<std::vector<double>> LSAM;
  for( size_t s=0; s<NSAM; ++s ){
    LSAM.push_back( std::vector<double>( NDIM ) );
    for( size_t k=0; k<NDIM; k++ )
      LSAM.back()[k] = LB[k] + ( UB[k] - LB[k] ) * gen();
  }

  return LSAM;
}

} // end namescape mc

#endif


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
      ( std::vector<double> const& measurements_, double const& variance_=0. )
      : measurements( measurements_ ), variance( variance_ )
      {}

    //! @brief Assignment operator
    Record& operator=
      ( Record const& rec )
      {
        measurements = rec.measurements;
        variance     = rec.variance;
        return *this;
      }

    //! @brief Vector of measurement replicates
    std::vector<double> measurements;

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
      ( std::vector<double> const& inputs_ )
      : inputs( inputs_ )
      {}

    //! @brief Constructor
    Experiment
      ( std::vector<double> const& inputs_, std::map<size_t,Record> const& outputs_ )
      : inputs( inputs_ ), outputs( outputs_ )
      {}

    //! @brief Assignment operator
    Experiment& operator=
      ( Experiment const& exp )
      {
        inputs  = exp.inputs;
        outputs = exp.outputs;
        return *this;
      }

    //! @brief Vector of input values
    std::vector<double> inputs;

    //! @brief Map of output measurements
    std::map<size_t,Record> outputs;
  };

protected:

  //! @brief pointer to DAG of equation
  FFGraph* _dag;

  //! @brief Size of model output
  size_t _ny;

  //! @brief Size of experimental control
  size_t _nu;

  //! @brief Size of model parameter
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
  std::vector<FFVar> _vOUT;

  //! @brief vector of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief vector of experimental controls
  std::vector<FFVar> _vCON;

  //! @brief vector of model parameter lower bounds
  std::vector<double> _vPARLB;

  //! @brief vector of model parameter upper bounds
  std::vector<double> _vPARUB;

  //! @brief vector of model parameter guesses
  std::vector<double> _vPARVAL;

  //! @brief matrix of model parameter scaling factors
  arma::mat _mPARSCA;

  //! @brief vector of model constants
  std::vector<FFVar> _vCST;

  //! @brief vector of experimental data: 
  std::vector<Experiment> _vDAT;

  //! @brief vector of cost regularization terms
  std::vector<FFVar> _vREG;

  //! @brief constraints (constraint lhs, type, constraint rhs)
  std::tuple< std::vector<FFVar>, std::vector<t_CTR>, std::vector<FFVar> > _vCTR;

public:

  //! @brief Class constructor
  BASE_PAREST()
    : _dag(nullptr), _ny(0), _nu(0), _np(0), _nc(0), _ng(0), _nd(0)
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

  //! @brief Get size of model outputs
  size_t ny
    ()
    const
    { return _ny; }

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

  //! @brief Get size of model constants
  size_t nc
    ()
    const
    { return _nc; }

  //! @brief Get size of cost regularizations
  size_t nr
    ()
    const
    { return _vREG.size(); }

  //! @brief Get size of model constraints
  size_t ng
    ()
    const
    { return std::get<0>(_vCTR).size(); }

  //! @brief Set model outputs for single model
  void set_model
    ( std::vector<FFVar> const& Y )
    {
      assert( !Y.empty() );
      _ny = Y.size();
      _vOUT = Y;
    }

  //! @brief Set nominal model parameters and parameter scaling (matrix format)
  void set_parameters
    ( std::vector<FFVar> const& P,
      std::vector<double> const& PLB,
      std::vector<double> const& PUB,
      arma::mat const& scaP )
    {
      set_parameters( P, PLB, PUB );
      if( scaP.n_rows == P.size() && scaP.n_cols == P.size() ) _mPARSCA = scaP;
    }

  //! @brief Set nominal model parameters and parameter scaling (vector format)
  void set_parameters
    ( std::vector<FFVar> const& P,
      std::vector<double> const& PLB=std::vector<double>(),
      std::vector<double> const& PUB=std::vector<double>(),
      std::vector<double> const& scaP=std::vector<double>() )
    {
      assert( !P.empty() && (!PLB.size() || PLB.size() == P.size()) && (!PUB.size() || PUB.size() == P.size()) );
      _np   = P.size();
      _vPAR = P;
      _vPARLB = PLB;
      _vPARUB = PUB;
      _vPARVAL.clear();

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

  //! @brief Set experimental controls
  void set_controls
    ( std::vector<FFVar> const& U )
    {
      assert( !U.empty() );
      _nu     = U.size();
      _vCON   = U;
    }

  //! @brief Set experimental controls
  void set_constants
    ( std::vector<FFVar> const& C )
    {
      assert( !C.empty() );
      _nc     = C.size();
      _vCST   = C;
    }

  //! @brief Set experimental controls
  void set_data
    ( std::vector<Experiment> const& D )
    {
      assert( !D.empty() );
      _vDAT = D;
      _nd   = 0;
      for( auto const& EXP : _vDAT )
        for( auto const& [ k, RECk ] : EXP.outputs )
          _nd += RECk.measurements.size();
    }

  //! @brief Add regularisation term in estimation objective
  void add_regularization
    ( FFVar const& R )
    {
      _vREG.push_back( R );
    }

  //! @brief Reset regularisation term in estimation objective
  void reset_regularization
    ( FFVar const& R )
    {
      _vREG.clear();
    }

  //! @brief Get constraints
  std::tuple< std::vector<FFVar>, std::vector<t_CTR>, std::vector<FFVar> > const& constraints
    ()
    const
    { return _vCTR; }

  //! @brief Add constraint
  void add_constraint
    ( FFVar const& lhs, t_CTR const type, FFVar const& rhs=FFVar(0.) )
    { std::get<0>(_vCTR).push_back( lhs  );
      std::get<1>(_vCTR).push_back( type );
      std::get<2>(_vCTR).push_back( rhs  ); }

  //! @brief Reset constraints
  void reset_constraints
    ()
    { std::get<0>(_vCTR).clear(); std::get<1>(_vCTR).clear(); std::get<2>(_vCTR).clear(); }

  //! @brief Set uniform sample within bounds
  static std::list<std::vector<double>> uniform_sample
    ( size_t NSAM, std::vector<double> const& LB, std::vector<double> const& UB );

private:

  //! @brief Private methods to block default compiler methods
  BASE_PAREST( BASE_PAREST const& ) = delete;
  BASE_PAREST& operator=( BASE_PAREST const& ) = delete;
};

inline std::list<std::vector<double>>
BASE_PAREST::uniform_sample
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


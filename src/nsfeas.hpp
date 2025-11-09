// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_NSFEAS Model-based Feasibility Analysis using Nested Sampling
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 1.1
\date 2025
\bug No known bugs.
*/

#ifndef MAGNUS__NSFEAS_HPP
#define MAGNUS__NSFEAS_HPP

#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <boost/math/distributions/chi_squared.hpp>

#include "base_mbfa.hpp"
#include "fffeas.hpp"
#include "ffexpr.hpp"

namespace mc
{
//! @brief C++ class for model-based feasibility analysis using nested sampling
////////////////////////////////////////////////////////////////////////
//! mc::NSFEAS is a C++ class for determining the feasible set of
//! model-based constraints using nested sampling
////////////////////////////////////////////////////////////////////////
class NSFEAS
: public virtual BASE_MBFA
{

protected:

  typedef FFGraph DAG;
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;

  //! @brief quasi-random number generator
  qrgen                                        _gen;

  //! @brief quasi-random number generator
  qrgen                                        _gen1d;

  //! @brief flag indicating if problem is setup before sampling
  bool                                         _is_setup;

  //! @brief DAG of model
  DAG*                                         _dag;

  //! @brief local copy of model constants
  std::vector<FFVar>                           _vCST;

  //! @brief local copy of model parameters
  std::vector<FFVar>                           _vPAR;

  //! @brief local copy of experimental controls
  std::vector<FFVar>                           _vCON;

  //! @brief vector of sampled control candidates
  std::vector<std::vector<double>>             _vCONSAM;

  //! @brief vector of sampled control proposals
  std::vector<std::vector<double>>             _vCONPROP;

  //! @brief local copy of constraints
  std::vector<FFVar>                           _vCTR;

  //! @brief local copy of likelihood
  std::vector<FFVar>                           _vLKH;

  //! @brief criteria
  std::vector<FFVar>                           _vVAL;

  //! @brief criteria values
  std::vector<double>                          _dVAL;

  //! @brief vector of sampled criteria values
  std::vector<std::vector<double>>             _vVALSAM;

  //! @brief criteria subgraph
  FFSubgraph                                   _sgVAL;

  //! @brief work array for criteria evaluations
  std::vector<double>                          _wkD;

  // Thread storage for criteria evaluations
  std::vector<FFGraph::Worker<double>>         _wkVAL;

  //! @brief data matrix for current nest
  arma::mat                                    _nestData;

  //! @brief centre vector of current nest
  arma::vec                                    _nestCentre;

  //! @brief lower bound of current nest
  arma::vec                                    _nestLB;

  //! @brief upper bound of current nest
  arma::vec                                    _nestUB;

  //! @brief shape matrix for current nest
  arma::mat                                    _nestShape;

  //! @brief map of live points for feasibility
  std::multimap<double,std::tuple<double const*,double,double>> _liveFEAS;

  //! @brief map of dead points for feasibility
  std::multimap<double,std::tuple<double const*,double,double>> _deadFEAS;

  //! @brief map of discarded points for feasibility
  std::multimap<double,std::tuple<double const*,double,double>> _discardFEAS;

  //! @brief map of live points for likelihood
  std::multimap<double,std::tuple<double const*,double,double>> _liveLKH;

  //! @brief map of dead points for likelihood
  std::multimap<double,std::tuple<double const*,double,double>> _deadLKH;

  //! @brief mass of live nest
  double                                       _liveNest;

  //! @brief mass of dead shell
  double                                       _deadShell;

  //! @brief total mass
  double                                       _totalMass;

  //! @brief nest mass
  double                                       _nestMass;

public:
  /** @defgroup NSFEAS Model-based Feasibility Analysis using Nested Sampling
   *  @{
   */
  //! @brief NLP solution status
  enum STATUS{
     NORMAL=0,	 //!< Normal completion
     INTERRUPT,  //!< Resource limit reached
     FAILURE,    //!< Terminated after numerical failure
     ABORT       //!< Aborted after critical error
  };

  //! @brief Constructor
  NSFEAS
    ()
    : _gen       ( sobol64(1), boost::uniform_01<double>() ),
      _gen1d     ( sobol64(1), boost::uniform_01<double>() ),
      _is_setup  ( false ),
      _dag       ( nullptr ),
      _liveNest  ( 0. ),
      _deadShell ( 0. ),
      _totalMass ( 0. ),
      _nestMass  ( 0. )
    {
      stats.reset();
    }

  //! @brief Destructor
  virtual ~NSFEAS
    ()
    {
      delete _dag;
    }

  //! @brief NSFEAS solver options
  struct Options
  {
    //! @brief Constructor
    Options
      ()
      {
        reset();
      }

    //! @brief Reset to default options
    void reset
      ()
      {
        FEASCRIT   = VAR;
        FEASTHRES  = 0.1;
        LKHCRIT    = VAR;
        LKHTHRES   = 0.1;
        LKHTOL     = 0.05;
        NUMLIVE    = std::pow(2,8);
        NUMPROP    = std::pow(2,4);
        ELLCONF    = 0.99;
        ELLMAG     = 0.30; // Feroz et al. (2009)
        ELLRED     = 0.20; // Feroz et al. (2009)
        MAXITER    = 0;
        MAXERR     = 0;
        MAXCPU     = 0;
        MAXTHREAD  = 1;
        DISPLEVEL  = 1;
        DISPITER   = NUMLIVE/10;
      }

    //! @brief Assignment operator
    Options& operator=
      ( Options const& options )
      {
        FEASCRIT   = options.FEASCRIT;
        FEASTHRES  = options.FEASTHRES;
        LKHCRIT    = options.LKHCRIT;
        LKHTHRES   = options.LKHTHRES;
        LKHTOL     = options.LKHTOL;
        NUMLIVE    = options.NUMLIVE;
        NUMPROP    = options.NUMPROP;
        ELLCONF    = options.ELLCONF;
        ELLMAG     = options.ELLMAG;
        ELLRED     = options.ELLRED;
        MAXITER    = options.MAXITER;
        MAXERR     = options.MAXERR;
        MAXCPU     = options.MAXCPU;
        MAXTHREAD  = options.MAXTHREAD;
        DISPLEVEL  = options.DISPLEVEL;
        DISPITER   = options.DISPITER;
        return *this;
      }

      //! @brief Enumeration for feasibility criterion
      enum TYPE{
        VAR=1,  //!< Value-at-risk, VaR
        CVAR    //!< Conditional value-at-risk, CVaR
      };

    //! @brief Selected feasibility criterion
    TYPE                     FEASCRIT;
    //! @brief Percentile feasibility violation threshold
    double                   FEASTHRES;
    //! @brief Selected likelihood criterion
    TYPE                     LKHCRIT;
    //! @brief Percentile likelihood threshold
    double                   LKHTHRES;
    //! @brief Stopping tolerance for probability mass
    double                   LKHTOL;
    //! @brief number of live points
    size_t                   NUMLIVE;
    //! @brief number of proposals
    size_t                   NUMPROP;
    //! @brief Chi-squared confidence in ellipsoidal nest
    double                   ELLCONF;
    //! @brief Initial magnification of ellipsoidal nest
    double                   ELLMAG;
    //! @brief Reduction factor of ellipsoidal nest
    double                   ELLRED;
    //! @brief maximal number of iterations
    size_t                   MAXITER;
    //! @brief maximal number of failed evaluations
    size_t                   MAXERR;
    //! @brief maximal walltime
    double                   MAXCPU;
    //! @brief maximal number of parallel threads
    size_t                   MAXTHREAD;
    //! @brief display level
    int                      DISPLEVEL;
    //! @brief display frequency
    size_t                   DISPITER;
  } options;

  //! @brief NSFEAS solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for NSFEAS exception handling
    enum TYPE{
      BADSIZE=0,	//!< Inconsistent dimensions
      NOSETUP,		//!< Problem was not setup
      BADCONST,		//!< Unspecified constants
      INTERN=-33	//!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr() const { return _ierr; }
    //! @brief Inline function returning the error description
    std::string what() const {
      switch( _ierr ){
        case BADSIZE:
          return "NSFEAS::Exceptions  Inconsistent dimensions";
        case NOSETUP:
          return "NSFEAS::Exceptions  Problem not setup";
        case BADCONST:
          return "NSFEAS::Exceptions  Unspecified constants";
        case INTERN:
        default:
          return "NSFEAS::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief NSFEAS solver statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime = std::chrono::microseconds(0);
        iter = numfct = numerr = numfeas = 0; }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "** EVALUATIONS: " << std::right << std::setw(10) << numfct
                                 << std::left  << " (" << numerr << " FAILED)" << std::endl
           << "** WALL TIME:   " << std::right << std::setw(10) << to_time( walltime ) << " SEC" << std::endl; }
    //! @brief Current iteration
    size_t iter;
    //! @brief Total function evaluations
    size_t numfct;
    //! @brief Number of failed likelihood evaluations
    size_t numerr;
    //! @brief Number of feasible points
    size_t numfeas;
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime;
    //! @brief Get current time point
    std::chrono::time_point<std::chrono::system_clock> start
      () const
      { return std::chrono::system_clock::now(); }
    //! @brief Get current time lapse with respect to start time point
    std::chrono::microseconds lapse
      ( std::chrono::time_point<std::chrono::system_clock> const& start ) const
      { return std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start ); }    
    //! @brief Convert microsecond ticks to time
    double to_time
      ( std::chrono::microseconds t ) const
      { return t.count() * 1e-6; }
  } stats;

  //! @brief Setup feasibility problem before solution
  bool setup
    ();

  //! @brief Sample feasible domain
  int sample
    ( std::vector<double> const& vcst=std::vector<double>(), bool const reset=true, std::ostream& os=std::cout );
/*
  //! @brief Export effort, support and fim to file
  bool file_export
    ( std::string const& name );
*/
  //! @brief Retrieve live points
  std::multimap<double,std::tuple<double const*,double,double>> const& live_points
    ()
    const
    { return _nl? _liveLKH: _liveFEAS; }

  //! @brief Retrieve dead points
  std::multimap<double,std::tuple<double const*,double,double>> const& dead_points
    ()
    const
    { return _nl? _deadLKH: _deadFEAS; }

  //! @brief Retrieve discarded points
  std::multimap<double,std::tuple<double const*,double,double>> const& discard_points
    ()
    const
    { return _discardFEAS; }

protected:

  //! @brief Append candidate points inside bounds or current nest
  bool _sample_ini
    ( size_t const nprop, std::chrono::time_point<std::chrono::system_clock> const& tstart,
      std::ostream& os );

  //! @brief Update current nest
  bool _update_nest
    ( std::multimap<double,std::tuple<double const*,double,double>> const& points, 
      double const& factor, std::ostream& os );

  //! @brief Sample within domain bounds using uniform Sobol' sampling
  void _sample_bounds
    ( std::ostream& os );

  //! @brief Sample current nest - return value: True if within bounds; False otherwise
  bool _sample_nest
    ( bool const bndcheck, std::ostream& os );

  //! @brief Test for criterion feasibility
  bool _feasible
    ( double const& crit, std::ostream& os )
    const;

  //! @brief Test for completion or interruption
  int _terminate
    ( size_t const it, bool const feas, std::chrono::time_point<std::chrono::system_clock> const& tstart,
      std::ostream& os )
    const;

  //! @brief Feasibility sampler
  int _sample_feas
    ( double& factor, std::chrono::time_point<std::chrono::system_clock> const& tstart, std::ostream& os );

  //! @brief Likelihood sampler
  int _sample_lkh
    ( double& factor, std::chrono::time_point<std::chrono::system_clock> const& tstart, std::ostream& os );
};

inline
bool
NSFEAS::setup
()
{
  stats.reset();
  _is_setup = false;

  if( !_nu || ( !_ng && _nl > 1 ) )
    throw Exceptions( Exceptions::BADSIZE );

  try{
    delete _dag; _dag = new DAG;
    _dag->options = BASE_MBFA::_dag->options;
    _dag->options.MAXTHREAD = options.MAXTHREAD;

    _vCON.resize( _nu );
    _dag->insert( BASE_MBFA::_dag, _nu, BASE_MBFA::_vCON.data(), _vCON.data() );
    _vCST.resize( _nc );
    _dag->insert( BASE_MBFA::_dag, _nc, BASE_MBFA::_vCST.data(), _vCST.data() );
    _vPAR.resize( _np );
    _dag->insert( BASE_MBFA::_dag, _np, BASE_MBFA::_vPAR.data(), _vPAR.data() );
    _vCTR.resize( _ng );
    _dag->insert( BASE_MBFA::_dag, _ng, BASE_MBFA::_vCTR.data(), _vCTR.data() );
    _vLKH.resize( _nl );
    _dag->insert( BASE_MBFA::_dag, _nl, BASE_MBFA::_vLKH.data(), _vLKH.data() );

#ifdef MAGNUS__NSFEAS_DEBUG
    if( _ng ){
      auto sgCTR = _dag->subgraph( _ng, _vCTR.data() );
      std::vector<FFExpr> exCTR = FFExpr::subgraph( _dag, sgCTR ); 
      for( size_t i=0; i<_ng; ++i )
        std::cout << "CTR[" << i << "] = " << exCTR[i] << std::endl;
    }
    if( _nl ){
      auto sgLKH = _dag->subgraph( _vLKH );
      std::vector<FFExpr> exLKH = FFExpr::subgraph( _dag, sgLKH ); 
      std::cout << "LKH = " << exLKH[0] << std::endl;
    }
#endif
  }
  catch(...){
    return false;
  }
  
  // Sobol samplers
  sobol64 eng( _nu );
  _gen.engine() = eng;
  
  sobol64 eng1d( 1 );
  _gen1d.engine() = eng1d;

  _is_setup = true;
  return true;
}

inline
bool
NSFEAS::_sample_ini
( size_t const nprop, std::chrono::time_point<std::chrono::system_clock> const& tstart,
  std::ostream& os )
{
  // Nest bounds
  _nestLB = arma::vec( _vCONLB );
  _nestUB = arma::vec( _vCONUB );
  size_t const NLIVE0 = _liveFEAS.size();

  // Sample nest in parallel
#ifndef MAGNUS__NSFEAS_THREAD_FOREACH_SAMPLE
  if( _dag->options.MAXTHREAD != 1 ){
#else
  if( _vPARVAL.size() <= 1 && _dag->options.MAXTHREAD != 1 ){
#endif
    // Sample new points
    for( size_t s=0; _vCONSAM.size() < NLIVE0 + nprop; ++s ){
      _sample_bounds( os );
    }

    // Evaluate new points
    try{
      _dag->veval( _sgVAL, _wkD, _wkVAL, _vVAL, _vVALSAM, _vCON, _vCONSAM );
    }
    catch(...){
      return false;
    }

    // Add new points to live set
    for( size_t s=0; s<_vVALSAM.size(); ++s, stats.numfct+=(_vPARVAL.size()>0?_vPARVAL.size():1) ){ // to account for failures
      if( !_dag->veval_err().at(s) ){
        _liveFEAS.insert( { DBL_MAX,
                          { _vCONSAM[s].data(), 0., -DBL_MAX } } );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        std::cout << "Failed scenario " << s << ":"
                  << std::right << std::scientific << std::setprecision(5)
                  << arma::rowvec( const_cast<double*>(_vCONSAM[s].data()), _nu, false );
#endif
        ++stats.numerr;
        continue;
      }
      _liveFEAS.insert( { _vVALSAM[s][options.FEASCRIT],
                        { _vCONSAM[s].data(), _vVALSAM[s][0], _vVALSAM[s][2+(int)options.LKHCRIT] } } );
      if( _feasible( _vVALSAM[s][options.FEASCRIT], os ) ) ++stats.numfeas;
    }
  }

  // Sample nest sequentially
  else{
    for( size_t s=0; _liveFEAS.size() < NLIVE0 + nprop; ++s, stats.numfct+=(_vPARVAL.size()>0?_vPARVAL.size():1) ){ // to account for failures
      // Sample new point
      _sample_bounds( os );
      std::vector<double> const& dCON = _vCONSAM.back();

      // Evaluate new point
      try{
        _dag->eval( _sgVAL, _wkD, _vVAL, _dVAL, _vCON, dCON );
        _liveFEAS.insert( { _dVAL[options.FEASCRIT], { dCON.data(), _dVAL[0], _dVAL[2+(int)options.LKHCRIT] } } );
        if( _feasible( _dVAL[options.FEASCRIT], os ) ) ++stats.numfeas;
      }
      catch(...){
        ++stats.numerr;
        continue;
      }

      if( ( options.MAXCPU > 0 && stats.to_time( stats.lapse( tstart ) ) >= options.MAXCPU )
       || ( options.MAXERR > 0 && stats.numerr >= options.MAXERR ) )
        return false;
    }
  }
  
  return true;
}

inline
void
NSFEAS::_sample_bounds
( std::ostream& os )
{
  _vCONSAM.push_back( std::vector<double>( _nu ) );
  for( size_t i=0; i<_nu; i++ )
    _vCONSAM.back()[i] = _nestLB(i) + ( _nestUB(i) - _nestLB(i) ) * _gen();
    //_vCONSAM.back()[i] = _nestLB(i) + ( _nestUB(i) - _nestLB(i) ) * arma::randu();
}

inline
bool
NSFEAS::_update_nest
( std::multimap<double,std::tuple<double const*,double,double>> const& points, 
  double const& factor, std::ostream& os )
{
  // Gather data points
  _nestData.set_size( _nu, points.size() );
  size_t c=0;
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Live data points:\n" << std::right << std::scientific << std::setprecision(5);
#endif
  for( auto const& [lkh,pcon] : points ){
    double* mem = _nestData.colptr( c++ );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
    std::cout << std::setw(12) << lkh << ":";
#endif
    for( size_t i=0; i<_nu; ++i ){
      mem[i] = std::get<0>(pcon)[i]; 
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << std::setw(12) << mem[i];
#endif
    }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
    std::cout << std::endl;
#endif
  }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Live data points:\n" << _nestData.t();
#endif

  // Nest bounds
  _nestLB = arma::min( _nestData, 1 );
  _nestUB = arma::max( _nestData, 1 );
  arma::vec corr = (_nestUB - _nestLB) * (0.5*factor);
  _nestLB -= corr;
  _nestUB += corr;
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << std::right << std::scientific << std::setprecision(5);
  for( size_t i=0; i<_nu; ++i ){
    std::cout << std::setw(13) << _nestLB(i)
              << std::setw(13) << _nestUB(i);
  }
  std::cout << std::endl;
#endif
  _nestLB = arma::max( _nestLB, arma::vec( _vCONLB.data(), _nu, false ) );
  _nestUB = arma::min( _nestUB, arma::vec( _vCONUB.data(), _nu, false ) );

  // Nest centre
  _nestCentre = arma::mean( _nestData, 1 );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Data centre:" << _nestCentre.t();
#endif

  // Nest shape
  arma::mat covData = arma::cov( _nestData.t(), 1 );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Data covariance:\n" << covData;
#endif
  if( !arma::chol( _nestShape, covData, "lower" ) ){
    if( options.DISPLEVEL )
      std::cout << "Cholesky factorization failed, rank = " << arma::rank( covData ) << std::endl;
    _nestCentre = 0.5 * ( _nestLB + _nestUB );
    _nestShape  = arma::diagmat( (0.5*std::sqrt(_nu)) * ( _nestUB - _nestLB ) );
    //return false;
  }
  else{
    _nestData.each_col() -= _nestCentre;

    arma::vec covMag( _nestData.n_cols );
    c=0;
    _nestData.each_col(
      [&]( arma::vec const& v )
      { arma::vec x(_nu); arma::solve( x, arma::trimatl(_nestShape), v ); covMag(c++) = arma::norm( x, 2 ); }
    );

#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
    std::cout << "Magnification:\n" << arma::max( covMag ) << std::endl; //covMag;
#endif

    _nestShape *= arma::max( covMag ) * (1+factor);

#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
    std::cout << "   " << std::right << std::scientific << std::setprecision(5);
    for( size_t i=0; i<_nu; ++i ){
      std::cout << std::setw(13) << _nestCentre(i)-_nestShape(i,i)
                << std::setw(13) << _nestCentre(i)+_nestShape(i,i);
    }
    std::cout << std::endl;

    arma::vec covChk( _nestData.n_cols );
    c=0;
    _nestData.each_col(
      [&]( arma::vec const& v )
      { arma::vec x(_nu); arma::solve( x, arma::trimatl(_nestShape), v ); covChk(c++) = arma::norm( x, 2 ); }
    );
    std::cout << "Check: " << arma::max( covChk ) << std::endl;
#endif
  }

  return true;
}

inline
bool
NSFEAS::_sample_nest
( bool const bndcheck, std::ostream& os )
{
  // Compare nest volume to box domain volume
  double logvoldom = 0., logvolnest = 0.;
  for( size_t i=0; i<_nu; ++i ){
    if( _nestUB(i) - _nestLB(i) > DBL_EPSILON )
      logvoldom += std::log( _nestUB(i) - _nestLB(i) );
    else
      logvoldom += std::log( DBL_EPSILON );
    if( _nestShape(i,i) > DBL_EPSILON )
      logvolnest += std::log( _nestShape(i,i) );
    else
      logvolnest += std::log( DBL_EPSILON );
  }

  // Sample domain instead of nest
  if( logvoldom <= logvolnest ){
    _sample_bounds( os );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
    std::cout << "Sampling of _nu-dimensional box domain:\n";
    std::cout << "Lower range: " << _nestLB.t();
    std::cout << "Upper range: " << _nestUB.t();
    std::cout << arma::rowvec( _vCONSAM.back().data(), _nu, false );
#endif
    return true;
  }
  
  // Generate point in _nu-dimensional unit hyperball
  //for( unsigned i=0; i<500; ++i ){
    //arma::vec vSAM = arma::normalise( arma::randn(_nu), 2 ) * std::pow( _gen1d(), 1/(double)_nu );
    arma::vec vSAM = arma::normalise( arma::randn(_nu), 2 ) * std::pow( arma::randu(), 1/(double)_nu );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
    std::cout << "Sampling of _nu-dimensional unit hyperball:\n";
    std::cout << vSAM.t();
#endif
  //}

#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Nest Centre:\n" << _nestCentre.t();
  std::cout << "Nest Shape:\n" << _nestShape;
  std::cout << "Nest Diameter:\n" << arma::diagvec(_nestShape).t();
#endif
  auto vBALL = vSAM;
  vSAM = _nestCentre + _nestShape * vSAM;
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << vSAM.t();
#endif
  for( size_t i=0; i<_nu && bndcheck; ++i )
    if( vSAM[i] < _vCONLB[i] || vSAM[i] > _vCONUB[i] ){ return false; }

  _vCONSAM.push_back( std::vector<double>( vSAM.memptr(), vSAM.memptr()+_nu ) );
  return true;
}

inline
bool
NSFEAS::_feasible
( double const& crit, std::ostream& os )
const
{
  switch( options.FEASCRIT ){
    case Options::VAR:
    case Options::CVAR: if( crit <= 0 ) return true;
  }
  return false;
}

inline
int
NSFEAS::_terminate
( size_t const it, bool const feas, std::chrono::time_point<std::chrono::system_clock> const& tstart,
  std::ostream& os )
const
{
  if( feas ){
    switch( options.FEASCRIT ){
      case Options::VAR:
      case Options::CVAR: if( _liveFEAS.rbegin()->first <= 0 ) return( 1 ); // completion for feasibility
    }
  }
  else{
    switch( options.LKHCRIT ){
      case Options::VAR:
      case Options::CVAR: if( _liveNest <= _totalMass + std::log(options.LKHTOL) ) return( 1 ); // completion for likelihood
    }
  }

  if( options.MAXCPU > 0. && stats.to_time( stats.lapse( tstart ) ) < options.MAXCPU ) return( 2 ); // interruption
  if( options.MAXITER > 0 && it >= options.MAXITER ) return( 2 ); // interruption

  return( 0 ); // continuation;
}

inline
int
NSFEAS::_sample_feas
( double& factor, std::chrono::time_point<std::chrono::system_clock> const& tstart, std::ostream& os )
{
  if( options.DISPLEVEL ){
    std::cout << std::endl
              << std::setw(7)  << "Iterate"
              << std::setw(15) << "Contour"
              << std::setw(6)  << "#Feas"
              << std::setw(9)  << "#Dead"
              << std::setw(15) << "Factor"
              << std::endl
              << std::setw(7+15+6+9+15) << std::setfill('-') << "-"
              << std::setfill(' ')
              << std::endl;
    }

  // Iteration for feasibility phase
  int flag = _terminate( 0, true, tstart, os );
  for( stats.iter=0; !flag; ++stats.iter ){

    // Update ellipsoidal nest
    if( !_update_nest( _liveFEAS, factor, os ) )
      return( 3 );

    if( options.DISPLEVEL && (!options.DISPITER || !(stats.iter % options.DISPITER)) ){
      std::cout << std::setw(7) << stats.iter
                << std::scientific << std::setprecision(4) << std::setw(15) << _liveFEAS.rbegin()->first
                << std::setw(6) << stats.numfeas
                << std::setw(9) << _deadFEAS.size()
                << std::scientific << std::setprecision(4) << std::setw(15) << factor;
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << std::right << std::scientific << std::setprecision(5);
      for( size_t i=0; i<_nu; ++i ){
        std::cout << std::setw(12) << _nestCentre(i)-_nestShape(i,i)
                  << std::setw(12) << _nestCentre(i)+_nestShape(i,i)
                  << std::setw(12) << _nestLB(i)
                  << std::setw(12) << _nestUB(i);
      }
#endif
      std::cout << std::endl;
    }

    // Sample proposals in nest
    size_t const NSAM0 = _vCONSAM.size();
    for( size_t s=0; !flag && _vCONSAM.size() < NSAM0 + options.NUMPROP; ++s ){
      _sample_nest( true, os );
    }

    // Evaluate proposals in parallel
#ifndef MAGNUS__NSFEAS_THREAD_FOREACH_SAMPLE
    if( _dag->options.MAXTHREAD != 1 ){
#else
    if( _vPARVAL.size() <= 1 && _dag->options.MAXTHREAD != 1 ){
#endif
      try{
        auto itSAM0 = _vCONSAM.begin(); std::advance( itSAM0, NSAM0 );
        //_vCONPROP.assign( itSAM0, _vCONSAM.end() );
        // Move elements to _vCONPROP to avoid unnecessary memory allocation
        _vCONPROP.assign( std::make_move_iterator( itSAM0 ), std::make_move_iterator( _vCONSAM.end() ) );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        for( auto const& dCON : _vCONPROP )
          std::cout << std::right << std::scientific << std::setprecision(5)
                    << arma::rowvec( const_cast<double*>(dCON.data()), _nu, false );
#endif
        _vCONSAM.erase( itSAM0, _vCONSAM.end() );
        _vVALSAM.resize( _vCONPROP.size() );
        _dag->veval( _sgVAL, _wkD, _wkVAL, _vVAL, _vVALSAM, _vCON, _vCONPROP );
        // Move elements back into _vCONSAM
        _vCONSAM.insert( _vCONSAM.end(), std::make_move_iterator( _vCONPROP.begin() ), std::make_move_iterator( _vCONPROP.end() ) );
      }
      catch(...){
        return FAILURE;
      }
    }
    
    // Evaluate proposals sequentially
    else{
      _vVALSAM.clear();
      _vVALSAM.reserve( options.NUMPROP );
      for( size_t s=0; s<options.NUMPROP; ++s ){
        std::vector<double> const& dCON = _vCONSAM[NSAM0+s];
        // Evaluate candidate
        try{
          _dag->eval( _sgVAL, _wkD, _vVAL, _dVAL, _vCON, dCON );
        }      
        catch(...){
          ++stats.numerr;
          continue;
        }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        std::cout << (_dVAL[options.FEASCRIT] <= _liveFEAS.rbegin()->first)
                  << std::right << std::scientific << std::setprecision(5) << std::setw(12) << _dVAL[options.FEASCRIT]
                  << arma::rowvec( const_cast<double*>(dCON.data()), _nu, false );
#endif
        _vVALSAM.push_back( _dVAL );
      }
    }
    
    // Update live set
    for( size_t s=0; s<_vVALSAM.size(); ++s, stats.numfct+=(_vPARVAL.size()>0?_vPARVAL.size():1) ){
      _dVAL = _vVALSAM[s];
      std::vector<double> const& dCON = _vCONSAM[NSAM0+s];
      // Check for failure during parallel evaluation
#ifndef MAGNUS__NSFEAS_THREAD_FOREACH_SAMPLE
      if( _dag->options.MAXTHREAD != 1 && !_dag->veval_err().at(s) ){
#else
      if( _vPARVAL.size() <= 1 && _dag->options.MAXTHREAD != 1 && !_dag->veval_err().at(s) ){
#endif
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        std::cout << "Failed "
                  << std::right << std::scientific << std::setprecision(5)
                  << arma::rowvec( const_cast<double*>(dCON.data()), _nu, false );
#endif
        ++stats.numerr;
        continue;
      }

      // Check for improvement for insertion
      if( _dVAL[options.FEASCRIT] > _liveFEAS.rbegin()->first ){
        _discardFEAS.insert( { _dVAL[options.FEASCRIT], { dCON.data(), _dVAL[0], _dVAL[2+(int)options.LKHCRIT] } } );
        continue;
      }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << "Replace live point" << std::endl;
#endif
      if( _feasible( _dVAL[options.FEASCRIT], os ) ) ++stats.numfeas;
      _liveFEAS.insert( { _dVAL[options.FEASCRIT], { dCON.data(), _dVAL[0], _dVAL[2+(int)options.LKHCRIT] } } );
      
      // Remove previous worse and add to dead points
      _nestMass = std::exp( -(double)_deadFEAS.size() / (double)options.NUMLIVE );
      _deadFEAS.insert( *_liveFEAS.rbegin() );
      auto itlast = _liveFEAS.end();
      _liveFEAS.erase( --itlast );

      // Update ellipsoid magnification factor
      factor = options.ELLMAG * std::pow( _nestMass, options.ELLRED );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << "Ellipsoid magnification factor: " << factor << std::endl;
#endif

      // Update termination flag
      flag = _terminate( stats.iter, true, tstart, os );
    }
  }

  if( options.DISPLEVEL ){
    std::cout << std::setw(7) << stats.iter+1
              << std::scientific << std::setprecision(4) << std::setw(15) << _liveFEAS.rbegin()->first
              << std::setw(6) << stats.numfeas
              << std::setw(9) << _deadFEAS.size()
              << std::scientific << std::setprecision(4) << std::setw(15) << factor
              << std::endl;
  }

  return flag;
}

inline
int
NSFEAS::_sample_lkh
( double& factor, std::chrono::time_point<std::chrono::system_clock> const& tstart, std::ostream& os )
{
  if( options.DISPLEVEL ){
    std::cout << std::endl
              << std::setw(7)  << "Iterate"
              << std::setw(15) << "Contour"
              << std::setw(9)  << "#Dead"
              << std::setw(15) << "Mass"
              << std::setw(15) << "Nest"
              << std::setw(15) << "Factor"
              << std::endl
              << std::setw(7+15+9+15+15+15) << std::setfill('-') << "-"
              << std::setfill(' ')
              << std::endl;
    }

  // Iteration for likelihood phase
  _nestMass = 1.;
  int flag = 0;
  for( ; !flag; ++stats.iter ){

    if( options.DISPLEVEL && (!options.DISPITER || !(stats.iter % options.DISPITER)) ){
      std::cout << std::setw(7) << stats.iter
                << std::scientific << std::setprecision(4) << std::setw(15) << _liveLKH.begin()->first
                << std::setw(9) << _deadLKH.size()
                << std::scientific << std::setprecision(4) << std::setw(15) << _totalMass
                << std::scientific << std::setprecision(4) << std::setw(15) << _liveNest
                << std::scientific << std::setprecision(4) << std::setw(15) << factor
                << std::endl;
    }

    // Update ellipsoidal nest
    if( !_update_nest( _liveLKH, factor, os ) )
      return( 3 );

    // Sample proposals in nest
    size_t const NSAM0 = _vCONSAM.size();
    for( size_t s=0; !flag && _vCONSAM.size() < NSAM0 + options.NUMPROP; ++s ){
      _sample_nest( true, os );
    }

    // Evaluate proposals in parallel
#ifndef MAGNUS__NSFEAS_THREAD_FOREACH_SAMPLE
    if( _dag->options.MAXTHREAD != 1 ){
#else
    if( _vPARVAL.size() <= 1 && _dag->options.MAXTHREAD != 1 ){
#endif
      try{
        auto itSAM0 = _vCONSAM.begin(); std::advance( itSAM0, NSAM0 );
        //_vCONPROP.assign( itSAM0, _vCONSAM.end() );
        // Move elements to _vCONPROP to avoid unnecessary memory allocation
        _vCONPROP.assign( std::make_move_iterator( itSAM0 ), std::make_move_iterator( _vCONSAM.end() ) );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        for( auto const& dCON : _vCONPROP )
          std::cout << std::right << std::scientific << std::setprecision(5)
                    << arma::rowvec( const_cast<double*>(dCON.data()), _nu, false );
#endif
        _vCONSAM.erase( itSAM0, _vCONSAM.end() );
        _vVALSAM.resize( _vCONPROP.size() );
        _dag->veval( _sgVAL, _wkD, _wkVAL, _vVAL, _vVALSAM, _vCON, _vCONPROP );
        // Move elements back into _vCONSAM
        _vCONSAM.insert( _vCONSAM.end(), std::make_move_iterator( _vCONPROP.begin() ), std::make_move_iterator( _vCONPROP.end() ) );
      }
      catch(...){
        return FAILURE;
      }
    }
    
    // Evaluate proposals sequentially
    else{
      _vVALSAM.clear();
      _vVALSAM.reserve( options.NUMPROP );
      for( size_t s=0; s<options.NUMPROP; ++s ){
        std::vector<double> const& dCON = _vCONSAM[NSAM0+s];
        // Evaluate candidate
        try{
          _dag->eval( _sgVAL, _wkD, _vVAL, _dVAL, _vCON, dCON );
        }      
        catch(...){
          ++stats.numerr;
          continue;
        }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        std::cout << (_dVAL[options.FEASCRIT] <= _liveFEAS.rbegin()->first)
                  << std::right << std::scientific << std::setprecision(5) << std::setw(12) << _dVAL[options.FEASCRIT]
                  << arma::rowvec( const_cast<double*>(dCON.data()), _nu, false );
#endif
        _vVALSAM.push_back( _dVAL );
      }
    }
    
    // Update live set
    for( size_t s=0; s<_vVALSAM.size(); ++s, stats.numfct+=(_vPARVAL.size()>0?_vPARVAL.size():1) ){
      _dVAL = _vVALSAM[s];
      std::vector<double> const& dCON = _vCONSAM[NSAM0+s];
      // Check for failure during parallel evaluation
#ifndef MAGNUS__NSFEAS_THREAD_FOREACH_SAMPLE
      if( _dag->options.MAXTHREAD != 1 && !_dag->veval_err().at(s) ){
#else
      if( _vPARVAL.size() <= 1 && _dag->options.MAXTHREAD != 1 && !_dag->veval_err().at(s) ){
#endif
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        std::cout << "Failed "
                  << std::right << std::scientific << std::setprecision(5)
                  << arma::rowvec( const_cast<double*>(dCON.data()), _nu, false );
#endif
        ++stats.numerr;
        continue;
      }

      // Check for feasibility and improvement for insertion
      if( ( _ng && !_feasible( _dVAL[options.FEASCRIT], os ) )
       || _dVAL[2+(int)options.LKHCRIT] < _liveLKH.begin()->first ){
        continue;
      }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << "Replace live point" << std::endl;
#endif

      // Update live set, dead set, and probability mass
      auto itdead = _deadLKH.insert( *_liveLKH.begin() );
      _liveLKH.insert( { _dVAL[2+(int)options.LKHCRIT], { dCON.data(), _dVAL[0], _dVAL[options.FEASCRIT] } } );
      _liveLKH.erase( _liveLKH.begin() );
      double nestMass = std::exp( -(double)_deadLKH.size() / (double)options.NUMLIVE );
      _deadShell = _liveLKH.begin()->first + std::log( _nestMass - nestMass );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << "_totalMass: " << log_sum_exp( _totalMass, _deadShell ) << " = "<< _totalMass << " + " << log_sum_exp( _totalMass, _deadShell ) - _totalMass << std::endl;
      assert( _totalMass <= log_sum_exp( _totalMass, _deadShell ) );
#endif
      _totalMass = ( _nestMass < 1. ? log_sum_exp( _totalMass, _deadShell ): _deadShell );
      _liveNest  = log_sum_exp( _liveLKH ) - std::log( options.NUMLIVE ) + std::log( nestMass );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << "_nestMass:  " << _nestMass  << " >> " << nestMass << std::endl;
      std::cout << "_deadShell: " << _deadShell << " = " << _liveLKH.begin()->first << " + " << std::log( _nestMass - nestMass ) << std::endl;
      std::cout << "_liveNest:  " << _liveNest  << " = " << log_sum_exp( _liveLKH ) << " - " << std::log( options.NUMLIVE ) << " + " <<  std::log( nestMass ) << std::endl;
      int dum; std::cout << "ENTER 1 TO CONTINUE\n"; std::cin >> dum;
#endif
      _nestMass  = nestMass;
      std::get<2>( itdead->second ) = _deadShell;

      // Update ellipsoid magnification factor
      factor = options.ELLMAG * std::pow( _nestMass, options.ELLRED );   
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
      std::cout << "Ellipsoid magnification factor: " << factor << std::endl;
#endif

      // Update termination flag
      flag = _terminate( stats.iter, false, tstart, os );
    }
  }

  if( options.DISPLEVEL ){
    std::cout << std::setw(7) << stats.iter+1
              << std::scientific << std::setprecision(4) << std::setw(15) << _liveLKH.begin()->first
              << std::setw(9) << _deadLKH.size()
              << std::scientific << std::setprecision(4) << std::setw(15) << _totalMass
              << std::scientific << std::setprecision(4) << std::setw(15) << _liveNest
              << std::scientific << std::setprecision(4) << std::setw(15) << factor
              << std::endl;
  }

  // Update probability mass and weights
  _totalMass = log_sum_exp( _totalMass, _liveNest );
  for( auto& [lkh,tup] : _liveLKH )
    std::get<2>( tup ) = _liveNest - std::log( options.NUMLIVE ) - _totalMass;
  for( auto& [lkh,tup] : _deadLKH )
    std::get<2>( tup ) -= _totalMass;

  if( options.DISPLEVEL )
    std::cout << std::setw(9+13) << std::setfill('-') << "-"
              << std::setfill(' ')
              << std::endl
              << "Evidence:"<< std::scientific << std::setprecision(4) << std::setw(13) << _totalMass
              << std::endl;

  return flag;
}

inline
int
NSFEAS::sample
( std::vector<double> const& vcst, bool const reset, std::ostream& os )
{

  if( !_is_setup )          throw Exceptions( Exceptions::NOSETUP );
  if( !options.NUMLIVE )    throw Exceptions( Exceptions::BADSIZE );
  auto&& tstart = stats.start();

  // Update constants
  if( vcst.size() && vcst.size() == _nc ) _vCSTVAL = vcst;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Set evaluation functions
  FFNSamp OpEval( options.FEASTHRES, options.LKHTHRES );
  FFVar** ppVAL = OpEval( _dag, &_vCON, _vPAR.size()?&_vPAR:nullptr, _vCST.size()?&_vCST:nullptr,
                          _vCTR.size()?&_vCTR:nullptr, _vLKH.size()?&_vLKH[0]:nullptr,
                          _vPARVAL.size()?&_vPARVAL:nullptr, _vPARWEI.size()?&_vPARWEI:nullptr,
                          _vCSTVAL.size()?&_vCSTVAL:nullptr );
  _vVAL  = { *ppVAL[0], *ppVAL[1], *ppVAL[2], *ppVAL[3], *ppVAL[4] };
  _sgVAL = _dag->subgraph( _vVAL );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  _dag->output( _sgVAL );
#endif

  // Reset parallel worker
  _wkVAL.clear();

  int flag = 1;
  double factor = options.ELLMAG;

  if( reset ){
    stats.reset();

    // Reset various sets
    _liveFEAS.clear();
    _deadFEAS.clear();
    _discardFEAS.clear();
    _vCONSAM.clear();
    _vCONSAM.reserve( options.NUMLIVE );

    // Reset Sobol sampler
    _gen.engine().seed( 0 );
    _gen1d.engine().seed( 0 );

    // Evaluate feasibility/likelihood criteria for every live point
    if( options.DISPLEVEL )
      os << "\n** INITIALIZING LIVE POINTS " << std::flush;

    flag = _sample_ini( options.NUMLIVE, tstart, os );

    if( options.DISPLEVEL )
      os << "(" << _liveFEAS.size() << (flag? ")": ") INTERRUPTED AT") 
         << std::right << std::fixed << std::setprecision(2)
         << std::setw(10) << stats.to_time( stats.lapse( tstart ) ) << " SEC"
         << std::endl;
    if( !flag ) return STATUS::INTERRUPT;
  }
  
  else{
    stats.numfeas = 0;
    for( auto const& [crit,pcon] : _liveFEAS ){ // to update feasible point count
      if( _feasible( crit, os ) ) ++stats.numfeas;
    }
  }

  _liveLKH.clear();
  _deadLKH.clear();
  _liveNest = _deadShell = _totalMass = 0.;
  
  if( _ng ){
    // Run feasibility sampler
    flag = _sample_feas( factor, tstart, os );
  }

  if( _nl ){
    // Initialise live nest for likelihood
    for( auto const& [feas,pcon] : _liveFEAS )
      _liveLKH.insert( { std::get<2>(pcon), { std::get<0>(pcon), std::get<1>(pcon), feas  } } );
    // Run likelihood sampler
    flag = _sample_lkh( factor, tstart, os );
  }
  
  // Termination
  stats.walltime += stats.lapse( tstart );
  switch( flag ){
    default:
    case 1: return STATUS::NORMAL;
    case 2: return STATUS::INTERRUPT;
    case 3: return STATUS::FAILURE;
  }
}
/*
inline
bool
EXPDES::file_export
( std::string const& name )
{
  auto itPARVAL = _vPARVAL.cbegin();
  for( size_t s=0; s<_vPARVAL.size(); ++s, ++itPARVAL ){
    std::ofstream ofile( name + "_" + std::to_string(s) + ".log" );
    if( !ofile ) return false;
    
    ofile << std::scientific << std::setprecision(6);
    for( size_t k=0; k<_vCONSAM.size(); ++k ){
      for( size_t i=0; i<itPARVAL->size(); ++i )
        ofile << (*itPARVAL)[i] << "  ";

      for( size_t i=0; i<_vCONSAM[k].size(); ++i )
        ofile << _vCONSAM[k][i] << "  ";

      ofile << ( _EOpt.count(k)? _EOpt[k]: 0 ) << "  ";

      switch( options.CRITERION ){
        case BRISK:
        case ODIST:
          for( size_t i=0; i<_vOUTSAM[s][k].n_rows; ++i )
            ofile << _vOUTSAM[s][k](i) << "  ";
          break;
          
        case AOPT:
        case DOPT:
        case EOPT:
        default:
          for( size_t i=0; i<_vFIMSAM[s][k].n_rows; ++i )
            for( size_t j=i; j<_vFIMSAM[s][k].n_cols; ++j )
              ofile << _vFIMSAM[s][k](i,j) << "  ";
          break;
      }
      ofile << std::endl;
    }
  }
  return true;
}
*/

} // end namespace mc

#endif

// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_NSFEAS Model-based Feasibility Analysis using Nested Sampling
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 1.0
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

  //! @brief flag indicating if problem is setup before sampling
  bool                                         _is_setup;

  //! @brief DAG of model
  DAG*                                         _dag;

  //! @brief local copy of model parameters
  std::vector<FFVar>                           _vPAR;

  //! @brief local copy of experimental controls
  std::vector<FFVar>                           _vCON;

  //! @brief vector of sampled control candidates
  std::vector<std::vector<double>>             _vCONSAM;

  //! @brief local copy of model outputs
  std::vector<FFVar>                           _vCTR;

  //! @brief likelihood criteria
  std::vector<FFVar>                           _vLKH;

  //! @brief likelihood values
  std::vector<double>                          _dLKH;

  //! @brief likelihood subgraph
  FFSubgraph                                   _sgLKH;

  //! @brief work array for constraint evaluations
  std::vector<double>                          _wkLKH;

  //! @brief data matrix for current nest
  arma::mat                                    _nestData;

  //! @brief centre vector of current nest
  arma::vec                                    _nestCentre;

  //! @brief shape matrix for current nest
  arma::mat                                    _nestShape;

  //! @brief map of live points
  std::multimap<double,double const*>          _liveCON;

  //! @brief map of dead points
  std::multimap<double,double const*>          _deadCON;

  //! @brief map of discarded points
  std::multimap<double,double const*>          _discardCON;

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
    : _gen      ( sobol64(1), boost::uniform_01<double>() ),
      _is_setup ( false ),
      _dag      ( nullptr )
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
        NUMLIVE    = std::pow(2,8);
        NUMPROP    = std::pow(2,4);
        ELLCONF    = 0.99;
        ELLMAG     = 0.30; // Feroz et al. (2009)
        ELLRED     = 0.20; // Feroz et al. (2009)
        MAXITER    = 0;
        MAXCPU     = 0;
        DISPLEVEL  = 1;
      }

    //! @brief Assignment operator
    Options& operator=
      ( Options const& options )
      {
        FEASCRIT   = options.FEASCRIT;
        FEASTHRES  = options.FEASTHRES;
        NUMLIVE    = options.NUMLIVE;
        NUMPROP    = options.NUMPROP;
        ELLCONF    = options.ELLCONF;
        ELLMAG     = options.ELLMAG;
        ELLRED     = options.ELLRED;
        MAXITER    = options.MAXITER;
        MAXCPU     = options.MAXCPU;
        DISPLEVEL  = options.DISPLEVEL;
        return *this;
      }

      //! @brief Enumeration for feasibility criterion
      enum TYPE{
        PR=0,   //!< Infeasibility probability
        VAR,    //!< Value-at-risk, VaR
        CVAR    //!< Conditional value-at-risk, CVaR
      };

    //! @brief Selected feasibility criterion
    TYPE                     FEASCRIT;
    //! @brief Percentile infeasibility threshold
    double                   FEASTHRES;
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
    //! @brief maximal walltime
    double                   MAXCPU;
    //! @brief Verbosity level
    int                      DISPLEVEL;
  } options;

  //! @brief NSFEAS solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for NSFEAS exception handling
    enum TYPE{
      BADSIZE=0,	//!< Inconsistent dimensions
      NOSETUP,		//!< Problem was not setup
      INTERN=-33	//!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr(){ return _ierr; }
    //! @brief Inline function returning the error description
    std::string what(){
      switch( _ierr ){
        case BADSIZE:
          return "NSFEAS::Exceptions  Inconsistent dimensions";
        case NOSETUP:
          return "NSFEAS::Exceptions  Problem not setup";
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
        numlkh = numerr = numfeas = 0; }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "#  LIKELIHOOD:" << std::right << std::setw(10) << numlkh << " / "
                               << std::left  << std::setw(10) << numerr << std::endl
           << "#  WALL TIME: " << std::right << std::setw(10) << to_time( walltime ) << " SEC" << std::endl
           << std::endl; }
    //! @brief Total likelihood evaluations
    size_t numlkh;
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
    ( bool const reset=true, std::ostream& os=std::cout );
/*
  //! @brief Export effort, support and fim to file
  bool file_export
    ( std::string const& name );
*/
  //! @brief Retrieve live points
  std::multimap<double,double const*> const& live_points
    ()
    const
    { return _liveCON; }

  //! @brief Retrieve dead points
  std::multimap<double,double const*> const& dead_points
    ()
    const
    { return _deadCON; }

  //! @brief Retrieve discarded points
  std::multimap<double,double const*> const& discard_points
    ()
    const
    { return _discardCON; }

protected:

  //! @brief Append candidate points inside bounds or current nest
  bool _sample_ini
    ( size_t const nprop, std::chrono::time_point<std::chrono::system_clock> const& tstart,
      std::ostream& os );

  //! @brief Update current nest
  bool _update_nest
    ( double const& f, std::ostream& os );

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
    ( size_t const it, std::chrono::time_point<std::chrono::system_clock> const& tstart,
      std::ostream& os )
    const;
};

inline
bool
NSFEAS::setup
()
{
  stats.reset();
  _is_setup = false;

  if( !_ng || !_nc )
    throw Exceptions( Exceptions::BADSIZE );

  try{
    delete _dag; _dag = new DAG;
    _dag->options = BASE_MBFA::_dag->options;

    _vCON.resize( _nc );
    _dag->insert( BASE_MBFA::_dag, _nc, BASE_MBFA::_vCON.data(), _vCON.data() );
    _vPAR.resize( _np );
    _dag->insert( BASE_MBFA::_dag, _np, BASE_MBFA::_vPAR.data(), _vPAR.data() );
    _vCTR.resize( _ng );
    _dag->insert( BASE_MBFA::_dag, _ng, BASE_MBFA::_vCTR.data(), _vCTR.data() );

#ifdef MAGNUS__NSFEAS_SETUP_DEBUG
    auto sgCTR = _dag->subgraph( _ng, _vCTR.data() );
    std::vector<FFExpr> exCTR = FFExpr::subgraph( _dag, sgCTR ); 
    for( size_t i=0; i<_ng; ++i )
      std::cout << "CTR[" << i << "] = " << exCTR[i] << std::endl;
#endif

    FFFeas OpFeas( options.FEASTHRES );
    FFVar** ppLKH = OpFeas( _dag, &_vCON, _vPAR.size()?&_vPAR:nullptr, &_vCTR,
                            _vPARVAL.size()?&_vPARVAL:nullptr, _vPARWEI.size()?&_vPARWEI:nullptr );
    _vLKH  = { *ppLKH[0], *ppLKH[1], *ppLKH[2] };
    _sgLKH = _dag->subgraph( _vLKH );
#ifdef MAGNUS__NSFEAS_SETUP_DEBUG
    _dag->output( _sgLKH );
#endif
  }
  catch(...){
    return false;
  }
  
  // Sobol sampler
  sobol64 eng( _nc );
  _gen.engine() = eng;

  _is_setup = true;
  return true;
}

inline
bool
NSFEAS::_sample_ini
( size_t const nprop, std::chrono::time_point<std::chrono::system_clock> const& tstart,
  std::ostream& os )
{
  // Sample nest
  size_t const NLIVE0 = _liveCON.size();
  for( size_t s=0; _liveCON.size() < NLIVE0 + nprop; ++s, ++stats.numlkh ){ // to account for failures
    // Sample new point
    _sample_bounds( os );
    std::vector<double> const& dCON = _vCONSAM.back();

    // Evaluate new point
    try{
      _dag->eval( _sgLKH, _wkLKH, _vLKH, _dLKH, _vCON, dCON );
      _liveCON.insert( { _dLKH[options.FEASCRIT], dCON.data() } );
      if( _feasible( _dLKH[options.FEASCRIT], os ) ) ++stats.numfeas;
    }
    catch(...){
      ++stats.numerr;
      continue;
    }

    if( options.DISPLEVEL > 1  && _liveCON.size() && !(_liveCON.size()%20) )
      os << "." << std::flush;

    if( options.MAXCPU > 0 && stats.to_time( stats.lapse( tstart ) ) >= options.MAXCPU )
      return false;
  }
  return true;
}

inline
void
NSFEAS::_sample_bounds
( std::ostream& os )
{
  _vCONSAM.push_back( std::vector<double>( _nc ) );
  for( size_t i=0; i<_nc; i++ )
    _vCONSAM.back()[i] = _vCONLB[i] + ( _vCONUB[i] - _vCONLB[i] ) * _gen();
    //_vCONSAM.back()[i] = _vCONLB[i] + ( _vCONUB[i] - _vCONLB[i] ) * arma::randu();
}

inline
bool
NSFEAS::_update_nest
( double const& f, std::ostream& os )
{
  // Gather data points
  _nestData.set_size( _nc, _liveCON.size() );
  size_t c=0;
  for( auto const& [lkh,pcon] : _liveCON ){
    double* mem = _nestData.colptr( c++ );
    for( size_t i=0; i<_nc; ++i ) mem[i] = pcon[i]; 
  }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Live data points:\n" << _nestData.t();
#endif

  // Nest centre
  _nestCentre = arma::mean( _nestData, 1 );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Data centre:\n" << _nestCentre.t();
#endif

  // Nest shape matrix
  boost::math::chi_squared dist( _nc );
  double const Chi2Crit = boost::math::quantile( dist, options.ELLCONF );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Critical chi-squared " << _nc << " DoF, 99%: " << Chi2Crit << std::endl;
#endif
  arma::mat covData = arma::cov( _nestData.t(), 1 );
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << "Data covariance:\n" << covData;
#endif
  return arma::chol( _nestShape, covData *= Chi2Crit * (1+f) );
}

inline
bool
NSFEAS::_sample_nest
( bool const bndcheck, std::ostream& os )
{
  // Generate point in _nc-dimensional unit hyperball
  //thread_local static sobol64 eng( 1 );
  //thread_local static qrgen gen( eng, boost::uniform_01<double>() );
  //gen.engine().seed( 0 );
  //for( unsigned i=0; i<500; ++i ){
    //arma::vec vRan = arma::normalise( arma::randn(_nc), 2 ) * std::pow( gen(), 1/(double)_nc );
    arma::vec vSAM = arma::normalise( arma::randn(_nc), 2 ) * std::pow( arma::randu(), 1/(double)_nc );
//#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
//    std::cout << "Sampling of _nc-dimensional unit hyperball:\n";
//    std::cout << vSAM.t();
//#endif
  //}

//#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
//  std::cout << "Nest Centre:\n" << _nestCentre.t();
//  std::cout << "Nest Shape:\n" << _nestShape;
//#endif
  vSAM = _nestCentre + _nestShape * vSAM;
  for( size_t i=0; i<_nc && bndcheck; ++i )
    if( vSAM[i] < _vCONLB[i] || vSAM[i] > _vCONUB[i] ) return false;
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
  std::cout << vSAM.t();
#endif

  _vCONSAM.push_back( std::vector<double>( vSAM.memptr(), vSAM.memptr()+_nc ) );
  return true;
}

inline
bool
NSFEAS::_feasible
( double const& crit, std::ostream& os )
const
{
  switch( options.FEASCRIT ){
    case Options::PR:   if( crit <= 1-options.FEASTHRES ) return true;
    case Options::VAR:
    case Options::CVAR: if( crit <= 0 ) return true;
  }
  return false;
}

inline
int
NSFEAS::_terminate
( size_t const it, std::chrono::time_point<std::chrono::system_clock> const& tstart,
  std::ostream& os )
const
{
  switch( options.FEASCRIT ){
    case Options::PR:   if( _liveCON.rbegin()->first <= 1-options.FEASTHRES ) return 1; // completion
    case Options::VAR:
    case Options::CVAR: if( _liveCON.rbegin()->first <= 0 ) return 1; // completion
  }

  if( options.MAXCPU > 0. && stats.to_time( stats.lapse( tstart ) ) < options.MAXCPU ) return 2; // interruption
  if( options.MAXITER > 0 && it >= options.MAXITER ) return 2; // interruption

  return 0; // continuation;
}
  
inline
int
NSFEAS::sample
( bool const reset, std::ostream& os )
{

  if( !_is_setup )       throw Exceptions( Exceptions::NOSETUP );
  if( !options.NUMLIVE ) throw Exceptions( Exceptions::BADSIZE );
  auto&& tstart = stats.start();

  FFFeas::Confidence = options.FEASTHRES;
  int flag = 1;
  if( reset ){
    stats.reset();

    // Reset various sets
    _liveCON.clear();
    _deadCON.clear();
    _vCONSAM.clear();
    _vCONSAM.reserve( options.NUMLIVE );

    // Reset Sobol sampler
    _gen.engine().seed( 0 );

    // Compute constraint feasibility for every live point
    if( options.DISPLEVEL )
      os << "** INITIALIZING LIVE POINTS" << std::endl;

    flag = _sample_ini( options.NUMLIVE, tstart, os );

    if( options.DISPLEVEL )
      os << "(" << _liveCON.size() << (flag? ")": ") INTERRUPTED AT") 
         << std::right << std::fixed << std::setprecision(2)
         << std::setw(10) << stats.to_time( stats.lapse( tstart ) ) << " SEC"
         << std::endl;
    if( !flag ) return STATUS::INTERRUPT;
  }
  
  else{
    stats.numfeas = 0;
    for( auto const& [crit,pcon] : _liveCON ){ // to update feasible point count
      if( _feasible( crit, os ) ) ++stats.numfeas;
    }
  }
 
  // Main iteration
  flag = 0;
  double f = options.ELLMAG; //Z = 0., 
  for( size_t it=0; !flag; ++it ){

    // Update ellipsoidal nest
    if( !_update_nest( f, os ) )
      return STATUS::FAILURE;

    size_t const NSAM0 = _vCONSAM.size();
    for( size_t s=0; _vCONSAM.size() < NSAM0 + options.NUMPROP; ++s ){

      // Sample candidate in nest
      if( !_sample_nest( true, os ) ) continue; // New sample must be within bounds
      std::vector<double> const& dCON = _vCONSAM.back();
      ++stats.numlkh;
      
      // Evaluate candidate
      try{
        _dag->eval( _sgLKH, _wkLKH, _vLKH, _dLKH, _vCON, dCON );

        // Check for improvement for insertion
        if( _dLKH[options.FEASCRIT] > _liveCON.rbegin()->first ){
          _discardCON.insert( { _dLKH[options.FEASCRIT], dCON.data() } );
          continue;
        }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
        std::cout << "Replace live point" << std::endl;
#endif
        if( _feasible( _dLKH[options.FEASCRIT], os ) ) ++stats.numfeas;
        _liveCON.insert( { _dLKH[options.FEASCRIT], dCON.data() } );
      
        // Remove previous worse and add to dead points
        _deadCON.insert( *_liveCON.rbegin() );
        auto itlast = _liveCON.end();
        _liveCON.erase( --itlast );
        
        // Update evidence
        //Z += ...
      }
      catch(...){
        ++stats.numerr;
        continue;
      }
    }

    if( options.DISPLEVEL ){
      std::cout << std::setw(5) << it
                << std::scientific << std::setprecision(4) << std::setw(15) << _liveCON.rbegin()->first
                << std::setw(6) << stats.numfeas
                << std::setw(9) << _deadCON.size()
                << std::scientific << std::setprecision(4) << std::setw(15) << f
                << std::endl;
    }

    // Update ellipsoid magnification factor
    f = options.ELLMAG * std::pow( std::exp( -(_deadCON.size()-1e0)/options.NUMLIVE ), options.ELLRED );   
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
    std::cout << "Ellipsoid magnification factor: " << f << std::endl;
#endif

    // Update termination flag
    flag = _terminate( it, tstart, os );
  }

  stats.walltime += stats.lapse( tstart );
  switch( flag ){
    default:
    case 1: return STATUS::NORMAL;
    case 2: return STATUS::INTERRUPT;
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

// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_PAREST Parameter Estimation in Mathematical Models
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 0.2
\date 2025
\bug No known bugs.
*/

#ifndef MAGNUS__PAREST_HPP
#define MAGNUS__PAREST_HPP

#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/fisher_f.hpp>
using boost::math::normal;
using boost::math::chi_squared;
using boost::math::students_t;
using boost::math::fisher_f;
using boost::math::quantile;
using boost::math::cdf;

#ifdef MC__USE_PROFIL
 #include "mcprofil.hpp"
 typedef INTERVAL I;
#else
 #ifdef MC__USE_FILIB
  #include "mcfilib.hpp"
  typedef filib::interval<double,filib::native_switched,filib::i_mode_extended> I;
 #else
  #ifdef MC__USE_BOOST
   #include "mcboost.hpp"
   typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
   typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
   typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
   typedef boost::numeric::interval<double,T_boost_policy> I;
  #else
   #include "interval.hpp"
   typedef mc::Interval I;
  #endif
 #endif
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
#endif

#include "base_parest.hpp"
#include "nsfeas.hpp"

#include "fflin.hpp"
#include "ffode.hpp"
#include "ffest.hpp"

namespace mc
{
//! @brief C++ class for parameter estimation in parametric models
////////////////////////////////////////////////////////////////////////
//! mc::PAREST is a C++ class for solving parameter estimation problems
//! for parametric models using MC++, CRONOS and CANON
////////////////////////////////////////////////////////////////////////
class PAREST
: public virtual BASE_PAREST
{

protected:

  typedef FFGraph DAG;

#if defined( MC__USE_SNOPT )
  typedef NLPSLV_SNOPT NLP;
#elif defined( MC__USE_IPOPT )
  typedef NLPSLV_IPOPT NLP;
#endif
  
private:

  //! @brief DAG of model
  DAG* _dag;

  //! @brief local copy of experimental controls
  std::vector<std::vector<FFVar>> _vCON;

  //! @brief local copy of model outputs
  std::vector<std::vector<FFVar>> _vOUT;

  //! @brief local copy of model constants
  std::vector<FFVar> _vCST;

  //! @brief local copy of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief local copy of cost regularizations
  std::vector<FFVar> _vREG;

  //! @brief local copy of model constraints (constraint lhs, type, constraint rhs)
  std::tuple< std::vector<FFVar>, std::vector<t_CTR>, std::vector<FFVar> > _vCTR;

  //! @brief output subgraph
  std::vector<FFSubgraph> _sgOUT;

  //! @brief work array for output evaluations
  std::vector<double> _wkOUT;

  //! @brief output values
  std::vector<double> _dOUT;

  //! @brief Structure holding estimated parameters
  SOLUTION_OPT _PAREst;

  //! @brief Sample parameters
  arma::mat _PARSam;

  //! @brief Sample weights
  arma::vec _WEISam;
  
  //! @brief current MLE criterion
  FFVar _MLECrit;

public:
  /** @defgroup PAREST Parameter Estimation of Parametric Models using MC++
   *  @{
   */
   
  //! @brief Constructor
  PAREST()
    : _dag     ( nullptr ),
      _MLECrit ( 0. )
    { stats.reset(); }

  //! @brief Destructor
  virtual ~PAREST()
    {
      delete   _dag;
    }

  //! @brief PAREST solver options
  struct Options
  {
    //! @brief Constructor
    Options()
      : NLPSLV(),
        NSSLV()
      { reset(); }

    //! @brief Reset to default options
    void reset
      ()
      {
        NLPSLV.reset();
#ifdef MC__USE_SNOPT
        NLPSLV.DISPLEVEL            = 0;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-6;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#elif  MC__USE_IPOPT
        NLPSLV.DISPLEVEL            = 0;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-5;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.HESSMETH             = NLP::Options::LBFGS;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#endif
        NSSLV.reset();
        NSSLV.DISPLEVEL             = 0;
        DISPLEVEL                   = 1;
        RNGSEED                     = -1;
      }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        DISPLEVEL   = options.DISPLEVEL;
        RNGSEED     = options.RNGSEED;
        NLPSLV      = options.NLPSLV;
        NSSLV       = options.NSSLV;
        return *this;
      }

    //! @brief Verbosity level
    int                       DISPLEVEL;
    //! @brief Random-number generator seed. =0: Sobol sampling; >0: seed set to specified value; <0: seed set to a value drawn from std::random_device
    int                       RNGSEED;
    //! @brief NLP gradient-based solver options
    typename NLP::Options     NLPSLV;
    //! @brief Nested sampling solver options
    NSFEAS::Options  NSSLV;
  } options;

  //! @brief PAREST solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for PAREST exception handling
    enum TYPE{
      BADSIZE=0,    //!< Inconsistent dimensions
      BADOPTION,    //!< Incorrect option
      BADCONST,     //!< Unspecified constants
      NOMODEL,	    //!< Unspecified model
      NODATA,	    //!< Unspecified data
      INTERN=-33    //!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr(){ return _ierr; }
    //! @brief Inline function returning the error description
    std::string what(){
      switch( _ierr ){
        case BADSIZE:
          return "PAREST::Exceptions  Inconsistent dimensions";
        case BADOPTION:
          return "PAREST::Exceptions  Incorrect option";
        case BADCONST:
          return "PAREST::Exceptions  Unspecified constants";
        case NOMODEL:
          return "PAREST::Exceptions  Unspecified model";
        case NODATA:
          return "PAREST::Exceptions  Unspecified data";
        case INTERN:
        default:
          return "PAREST::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief PAREST solver statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime_all = walltime_setup = walltime_slvmle = walltime_slvns =
        std::chrono::microseconds(0); }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "#  WALL-CLOCK TIMES" << std::endl
           << "#  SOLVER SETUP:          " << std::setw(10) << to_time( walltime_setup )   << " SEC" << std::endl
           << "#  GRADIENT-BASED SOLVE:  " << std::setw(10) << to_time( walltime_slvmle )  << " SEC" << std::endl
           << "#  NESTED SAMPLING SOLVE: " << std::setw(10) << to_time( walltime_slvns )   << " SEC" << std::endl
           << "#  TOTAL:                 " << std::setw(10) << to_time( walltime_all )     << " SEC" << std::endl
           << std::endl; }
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime_all;
    //! @brief Cumulated wall-clock time used for problem setup (in microseconds)
    std::chrono::microseconds walltime_setup;
    //! @brief Cumulated wall-clock time used by gradient-based NLP solver (in microseconds)
    std::chrono::microseconds walltime_slvmle;
    //! @brief Cumulated wall-clock time used by nested sampling solver (in microseconds)
    std::chrono::microseconds walltime_slvns;
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

  //! @brief Setup estimation problem before solution
  bool setup
    ();

  //! @brief Solve set-membership estimation problem 
  int sme_solve
    ( double const& conf=0.95, std::vector<double> const& C0=std::vector<double>(),
      std::ostream& os=std::cout );

  //! @brief Solve Bayesian parameter estimation problem 
  int bpe_solve
    ( std::vector<double> const& C0=std::vector<double>(),
      std::ostream& os=std::cout );

  //! @brief Solve gradient-based parameter estimation problem 
  int mle_solve
    ( std::vector<double> const& P0, std::vector<double> const& C0=std::vector<double>(),
      std::ostream& os=std::cout );

  //! @brief Multi-start solve gradient-based parameter estimation problem 
  int mle_solve
    ( size_t const nsam, std::vector<double> const& C0=std::vector<double>(),
      std::ostream& os=std::cout );

  //! @brief Compute chi-squared (good-of-fit) test
  std::tuple<double,double,double> chi2_test
    ( double const& conf=0.95, std::ostream& os=std::cout );

  //! @brief Compute chi-squared (good-of-fit) test
  std::tuple<double,double,double> chi2_test
    ( double const& conf, std::vector<double> const& P0,
      std::vector<double> const& C0=std::vector<double>(), std::ostream& os=std::cout );

  //! @brief Sample confidence region using bootstrapping around given data set
  void bootstrap_sample
    ( std::vector<std::vector<Experiment>> const& data, size_t const nsam,
      std::ostream& os=std::cout );

  //! @brief Sample confidence region using bootstrapping around given data set
  void bootstrap_sample
    ( std::vector<Experiment> const& data, size_t const nsam,
      std::ostream& os=std::cout );

  //! @brief Sample confidence region using bootstrapping around best parameter estimates
  void bootstrap_sample
    ( size_t const nsam, std::ostream& os=std::cout );

  //! @brief Return covariance from sampled parameter posterior
  arma::mat cov_sample
    ( std::ostream& os=std::cout )
    const;

  //! @brief Return covariance for given parameter sample
  arma::mat cov_sample
    ( arma::mat PARSam, std::ostream& os=std::cout )
    const;

  //! @brief Compute covariance matrix using linearization (Wald)
  arma::mat cov_linearized
    ( std::ostream& os=std::cout );

  //! @brief Compute parameter confidence intervals
  arma::vec conf_interval
    ( arma::mat const& covmat, double const& conflevel=0.95, std::string const& type="T",
      std::ostream& os=std::cout );

  //! @brief Sample parameter (i,j) confidence ellipsoid
  arma::mat conf_ellipsoid
    ( arma::mat const& covmat, size_t const i, size_t const j, double const& conflevel=0.95,
      std::string const& type="F", size_t const nsam=50, std::ostream& os=std::cout );

  //! @brief Return weighted samples in level*100% HPD region
  std::pair<arma::mat,arma::vec> hpd_region
    ( double const& level=0.95, std::ostream& os=std::cout )
    const;

  //! @brief Return lower and upper range of level*100% HPD intervals
  std::pair<arma::vec,arma::vec> hpd_interval
    ( double const& level=0.95, std::ostream& os=std::cout )
    const;

  //! @brief Return level*100% quantile of parameter posterior
  arma::vec hpd_quantile
    ( double const& level, std::ostream& os=std::cout )
    const;

  //! @brief Return min range of level*100% HPD region
  arma::vec hpd_min
    ( double const& level, std::ostream& os=std::cout )
    const;

  //! @brief Return max range of level*100% HPD region
  arma::vec hpd_max
    ( double const& level, std::ostream& os=std::cout )
    const;

  //! @brief Return mean of parameter posterior
  arma::vec hpd_mean
    ( std::ostream& os=std::cout )
    const;

  //! @brief Best parameter estimates
  std::vector<double> const& par_best
    ()
    const
    { return _PAREst.x; }

  //! @brief Set of parameter samples
  arma::mat const& par_sample
    ()
    const
    { return _PARSam; }

  //! @brief Set of weight samples
  arma::vec const& wei_sample
    ()
    const
    { return _WEISam; }

private:

  //! @brief Compute chi-squared (good-of-fit) test
  std::tuple<double,double,double> _chi2_test
    ( double const& conf, std::vector<double> const& P0, std::vector<double> const& C0,
      int const disp, std::ostream& os );

  //! @brief Return position of last weighted samples in HPD region at confidence level conf*100%
  size_t _pos
    ( double const& level, arma::vec const& wei )
    const;

  //! @brief Return mean of weighted samples in log space
  double _mean
    ( arma::vec const& par, arma::vec const& wei )
    const;

  //! @brief Return quantile at confidence level conf*100% in 1d weighted sample
  double _quantile
    ( double const& level, arma::vec const& val, arma::vec const& wei )
    const;

  //! @brief Return conf*100% quantile of parameter posterior
  arma::vec _hpd_quantile
    ( double const& level )
    const;

};

inline
bool
PAREST::setup
()
{
  stats.reset();
  auto&& t_setup = stats.start();

  if( !_nm ) throw Exceptions( Exceptions::NOMODEL );
  if( !_np ) throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;
  _dag->options = BASE_PAREST::_dag->options;

  _vPAR.resize( _np );
  _dag->insert( BASE_PAREST::_dag, _np, BASE_PAREST::_vPAR.data(), _vPAR.data() );
  _vCST.resize( _nc );
  _dag->insert( BASE_PAREST::_dag, _nc, BASE_PAREST::_vCST.data(), _vCST.data() );

  _sgOUT.clear();
  _sgOUT.resize( _nm );
  _vOUT.resize( _nm );
  _vCON.resize( _nm );
  size_t nytot = 0;
  for( size_t m=0; m<_nm; nytot+=_ny[m++] ){
    if( _nu[m] ){
      _vCON[m].resize( _nu[m] );
      _dag->insert( BASE_PAREST::_dag, _nu[m], BASE_PAREST::_vCON[m].data(), _vCON[m].data() );
    }
    if( _ny[m] ){
      _vOUT[m].resize( _ny[m] );
      _dag->insert( BASE_PAREST::_dag, _ny[m], BASE_PAREST::_vOUT[m].data(), _vOUT[m].data() );

#ifdef MAGNUS__PAREST_SETUP_DEBUG
      _sgOUT[m] = _dag->subgraph( _ny[m], _vOUT[m].data() );
      std::vector<FFExpr> exOUT = FFExpr::subgraph( _dag, _sgOUT[m] ); 
      for( size_t i=0; i<_ny[m]; ++i )
        std::cout << "OUT[" << m << "][" << i << "] = " << exOUT[i] << std::endl;
#endif
    }
  }
  if( !nytot ) throw Exceptions( Exceptions::NOMODEL );

  _vREG.resize( _nr );
  if( _nr )
    _dag->insert( BASE_PAREST::_dag, _nr, BASE_PAREST::_vREG.data(), _vREG.data() );
  
  std::get<0>(_vCTR).resize( _ng ); 
  std::get<2>(_vCTR).resize( _ng ); 
  std::get<1>(_vCTR) = std::get<1>(BASE_PAREST::_vCTR);
  if( _ng ){
    _dag->insert( BASE_PAREST::_dag, _ng, std::get<0>(BASE_PAREST::_vCTR).data(), std::get<0>(_vCTR).data() );
    _dag->insert( BASE_PAREST::_dag, _ng, std::get<2>(BASE_PAREST::_vCTR).data(), std::get<2>(_vCTR).data() );
  }

  if( options.RNGSEED < 0 )
    arma::arma_rng::set_seed_random();
  else
    arma::arma_rng::set_seed( options.RNGSEED ); 

  stats.walltime_setup += stats.lapse( t_setup );
  stats.walltime_all   += stats.lapse( t_setup );
  return true;
}

inline
int
PAREST::sme_solve
( double const& conf, std::vector<double> const& C0, std::ostream& os )
{
  _WEISam.reset();
  _PARSam.reset();
  _PAREst.reset();
  
  if( !_nd  )          throw Exceptions( Exceptions::NODATA );
  if( !_vOUT.size()  ) throw Exceptions( Exceptions::NOMODEL );

  auto&& t_slvns = stats.start();

  // Update constants
  if( C0.size() && C0.size() == _nc ) _vCSTVAL = C0;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Nested sampling
  NSFEAS PE;

  PE.options = options.NSSLV;
  PE.set_dag( _dag );
  PE.set_control( _vPAR, _vPARLB, _vPARUB );
  PE.set_constant( _vCST, _vCSTVAL );

  // Adequacy constraint
  FFMLE OpMLE;
  _MLECrit = 0.;
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_vCSTVAL, &_vDAT[m] );
  }
  chi_squared dist( _nd-_np );
  double Chi2Crit = quantile( dist, conf );
  FFLin<I> OpSum;
  if( _nr ) PE.add_constraint( 2*_MLECrit + OpSum( _vREG, 1. ), LE, Chi2Crit );
  else      PE.add_constraint( 2*_MLECrit, LE, Chi2Crit );

  // Other constraints
  for( size_t g=0; g<_ng; ++g )
    PE.add_constraint( std::get<0>(_vCTR).at(g), std::get<1>(_vCTR).at(g), std::get<2>(_vCTR).at(g) );

  PE.setup();
  int iflag = PE.sample();

  if( options.NSSLV.DISPLEVEL )
    PE.stats.display();

  if( iflag == PE.NORMAL ){
    double const* pPAROPT = std::get<0>(PE.live_points().cbegin()->second);    
    _PAREst.x.assign( pPAROPT, pPAROPT+_np );
    _PAREst.f = { std::get<1>(_chi2_test( conf, _PAREst.x, C0, 0, os )) };
    if( options.DISPLEVEL ){
      os << "\n# MAXIMUM ADEQUACY:\n"
         << "STATUS: " << iflag << std::endl
         << std::scientific << std::setprecision(6) << std::right;
      for( unsigned int i=0; i<_np; i++ )
        os << "P[" << i << "]: " << std::setw(13) << _PAREst.x[i] << std::endl;
      os << "GOODNESS-OF-FIT: " << std::fixed << std::setprecision(0) << _PAREst.f[0]*1e2 << "% CONFIDENCE LEVEL"  << std::setw(13) << std::endl;
    }

    if( options.DISPLEVEL )
      os << "\n# SAMPLED ADEQUACY REGION:\n";

    size_t isam = 0;
    _PARSam.set_size( PE.live_points().size(), _np );
    for( auto it = PE.live_points().cbegin(); it != PE.live_points().cend(); ++it, ++isam ){
      auto const& [feas,tup] = *it;
      _PARSam.row(isam) = arma::mat( const_cast<double*>(std::get<0>(tup)), 1, _np, false );
      if( options.DISPLEVEL > 1 ){
        os << std::setw(5) << std::right << isam << ": "
           << std::scientific << std::setprecision(6) << std::setw(14) << feas << " | ";
        auto vecx = arma::vec( const_cast<double*>(std::get<0>(tup)), _np, false );
        vecx.t().raw_print( os );
      }
    }
    
    if( options.DISPLEVEL == 1 ){
      os << std::scientific << std::setprecision(5);
      for( size_t r=0; r<_PARSam.n_rows; ++r ){
        os << std::setw(5) << std::right << r << ": ";
        arma::rowvec&& par = _PARSam.row(r);
        par.raw_print( os );
        if( r == 2 ){
          os << "   ..." << std::endl;
          r += _PARSam.n_rows - 6;
        }
      }
    }
  }

  stats.walltime_slvns += stats.lapse( t_slvns );
  stats.walltime_all   += stats.lapse( t_slvns );

  return iflag;
}

inline
int
PAREST::bpe_solve
( std::vector<double> const& C0, std::ostream& os )
{
  _WEISam.reset();
  _PARSam.reset();
  _PAREst.reset();
  
  if( !_nd  )          throw Exceptions( Exceptions::NODATA );
  if( !_vOUT.size()  ) throw Exceptions( Exceptions::NOMODEL );

  auto&& t_slvns = stats.start();

  // Update constants
  if( C0.size() && C0.size() == _nc ) _vCSTVAL = C0;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Nested sampling
  NSFEAS PE;

  PE.options = options.NSSLV;
  PE.set_dag( _dag );
  PE.set_control( _vPAR, _vPARLB, _vPARUB );
  PE.set_constant( _vCST, _vCSTVAL );

  // Constraints
  for( size_t g=0; g<_ng; ++g )
    PE.add_constraint( std::get<0>(_vCTR).at(g), std::get<1>(_vCTR).at(g), std::get<2>(_vCTR).at(g) );

  // Log-likelihood function
  FFMLE OpMLE;
  _MLECrit = 0.;
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    _MLECrit -= OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_vCSTVAL, &_vDAT[m] );
  }
  FFLin<I> OpSum;
  if( _nr ) PE.set_loglikelihood( _MLECrit - OpSum( _vREG, 1. ) );
  else      PE.set_loglikelihood( _MLECrit );

  PE.setup();
  int iflag = PE.sample();

  if( options.NSSLV.DISPLEVEL )
    PE.stats.display();

  if( iflag == PE.NORMAL ){
    double const* pPAROPT = std::get<0>(PE.live_points().crbegin()->second);    
    _PAREst.x.assign( pPAROPT, pPAROPT+_np );
    _PAREst.f = { -PE.live_points().crbegin()->first };
    if( options.DISPLEVEL ){
      os << "\n# MAXIMUM A POSTERIORI:\n"
         << "STATUS: " << iflag << std::endl
         << std::scientific << std::setprecision(6) << std::right;
      for( unsigned int i=0; i<_np; i++ )
        os << "P[" << i << "]: " << std::setw(13) << _PAREst.x[i] << std::endl;
      os << "LIKELIHOOD: " << std::setw(13) << _PAREst.f[0] << std::endl;
    }

    if( options.DISPLEVEL > 1 )
      os << "\n# SAMPLED POSTERIOR:\n";
    size_t isam = 0;
    _PARSam.set_size( PE.live_points().size() + PE.dead_points().size(), _np );
    _WEISam.set_size( PE.live_points().size() + PE.dead_points().size() );
    double WEItot = 0.;
    for( auto it = PE.live_points().crbegin(); it != PE.live_points().crend(); ++it, ++isam ){
      auto const& [lkh,tup] = *it;
      _PARSam.row(isam) = arma::mat( const_cast<double*>(std::get<0>(tup)), 1, _np, false );
      _WEISam(isam) = std::get<2>(tup);
      WEItot = (it != PE.live_points().crbegin()? log_sum_exp( WEItot, std::get<2>(tup) ): std::get<2>(tup) ); 
      if( options.DISPLEVEL > 1 ){
        os << std::setw(5) << std::right << isam << ": "
           << std::scientific << std::setprecision(6) << std::setw(14) << lkh << " | "
           << std::setw(14) << std::get<2>(tup) << " | " << std::setw(14) << WEItot << " | ";
        auto vecx = arma::vec( const_cast<double*>(std::get<0>(tup)), _np, false );
        vecx.t().raw_print( os );
      }
    }
    for( auto it = PE.dead_points().crbegin(); it != PE.dead_points().crend(); ++it, ++isam ){
      auto const& [lkh,tup] = *it;
      _PARSam.row(isam) = arma::mat( const_cast<double*>(std::get<0>(tup)), 1, _np, false );
      _WEISam(isam) = std::get<2>(tup);
      WEItot = log_sum_exp( WEItot, std::get<2>(tup) ); 
      if( options.DISPLEVEL > 1 ){
        os << std::setw(5) << std::right << isam << ": "
           << std::scientific << std::setprecision(6) << std::setw(14) << lkh << " | "
           << std::setw(14) << std::get<2>(tup) << " | " << std::setw(14) << WEItot << " | ";
        auto vecx = arma::vec( const_cast<double*>(std::get<0>(tup)), _np, false );
        vecx.t().raw_print( os );
      }
    }
  }

  stats.walltime_slvns += stats.lapse( t_slvns );
  stats.walltime_all   += stats.lapse( t_slvns );

  return iflag;
}

inline
size_t
PAREST::_pos
( double const& level, arma::vec const& wei )
const
{
  // compute cumulated probability mass wei up to level
  if( !wei.n_rows ) return 0;

  size_t pos  = 0;
  double mass = wei( pos );
  for( ; mass < std::log( level ); ){
    ++pos;
#ifdef MAGNUS__PAREST_CONF_DEBUG
    std::cout << "pos: " << pos << "  mass: " << mass << "  threshold: " <<  std::log(level) << std::endl;
#endif
    if( pos == wei.n_rows ) break;
    mass = log_sum_exp( mass, wei( pos ) );
  }

  return pos;
}

inline
std::pair<arma::mat,arma::vec>
PAREST::hpd_region
( double const& level, std::ostream& os )
const
{
  // compute cumulated probability mass _WEISam up to level
  arma::mat PARHpd;
  arma::vec WEIHpd;
  if( _WEISam.empty() ){
    size_t pos = std::round( _PARSam.n_rows * level );
    PARHpd = _PARSam.submat( 0,0, pos,_np-1 );
    WEIHpd.resize( pos+1 ).fill( -std::log(pos+1) );
  }
  else{
    size_t pos = _pos( level, _WEISam );
    PARHpd = _PARSam.submat( 0,0, pos,_np-1 );
    WEIHpd = _WEISam.rows( 0, pos );
  }
  
  if( options.DISPLEVEL ){
    os << std::endl << "# " << std::fixed << std::setprecision(0) << level*1e2
       << "% HPD PARAMETER REGION:\n"
       << std::scientific << std::setprecision(5);
    for( size_t r=0; r<PARHpd.n_rows; ++r ){
      os << std::setw(5) << std::right << r << ": ";
      os << std::setw(12) << WEIHpd(r) << " | ";
      arma::rowvec&& par = PARHpd.row(r);
      par.raw_print( os );
      if( r == 2 ){
        os << "   ..." << std::endl;
        r += PARHpd.n_rows - 6;
      }
    }
  }

  return std::make_pair( PARHpd, WEIHpd );
}

inline
arma::vec
PAREST::hpd_min
( double const& level, std::ostream& os )
const
{
  // compute cumulated probability mass _WEISam up to level
  if( _WEISam.empty() ) return arma::vec();
  size_t pos = _pos( level, _WEISam );

  // extract corresponding top part of _PARSam and determine minimum
  auto&& PARMin = arma::min(_PARSam.submat( 0,0, pos,_np-1 ),0);

  if( options.DISPLEVEL ){
    os << std::endl << "# MINIMUM RANGE OF " << std::fixed << std::setprecision(0) << level*1e2
       << "% HPD PARAMETER REGION:\n"
       << std::scientific << std::setprecision(5);
    PARMin.raw_print( os );
  }

  return PARMin.t();
}

inline
arma::vec
PAREST::hpd_max
( double const& level, std::ostream& os )
const
{
  // compute cumulated probability mass _WEISam up to level
  if( _WEISam.empty() ) return arma::vec();
  size_t pos = _pos( level, _WEISam );

  // extract corresponding top part of _PARSam and determine maximum
  auto&& PARMax = arma::max(_PARSam.submat( 0,0, pos,_np-1 ),0);

  if( options.DISPLEVEL ){
    os << std::endl << "# MAXIMUM RANGE OF " << std::fixed << std::setprecision(0) << level*1e2
       << "% HPD PARAMETER REGION:\n"
       << std::scientific << std::setprecision(5);
    PARMax.raw_print( os );
  }

  return PARMax.t();
}

inline
double
PAREST::_quantile
( double const& level, arma::vec const& val, arma::vec const& wei )
const
{
  assert( !wei.n_rows || val.n_rows == wei.n_rows );
  
  // create map { val: wei }
  std::multimap<double,double> map;
  for( size_t i=0; i<val.n_rows; ++i )
    map.insert({ val(i), wei.n_rows? wei(i): -std::log(val.n_rows) });

#ifdef MAGNUS__PAREST_CONF_DEBUG
  std::cout << std::endl << std::scientific << std::setprecision(5) << std::right;
  double m = 1.;
  bool first = true;
  for( auto const& [v,w] : map ){
    m = (first? w: log_sum_exp( m, w ));
    first = false;
    std::cout << std::setw(13) << v << std::setw(13) << m << std::endl;
  }
#endif

  // locate desired quantile
  double mass = map.cbegin()->second;
  for( auto it = map.cbegin(); ; ){
    if( mass >= std::log( level ) ) return it->first;
    ++it;
    if( it == map.cend() ) break;
    mass = log_sum_exp( mass, it->second );
  }
  return map.crend()->first;
}

inline
arma::vec
PAREST::_hpd_quantile
( double const& level )
const
{
  if( _PARSam.empty() ) return arma::vec();
  assert( _PARSam.n_cols == _np );
  
  arma::vec PARQuan( _np );
  size_t p=0;
  _PARSam.each_col(
    [&]( arma::vec const& v )
    { PARQuan(p++) = _quantile( level, v, _WEISam ); }
  );

  return PARQuan;
}

inline
arma::vec
PAREST::hpd_quantile
( double const& level, std::ostream& os )
const
{ 
  auto&& PARQuan = _hpd_quantile( level );

  if( options.DISPLEVEL ){
    os << std::endl << "# " << std::fixed << std::setprecision(1) << level*1e2
       << "% QUANTILE OF PARAMETER POSTERIOR:\n"
       << std::scientific << std::setprecision(5);
    PARQuan.t().raw_print( os );
  }

  return PARQuan;
}

inline
arma::vec
PAREST::hpd_mean
( std::ostream& os )
const
{
  if( _PARSam.empty() ) return arma::vec();
  assert( _PARSam.n_cols == _np );

  arma::vec PARMean( _np );
  size_t p=0;
  _PARSam.each_col(
    [&]( arma::vec const& par )
    { PARMean(p++) = _mean( par, _WEISam ); }
  );
  
  if( options.DISPLEVEL ){
    os << std::endl << "# MEAN OF PARAMETER POSTERIOR:\n"
       << std::scientific << std::setprecision(5);
    PARMean.t().raw_print( os );
  }

  return PARMean;
}

inline
double
PAREST::_mean
( arma::vec const& par, arma::vec const& wei )
const
{
  assert( !wei.n_rows || wei.n_rows == par.n_rows );

  // compute mean in log space
  ;
  double mean = ( wei.n_rows? wei( 0 ) + std::log( par( 0 ) ): std::log( par( 0 ) / par.n_rows ) );
  for( size_t pos = 0; ; ){
    ++pos;
    if( pos == par.n_rows ) break;
    if( wei.n_rows ) mean = log_sum_exp( mean, wei( pos ) + std::log( par( pos ) ) );
    else             mean = log_sum_exp( mean, std::log( par( pos ) / par.n_rows ) );
  }

  return std::exp( mean );
}

inline
std::pair<arma::vec,arma::vec>
PAREST::hpd_interval
( double const& level, std::ostream& os )
const
{
  auto&& PARLb = _hpd_quantile( 0.5*(1-level) );
  auto&& PARUb = _hpd_quantile( 1-0.5*(1-level) );

  if( options.DISPLEVEL ){
    os << std::endl << "# " << std::fixed << std::setprecision(0) << level*1e2
       << "% HPD PARAMETER INTERVALS:\n"
       << std::scientific << std::setprecision(5);
    PARLb.t().raw_print( os );
    PARUb.t().raw_print( os );
  }

  return std::make_pair( PARLb, PARUb );
}

inline
int
PAREST::mle_solve
( std::vector<double> const& P0, std::vector<double> const& C0, std::ostream& os )
{
  _PAREst.reset();

  if( !_nd  )          throw Exceptions( Exceptions::NODATA );
  if( !_vOUT.size()  ) throw Exceptions( Exceptions::NOMODEL );

  auto&& t_slvmle = stats.start();

  // Update constants
  if( C0.size() && C0.size() == _nc ) _vCSTVAL = C0;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Local NLP optimization
  NLP PE;

  PE.options = options.NLPSLV;
  PE.set_dag( _dag );
  PE.add_var( _vPAR, _vPARLB, _vPARUB );
  PE.add_par( _vCST );

  FFMLE OpMLE;
  _MLECrit = 0.;
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_vCSTVAL, &_vDAT[m] );
  }
  FFLin<I> OpSum;
  if( _nr )
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit + OpSum( _vREG, 1. ) );
  else
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit );
    
  for( size_t g=0; g<_ng; ++g )
    PE.add_ctr( std::get<1>(_vCTR).at(g), std::get<0>(_vCTR).at(g)-std::get<2>(_vCTR).at(g) );

  PE.setup();
  int iflag = PE.solve( P0.data(), nullptr, nullptr, _vCSTVAL.data() );

  if( options.DISPLEVEL > 1 )
    os << "\n#  FEASIBLE:   " << PE.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << PE.is_stationary( 1e-6 ) << std::endl;

  _PAREst = PE.solution();
  if( options.DISPLEVEL ){
    os << "\n# MAXIMUM LIKELIHOOD ESTIMATE:\n"
       << "STATUS: " << iflag << std::endl
       << std::scientific << std::setprecision(6) << std::right;
    for( unsigned int i=0; i<_np; i++ )
      os << "P[" << i << "]: " << std::setw(13) << _PAREst.x[i] << std::endl;
    os << "LIKELIHOOD: " << std::setw(13) << _PAREst.f[0] << std::endl;
  }

  stats.walltime_slvmle += stats.lapse( t_slvmle );
  stats.walltime_all    += stats.lapse( t_slvmle );

  return iflag;
}

inline
int
PAREST::mle_solve
( size_t const nsam, std::vector<double> const& C0, std::ostream& os )
{
  _PAREst.reset();

  auto&& t_slvmle = stats.start();
  
  // Update constants
  if( C0.size() && C0.size() == _nc ) _vCSTVAL = C0;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Local NLP optimization
  NLP PE;

  PE.options = options.NLPSLV;
  PE.set_dag( _dag );
  PE.add_var( _vPAR, _vPARLB, _vPARUB );
  PE.add_par( _vCST );

  FFMLE OpMLE;
  _MLECrit = 0.;
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_vCSTVAL, &_vDAT[m] );
  }
  FFLin<I> OpSum;
  if( _nr )
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit + OpSum( _vREG, 1. ) );
  else
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit );
    
  for( size_t g=0; g<_ng; ++g )
    PE.add_ctr( std::get<1>(_vCTR).at(g), std::get<0>(_vCTR).at(g)-std::get<2>(_vCTR).at(g) );

  PE.setup();
  int iflag = PE.solve( nsam, _vPARLB.data(), _vPARUB.data(), _vCSTVAL.data(), nullptr, 1 );

  if( options.DISPLEVEL > 1 )
    os << "\n#  FEASIBLE:   " << PE.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << PE.is_stationary( 1e-6 ) << std::endl;

  _PAREst = PE.solution();
  if( options.DISPLEVEL ){
    os << "\n# MAXIMUM LIKELIHOOD ESTIMATE:\n"
       << "STATUS: " << iflag << std::endl
       << std::scientific << std::setprecision(6) << std::right;
    for( unsigned int i=0; i<_np; i++ )
      os << "P[" << i << "]: " << std::setw(13) << _PAREst.x[i] << std::endl;
    os << "LIKELIHOOD: " << std::setw(13) << _PAREst.f[0] << std::endl;
  }

  stats.walltime_slvmle += stats.lapse( t_slvmle );
  stats.walltime_all    += stats.lapse( t_slvmle );

  return iflag;
}

inline
std::tuple<double,double,double>
PAREST::chi2_test
( double const& conf, std::ostream& os )
{
  return _chi2_test( conf, _PAREst.x, _PAREst.p, options.DISPLEVEL, os );
}

inline
std::tuple<double,double,double>
PAREST::chi2_test
( double const& conf, std::vector<double> const& P0, std::vector<double> const& C0, std::ostream& os )
{
  return _chi2_test( conf, _PAREst.x, _PAREst.p, options.DISPLEVEL, os );
}

inline
std::tuple<double,double,double>
PAREST::_chi2_test
( double const& conf, std::vector<double> const& P0, std::vector<double> const& C0, 
  int const disp, std::ostream& os )
{
  double Chi2Val = 0./0.;
  chi_squared dist( _nd-_np );
  double Chi2Crit = quantile( dist, conf );

  // Update constants
  if( C0.size() && C0.size() == _nc ) _vCSTVAL = C0;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Compute MLE residual
  FFMLE OpMLE;
  _MLECrit = 0.;
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_vCSTVAL, &_vDAT[m] );
  }
  try{
    _dag->eval( _wkOUT, 1, &_MLECrit, &Chi2Val, _np, _vPAR.data(), P0.data(), _nc, _vCST.data(), _vCSTVAL.data() );
    Chi2Val *= 2;
  }
  catch(...){
    return std::make_tuple( Chi2Val, 0., Chi2Crit );
  }

  if( disp > 0 ){
    os << "\n# CHI-SQUARED TEST: " << std::scientific << std::setprecision(3) << Chi2Val
       << (Chi2Val < Chi2Crit? " < ": " > ") << Chi2Crit << " CRITICAL CHI_SQUARED VALUE (" << std::fixed << std::setprecision(0) << conf*1e2 << "%, " << _nd-_np << " DOF)"
       << std::endl;
  }
  
  double Chi2Conf = cdf( dist, Chi2Val );
  if( disp > 0 )
    os << "# CHI-SQUARED TEST PASSED WITH >" << std::fixed << std::setprecision(0) << Chi2Conf*1e2 << "% CONFIDENCE LEVEL"
       << std::endl;

  return std::make_tuple( Chi2Val, Chi2Conf, Chi2Crit );
}

inline
void
PAREST::bootstrap_sample
( size_t const nsam, std::ostream& os )
{  
  _WEISam.reset();
  _PARSam.reset();

  if( NLP::get_status( _PAREst.stat ) != NLP::SUCCESSFUL 
   && NLP::get_status( _PAREst.stat ) != NLP::FAILURE )
    return;

  // Simulate model at MLE estimate
  double MLERes = 0./0.;
  FFMLE OpMLE;
  _MLECrit = 0.;
  std::vector<FFVar> vMLECrit( _nm, 0. );
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    vMLECrit[m] = OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_PAREst.p, &_vDAT[m] );
    _MLECrit += vMLECrit[m];
  }

  try{
    _dag->eval( _wkOUT, 1, &_MLECrit, &MLERes, _np, _vPAR.data(), _PAREst.x.data(),
                _nc, _vCST.data(), _PAREst.p.data() );
  }
  catch(...){
    return;
  }

  auto vDAT0 = _vDAT;
  for( size_t m=0, e=0; m<_nm; ++m, e=0 ){
    if( !_ny[m] ) continue;
    auto const& dOUT = dynamic_cast<FFMLE const*>(vMLECrit[m].opdef().first)->dOUT();
    for( auto& EXP : vDAT0[m] ){
      for( auto& [ k, RECk ] : EXP.output )
        for( auto& YMk : RECk.measurement )
          YMk = dOUT.at(e).at(k);
      ++e;
    }
  }

  return bootstrap_sample( vDAT0, nsam, os );
}

inline
void
PAREST::bootstrap_sample
( std::vector<Experiment> const& data, size_t const nsam, std::ostream& os )
{
  std::vector<std::vector<Experiment>> vDAT0;
  for( auto const& exp : data )
    _add_data( vDAT0, exp );

  return bootstrap_sample( vDAT0, nsam, os );
}

inline
void
PAREST::bootstrap_sample
( std::vector<std::vector<Experiment>> const& data, size_t const nsam, std::ostream& os )
{
  _PARSam.reset();
  _WEISam.reset();

  // Define MLE problem
  NLP PE;

  PE.options = options.NLPSLV;
  PE.set_dag( _dag );
  PE.add_var( _vPAR, _vPARLB, _vPARUB );
  PE.add_par( _vCST );

  FFLin<I> OpSum;

  for( size_t g=0; g<_ng; ++g )
    PE.add_ctr( std::get<1>(_vCTR).at(g), std::get<0>(_vCTR).at(g)-std::get<2>(_vCTR).at(g) );

  // Number of measurement for dataset
  size_t nd = 0;
  for( size_t m=0; m<_nm; ++m )
    for( auto& EXP : data[m] )
      for( auto& [ k, RECk ] : EXP.output )
        nd += RECk.measurement.size();

  // Initialize Sobol sampler
  std::vector<double> vSAM(nd), Ym_noise(nd);
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( nd ); // sample nd-dimensional space
  qrgen noise( eng, boost::uniform_01<double>() );
  noise.engine().seed( 0 );

  std::multimap<double,std::vector<double>> PARBoot;

  // Apply bootstrapping to MLE problem
  for( size_t isam=0; isam<nsam; ++isam ){

    // Add measurement noise
    auto vDATn = data;
    double lkh = 0.;
    bool first = true;
    for( size_t m=0, e=0; m<_nm; ++m, e=0 ){
      for( auto& EXP : vDATn[m] ){
        for( auto& [ k, RECk ] : EXP.output ){
#ifdef MAGNUS__PAREST_CONF_DEBUG
          size_t r = 0;
#endif
          for( auto& YMk : RECk.measurement ){
            double const dY = ( options.RNGSEED?
                                arma::randn( arma::distr_param( 0., std::sqrt(RECk.variance) ) ):
                                quantile( normal( 0., std::sqrt(RECk.variance) ), noise() ) );
            YMk += dY;
#ifdef MAGNUS__PAREST_CONF_DEBUG
            std::cout << "YM(" << isam << ")[" << std::to_string(m) << "," << std::to_string(e) << "]["
                      << std::to_string(k) << "][" << std::to_string(r++) << "] = "
                      << YMk << std::endl;
#endif
            lkh = ( first? arma::log_normpdf( dY, 0., std::sqrt(RECk.variance) ):
                           log_sum_exp( lkh, arma::log_normpdf( dY, 0., std::sqrt(RECk.variance) ) ) );
            first = false;
#ifdef MAGNUS__PAREST_CONF_DEBUG
            std::cout << dY << "  " << arma::normpdf( dY, 0., std::sqrt(RECk.variance) ) << "  " << arma::log_normpdf( dY, 0., std::sqrt(RECk.variance) ) << "  " << lkh << std::endl;
#endif
          }
        }
        ++e;
      }
    }
#ifdef MAGNUS__PAREST_CONF_DEBUG
    std::cout << "Log Lkh = " << lkh << std::endl;
#endif
    
    // Update MLE objective - could also use set_obj_lazy - and solve from MLE estimate
    FFMLE OpMLE;
    _MLECrit = 0.;
    for( size_t m=0; m<_nm; ++m ){
      if( !_ny[m] ) continue;
      _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_PAREst.p, &vDATn[m] );
    }
    if( _nr )
      PE.set_obj( mc::BASE_OPT::MIN, _MLECrit + OpSum( _vREG, 1. ) );
    else
      PE.set_obj( mc::BASE_OPT::MIN, _MLECrit );
    
    PE.setup();
    PE.solve( _PAREst.x.data(), nullptr, nullptr, _PAREst.p.data() );

    if( options.DISPLEVEL > 2 )
      os << "#  SAMPLE:     " << isam                     << std::endl
         << "#  FEASIBLE:   " << PE.is_feasible( 1e-6 )   << std::endl
         << "#  STATIONARY: " << PE.is_stationary( 1e-6 ) << std::endl
         << std::endl;

    if( PE.get_status() == NLP::SUCCESSFUL 
     || ( ( PE.get_status() == NLP::FAILURE
         || PE.get_status() == NLP::INTERRUPTED )
         && PE.is_feasible( options.NLPSLV.FEASTOL ) ) ){
      PARBoot.insert({ lkh, PE.solution().x });
      //_PARSam.insert_rows( _PARSam.n_rows, arma::mat( const_cast<double*>(PE.solution().x.data()), 1, _np, false ) );
      if( options.DISPLEVEL > 1 ){
        os << std::setw(5) << std::right << isam << ": "
           << std::scientific << std::setprecision(5) << std::setw(12) << PE.solution().f[0] << " | ";
        auto vecx = arma::vec( PE.solution().x );
        vecx.t().raw_print( os );
      }
    }
  }

  _PARSam.resize( PARBoot.size(), _np );
  _WEISam.resize( PARBoot.size() );
  size_t r = 0;
  for( auto it = PARBoot.crbegin(); it != PARBoot.crend(); ++it, ++r ){
    auto& [lkh,par] = *it;
    _PARSam.row(r) = arma::rowvec( const_cast<double*>(par.data()), _np, false );
    _WEISam[r] = -std::log( PARBoot.size() );//lkh;
  }

  if( options.DISPLEVEL == 1 ){
    os << "\n# BOOTSTRAPPED CONFIDENCE REGION:\n";
    os << std::scientific << std::setprecision(5);
    for( size_t r=0; r<_PARSam.n_rows; ++r ){
      os << std::setw(5) << std::right << r << ": ";
      //os << std::setw(12) << _WEISam(r) << " | ";
      arma::rowvec&& par = _PARSam.row(r);
      par.raw_print( os );
      if( r == 2 ){
        os << "   ..." << std::endl;
        r += _PARSam.n_rows - 6;
      }
    }
  }
}

inline
arma::mat
PAREST::cov_sample
( std::ostream& os )
const
{
  return cov_sample( _PARSam, os ); 
}

inline
arma::mat
PAREST::cov_sample
( arma::mat PARSam, std::ostream& os )
const
{
  if( !PARSam.n_rows ) return arma::mat();

  // Estimate covariance around ML estimates
  arma::mat COVSam = arma::cov(PARSam);

  if( options.DISPLEVEL > 0 ){
    os << std::scientific << std::setprecision(5);
    COVSam.raw_print( os, "\n# PARAMETER COVARIANCE MATRIX (FROM SAMPLE):" );
  }

  return COVSam;
}

inline
arma::mat
PAREST::cov_linearized
( std::ostream& os )
{
  if( NLP::get_status( _PAREst.stat ) != NLP::SUCCESSFUL 
   && NLP::get_status( _PAREst.stat ) != NLP::FAILURE )
    return arma::mat();

  // Compute confidence ellipsoids in dedicated 
  DAG dagmle;
  dagmle.options = BASE_PAREST::_dag->options;

  std::vector<mc::FFVar> vCSTMLE( _nc );
  dagmle.insert( BASE_PAREST::_dag, _nc, BASE_PAREST::_vCST.data(), vCSTMLE.data() );
  std::vector<mc::FFVar> vPARMLE( _np );
  dagmle.insert( BASE_PAREST::_dag, _np, BASE_PAREST::_vPAR.data(), vPARMLE.data() );

  // Define extended MLE criterion with measurements as variables
  std::vector<mc::FFVar> vUMLE, vYMLE, vYMMLE;
  std::vector<double> dUMLE, dYMMLE;
  arma::mat mYCOV( _nd, _nd, arma::fill::zeros );

  mc::FFVar FMLE( 0. );
  for( size_t m=0, e=0, d=0; m<_nm; ++m, e=0 ){

    if( !_ny[m] ) continue;
    std::vector<mc::FFVar> vCONMLE( _nu[m] );
    if( _nu[m] ) dagmle.insert( BASE_PAREST::_dag, _nu[m], BASE_PAREST::_vCON[m].data(), vCONMLE.data() );
    std::vector<mc::FFVar> vOUTMLE( _ny[m] );
    dagmle.insert( BASE_PAREST::_dag, _ny[m], BASE_PAREST::_vOUT[m].data(), vOUTMLE.data() );

    for( auto& EXP : _vDAT[m] ){

      std::vector<mc::FFVar> vUMLE_e;
      for( size_t u=0; u<_nu[m]; ++u ){
        mc::FFVar UMLE_e_u( &dagmle, "U["+std::to_string(m)+","+std::to_string(e)+"]["+std::to_string(u)+"]" );
        vUMLE_e.push_back( UMLE_e_u );
      }
      vUMLE.insert( vUMLE.end(), vUMLE_e.cbegin(), vUMLE_e.cend() );
      dUMLE.insert( dUMLE.end(), EXP.control.cbegin(), EXP.control.cend() );

      std::vector<mc::FFVar> vYMLE_e = dagmle.compose( vOUTMLE, vCONMLE, vUMLE_e );
      for( auto& [ k, RECk ] : EXP.output ){
        for( unsigned r=0; r<RECk.measurement.size(); ++r, ++d ){
          mc::FFVar YMMLE_e_k_r( &dagmle, "YM["+std::to_string(m)+","+std::to_string(e)+"]["+std::to_string(k)+"]["+std::to_string(r)+"]" );
          vYMMLE.push_back( YMMLE_e_k_r );
          FMLE -= mc::sqr( vYMLE_e.at(k) - YMMLE_e_k_r ) / RECk.variance;
          mYCOV(d,d) = RECk.variance;
        }
        dYMMLE.insert( dYMMLE.end(), RECk.measurement.cbegin(), RECk.measurement.cend() );
      }
      ++e;
    }
  }
  FMLE /= 2.;
  //assert( vUMLE.size() == _nu*_vDAT.size() );
  assert( vYMMLE.size() == _nd );

  // Add any regularization terms
  std::vector<mc::FFVar> vREGMLE( _nr );
  if( _nr ){
    dagmle.insert( BASE_PAREST::_dag, _nr, BASE_PAREST::_vREG.data(), vREGMLE.data() );
    FFLin<I> OpSum;
    FMLE -= OpSum( vREGMLE, 1. );
  }
  
  // Add active general constraints
  std::vector<FFVar> vCTR( _ng ), vCTRMLE( _ng );
  for( size_t g=0; g<_ng; ++g )
    vCTR[g] = std::get<0>(BASE_PAREST::_vCTR)[g] - std::get<2>(BASE_PAREST::_vCTR)[g];
  if( _ng ) dagmle.insert( BASE_PAREST::_dag, _ng, vCTR.data(), vCTRMLE.data() );

  std::vector<mc::FFVar> vMULMLE;
  std::vector<double> dMULMLE;
  size_t na=0;
  for( size_t g=0; g<_ng; ++g ){  
    switch( std::get<1>(BASE_PAREST::_vCTR)[g] ){
      case BASE_OPT::LE:
      case BASE_OPT::GE:
        if( std::fabs( _PAREst.f[g+1] ) > options.NLPSLV.FEASTOL ) continue;
      case BASE_OPT::EQ: 
        vMULMLE.push_back( mc::FFVar( &dagmle, "MG["+std::to_string(g)+"]" ) );
        dMULMLE.push_back( _PAREst.uf[g+1] );
        FMLE += vCTRMLE[g] * vMULMLE[na++];
        break;
    }
  }

  // Add active bound constraints
  for( size_t p=0; p<_np; ++p ){  
    if( ( std::fabs( _PAREst.x[p] - _vPARLB[p] ) > options.NLPSLV.FEASTOL )
     && ( std::fabs( _PAREst.x[p] - _vPARUB[p] ) > options.NLPSLV.FEASTOL ) )  continue;
    vMULMLE.push_back( mc::FFVar( &dagmle, "MP["+std::to_string(p)+"]" ) );
    dMULMLE.push_back( _PAREst.ux[p] );
    FMLE += vPARMLE[p] * vMULMLE[na++]; // omit bound constant since to be differentiated
  }

#ifdef MAGNUS__PAREST_CONF_DEBUG
  auto sgFMLE = dagmle.subgraph( 1, &FMLE );
  std::vector<FFExpr> exFMLE = FFExpr::subgraph( &dagmle, sgFMLE ); 
  std::cout << "FMLE = " << exFMLE[0] << std::endl;
#endif

  // Parameter linearized covariance matrix
  //mc::FFODE::options.DIFF = mc::FFODE::Options::SYM_P;
  mc::FFODE::options.SYMDIFF = vPARMLE;
  auto pDFMLE = dagmle.FAD( 1, &FMLE, _np, vPARMLE.data(), na, vMULMLE.data(), _nd, vYMMLE.data() );
#ifdef MAGNUS__PAREST_CONF_DEBUG
  auto sgDFMLE = dagmle.subgraph( _np+na, pDFMLE );
  std::vector<FFExpr> exDFMLE = FFExpr::subgraph( &dagmle, sgDFMLE );
  for( unsigned k=0; k<_np+na; ++k )
    std::cout << "DFMLE(" << k << ") = " << exDFMLE[k] << std::endl;
  std::vector<double> dDFMLE( _np+na );
  dagmle.eval( _np+na, pDFMLE, dDFMLE.data(), _np, vPARMLE.data(), _PAREst.x.data(),
               vUMLE.size(), vUMLE.data(), dUMLE.data(), _nd, vYMMLE.data(), dYMMLE.data(),
               _nc, vCSTMLE.data(), _PAREst.p.data(), na, vMULMLE.data(), dMULMLE.data() );
  std::cout << "Lagrangian gradient\n " << arma::trans( arma::vec( dDFMLE.data(), _np+na, false ) );
#endif

  //mc::FFODE::options.DIFF = mc::FFODE::Options::NUM_P;
  mc::FFODE::options.SYMDIFF.clear();
  auto tD2FMLE = dagmle.SFAD( _np+na+_nd, pDFMLE, _np, _vPAR.data(), na, vMULMLE.data() );
  size_t const ne = std::get<0>( tD2FMLE );
  std::vector<double> dD2FMLE( ne );
#ifdef MAGNUS__PAREST_CONF_DEBUG
  auto sgD2FMLE = dagmle.subgraph( ne, std::get<3>(tD2FMLE) );
  std::vector<FFExpr> exD2FMLE = FFExpr::subgraph( &dagmle, sgD2FMLE );
  for( unsigned k=0; k<ne; ++k )
    std::cout << "D2FMLE(" << std::get<1>(tD2FMLE)[k] << "," <<  std::get<2>(tD2FMLE)[k] << ") = "
              << exD2FMLE[k] << std::endl;
#endif

  dagmle.eval( ne, std::get<3>(tD2FMLE), dD2FMLE.data(), _np, vPARMLE.data(), _PAREst.x.data(),
               vUMLE.size(), vUMLE.data(), dUMLE.data(), _nd, vYMMLE.data(), dYMMLE.data(),
               _nc, vCSTMLE.data(), _PAREst.p.data(), na, vMULMLE.data(), dMULMLE.data() );

  arma::mat mD2FMLEDYDP( _nd, _np+na );//, arma::fill::zeros );
  arma::mat mD2FMLEDP2( _np+na, _np+na, arma::fill::zeros );
  for( unsigned k=0; k<ne; ++k ){
    unsigned i = std::get<1>(tD2FMLE)[k];
    unsigned j = std::get<2>(tD2FMLE)[k];
    if( i < _np+na ) mD2FMLEDP2(i,j)         = dD2FMLE[k];
    else             mD2FMLEDYDP(i-_np-na,j) = dD2FMLE[k];
  }
#ifdef MAGNUS__PAREST_CONF_DEBUG
  std::cout << "d2FMLE/dP2 =\n"   << mD2FMLEDP2;
  std::cout << "d2FMLE/dYdP =\n " << mD2FMLEDYDP;
  std::cout << "Measurement covariance\n " << mYCOV;
#endif

  arma::mat mA = arma::inv( mD2FMLEDP2 ) * arma::trans( mD2FMLEDYDP );
  arma::mat mPCOV = mA * mYCOV * arma::trans(mA);

#ifdef MAGNUS__PAREST_CONF_DEBUG
  mPCOV.raw_print( os, "Parameter-multiplier covariance:" );
  mPCOV.submat( 0, 0, _np-1, _np-1 ).raw_print( os, "Parameter-covariance:" );
  //std::cout << "Eivenvalues\n " << arma::trans( arma::eig_sym( mPCOV ) );
  //std::cout << "Rank: " << arma::rank( mPCOV ) << std::endl;
#endif
  delete[] pDFMLE;
  delete[] std::get<1>(tD2FMLE);
  delete[] std::get<2>(tD2FMLE);
  delete[] std::get<3>(tD2FMLE);

  if( options.DISPLEVEL ){
    os << std::scientific << std::setprecision(5);
    mPCOV.submat( 0,0, _np-1,_np-1 ).raw_print( os, "\n# PARAMETER COVARIANCE MATRIX (FROM LINEARIZATION):" );
  }

  return mPCOV.submat( 0, 0, _np-1, _np-1 );
}

inline
arma::vec
PAREST::conf_interval
( arma::mat const& covmat, double const& conflevel, std::string const& type,
  std::ostream& os )
{
#ifdef MAGNUS__PAREST_CONF_DEBUG
  std::cout << "Parameter standard deviations\n " << arma::trans( arma::sqrt( covmat.diag() ) );
#endif

  arma::vec cint = arma::diagvec( arma::sqrt( covmat ) );
  if( type == "Z" )
    cint *= quantile( normal(), (1+conflevel)/2 );
  else if( type == "T" )
    cint *= quantile( students_t( _nd-_np ), (1+conflevel)/2 );
  else
    throw Exceptions( Exceptions::BADOPTION );

  if( options.DISPLEVEL ){
    os << std::endl << "# " << std::fixed << std::setprecision(0) << conflevel*1e2
       << "% PARAMETER CONFIDENCE INTERVALS:\n"
       << std::scientific << std::setprecision(5);
    cint.t().raw_print( os );
  }

  return cint;
}

inline
arma::mat
PAREST::conf_ellipsoid
( arma::mat const& covmat, size_t const i, size_t const j, double const& conflevel,
  std::string const& type, size_t const nsam, std::ostream& os )
{
  arma::mat CRSam;

  // Compute limit
  double rhs;
  arma::vec cint = arma::diagvec( arma::sqrt( covmat ) );

  if( type == "C" )
    rhs = quantile( chi_squared( _np ), conflevel );

  else if( type == "F" ){
    FFMLE OpMLE;
    _MLECrit = 0.;
    for( size_t m=0; m<_nm; ++m ){
      if( !_ny[m] ) continue;
      _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_PAREst.p, &_vDAT[m] );
    }
    try{
      _dag->eval( _wkOUT, 1, &_MLECrit, &rhs, _np, _vPAR.data(), _PAREst.x.data(),
                  _nc, _vCST.data(), _PAREst.p.data() );
      rhs *= 2. * _np/double(_nd-_np) * quantile( fisher_f( _np, _nd-_np ), conflevel );
    }
    catch(...){
      return CRSam;
    }
  }

  else
    throw Exceptions( Exceptions::BADOPTION );
#ifdef MAGNUS__PAREST_CONF_DEBUG
  std::cout << "Ellipsoid limit: " << rhs << std::endl;
#endif

  arma::rowvec range = arma::linspace<arma::rowvec>( 0, 2*arma::datum::pi, nsam );
  arma::mat points( 2, nsam );
  points.row(0) = arma::cos( range );
  points.row(1) = arma::sin( range );
#ifdef MAGNUS__PAREST_CONF_DEBUG
  points.raw_print( os, "Points sampled on a circle of radius 1:\n" );
#endif

  arma::uvec ij( {i,j} );
#ifdef MAGNUS__PAREST_CONF_DEBUG
  std::cout << "Square-root of (" << i << "," << j << ") covariance matrix:\n" << arma::sqrtmat_sympd( covmat.submat(ij,ij) );
  std::cout << arma::sqrtmat_sympd( covmat.submat(ij,ij) ) * arma::sqrtmat_sympd( covmat.submat(ij,ij) );
#endif

  CRSam = points.t() * arma::sqrtmat_sympd( covmat.submat(ij,ij) ) * std::sqrt(rhs);
  CRSam += arma::kron( arma::conv_to<arma::mat>::from(arma::vec( _PAREst.x.data(), _np, false ) ).rows( ij ),
                       arma::mat( 1, nsam, arma::fill::ones ) ).t();
  if( options.DISPLEVEL ){
    os << std::endl << "# " << std::fixed << std::setprecision(0) << conflevel*1e2
       << "% CONFIDENCE ELLIPSOID FOR PARAMETERS (" << i << "," << j << "):\n"
       << std::scientific << std::setprecision(5);
    CRSam.brief_print( os );
  }

  return CRSam;
}

// IMPLEMENT LIKELIHOOD RATIO CONFIDENCE REGION WHEN NESTED SAMPLING AVAILABLE IN MAGNUS

} // end namespace mc
#endif

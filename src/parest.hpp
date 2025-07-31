// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_PAREST Parameter Estimation in Parametric Models with MC++
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

  //! @brief Structure holding NLP intermediate solution
  SOLUTION_OPT _MLEOpt;

  //! @brief Sampled confidence current MLE criterion
  arma::mat _CRSam;
  
  //! @brief current MLE criterion
  FFVar _MLECrit;

public:
  /** @defgroup PAREST Parameter Estimation of Parametric Models using MC++
   *  @{
   */
   
  //! @brief Constructor
  PAREST()
    : _dag(nullptr),
      _MLECrit(0.)
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
      : NLPSLV()
      { reset(); }

    //! @brief Reset to default options
    void reset
      ()
      {
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
        DISPLEVEL                   = 1;
        RNGSEED                     = -1;
      }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        DISPLEVEL   = options.DISPLEVEL;
        RNGSEED     = options.RNGSEED;
        NLPSLV      = options.NLPSLV;
        return *this;
      }

    //! @brief Verbosity level
    int                      DISPLEVEL;
    //! @brief Random-number generator seed. >=0: seed set to specified value; <0: seed set to a value drawn from std::random_device
    int                      RNGSEED;
    //! @brief NLP gradient-based solver options
    typename NLP::Options    NLPSLV;
  } options;

  //! @brief PAREST solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for PAREST exception handling
    enum TYPE{
      BADSIZE=0,    //!< Inconsistent dimensions
      BADOPTION,    //!< Incorrect option
      NOMODEL,	    //!< unspecified model
      NODATA,	    //!< unspecified data
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
      { walltime_all = walltime_setup = walltime_slvmle =
        std::chrono::microseconds(0); }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "#  WALL-CLOCK TIMES" << std::endl
           << "#  SOLVER SETUP:         " << std::setw(10) << to_time( walltime_setup )   << " SEC" << std::endl
           << "#  GRADIENT-BASED SOLVE: " << std::setw(10) << to_time( walltime_slvmle )  << " SEC" << std::endl
           << "#  TOTAL:                " << std::setw(10) << to_time( walltime_all )     << " SEC" << std::endl
           << std::endl; }
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime_all;
    //! @brief Cumulated wall-clock time used for problem setup (in microseconds)
    std::chrono::microseconds walltime_setup;
    //! @brief Cumulated wall-clock time used by gradient-based NLP solver (in microseconds)
    std::chrono::microseconds walltime_slvmle;
    //! @brief Get current time point
    std::chrono::time_point<std::chrono::system_clock> start
      () const
      { return std::chrono::system_clock::now(); }
    //! @brief Get current time lapse with respect to start time point
    std::chrono::microseconds walltime
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

  //! @brief Compute covariance matrix using bootstrapping around given data set
  arma::mat cov_bootstrap
    ( std::vector<std::vector<Experiment>> const& data, size_t const nsam,
      std::ostream& os=std::cout );

  //! @brief Compute covariance matrix using bootstrapping around given data set
  arma::mat cov_bootstrap
    ( std::vector<Experiment> const& data, size_t const nsam,
      std::ostream& os=std::cout );

  //! @brief Compute covariance matrix using bootstrapping around maximum likelihood fit
  arma::mat cov_bootstrap
    ( size_t const nsam, std::ostream& os=std::cout );

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

  //! @brief MLE solution
  SOLUTION_OPT const& mle
    ()
    const
    { return _MLEOpt; }

  //! @brief Sampled confidence region
  arma::mat const& crsam
    ()
    const
    { return _CRSam; }
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

  if( _nc ){
    _vCST.resize( _nc );
    _dag->insert( BASE_PAREST::_dag, _nc, BASE_PAREST::_vCST.data(), _vCST.data() );
  }

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

  stats.walltime_setup += stats.walltime( t_setup );
  stats.walltime_all   += stats.walltime( t_setup );
  return true;
}

inline
int
PAREST::mle_solve
( std::vector<double> const& P0, std::vector<double> const& C0, std::ostream& os )
{
  if( !_nd  )          throw Exceptions( Exceptions::NODATA );
  if( !_vOUT.size()  ) throw Exceptions( Exceptions::NOMODEL );

  auto&& t_slvmle = stats.start();

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
    _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &C0, &_vDAT[m] );
  }
  FFLin<I> OpSum;
  if( _nr )
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit + OpSum( _vREG, 1. ) );
  else
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit );
    
  for( size_t g=0; g<_ng; ++g )
    PE.add_ctr( std::get<1>(_vCTR).at(g), std::get<0>(_vCTR).at(g)-std::get<2>(_vCTR).at(g) );

  PE.setup();
  int iflag = PE.solve( P0.data(), nullptr, nullptr, C0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << PE.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << PE.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( options.DISPLEVEL )
    os << "# PARAMETER ESTIMATION SOLUTION:\n" << PE.solution();

  _MLEOpt = PE.solution();

  stats.walltime_slvmle += stats.walltime( t_slvmle );
  stats.walltime_all    += stats.walltime( t_slvmle );

  return iflag;
}

inline
int
PAREST::mle_solve
( size_t const nsam, std::vector<double> const& C0, std::ostream& os )
{
  auto&& t_slvmle = stats.start();

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
    _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &C0, &_vDAT[m] );
  }
  FFLin<I> OpSum;
  if( _nr )
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit + OpSum( _vREG, 1. ) );
  else
    PE.set_obj( mc::BASE_OPT::MIN, _MLECrit );
    
  for( size_t g=0; g<_ng; ++g )
    PE.add_ctr( std::get<1>(_vCTR).at(g), std::get<0>(_vCTR).at(g)-std::get<2>(_vCTR).at(g) );

  PE.setup();
  int iflag = PE.solve( nsam, _vPARLB.data(), _vPARUB.data(), C0.data(), nullptr, 1 );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << PE.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << PE.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( options.DISPLEVEL )
    os << "# PARAMETER ESTIMATION SOLUTION:\n" << PE.solution();

  _MLEOpt = PE.solution();

  stats.walltime_slvmle += stats.walltime( t_slvmle );
  stats.walltime_all    += stats.walltime( t_slvmle );

  return iflag;
}

inline
std::tuple<double,double,double>
PAREST::chi2_test
( double const& conf, std::ostream& os )
{
  return chi2_test( conf, _MLEOpt.x, _MLEOpt.p, os );
}

inline
std::tuple<double,double,double>
PAREST::chi2_test
( double const& conf, std::vector<double> const& P0, std::vector<double> const& C0, std::ostream& os )
{
  double Chi2Val = 0./0.;
  chi_squared dist( _nd-_np );
  double Chi2Crit = quantile( dist, conf );

  // Compute MLE residual
  FFMLE OpMLE;
  _MLECrit = 0.;
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &C0, &_vDAT[m] );
  }
  try{
    _dag->eval( _wkOUT, 1, &_MLECrit, &Chi2Val, _np, _vPAR.data(), P0.data(), _nc, _vCST.data(), C0.data() );
    Chi2Val *= 2;
  }
  catch(...){
    return std::make_tuple( Chi2Val, 0., Chi2Crit );
  }

  if( options.DISPLEVEL > 0 ){
    os << "\n# CHI-SQUARED TEST: " << std::scientific << std::setprecision(3) << Chi2Val
       << (Chi2Val < Chi2Crit? " < ": " > ") << Chi2Crit << " CRITICAL CHI_SQUARED VALUE (95%, " << _nd-_np << " DOF)"
       << std::endl;
  }
  
  double Chi2Conf = cdf( dist, Chi2Val );
  if( options.DISPLEVEL > 0 )
    os << "# CHI-SQUARED TEST PASSED WITH >" << std::fixed << std::setprecision(1) << Chi2Conf*1e2 << "% CONFIDENCE LEVEL"
       << std::endl;

  return std::make_tuple( Chi2Val, Chi2Conf, Chi2Crit );
}

inline
arma::mat
PAREST::cov_bootstrap
( size_t const nsam, std::ostream& os )
{  
  _CRSam.reset();

  if( NLP::get_status( _MLEOpt.stat ) != NLP::SUCCESSFUL 
   && NLP::get_status( _MLEOpt.stat ) != NLP::FAILURE )
    return _CRSam;

  // Simulate model at MLE estimate
  double MLERes = 0./0.;
  FFMLE OpMLE;
  _MLECrit = 0.;
  std::vector<FFVar> vMLECrit( _nm, 0. );
  for( size_t m=0; m<_nm; ++m ){
    if( !_ny[m] ) continue;
    vMLECrit[m] = OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_MLEOpt.p, &_vDAT[m] );
    _MLECrit += vMLECrit[m];
  }

  try{
    _dag->eval( _wkOUT, 1, &_MLECrit, &MLERes, _np, _vPAR.data(), _MLEOpt.x.data(),
                _nc, _vCST.data(), _MLEOpt.p.data() );
  }
  catch(...){
    return _CRSam;
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

  return cov_bootstrap( vDAT0, nsam, os );
}

inline
arma::mat
PAREST::cov_bootstrap
( std::vector<Experiment> const& data, size_t const nsam, std::ostream& os )
{
  std::vector<std::vector<Experiment>> vDAT0;
  for( auto const& exp : data )
    _add_data( vDAT0, exp );

  return cov_bootstrap( vDAT0, nsam, os );
}

inline
arma::mat
PAREST::cov_bootstrap
( std::vector<std::vector<Experiment>> const& data, size_t const nsam, std::ostream& os )
{
  _CRSam.reset();

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

  // Apply bootstrapping to MLE problem
  for( size_t isam=0; isam<nsam; ++isam ){

    // Add measurement noise
    auto vDATn = data;
    auto mean = arma::vec( 1, arma::fill::zeros );
    auto var  = arma::mat( 1, 1, arma::fill::none );
    for( size_t m=0, e=0; m<_nm; ++m, e=0 ){
      for( auto& EXP : vDATn[m] ){
        for( auto& [ k, RECk ] : EXP.output ){
#ifdef MAGNUS__PAREST_CONF_DEBUG
          size_t r = 0;
#endif
          for( auto& YMk : RECk.measurement ){
            var(0,0) = RECk.variance;
            //arma::mat dYk = arma::mvnrnd( mean, var );
            //YMk += dYk(0,0);
            YMk += quantile( normal( 0, std::sqrt(RECk.variance) ), noise() );
#ifdef MAGNUS__PAREST_CONF_DEBUG
            std::cout << "YM(" << isam << ")[" << std::to_string(m) << "," << std::to_string(e) << "]["
                      << std::to_string(k) << "][" << std::to_string(r++) << "] = "
                      << YMk << std::endl;
#endif
          }
        }
        ++e;
      }
    }
    
    // Update MLE objective - could also use set_obj_lazy - and solve from MLE estimate
    FFMLE OpMLE;
    _MLECrit = 0.;
    for( size_t m=0; m<_nm; ++m ){
      if( !_ny[m] ) continue;
      _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_MLEOpt.p, &vDATn[m] );
    }
    if( _nr )
      PE.set_obj( mc::BASE_OPT::MIN, _MLECrit + OpSum( _vREG, 1. ) );
    else
      PE.set_obj( mc::BASE_OPT::MIN, _MLECrit );
    
    PE.setup();
    PE.solve( _MLEOpt.x.data(), nullptr, nullptr, _MLEOpt.p.data() );

    if( options.DISPLEVEL > 2 )
      os << "#  SAMPLE:     " << isam                     << std::endl
         << "#  FEASIBLE:   " << PE.is_feasible( 1e-6 )   << std::endl
         << "#  STATIONARY: " << PE.is_stationary( 1e-6 ) << std::endl
         << std::endl;

    if( PE.get_status() == NLP::SUCCESSFUL 
     || ( ( PE.get_status() == NLP::FAILURE
         || PE.get_status() == NLP::INTERRUPTED )
         && PE.is_feasible( options.NLPSLV.FEASTOL ) ) ){
      _CRSam.insert_rows( _CRSam.n_rows, arma::mat( const_cast<double*>(PE.solution().x.data()), 1, _np, false ) );
      if( options.DISPLEVEL > 1 ){
        os << std::setw(5) << std::right << isam << ": "
           << std::scientific << std::setprecision(5) << std::setw(12) << PE.solution().f[0] << " | ";
        auto vecx = arma::vec( PE.solution().x );
        vecx.t().raw_print( os );
      }
    }
  }

  // Estimate covariance around ML estimates
  arma::mat covSam = _CRSam - arma::kron( arma::conv_to<arma::mat>::from(arma::vec( _MLEOpt.x.data(), _np, false ) ), arma::mat( 1, _CRSam.n_rows, arma::fill::ones ) ).t();
  covSam = ( covSam.t() * covSam ) / covSam.n_rows;
  //arma::mat covSam = arma::cov(_CRSam);

  if( options.DISPLEVEL > 0 ){
    os << std::scientific << std::setprecision(5);
    // os << "\nSampled confidence region:\n" << _CRSam;
    // _CRSam_c.raw_print( os, "\nCENTERED BOOTSTRAP CONFIDENCE REGION:" );
    covSam.raw_print( os, "\n# PARAMETER COVARIANCE MATRIX (VIA BOOTSTRAPPING):" );
    //arma::cov(_CRSam).raw_print( os, "\nPARAMETER COVARIANCE MATRIX (VIA BOOTSTRAPPING):" );
  }

  return covSam;
}

inline
arma::mat
PAREST::cov_linearized
( std::ostream& os )
{
  if( NLP::get_status( _MLEOpt.stat ) != NLP::SUCCESSFUL 
   && NLP::get_status( _MLEOpt.stat ) != NLP::FAILURE )
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
        if( std::fabs( _MLEOpt.f[g+1] ) > options.NLPSLV.FEASTOL ) continue;
      case BASE_OPT::EQ: 
        vMULMLE.push_back( mc::FFVar( &dagmle, "MG["+std::to_string(g)+"]" ) );
        dMULMLE.push_back( _MLEOpt.uf[g+1] );
        FMLE += vCTRMLE[g] * vMULMLE[na++];
        break;
    }
  }

  // Add active bound constraints
  for( size_t p=0; p<_np; ++p ){  
    if( ( std::fabs( _MLEOpt.x[p] - _vPARLB[p] ) > options.NLPSLV.FEASTOL )
     && ( std::fabs( _MLEOpt.x[p] - _vPARUB[p] ) > options.NLPSLV.FEASTOL ) )  continue;
    vMULMLE.push_back( mc::FFVar( &dagmle, "MP["+std::to_string(p)+"]" ) );
    dMULMLE.push_back( _MLEOpt.ux[p] );
    FMLE += vPARMLE[p] * vMULMLE[na++]; // omit bound constant since to be differentiated
  }

#ifdef MAGNUS__PAREST_CONF_DEBUG
  auto sgFMLE = dagmle.subgraph( 1, &FMLE );
  std::vector<FFExpr> exFMLE = FFExpr::subgraph( &dagmle, sgFMLE ); 
  std::cout << "FMLE = " << exFMLE[0] << std::endl;
#endif

  // Parameter linearized covariance matrix
  mc::FFODE::options.DIFF = mc::FFODE::Options::SYM_P;
  auto pDFMLE = dagmle.FAD( 1, &FMLE, _np, vPARMLE.data(), na, vMULMLE.data(), _nd, vYMMLE.data() );
#ifdef MAGNUS__PAREST_CONF_DEBUG
  auto sgDFMLE = dagmle.subgraph( _np+na, pDFMLE );
  std::vector<FFExpr> exDFMLE = FFExpr::subgraph( &dagmle, sgDFMLE );
  for( unsigned k=0; k<_np+na; ++k )
    std::cout << "DFMLE(" << k << ") = " << exDFMLE[k] << std::endl;
  std::vector<double> dDFMLE( _np+na );
  dagmle.eval( _np+na, pDFMLE, dDFMLE.data(), _np, vPARMLE.data(), _MLEOpt.x.data(),
               vUMLE.size(), vUMLE.data(), dUMLE.data(), _nd, vYMMLE.data(), dYMMLE.data(),
               _nc, vCSTMLE.data(), _MLEOpt.p.data(), na, vMULMLE.data(), dMULMLE.data() );
  std::cout << "Lagrangian gradient\n " << arma::trans( arma::vec( dDFMLE.data(), _np+na, false ) );
#endif

  mc::FFODE::options.DIFF = mc::FFODE::Options::NUM_P;
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

  dagmle.eval( ne, std::get<3>(tD2FMLE), dD2FMLE.data(), _np, vPARMLE.data(), _MLEOpt.x.data(),
               vUMLE.size(), vUMLE.data(), dUMLE.data(), _nd, vYMMLE.data(), dYMMLE.data(),
               _nc, vCSTMLE.data(), _MLEOpt.p.data(), na, vMULMLE.data(), dMULMLE.data() );

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
    mPCOV.submat( 0, 0, _np-1, _np-1 ).raw_print( os, "\n# PARAMETER COVARIANCE MATRIX (VIA LINEARIZATION):" );
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
      _MLECrit += OpMLE( _vPAR.data(), _dag, _vPAR, _vCST, _vCON[m], _vOUT[m], &_MLEOpt.p, &_vDAT[m] );
    }
    try{
      _dag->eval( _wkOUT, 1, &_MLECrit, &rhs, _np, _vPAR.data(), _MLEOpt.x.data(),
                  _nc, _vCST.data(), _MLEOpt.p.data() );
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
  CRSam += arma::kron( arma::conv_to<arma::mat>::from(arma::vec( _MLEOpt.x.data(), _np, false ) ).rows( ij ),
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

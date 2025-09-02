// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_MODISCR Design of Experiments for Model Discrimination
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 0.2
\date 2025
\bug No known bugs.
*/

#ifndef MAGNUS__MODISCR_HPP
#define MAGNUS__MODISCR_HPP

#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <algorithm>

#if defined( MC__USE_PROFIL )
 #include "mcprofil.hpp"
#elif defined( MC__USE_BOOST )
 #include "mcboost.hpp"
#elif defined( MC__USE_FILIB )
 #include "mcfilib.hpp"
#else
 #include "interval.hpp"
#endif

#ifdef MC__USE_GUROBI
 #include "mipslv_gurobi.hpp"
#elif  MC__USE_CPLEX
 #include "mipslv_cplex.hpp"
#endif

#ifdef MC__USE_SNOPT
 #include "nlpslv_snopt.hpp"
#elif  MC__USE_IPOPT
 #include "nlpslv_ipopt.hpp"
#endif

#include "minlpslv.hpp"

#include "base_mbdoe.hpp"
#include "fflin.hpp"
#include "ffdoe.hpp"
#include "ffode.hpp"

namespace mc
{
//! @brief C++ class for design of experiments for model discrimination
////////////////////////////////////////////////////////////////////////
//! mc::MODISCR is a C++ class for solving design of experiments for
//! model discrimination using MC++, CRONOS and CANON
////////////////////////////////////////////////////////////////////////
class MODISCR
: public virtual BASE_MBDOE
{

protected:

#if defined( MC__USE_PROFIL )
 typedef ::INTERVAL I;
#elif defined( MC__USE_BOOST )
 typedef boost::numeric::interval_lib::save_state<boost::numeric::interval_lib::rounded_transc_opp<double>> T_boost_round;
 typedef boost::numeric::interval_lib::checking_base<double> T_boost_check;
 typedef boost::numeric::interval_lib::policies<T_boost_round,T_boost_check> T_boost_policy;
 typedef boost::numeric::interval<double,T_boost_policy> I;
#elif defined( MC__USE_FILIB )
 typedef filib::interval<double> I;
#else
 typedef Interval I;
#endif

  typedef FFGraph DAG;

#if defined( MC__USE_GUROBI )
  typedef MIPSLV_GUROBI<I> MIP;
#elif defined( MC__USE_CPLEX )
  typedef MIPSLV_CPLEX<I> MIP;
#endif
#if defined( MC__USE_SNOPT )
  typedef NLPSLV_SNOPT NLP;
#elif defined( MC__USE_IPOPT )
  typedef NLPSLV_IPOPT NLP;
#endif
  typedef MINLPSLV<I,NLP,MIP> MINLP;
  
private:

  //! @brief DAG of model
  DAG* _dag;

  //! @brief DAG of MBDoE problems
  DAG* _dagdoe;

  //! @brief local copy of model parameters
  std::vector<FFVar> _vPAR;

  //! @brief local copy of model constants
  std::vector<FFVar> _vCST;

  //! @brief local copy of experimental controls
  std::vector<FFVar> _vCON;

  //! @brief vector of experimental control samples
  std::vector<std::vector<double>> _vCONSAM;

  //! @brief local copy of model outputs
  std::vector<std::vector<FFVar>> _vOUT;

  //! @brief local copy of model outputs as single vector
  std::vector<FFVar> _vOUTVEC;

  //! @brief output subgraph
  std::vector<FFSubgraph> _sgOUT;

  //! @brief work array for output evaluations
  std::vector<double> _wkOUT;

  //! @brief output values
  std::vector<std::vector<double>> _dOUT;
  
  //! @brief vector of response vectors: 
  std::vector<std::vector<std::vector<arma::vec>>> _vOUTSAM;

public:
  /** @defgroup MODISCR Design of Experiments for Model Discrimination
   *  @{
   */
   
  //! @brief Constructor
  MODISCR()
    : _dag(nullptr), _dagdoe(nullptr),
      _VOpt(0./0.)
    { stats.reset(); }

  //! @brief Destructor
  virtual ~MODISCR()
    {
      delete   _dag;
      delete   _dagdoe;
    }

  //! @brief MODISCR solver options
  struct Options
  {
    //! @brief Constructor
    Options()
      : MINLPSLV(), NLPSLV()
      {
        reset();
      }

    //! @brief Reset to default options
    void reset
      ()
      {
        CRITERION                   = BASE_MBDOE::BRISK;
        RISK                        = AVERSE;
        CVARTHRES                   = 0.10;
        MINDIST                     = 1e-6;
        MAXITER                     = 4;
        TOLITER                     = 1e-4;
        DISPLEVEL                   = 1;
#ifdef MC__USE_SNOPT
        NLPSLV.DISPLEVEL            = DISPLEVEL;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-6;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#elif  MC__USE_IPOPT
        NLPSLV.DISPLEVEL            = DISPLEVEL;
        NLPSLV.MAXITER              = 500;
        NLPSLV.FEASTOL              = 1e-6;
        NLPSLV.OPTIMTOL             = 1e-5;
        NLPSLV.GRADMETH             = NLP::Options::FSYM;
        NLPSLV.HESSMETH             = NLP::Options::LBFGS;
        NLPSLV.GRADCHECK            = 0;
        NLPSLV.MAXTHREAD            = 0;
#endif
        MINLPSLV.reset();
        MINLPSLV.SEARCHALG          = MINLP::Options::OA;
        MINLPSLV.DISPLEVEL          = DISPLEVEL;
        MINLPSLV.CVRTOL             = 1e-5;
        MINLPSLV.CVATOL             = 1e-8;
        MINLPSLV.FEASTOL            = 1e-6;
        MINLPSLV.FEASPUMP           = 0;
        MINLPSLV.ROOTCUT            = 1;
        MINLPSLV.TIMELIMIT          = 36e2;
        MINLPSLV.LINMETH            = MINLP::Options::CVX;
        MINLPSLV.MAXITER            = 200;
        MINLPSLV.MSLOC              = 1;
        MINLPSLV.CPMAX              = 5;
        MINLPSLV.NLPSLV.DISPLEVEL   = 0;
#ifdef MC__USE_GUROBI
        MINLPSLV.MIPSLV.DISPLEVEL   = 0;
        MINLPSLV.MIPSLV.THREADS     = 0;
        MINLPSLV.MIPSLV.MIPRELGAP   = 1e-6;
        MINLPSLV.MIPSLV.MIPABSGAP   = 1e-9;
        MINLPSLV.MIPSLV.OUTPUTFILE  = "";//"doe.lp";
#elif  MC__USE_CPLEX
        throw std::runtime_error("Error: CPLEX solver not yet implemented");
#endif
        FFDOEBase::type             = CRITERION;
      }
    //! @brief Assignment operator
    Options& operator= ( Options const& options ){
        CRITERION   = options.CRITERION;
        RISK        = options.RISK;
        CVARTHRES   = options.CVARTHRES;
        MINDIST     = options.MINDIST;
        MAXITER     = options.MAXITER;
        TOLITER     = options.TOLITER;
        DISPLEVEL   = options.DISPLEVEL;
        MINLPSLV    = options.MINLPSLV;
        NLPSLV      = options.NLPSLV;
        return *this;
      }
    //! @brief Enumeration type for risk attitude
    enum RISK_TYPE{
      NEUTRAL=0, //!< Perform a risk-neutral average design
      AVERSE     //!< Perform a risk-averse CVaR design
    };
    //! @brief Selected DOE criterion
    BASE_MBDOE::TYPE         CRITERION;
    //! @brief Selected risk attitude
    RISK_TYPE                RISK;
    //! @brief Percentile threshold for CVaR calculation
    double                   CVARTHRES;
    //! @brief Minimal relative mean-absolute distance between support points after refinement
    double                   MINDIST;
   //! @brief Maximal iteration of effort-based and gradient-based solves
    int                      MAXITER;
   //! @brief Stopping tolerance for effort-based and gradient-based iteration
    double                   TOLITER;
    //! @brief Verbosity level
    int                      DISPLEVEL;
    
    //! @brief MINLP effort-based solver options
    typename MINLP::Options  MINLPSLV;
    //! @brief NLP gradient-based solver options
    typename NLP::Options    NLPSLV;
  } options;

  //! @brief MODISCR solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for MODISCR exception handling
    enum TYPE{
      BADSIZE=0,    //!< Inconsistent dimensions
      NOMODEL,	    //!< Unspecified model
      BADCONST,	    //!< Unspecified constants
      BADCRIT,      //!< Misspecified design criterion
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
          return "MODISCR::Exceptions  Inconsistent dimensions";
        case NOMODEL:
          return "MODISCR::Exceptions  Unspecified model";
        case BADCONST:
          return "MODISCR::Exceptions  Unspecified constants";
        case BADCRIT:
          return "MODISCR::Exceptions  Misspecified design criterion";
        case INTERN:
        default:
          return "MODISCR::Exceptions  Internal error";
      }
    }
  private:
    TYPE _ierr;
  };

  //! @brief MODISCR solver statistics
  struct Stats{
    //! @brief Reset statistics
    void reset()
      { walltime_all = walltime_setup = walltime_samgen = walltime_slvnlp = walltime_slvmip =
        std::chrono::microseconds(0); }
    //! @brief Display statistics
    void display
      ( std::ostream&os=std::cout )
      { os << std::fixed << std::setprecision(2) << std::right
           << std::endl
           << "#  WALL-CLOCK TIMES" << std::endl
           << "#  SOLVER SETUP:         " << std::setw(10) << to_time( walltime_setup )   << " SEC" << std::endl
           << "#  SCENARIO GENERATION:  " << std::setw(10) << to_time( walltime_samgen )  << " SEC" << std::endl
           << "#  EFFORT-BASED SOLVE:   " << std::setw(10) << to_time( walltime_slvmip )  << " SEC" << std::endl
           << "#  GRADIENT-BASED SOLVE: " << std::setw(10) << to_time( walltime_slvnlp )  << " SEC" << std::endl
           << "#  TOTAL:                " << std::setw(10) << to_time( walltime_all )     << " SEC" << std::endl
           << std::endl; }
    //! @brief Total wall-clock time (in microseconds)
    std::chrono::microseconds walltime_all;
    //! @brief Cumulated wall-clock time used for problem setup (in microseconds)
    std::chrono::microseconds walltime_setup;
    //! @brief Cumulated wall-clock time used by sample generation and scenario reduction (in microseconds)
    std::chrono::microseconds walltime_samgen;
    //! @brief Cumulated wall-clock time used by gradient-based NLP solver (in microseconds)
    std::chrono::microseconds walltime_slvnlp;
    //! @brief Cumulated wall-clock time used by effort-based MINLP solver (in microseconds)
    std::chrono::microseconds walltime_slvmip;
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

  //! @brief Setup MODISCR problem before solution
  bool setup
    ();

  //! @brief Evaluate performance of experimental campaign
  std::pair<double,bool> evaluate_design
    ( std::list<std::pair<double,std::vector<double>>> const& Campaign, std::string const& type="",
      std::vector<double> const& vcst=std::vector<double>(), std::ostream& os=std::cout );

  //! @brief Generate <a>NSAM</a> initial supports
  bool sample_support
    ( size_t const NSAM, std::vector<double> const& vcst=std::vector<double>(),
      std::ostream& os=std::cout );

  //! @brief Solve effort-based exact experiment design with <a>NEXP</a> supports
  int effort_solve
    ( size_t const NEXP, bool const exact=true, 
     std::map<size_t,double> const& EIni=std::map<size_t,double>(),
     std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports 
  int gradient_solve
    ( std::map<size_t,double> const& EOpt,  std::vector<double> const& vcst=std::vector<double>(),
      bool const update=true, std::ostream& os=std::cout );

  //! @brief Solve combined effort- and gradient-based experiment design with <a>NEXP</a> supports 
  int combined_solve
    ( size_t const NEXP,  std::vector<double> const& vcst=std::vector<double>(),
     bool const exact=true, std::map<size_t,double> const& EIni=std::map<size_t,double>(),
     std::ostream& os=std::cout );

  //! @brief Export effort, support and fim to file
  bool file_export
    ( std::string const& name );

  //! @brief Retrieve optimized efforts  
  std::map<size_t,double> const& effort
    ()
    const
    { return _EOpt; }

  //! @brief Retrieve optimized supports
  std::map<size_t,std::vector<double>> const& support
    ()
    const
    { return _SOpt; }
  
  //! @brief Retrieve optimized criterion
  double criterion
    ()
    const
    { return _VOpt; }

  //! @brief Retrieve optimized campaign
  std::list<std::pair<double,std::vector<double>>> campaign
    ()
    const;

  //! @brief Retrieve sampled controls
  std::vector< std::vector< double > > const& control_sample
    ()
    const
    { return _vCONSAM; }


  //! @brief Retrieve sampled outputs
  std::vector<std::vector<std::vector<arma::vec>>> const& output_sample
    ()
    const
    { return _vOUTSAM; }

protected:

  //! @brief map of current optimal efforts
  std::set<std::pair<size_t,size_t>> _sPARSEL;

  //! @brief current optimal criterion
  double _VOpt;

  //! @brief map of current optimal efforts
  std::map<size_t,double> _EOpt;
  
  //! @brief map of current optimal supports
  std::map<size_t,std::vector<double>> _SOpt;
  
  //! @brief vector of current optimal values for risk-averse variables
  std::vector<double> _ROpt;

  //! @brief Create local copy of output model for output prediction
  void _setup_out
    ();

  //! @brief Generate output samples for <a>NSAM</a> initial supports
  bool _sample_out
    ( size_t const NSAM, std::ostream& os=std::cout );
  
  //! @brief Append output under given control scenario for all uncertainty scenarios
  bool _append_out
    ( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
      std::vector<std::vector<double>>& Output, std::vector<std::vector<std::vector<arma::vec>>>& Response,
      std::ostream& os=std::cout );

  //! @brief Set Bayes risk-neutral criterion in effort-based MINLP model
  void _effort_set_BRNeutral
    ( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF )
    const;

  //! @brief Set Bayes risk-averse criterion in effort-based MINLP model
  void _effort_set_BRAverse
    ( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF, std::vector<double>& E0 )
    const;

  //! @brief Evaluate Bayes risk-neutral criterion
  double _evaluate_BRNeutral
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double> const& UTOT0, std::ostream& os );

  //! @brief Evaluate Bayes risk-averse criterion
  double _evaluate_BRAverse
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double> const& UTOT0, std::ostream& os );

  //! @brief Set Bayes risk-neutral criterion in gradient-based NLP model
  void _refine_set_BRNeutral
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double> const& UTOT0, std::ostream& os );

  //! @brief Set Bayes risk-averse criterion in gradient-based NLP model
  void _refine_set_BRAverse
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double>& UTOT0, std::ostream& os );

  //! @brief Generate samples for refined supports
  bool _update_supports
    ( std::map<size_t,double> const EOpt, std::map<size_t,std::vector<double>> const SOpt,
      std::ostream& os );

  //! @brief Determine if support <a>supp</a> is redundant with an existing support
  size_t _redundant_support
    ( std::vector<double> const& supp );

  //! @brief Mean-absolute error between two supports
  double _mae_support
    ( std::vector<double> const& s1, std::vector<double> const& s2 );

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit, std::map<size_t,double> const& eff,
      std::map<size_t,std::vector<double>> const& supp, std::ostream& os )
    const;

  //! @brief Display current efforts and supports
  void _display_design
    ( std::string const& title, double const& crit,
      std::list<std::pair<double,std::vector<double>>> const& campaign,
      std::ostream& os )
    const;
};

inline
bool
MODISCR::setup
()
{
  stats.reset();
  auto&& t_setup = stats.start();

  if( !_nm || !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  switch( options.CRITERION ){
   case BRISK:
    _setup_out();
    break;

   default:
    throw Exceptions( Exceptions::BADCRIT );
  }

  stats.walltime_setup += stats.walltime( t_setup );
  stats.walltime_all   += stats.walltime( t_setup );
  return true;
}

inline
void
MODISCR::_setup_out
()
{
  if( _nm != BASE_MBDOE::_vOUT.size() || _ny != BASE_MBDOE::_vOUT[0].size() || !BASE_MBDOE::_vCON.size() )
    throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;
  _dag->options = BASE_MBDOE::_dag->options;

  _vCON.resize( _nu );
  _dag->insert( BASE_MBDOE::_dag, _nu, BASE_MBDOE::_vCON.data(), _vCON.data() );
  _vPAR.resize( _np );
  _dag->insert( BASE_MBDOE::_dag, _np, BASE_MBDOE::_vPAR.data(), _vPAR.data() );
  _vCST.resize( _nc );
  _dag->insert( BASE_MBDOE::_dag, _nc, BASE_MBDOE::_vCST.data(), _vCST.data() );
  _vOUT.resize( _nm );
  _vOUTVEC.clear(); _vOUTVEC.reserve( _nm * _ny );
  _sgOUT.resize( _nm );
  for( size_t m=0; m<_nm; ++m ){
    _vOUT[m].resize( _ny );
    _dag->insert( BASE_MBDOE::_dag, _ny, BASE_MBDOE::_vOUT[m].data(), _vOUT[m].data() );
    _sgOUT[m].clear();
#ifdef MAGNUS__MODISCR_SETUP_DEBUG
    _sgOUT[m] = _dag->subgraph( _ny, _vOUT[m].data() );
    std::vector<FFExpr> exOUT = FFExpr::subgraph( _dag, _sgOUT[m] ); 
    for( size_t i=0; i<_ny; ++i )
      std::cout << "OUT[" << i << "] = " << exOUT[i] << std::endl;
#endif
    _vOUTVEC.insert( _vOUTVEC.end(), _vOUT[m].cbegin(), _vOUT[m].cend() );
  }
}

inline
bool
MODISCR::sample_support
( size_t const NSAM, std::vector<double> const& vcst, std::ostream& os )
{
  auto&& t_samgen = stats.start();

  // Update constants
  if( vcst.size() && vcst.size() == _nc ) _vCSTVAL = vcst;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Control samples
  typedef boost::random::sobol_engine< boost::uint_least64_t, 64u > sobol64;
  typedef boost::variate_generator< sobol64, boost::uniform_01< double > > qrgen;
  sobol64 eng( _nu );
  qrgen gen( eng, boost::uniform_01<double>() );
  gen.engine().seed( 0 );

  _vCONSAM.clear();
  _vCONSAM.reserve( NSAM );
  for( size_t s=0; s<NSAM; ++s ){
    _vCONSAM.push_back( std::vector<double>( _nu ) );
    for( size_t i=0; i<_nu; i++ )
      _vCONSAM.back()[i] = _vCONLB[i] + ( _vCONUB[i] - _vCONLB[i] ) * gen();
  }

  // Observation samples
  bool flag = false;
  switch( options.CRITERION ){
    case BRISK:
      flag = _sample_out( NSAM, os );
      break;

   default:
    throw Exceptions( Exceptions::BADCRIT );
  }

  if( options.DISPLEVEL )
    os << std::endl;

  stats.walltime_samgen += stats.walltime( t_samgen );
  stats.walltime_all    += stats.walltime( t_samgen );
  return flag;
}

inline
bool
MODISCR::_sample_out
( size_t const NSAM, std::ostream& os )
{
  auto&& tstart = stats.start();
  int DISPFREQ = (_ne0+NSAM)/20;
  if( options.DISPLEVEL )
    os << "** GENERATING SUPPORT SAMPLES     |" << std::flush;

  // Reset and resize output/intermediate containers
  size_t NUNC = _vPARVAL.size();
  _vOUTSAM.clear();
  _vOUTSAM.resize( NUNC );
  _dOUT.resize( NUNC );

  // Compute responses for every a priori control and uncertainty scenarios
  for( size_t e=0; e<_ne0; ++e ){
    if( !_append_out( _vCONAP[e], _vPARVAL, _dOUT, _vOUTSAM, os ) )
      return false;
    if( options.DISPLEVEL  && !(e%DISPFREQ) )
      os << "=" << std::flush;
  }

  // Compute responses for every control samples and uncertainty scenarios
  for( size_t e=0; e<NSAM; ++e ){
    if( !_append_out( _vCONSAM[e], _vPARVAL, _dOUT, _vOUTSAM, os ) )
        return false;
    if( options.DISPLEVEL  && !((e+_ne0)%DISPFREQ) )
      os << "=" << std::flush;
  }
  if( options.DISPLEVEL )
    os << "| " << NUNC*(_ne0+NSAM)
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.walltime( tstart ) ) << " SEC" << std::flush;

  return true;
}

inline
bool
MODISCR::_append_out
( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
  std::vector<std::vector<double>>& Output, std::vector<std::vector<std::vector<arma::vec>>>& Response,
  std::ostream& os )
{
  if( !_nm || !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  for( size_t m=0; m<_nm; ++m ){
    // Compute Output of model m for Control in all uncertainty scenarios Parameter
    try{
      if( _nc ) _dag->veval( _sgOUT[m], _wkOUT, _vOUT[m], Output, _vPAR, Parameter, _vCON, Control, _vCST, _vCSTVAL );
      else      _dag->veval( _sgOUT[m], _wkOUT, _vOUT[m], Output, _vPAR, Parameter, _vCON, Control );
    }
    catch(...){
      return false;
    }

    // Append Output to Response
    for( size_t s=0; s<Output.size(); ++s ){
      if( Response[s].size() != _nm ) Response[s].resize( _nm );
      arma::vec vOut( Output[s] );
      if( Response[s][m].size() && arma::size( Response[s][m].back() ) != arma::size( vOut ) )
        throw Exceptions( Exceptions::BADSIZE );
      Response[s][m].push_back( vOut );
#ifdef MAGNUS__MODISCR_SAMPLE_DEBUG
      std::cout << "OUT[" << s << "][" << m << "][" << Response[s][m].size() << "]:" << std::endl << arma::trans(vOut);
#endif
    }
  }

  return true;
}

inline
size_t
MODISCR::_redundant_support
( std::vector<double> const& suppref )
{
  size_t pos=0;
  for( auto const& supp : _vCONSAM ){
    if( _mae_support( supp, suppref ) < options.MINDIST )
      return pos;
    ++pos;
  }
  return pos;
}

inline
double
MODISCR::_mae_support
( std::vector<double> const& s1, std::vector<double> const& s2 )
{
  if( s1.size() != s2.size() || s1.size() != _vCONLB.size() )
    return 0./0.; // NaN
  double mae=0;
  for( auto it1 = s1.cbegin(), it2 = s2.cbegin(), itLB = _vCONLB.cbegin(), itUB = _vCONUB.cbegin();
       it1 != s1.end();
       ++it1, ++it2, ++itLB, ++itUB )
    mae += std::fabs( *it1 - *it2 ) / std::fabs( *itUB - *itLB );
  mae /= s1.size();
#ifdef MAGNUS__MODISCR_SAMPLE_DEBUG
  std::cout << "mae: " << std::scientific << std::setprecision(7) << mae << std::endl;
#endif
  return mae;
}

inline
bool
MODISCR::_update_supports
( std::map<size_t,double> const EOpt, std::map<size_t,std::vector<double>> const SOpt,
  std::ostream& os )
{
  auto&& t_samgen = stats.start();
  if( options.DISPLEVEL )
    os << "** REFINING SUPPORT SAMPLES" << std::endl;

  size_t posSupp = _vCONSAM.size(), newSupp = 0;
  auto itE = EOpt.cbegin();
  auto itS = SOpt.cbegin();
  _EOpt.clear();
  _SOpt.clear();
  for( ; itS != SOpt.cend(); ++itS, ++itE ){
    auto const& eff  = itE->second; 
    auto const& supp = itS->second;
    size_t pos = _redundant_support( supp );
    // Refined support is redundant
    if( pos < _vCONSAM.size() ){
      if( options.DISPLEVEL > 1 )
        os << "   REFINED SUPPORT REDUNDANT WITH #" << pos << std::endl;
      auto itR = _EOpt.find( pos );
      // Redundant support not present
      if( itR == _EOpt.end() ){
        _EOpt[pos] = eff;
        _SOpt[pos] = supp;
      }
      // Redundant support already present
      else
        _EOpt[pos] += eff;      
    }
    // Refined support is distinct
    else{
      _EOpt[_vCONSAM.size()] = eff;
      _SOpt[_vCONSAM.size()] = supp;
      _vCONSAM.push_back( supp );
      ++newSupp;
    }
  }

  // Add samples for refined supports
  for( size_t s=posSupp; s<posSupp+newSupp; ++s ){

    switch( options.CRITERION ){
      case BRISK:
        if( !_append_out( _vCONSAM[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
          return false;
        break;
      default:
        throw Exceptions( Exceptions::BADCRIT );
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  stats.walltime_samgen += stats.walltime( t_samgen );
  stats.walltime_all    += stats.walltime( t_samgen );
  return true;
}

inline
bool
MODISCR::file_export
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

      for( size_t m=0; m<_vOUTSAM[s].size(); ++m )
      switch( options.CRITERION ){
        case BRISK:
          for( size_t i=0; i<_vOUTSAM[s][m][k].n_rows; ++i )
            ofile << _vOUTSAM[s][m][k](i) << "  ";
          break;

        default:
          throw Exceptions( Exceptions::BADCRIT );
      }
      ofile << std::endl;
    }
  }
  return true;
}

inline
int
MODISCR::combined_solve
( size_t const NEXP, std::vector<double> const& vcst, bool const exact,
  std::map<size_t,double> const& EIni, std::ostream& os )
{
  _EOpt = EIni;
  double VLast;
  int flag = -33;
  
  for( int it=0; ; ){
    flag = effort_solve( NEXP, exact, _EOpt, os );
    if( it && std::fabs( VLast - _VOpt ) < options.TOLITER * std::fabs( VLast + _VOpt ) / 2 ){
      if( options.DISPLEVEL )
        os << "** CONVERGENCE TOLERANCE SATISFIED" << std::endl;
      break;
    }
    else if( _EOpt.empty() ){
      if( options.DISPLEVEL )
        os << "** EFFORT-BASED OPTIMIZATION FAILED" << std::endl;
      break;    
    }

    flag = gradient_solve( _EOpt, vcst, true, os );
    VLast = _VOpt;
    if( ++it >= options.MAXITER ){
      if( options.DISPLEVEL )
        os << "** MAXIMUM ITERATION LIMIT REACHED" << std::endl;
      break;
    }
  }
  
  return flag;
}

inline
void
MODISCR::_effort_set_BRNeutral
( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF )
const
{

  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF, 1. ) - (int)NEXP );

  size_t const NUNC  = _vPARVAL.size();
  std::vector<FFVar const*> vBRCrit( NUNC );
  FFBREff OpBRCrit;
  for( size_t s=0; s<NUNC; ++s )
    vBRCrit[s] = &OpBRCrit( EFF, &(_vOUTSAM[s]), &_vEFFAP );

  doe.set_obj( BASE_OPT::MIN, Sum( NUNC, vBRCrit.data(), _vPARWEI.data() ) );
}

inline
void
MODISCR::_effort_set_BRAverse
( MINLP& doe, size_t const NEXP, std::vector<FFVar> const& EFF, std::vector<double>& E0 )
const
{
  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF, 1. ) - (int)NEXP );

  size_t const NUNC  = _vPARVAL.size();
  std::vector<FFVar> DELTA( NUNC );
  for( auto& Dk : DELTA )
    Dk.set( _dagdoe );
  doe.add_var( NUNC, DELTA.data(), 0e0 );//, 1e2 );

  FFVar VaR( _dagdoe );
  doe.add_var( VaR );//, -1e2, 1e2 );

  std::vector<FFVar const*> vBRCrit( NUNC );
  FFBREff OpBRCrit;
  for( size_t s=0; s<NUNC; ++s ){
    vBRCrit[s] = &OpBRCrit( EFF, &(_vOUTSAM[s]), &_vEFFAP );
    doe.add_ctr( BASE_OPT::GE, VaR + DELTA[s] - *(vBRCrit[s]) );
  }

  doe.set_obj( BASE_OPT::MIN, VaR + Sum( DELTA, _vPARWEI ) / options.CVARTHRES );

  E0.insert( E0.end(), NUNC+1, 0e0 );
}

inline
int
MODISCR::effort_solve
( size_t const NEXP, bool const exact, std::map<size_t,double> const& EIni,
  std::ostream& os )
{
  auto&& t_slvmip = stats.start();

  // Define MINLP DAG
  delete _dagdoe; _dagdoe = new DAG;

  size_t const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF( NSUPP );
  for( auto& Ek : EFF )
    Ek.set( _dagdoe );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, (double)NEXP/(double)NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [ndx,eff] : EIni )
      E0[ndx] = eff;
  }
  
  // Update external operations
  FFDOEBase::set_weighting( _vOUTWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::type = options.CRITERION;

  // Convex MINLP optimization
  MINLP doe;
  doe.options = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( EFF, 0e0, NEXP, exact );

  switch( options.CRITERION ){
    case BRISK:
      switch( options.RISK){
        case Options::NEUTRAL:
          _effort_set_BRNeutral( doe, NEXP, EFF );
          break;
        case Options::AVERSE:
          _effort_set_BRAverse( doe, NEXP, EFF, E0 );
          break;
        default:
          throw Exceptions( Exceptions::BADCRIT );
      }
      break;

    default:
      throw Exceptions( Exceptions::BADCRIT );
  }
  
  doe.setup();
  //doe.optimize( E0.data() );
  //doe.optimize( E0.data(), nullptr, nullptr, effort_apportion );
  doe.optimize( E0.data(), nullptr, nullptr, effort_rounding );

  if( options.DISPLEVEL > 1 )
    doe.stats.display();

  _EOpt.clear();
  _SOpt.clear();
  _ROpt.clear();
  _VOpt = BASE_OPT::BASE_OPT::INF;
  if( doe.get_status() == MINLP::SUCCESSFUL 
   || doe.get_status() == MINLP::INTERRUPTED ){
    size_t isupp = 0;
    for( auto const& Ek : doe.get_incumbent().x ){
      if( isupp >= NSUPP ){
        _ROpt.push_back( Ek );
      }
      else if( Ek > 1e-3 ){
        _EOpt[isupp] = Ek;
        _SOpt[isupp] = _vCONSAM[isupp];
      }
      ++isupp;
    }
    _VOpt = doe.get_incumbent().f[0];
  }

  if( options.DISPLEVEL )
    if( exact ) _display_design( "EFFORT-BASED EXACT DESIGN", _VOpt, _EOpt, _SOpt, os ); 
    else        _display_design( "EFFORT-BASED CONTINUOUS DESIGN", _VOpt, _EOpt, _SOpt, os ); 

  stats.walltime_slvmip += stats.lapse( t_slvmip );
  stats.walltime_all    += stats.lapse( t_slvmip );
  
  return doe.get_status();
}

inline
std::list<std::pair<double,std::vector<double>>>
MODISCR::campaign
()
const
{
  assert( _EOpt.size() == _SOpt.size() );
  std::list<std::pair<double,std::vector<double>>> C;
  auto iteff = _EOpt.cbegin();
  auto itsup = _SOpt.cbegin();
  for( ; iteff != _EOpt.cend(); ++iteff, ++itsup )
    C.push_back( { iteff->second, itsup->second } );
  return C;
}

inline
double
MODISCR::_evaluate_BRNeutral
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double> const& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() ){
    auto DISPLEVEL  = options.DISPLEVEL;
    options.DISPLEVEL = 0;
    _sample_out( 0, os );
    options.DISPLEVEL = DISPLEVEL;
  }

  // Define Bayes risk criterion
  FFBRMMCrit OpBR;
  FFVar const*const* ppBR = OpBR( UTOT.data(), _dag, _nm, &_vPAR, &_vCST, &_vCON, &_vOUTVEC, &EOpt,
                                  &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP );
#ifdef MAGNUS__MODISCR_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppBR[0] ), " ppBRMM[0]" );
#endif

  size_t const NUNC = _vPARVAL.size();
  FFLin<I> Sum;
  FFVar FBR = Sum( NUNC, ppBR, _vPARWEI.data() );

  // Evaluate average BR criterion
  double DBR = 0./0.;
  _dagdoe->eval( 1, &FBR, &DBR, UTOT.size(), UTOT.data(), UTOT0.data() );
#ifdef MAGNUS__MODISCR_SOLVE_DEBUG
  auto&& FGRADBR  = _dagdoe->FAD( { FBR }, UTOT );
  std::vector<double> DGRADBR;
  _dagdoe->eval( FGRADBR, DGRADBR, UTOT, UTOT0 );
  double const RTOL = 1e-5, ATOL = 1e-5;
  for( size_t i=0; i<UTOT.size(); i++ ){
    auto UTOT0_pert = UTOT0;
    UTOT0_pert[i] = UTOT0[i] * ( 1 + RTOL ) + ATOL;
    double DBR_plus;
    _dagdoe->eval( 1, &FBR, &DBR_plus, UTOT.size(), UTOT.data(), UTOT0_pert.data() );
    UTOT0_pert[i] = UTOT0[i] * ( 1 - RTOL ) - ATOL;
    double DBR_minus;
    _dagdoe->eval( 1, &FBR, &DBR_minus, UTOT.size(), UTOT.data(), UTOT0_pert.data() );
    std::cout << std::scientific << std::setprecision(5);
    std::cout << "d_UTOT[" << i << "] = " << 2*(UTOT0[i]-UTOT0_pert[i]) << std::endl;
    std::cout << "grad BR[" << i << "] = " << (DBR_plus-DBR_minus)/(2*(UTOT0[i]-UTOT0_pert[i]))
                                 << "  <?>  " << DGRADBR[i] << std::endl;
  }
#endif

  return DBR;
}

inline
double
MODISCR::_evaluate_BRAverse
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double> const& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() ){
    auto DISPLEVEL  = options.DISPLEVEL;
    options.DISPLEVEL = 0;
    _sample_out( 0, os );
    options.DISPLEVEL = DISPLEVEL;
  }

  // Define Bayes risk criterion
  FFBRMMCrit OpBR;
  FFVar const*const* ppBR = OpBR( UTOT.data(), _dag, _nm, &_vPAR, &_vCST, &_vCON, &_vOUTVEC, &EOpt,
                                  &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP );
#ifdef MAGNUS__MODISCR_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppBR[0] ), " ppBRMM[0]" );
#endif

  size_t const NUNC = _vPARVAL.size();
  std::vector<double> DBR( NUNC );
  std::vector<FFVar>  FBR( NUNC );
  for( size_t s=0; s<NUNC; ++s )
    FBR[s] = *ppBR[s];

  // Evaluate cost function
  _dagdoe->eval( FBR, DBR, UTOT, UTOT0 );

  std::map<double,double> SA;
  for( size_t s=0; s<NUNC; ++s )
    SA[ DBR[s] ] = _vPARWEI[s];

  double prsum = 0., VaR = 0.;
  for( auto const& [BR,pr] : SA ){
    VaR = BR;
    if( prsum + pr > 1. - options.CVARTHRES ) break;
    prsum += pr;
  }
  std::cout << "VaR = " << VaR << std::endl;

  double CVaR = VaR;
  for( auto const& [BR,pr] : SA ){
    if( BR < VaR ) continue;
    CVaR += ( BR - VaR ) * pr / options.CVARTHRES;
  }
  std::cout << "CVaR = " << CVaR << std::endl;

  return CVaR;
}

inline
std::pair<double,bool>
MODISCR::evaluate_design
( std::list<std::pair<double,std::vector<double>>> const& Campaign, std::string const& type,
  std::vector<double> const& vcst, std::ostream& os )
{
  // Define evaluation DAG
  delete _dagdoe; _dagdoe = new DAG;

  size_t const NSUP  = Campaign.size();
  size_t const NUTOT = _nu * NSUP;
  std::vector<FFVar> UTOT(NUTOT);  // Concatenate experimental controls
  for( size_t i=0; i<NUTOT; i++ )
    UTOT[i].set( _dagdoe );

  std::vector<double> UTOT0;
  UTOT0.reserve(NUTOT);
  std::map<size_t,double> EOpt;
  size_t ieff = 0;
  for( auto const& [eff,supp] : Campaign ){
    EOpt[ieff++] = eff;
    UTOT0.insert( UTOT0.end(), supp.cbegin(), supp.cend() );
  }

  // Update constants
  if( vcst.size() && vcst.size() == _nc ) _vCSTVAL = vcst;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Update external operations
  FFDOEBase::set_weighting( _vOUTWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::type = options.CRITERION;

  // Evaluate design
  std::string header( type.empty()? "DESIGN PERFORMANCE": type + " DESIGN PERFORMANCE" );
  double crit = 0./0.;

  try{
    switch( options.CRITERION ){
      case BRISK:
        switch( options.RISK){
          case Options::NEUTRAL:
            crit = _evaluate_BRNeutral( EOpt, UTOT, UTOT0, os );
            break;
          case Options::AVERSE:
            crit = _evaluate_BRAverse( EOpt, UTOT, UTOT0, os );
            break;
          default:
            throw Exceptions( Exceptions::BADCRIT );
        }
        break;

      default:
        throw Exceptions( Exceptions::BADCRIT );
    }
  }

  catch(...){
    if( options.DISPLEVEL )
      _display_design( header, 0./0., std::list<std::pair<double,std::vector<double>>>(), os ); 
    return std::make_pair( 0./0., false ); // NaN
  }

  if( options.DISPLEVEL )
    _display_design( header, crit, Campaign, os ); 
  return std::make_pair( crit, true );
}

inline
void
MODISCR::_refine_set_BRNeutral
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double> const& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() )
    _sample_out( 0, os );

  // Define Bayes risk objective
  FFBRMMCrit OpBR;
  FFVar const*const* ppBR = OpBR( UTOT.data(), _dag, _nm, &_vPAR, &_vCST, &_vCON, &_vOUTVEC, &EOpt,
                                  &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP );
#ifdef MAGNUS__MODISCR_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppBR[0] ), " BRCrit" );
#endif

  size_t const NUNC = _vPARVAL.size();
  FFLin<I> Sum;
  FFVar FBR = Sum( NUNC, ppBR, _vPARWEI.data() );
#ifdef MAGNUS__MODISCR_SOLVE_DEBUG
  double DBR;
  _dagdoe->eval( 1, &FBR, &DBR, UTOT.size(), UTOT.data(), UTOT0.data() );
  std::cout << "Crit = " << DBR << std::endl;
  auto&& FGRADBR  = _dagdoe->FAD( { FBR }, UTOT );
  std::vector<double> DGRADBR;
  _dagdoe->eval( FGRADBR, DGRADBR, UTOT, UTOT0 );
  std::cout << "Grad Cit AD = " << arma::trans( arma::vec( DGRADBR.data(), DGRADBR.size(), false ) ) << std::endl;
  double const TOL = 1e-3;
  for( size_t i=0; i<UTOT.size(); ++i ){
    std::vector<double> UTOT1 = UTOT0;
    UTOT1[i] = UTOT0[i] + std::fabs(UTOT0[i])*TOL + TOL;
    _dagdoe->eval( 1, &FBR, &DBR, UTOT.size(), UTOT.data(), UTOT1.data() );
    DGRADBR[i] = DBR;
    UTOT1[i] = UTOT0[i] - std::fabs(UTOT0[i])*TOL - TOL;
    _dagdoe->eval( 1, &FBR, &DBR, UTOT.size(), UTOT.data(), UTOT1.data() );
    DGRADBR[i] -= DBR;
    DGRADBR[i] /= std::fabs(UTOT0[i])*2*TOL + 2*TOL;
  }
  std::cout << "Grad Cit FD = " << arma::trans( arma::vec( DGRADBR.data(), DGRADBR.size(), false ) ) << std::endl;
  { int dum; std::cout << "Paused"; std::cin >> dum; }
#endif
  doeref.set_obj( BASE_OPT::MIN, FBR ); // maximize average Bayes risk criterion
}

inline
void
MODISCR::_refine_set_BRAverse
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double>& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() )
    _sample_out( 0, os );

  // Define Bayes risk objective
  FFBRMMCrit OpBR;
  FFVar const*const* ppBR = OpBR( UTOT.data(), _dag, _nm, &_vPAR, &_vCST, &_vCON, &_vOUTVEC, &EOpt,
                                  &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP );
#ifdef MAGNUS__MODISCR_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppBR[0] ), " BRCrit" );
#endif

  size_t const NUNC = _vPARVAL.size();
  std::vector<FFVar> DELTA( NUNC );
  FFVar VaR( _dagdoe );
  for( auto& Dk : DELTA )
    Dk.set( _dagdoe );
  if( _ROpt.size() == NUNC+1 )
    for( auto const& r0 : _ROpt ) UTOT0.push_back( r0 ); 
  else
    UTOT0.resize( UTOT.size()+NUNC+1, 0e0 );
  doeref.add_var( DELTA, 0e0 );//, 1e2 );
  doeref.add_var( VaR );//, -1e2, 1e2 );

  // Minimize risk-averse Bayes risk criterion
  FFLin<I> Sum;
  doeref.set_obj( BASE_OPT::MIN, VaR + Sum( NUNC, DELTA.data(), _vPARWEI.data() ) / options.CVARTHRES );  
  for( size_t s=0; s<NUNC; s++ )
    doeref.add_ctr( BASE_OPT::GE, VaR + DELTA[s] - *ppBR[s] );
}

inline
int
MODISCR::gradient_solve
( std::map<size_t,double> const& EOpt,  std::vector<double> const& vcst,
  bool const update, std::ostream& os )
{
  auto&& t_slvnlp = stats.start();

  // Define NLP DAG
  delete _dagdoe; _dagdoe = new DAG;
  
  size_t const NSUP = EOpt.size();
  size_t const NUTOT = _nu * NSUP;
  std::vector<FFVar> UTOT(NUTOT);  // Concatenated experimental controls
  for( size_t i=0; i<NUTOT; i++ )
    UTOT[i].set( _dagdoe );

  std::vector<double> UTOT0, UTOTLB, UTOTUB;
  UTOT0.reserve(NUTOT);
  UTOTLB.reserve(NUTOT);
  UTOTUB.reserve(NUTOT);
  for( auto const& [ndx,eff] : EOpt ){
    UTOT0.insert( UTOT0.end(), _vCONSAM[ndx].cbegin(), _vCONSAM[ndx].cend() );
#ifdef MAGNUS__MODISCR_SOLVE_DEBUG
    std::cout << "C[" << ndx << "] = ";
    for( size_t i=0; i<_nu; i++ )
      std::cout << _vCONSAM[ndx][i] << "  ";
    std::cout << std::endl;
#endif
    UTOTLB.insert( UTOTLB.end(), _vCONLB.cbegin(), _vCONLB.cend() );
    UTOTUB.insert( UTOTUB.end(), _vCONUB.cbegin(), _vCONUB.cend() );
  }

  // Update constants
  if( vcst.size() && vcst.size() == _nc ) _vCSTVAL = vcst;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Update external operations
  FFDOEBase::set_weighting( _vOUTWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::type = options.CRITERION;

  // Local NLP optimization
  NLP doeref;
  doeref.options = options.NLPSLV;
  doeref.set_dag( _dagdoe );
  doeref.add_var( UTOT, UTOTLB, UTOTUB );

  switch( options.CRITERION ){
    case BRISK:
      switch( options.RISK){
        case Options::NEUTRAL:
          _refine_set_BRNeutral( doeref, EOpt, UTOT, UTOT0, os );
          break;
        case Options::AVERSE:
          _refine_set_BRAverse( doeref, EOpt, UTOT, UTOT0, os );
          break;
        default:
          throw Exceptions( Exceptions::BADCRIT );
      }
      break;

    default:
      throw Exceptions( Exceptions::BADCRIT );
  }

  doeref.setup();
  doeref.solve( UTOT0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << doeref.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << doeref.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  if( update ){
    _SOpt.clear();
    _VOpt = 0./0.;
 
    if( doeref.get_status() == NLP::SUCCESSFUL
     || doeref.get_status() == NLP::FAILURE
     || doeref.get_status() == NLP::INTERRUPTED ){
      double const* dC = doeref.solution().x.data();
      for( auto const& [ndx,eff] : EOpt ){
        _SOpt[ndx] = std::vector<double>( dC, dC+_nu );
        dC += _nu;
      }
      _update_supports( _EOpt, _SOpt, os );
      _VOpt = doeref.solution().f[0];
      size_t const NEXTRA = doeref.solution().x.size() - NUTOT;
      if( NEXTRA > 0 )
        _ROpt.assign( dC, dC+NEXTRA );
    }
  }

  if( options.DISPLEVEL )
    _display_design( "GRADIENT-BASED REFINED DESIGN", _VOpt, EOpt, _SOpt, os ); 

  stats.walltime_slvnlp += stats.walltime( t_slvnlp );
  stats.walltime_all    += stats.walltime( t_slvnlp );

  return doeref.get_status();
}

inline
void
MODISCR::_display_design
( std::string const& title, double const& crit, std::map<size_t,double> const& eff,
  std::map<size_t,std::vector<double>> const& supp, std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( eff.empty() ){
     os << " FAILED" << std::endl;
     return;
  } 
   
  os  << std::scientific << std::setprecision(5) << crit << std::endl;
  for( auto const& [i,s] : supp ){
    os << "   SUPPORT #" << i << ": " << std::fixed << std::setprecision(2) << eff.at(i) << " x [ "
       << std::scientific << std::setprecision(5);
      for( auto Ck : s )
        os << Ck << " ";
      os << "]" << std::endl;
  }
  os << std::endl;
}

inline
void
MODISCR::_display_design
( std::string const& title, double const& crit,
  std::list<std::pair<double,std::vector<double>>> const& campaign,
  std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( campaign.empty() ){
     os << " FAILED" << std::endl;
     return;
  } 
   
  os  << std::scientific << std::setprecision(5) << crit << std::endl;
  size_t i=0;
  for( auto const& [eff,supp] : campaign ){
    os << "   SUPPORT #" << i++ << ": " << std::fixed << std::setprecision(2) << eff << " x [ "
       << std::scientific << std::setprecision(5);
      for( auto Ck : supp )
        os << Ck << " ";
      os << "]" << std::endl;
  }
  os << std::endl;
}

} // end namespace mc

#endif

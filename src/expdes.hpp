// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

/*!
\page page_EXPDES Model-based Design of Experiments with MC++
\author Benoit Chachuat <tt>(b.chachuat@imperial.ac.uk)</tt>
\version 2.1
\date 2025
\bug No known bugs.
*/

#ifndef MAGNUS__EXPDES_HPP
#define MAGNUS__EXPDES_HPP

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
#include "nsfeas.hpp"

#include "base_mbdoe.hpp"

#include "fflin.hpp"
#include "ffdoe.hpp"
#include "ffode.hpp"

#define MAGNUS__EXPDES_USE_OPFIM

namespace mc
{
//! @brief C++ class for design of experiments for model parameter precision
////////////////////////////////////////////////////////////////////////
//! mc::EXPDES is a C++ class for solving design of experiments for
//! model parameter precision using MC++, CRONOS and CANON
////////////////////////////////////////////////////////////////////////
class EXPDES
: public virtual BASE_MBDOE,
  protected virtual NSFEAS
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
  std::vector<FFVar> _vOUT;

  //! @brief local copy of model constraints
  std::vector<FFVar> _vCTR;
  
  //! @brief vector of FIM entries
  std::vector<FFVar> _vFIM;

  //! @brief output subgraph
  FFSubgraph _sgOUT;

  //! @brief work array for output evaluations
  std::vector<double> _wkOUT;

  //! @brief output values
  std::vector<std::vector<double>> _dOUT;
  
  //! @brief vector of response vectors
  std::vector< std::vector< arma::vec > > _vOUTSAM;

  //! @brief FIM subgraph
  FFSubgraph _sgFIM;

  //! @brief work array for FIM evaluations
  std::vector<double> _wkFIM;

  //! @brief FIM values
  std::vector<std::vector<double>> _dFIM;
  
  //! @brief vector of atom matrices
  std::vector< std::vector< arma::mat > > _vFIMSAM;

public:
  /** @defgroup EXPDES Model-based Design of Experiments using MC++
   *  @{
   */
   
  //! @brief Constructor
  EXPDES()
    : _dag(nullptr), _dagdoe(nullptr),
      _VOpt(0./0.)
    { stats.reset(); }

  //! @brief Destructor
  virtual ~EXPDES()
    {
      delete   _dag;
      delete   _dagdoe;
    }

  //! @brief EXPDES solver options
  struct Options
  {
    //! @brief Constructor
    Options()
      {
        reset();
      }

    //! @brief Reset to default options
    void reset
      ()
      {
        CRITERION                   = BASE_MBDOE::DOPT;
        RISK                        = NEUTRAL;
        CVARTHRES                   = 0.25;
        UNCREDUC                    = 1e-2;
        FIMSTOL                     = 1e-7;
        IDWTOL                      = 1e-3;
        FEASTHRES                   = 1e-1;
        FEASPROP                    = 16;
        MINDIST                     = 1e-6;
        MAXITER                     = 4;
        TOLITER                     = 1e-4;
        MAXTHREAD                   = 1;
        DISPLEVEL                   = 1;
        NLPSLV.reset();
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
        UNCREDUC    = options.UNCREDUC;
        FIMSTOL     = options.FIMSTOL;
        IDWTOL      = options.IDWTOL;
        FEASTHRES   = options.FEASTHRES;
        FEASPROP    = options.FEASPROP;
        MINDIST     = options.MINDIST;
        MAXITER     = options.MAXITER;
        TOLITER     = options.TOLITER;
        MAXTHREAD   = options.MAXTHREAD;
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
    //! @brief Uncertainty scenario reduction within threshold (if >=0) or to k-nearest neighboors (<0)
    double                   UNCREDUC;
    //! @brief Tolerance for singular value in FIM
    double                   FIMSTOL;
    //! @brief Tolerance for inverse distance weighting measure
    double                   IDWTOL;
    //! @brief Maximum constraint violation percentile
    double                   FEASTHRES;
    //! @brief number of proposals in nested sampling of constraints
    size_t                   FEASPROP;
    //! @brief Minimal relative mean-absolute distance between support points after refinement
    double                   MINDIST;
   //! @brief Maximal iteration of effort-based and gradient-based solves
    int                      MAXITER;
   //! @brief Stopping tolerance for effort-based and gradient-based iteration
    double                   TOLITER;
   //! @brief Maximal number of parallel threads
    size_t                   MAXTHREAD;
    //! @brief Verbosity level
    int                      DISPLEVEL;
    
    //! @brief MINLP effort-based solver options
    typename MINLP::Options  MINLPSLV;
    //! @brief NLP gradient-based solver options
    typename NLP::Options    NLPSLV;
  } options;

  //! @brief EXPDES solver exceptions
  class Exceptions
  {
  public:
    //! @brief Enumeration type for EXPDES exception handling
    enum TYPE{
      BADSIZE=0,    //!< Inconsistent dimensions
      NOMODEL,	    //!< unspecified model
      BADCONST,	    //!< Unspecified constants
      BADCRIT,      //!< Misspecified design criterion
      BADNEXP,      //!< Inconsistent campaign size
      INTERN=-33    //!< Internal error
    };
    //! @brief Constructor for error <a>ierr</a>
    Exceptions( TYPE ierr ) : _ierr( ierr ){}
    //! @brief Inline function returning the error flag
    int ierr
      ()
      const
      { return _ierr; }
    //! @brief Inline function returning the error description
    std::string what
      () 
      const
      {
        switch( _ierr ){
          case BADSIZE:
            return "EXPDES::Exceptions  Inconsistent dimensions";
          case NOMODEL:
            return "EXPDES::Exceptions  Unspecified model";
          case BADCONST:
            return "EXPDES::Exceptions  Unspecified constants";
          case BADCRIT:
            return "EXPDES::Exceptions  Unspecified constants";
          case BADNEXP:
            return "EXPDES::Exceptions  Inconsistent campaign size";
          case INTERN:
          default:
            return "EXPDES::Exceptions  Internal error";
        }
      }
  private:
    TYPE _ierr;
  };

  //! @brief EXPDES solver statistics
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
    std::chrono::microseconds lapse
      ( std::chrono::time_point<std::chrono::system_clock> const& start ) const
      { return std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::system_clock::now() - start ); }    
    //! @brief Convert microsecond ticks to time
    double to_time
      ( std::chrono::microseconds t ) const
      { return t.count() * 1e-6; }
  } stats;

  //! @brief Setup EXPDES problem before solution
  bool setup
    ( size_t const ndxmod=0 );

  //! @brief Evaluate performance of experimental campaign
  std::tuple<double,std::vector<double>,bool> evaluate_design
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

  //! @brief Solve effort-based exact experiment design with increasing number of supports <a>vNEXP</a>
  int effort_solve
    ( std::vector<size_t> const& vNEXP, bool const exact=true,
      std::map<size_t,double> const& EIni=std::map<size_t,double>(),
      std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports 
  int gradient_solve
    ( std::map<size_t,double> const& EOpt, std::vector<double> const& vcst=std::vector<double>(),
      bool const update=true, std::ostream& os=std::cout );

  //! @brief Solve gradient-based experiment design for refinement of <a>EOpt</a> supports 
  int gradient_solve
    ( std::list<std::pair<double,std::vector<double>>> const& Campaign,
      std::vector<double> const& vcst=std::vector<double>(), bool const update=true, 
      std::ostream& os=std::cout );

  //! @brief Solve combined effort- and gradient-based experiment design with <a>NEXP</a> supports 
  int combined_solve
    ( size_t const NEXP, std::vector<double> const& vcst=std::vector<double>(),
     bool const exact=true,  std::map<size_t,double> const& EIni=std::map<size_t,double>(),
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
  std::vector<std::vector<double>> const& control_sample
    ()
    const
    { return _vCONSAM; }

  //! @brief Retrieve sampled outputs
  std::vector<std::vector<arma::vec>> const& output_sample
    ()
    const
    { return _vOUTSAM; }

  //! @brief Retrieve sampled FIMs
  std::vector<std::vector<arma::mat>> const& fim_sample
    ()
    const
    { return _vFIMSAM; }

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
    ( size_t const ndxmod );

  //! @brief Create local copy of output model for FIM prediction
  void _setup_fim
    ( size_t const ndxmod );

  //! @brief Generate <a>NSAM</a> feasible supports
  bool _sample_support_nsfeas
    ( size_t const NSAM, std::ostream& os );

  //! @brief Generate output samples for <a>NSAM</a> initial supports
  bool _sample_out
    ( size_t const NSAM, std::ostream& os=std::cout );

  //! @brief Select uncertainty scenario subset based on highest value
  void _sample_rank
    ( std::set<std::pair<size_t,size_t>>& parsel,
      std::vector<double> const& E0, std::ostream& os );

  //! @brief Select uncertainty scenario subset based on nearest-neighbors criterion
  void _sample_select
    ( size_t const start, size_t const inc, std::set<std::pair<size_t,size_t>>& parsel,
      std::vector<double> const& E0, std::ostream& os );
  
  //! @brief Append output under given control scenario for all uncertainty scenarios
  bool _append_out
    ( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
      std::vector<std::vector<double>>& Output, std::vector<std::vector<arma::vec>>& Response,
      std::ostream& os=std::cout );

  //! @brief Generate FIM samples for <a>NSAM</a> initial supports
  bool _sample_fim
    ( size_t const NSAM, std::ostream& os=std::cout );

  //! @brief Append FIM (columnwise, lower-triangular) under given control scenario for all uncertainty scenarios
  bool _append_fim
    ( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
      std::vector<std::vector<double>>& FIM, std::vector<std::vector<arma::mat>>& Response,
      std::ostream& os=std::cout );

  //! @brief Set Bayes Risk criterion in effort-based MINLP model
  void _effort_set_BRisk
    ( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF )
    const;

  //! @brief Set output distance criterion in effort-based MINLP model
  void _effort_set_ODist
    ( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF,
      std::vector<double>& E0, std::vector<I>& EBND )
    const;

  //! @brief Set FIM-based risk-neutral criterion in effort-based MINLP model
  void _effort_set_FIMNeutral
    ( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF )
    const;

  //! @brief Set FIM-based risk-averse criterion in effort-based MINLP model
  void _effort_set_FIMAverse
    ( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF,
      std::vector<double>& E0, std::vector<I>& EBND )
    const;

  //! @brief Evaluate constraints
  void _evaluate_constraints
    ( std::vector<double>& DCTR, size_t const NSUP, size_t const NUNC,
      double const* DCTRSAM, std::ostream& os );

  //! @brief Evaluate Bayes Risk criterion
  std::pair<double,std::vector<double>> _evaluate_BRisk
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double> const& UTOT0, std::ostream& os );

  //! @brief Evaluate output distance criterion
  std::pair<double,std::vector<double>> _evaluate_ODist
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double> const& UTOT0, std::ostream& os );

  //! @brief Evaluate FIM-based risk-neutral criterion
  std::pair<double,std::vector<double>> _evaluate_FIMNeutral
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT, 
      std::vector<double> const& UTOT0, std::ostream& os );

  //! @brief Evaluate FIM-based risk-averse criterion
  std::pair<double,std::vector<double>> _evaluate_FIMAverse
    ( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double> const& UTOT0, std::ostream& os );

  //! @brief Set constraints in gradient-based NLP model
  void _refine_set_constraints
    ( NLP& doeref, size_t const NE, FFVar const*const* ppCTRSAM,
      std::vector<double>& UTOT0, std::ostream& os );

  //! @brief Set Bayes Risk criterion in gradient-based NLP model
  void _refine_set_BRisk
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double>& UTOT0, std::ostream& os );

  //! @brief Set output distance criterion in gradient-based NLP model
  void _refine_set_ODist
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double>& UTOT0, std::ostream& os );

  //! @brief Set FIM-based risk-neutral criterion in gradient-based NLP model
  void _refine_set_FIMNeutral
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double>& UTOT0, std::ostream& os );

  //! @brief Set FIM-based risk-averse criterion in gradient-based NLP model
  void _refine_set_FIMAverse
    ( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
      std::vector<double>& UTOT0, std::ostream& os );

  //! @brief Run gradient-based refinement
  int _gradient_solve
    ( std::map<size_t,double> const& EOpt, std::vector<double>& UTOT0,
      std::vector<double> const& UTOTLB, std::vector<double> const& UTOTUB,
      std::vector<double> const& vcst, bool const update, std::ostream& os );

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
EXPDES::setup
( size_t const ndxmod )
{
  stats.reset();
  auto&& t_setup = stats.start();

  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  switch( options.CRITERION ){
   case BRISK:
   case ODIST:
    _setup_out( ndxmod );
    break;

   case AOPT:
   case DOPT:
   case EOPT:
    _setup_fim( ndxmod );
    break;

   default:
    throw Exceptions( Exceptions::BADCRIT );
  }

  stats.walltime_setup += stats.lapse( t_setup );
  stats.walltime_all   += stats.lapse( t_setup );
  return true;
}

inline
void
EXPDES::_setup_out
( size_t const ndxmod )
{
  if( ndxmod >= _nm || !_ny || _ny != BASE_MBDOE::_vOUT[ndxmod].size() || !BASE_MBDOE::_vCON.size() )
    throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;
  _dag->options = BASE_MBDOE::_dag->options;
  _dag->options.MAXTHREAD = options.MAXTHREAD;
  _wkOUT.clear();
  
  _vCON.resize( _nu );
  _dag->insert( BASE_MBDOE::_dag, _nu, BASE_MBDOE::_vCON.data(), _vCON.data() );
  _vPAR.resize( _np );
  _dag->insert( BASE_MBDOE::_dag, _np, BASE_MBDOE::_vPAR.data(), _vPAR.data() );
  _vCST.resize( _nc );
  _dag->insert( BASE_MBDOE::_dag, _nc, BASE_MBDOE::_vCST.data(), _vCST.data() );
  _vOUT.resize( _ny );
  _dag->insert( BASE_MBDOE::_dag, _ny, BASE_MBDOE::_vOUT[ndxmod].data(), _vOUT.data() );
  _sgOUT.clear();

#ifdef MAGNUS__EXPDES_SETUP_DEBUG
  _sgOUT = _dag->subgraph( _ny, _vOUT.data() );
  std::vector<FFExpr> exOUT = FFExpr::subgraph( _dag, _sgOUT ); 
  for( size_t i=0; i<_ny; ++i )
    std::cout << "OUT[" << i << "] = " << exOUT[i] << std::endl;
#endif

  if( _ng ){
    _vCTR.resize( _ng );
    _dag->insert( BASE_MBDOE::_dag, _ng, BASE_MBDOE::_vCTR.data(), _vCTR.data() );

#ifdef MAGNUS__EXPDES_SETUP_DEBUG
    auto sgCTR = _dag->subgraph( _ng, _vCTR.data() );
    std::vector<FFExpr> exCTR = FFExpr::subgraph( _dag, sgCTR ); 
    for( size_t i=0; i<_ng; ++i )
      std::cout << "CTR[" << i << "] = " << exCTR[i] << std::endl;
#endif
  }
  else
    _vCTR.clear();
}

inline
void
EXPDES::_setup_fim
( size_t const ndxmod )
{
  if( ndxmod >= _nm || !_ny || _ny != BASE_MBDOE::_vOUT[ndxmod].size() || !BASE_MBDOE::_vCON.size() )
    throw Exceptions( Exceptions::BADSIZE );

  delete _dag; _dag = new DAG;
#ifdef MAGNUS__EXPDES_SETUP_DEBUG
  std::cout << "BASE_MBDOE::_dag: " << BASE_MBDOE::_dag << std::endl;
  if( BASE_MBDOE::_dag ) std::cout << *BASE_MBDOE::_dag;
#endif
  _dag->options = BASE_MBDOE::_dag->options;
  _dag->options.MAXTHREAD = options.MAXTHREAD;
  _wkFIM.clear();
    
  _vCON.resize( _nu );
  _dag->insert( BASE_MBDOE::_dag, _nu, BASE_MBDOE::_vCON.data(), _vCON.data() );
  _vPAR.resize( _np );
  _dag->insert( BASE_MBDOE::_dag, _np, BASE_MBDOE::_vPAR.data(), _vPAR.data() );
  _vCST.resize( _nc );
  _dag->insert( BASE_MBDOE::_dag, _nc, BASE_MBDOE::_vCST.data(), _vCST.data() );
  _vOUT.resize( _ny );
  _dag->insert( BASE_MBDOE::_dag, _ny, BASE_MBDOE::_vOUT[ndxmod].data(), _vOUT.data() );

  FFODE::options.SYMDIFF = _vPAR;
  FFVar* y_p = _dag->FAD( _ny, _vOUT.data(), _np, _vPAR.data(), true ); // Jacobian in dense format
  FFODE::options.SYMDIFF.clear();

#ifdef MAGNUS__EXPDES_USE_OPFIM
  FFFIM OpFIM;
  FFVar** ppFIM = OpFIM( _np, _ny, y_p, &_vOUTVAR );
  _vFIM.resize( _np*(_np+1)/2 );
  for( size_t i=0, ij=0; i<_np; ++i )
    for( size_t j=i; j<_np; ++j, ++ij )
      _vFIM[ij] = *ppFIM[ij];
#else
  _vFIM.assign( _np*(_np+1)/2, 0. );
  for( size_t k=0; k<_ny; k++ )
    for( size_t i=0, ij=0; i<_np; ++i )
      for( size_t j=i; j<_np; ++j, ++ij ){
        if( _vOUTVAR.size() == _ny )
          _vFIM[ij] += (y_p[_ny*i+k] * y_p[_ny*j+k]) / _vOUTVAR[k];
        else
          _vFIM[ij] += y_p[_ny*i+k] * y_p[_ny*j+k];
      }
#endif
  _dFIM.resize( _vFIM.size() );
  delete[] y_p;
  _sgFIM.clear();
  
#ifdef MAGNUS__EXPDES_SETUP_DEBUG
  _sgFIM = _dag->subgraph( _vFIM.size(), _vFIM.data() );
  std::vector<FFExpr> exFIM = FFExpr::subgraph( _dag, _sgFIM ); 
  for( size_t i=0, ij=0; i<_np; ++i )
    for( size_t j=i; j<_np; ++j, ++ij )
      std::cout << "FIM[" << i << "][" << j << "] = " << exFIM[ij] << std::endl;
#endif

  if( _ng ){
    _vCTR.resize( _ng );
    _dag->insert( BASE_MBDOE::_dag, _ng, BASE_MBDOE::_vCTR.data(), _vCTR.data() );

#ifdef MAGNUS__EXPDES_SETUP_DEBUG
    auto sgCTR = _dag->subgraph( _ng, _vCTR.data() );
    std::vector<FFExpr> exCTR = FFExpr::subgraph( _dag, sgCTR ); 
    for( size_t i=0; i<_ng; ++i )
      std::cout << "CTR[" << i << "] = " << exCTR[i] << std::endl;
#endif
  }
  else
    _vCTR.clear();
}

inline
bool
EXPDES::_sample_support_nsfeas
( size_t const NSAM, std::ostream& os )
{
  // Set nested sampling options
  NSFEAS::options.DISPLEVEL = options.DISPLEVEL;
  NSFEAS::options.FEASCRIT  = NSFEAS::Options::CVAR;
  NSFEAS::options.FEASTHRES = options.FEASTHRES;
  NSFEAS::options.NUMPROP   = options.FEASPROP;
  NSFEAS::options.NUMLIVE   = NSAM;
  NSFEAS::options.MAXTHREAD = options.MAXTHREAD;
  
  NSFEAS::setup();
  if( NSFEAS::sample( _vCSTVAL, true, os ) != NSFEAS::STATUS::NORMAL
   || NSFEAS::_liveFEAS.size() < NSAM )
    return false;

  if( options.DISPLEVEL )
    NSFEAS::stats.display();
  
  _vCONSAM.clear();
  _vCONSAM.reserve( NSAM );
  for( auto const& [lkh,pcon] : NSFEAS::_liveFEAS ){
    _vCONSAM.push_back( std::vector<double>( std::get<0>(pcon), std::get<0>(pcon)+_nu ) );
    if( _vCONSAM.size() < NSAM ) continue;
    break;
  }
  return true;
}

inline
bool
EXPDES::sample_support
( size_t const NSAM, std::vector<double> const& vcst, std::ostream& os )
{
  auto&& t_samgen = stats.start();

  // Update constants
  if( vcst.size() && vcst.size() == _nc ) _vCSTVAL = vcst;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Control samples
  if( !_ng ) uniform_sample( _vCONSAM, NSAM, _vCONLB, _vCONUB );
  else if( !_sample_support_nsfeas( NSAM, os ) ) return false;

  // Observation samples
  bool flag = false;
  switch( options.CRITERION ){
    case BRISK:
    case ODIST:
      flag = _sample_out( NSAM, os );
      break;
      
    case AOPT:
    case DOPT:
    case EOPT:
      flag = _sample_fim( NSAM, os );
      break;

   default:
    throw Exceptions( Exceptions::BADCRIT );
  }

  if( options.DISPLEVEL )
    os << std::endl;

  stats.walltime_samgen += stats.lapse( t_samgen );
  stats.walltime_all    += stats.lapse( t_samgen );
  return flag;
}

inline
bool
EXPDES::_sample_out
( size_t const NSAM, std::ostream& os )
{
  auto&& tstart = stats.start();
  int DISPFREQ = (_ne0+NSAM > 20? (_ne0+NSAM)/20: 1);
  if( options.DISPLEVEL )
    os << "** GENERATING SUPPORT SAMPLES     |" << std::flush;

  // Reset and resize output/intermediate containers
  size_t NUNC = _vPARVAL.size();
  _vOUTSAM.clear();
  _vOUTSAM.resize( NUNC );
  _dOUT.resize( NUNC );

  // Compute responses for every a priori control and uncertainty scenarios
  for( size_t s=0; s<_ne0; ++s ){
    if( !_append_out( _vCONAP[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
      return false;
    if( options.DISPLEVEL  && !(s%DISPFREQ) )
      os << "=" << std::flush;
  }

  // Compute responses for every control samples and uncertainty scenarios
  for( size_t s=0; s<NSAM; ++s ){
    if( !_append_out( _vCONSAM[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
        return false;
    if( options.DISPLEVEL  && !((s+_ne0)%DISPFREQ) )
      os << "=" << std::flush;
  }
  if( options.DISPLEVEL )
    os << "| " << NUNC*(_ne0+NSAM)
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.lapse( tstart ) ) << " SEC" << std::flush;

  if( !NSAM
   || options.CRITERION == ODIST
   || options.UNCREDUC  == 0
   || options.UNCREDUC  >= 1.
   || -options.UNCREDUC >= NUNC-1 )
    return true;

  // Perform scenario reduction for Bayes risk
  tstart = stats.start();
  if( options.DISPLEVEL )
    os << std::endl
       << "** REDUCING UNCERTAINTY SCENARIOS | " << std::flush;
       
  //FFDOEBase BRCrit;
  FFDOEBase::set_noise( _vOUTVAR );
  std::vector<double> E0( NSAM, 1./(double)NSAM );

  // Select sample pairs leading to given %age of BR full criterion
  if( options.UNCREDUC > 0. )
    _sample_rank( _sPARSEL, E0, os );

  // Select sample pairs as nearest neighbours 
  else{
#ifdef MC__USE_THREAD
    size_t const NOTHREADS = ( _dag->options.MAXTHREAD>0? _dag->options.MAXTHREAD: std::thread::hardware_concurrency() );
    std::vector<std::thread> vth( NOTHREADS-1 ); // Main thread also runs evaluations
    std::vector<std::set<std::pair<size_t,size_t>>> vsel( NOTHREADS-1 );

    // Dispatch neightbors selection on auxiliary thread
    for( size_t th=1; th<NOTHREADS; th++ )
      vth[th-1] = std::thread( &EXPDES::_sample_select, this, th, NOTHREADS, std::ref(vsel[th-1]),
                               std::cref(E0), std::ref(os) );

    // Run evaluations on main thread as well
    _sPARSEL.clear();
    _sample_select( 0, NOTHREADS, _sPARSEL, E0, os ); 

    // Join all the threads to the main one
    for( size_t th=1; th<NOTHREADS; th++ ){
      vth[th-1].join();
      _sPARSEL.insert( vsel[th-1].cbegin(), vsel[th-1].cend() );
    }
#else
    _sample_select( 0, 1, _sPARSEL, E0, os ); 
#endif
  }

  if( options.DISPLEVEL )
    os << _sPARSEL.size() << " PAIRS"
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.lapse( tstart ) ) << " SEC" << std::flush;

#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
  std::cout << "PARSEL[" << _sPARSEL.size() << "]: ";
  for( auto const& [j,k] : _sPARSEL )
    std::cout << " (" << j << "," << k << ")";
  std::cout << std::endl;
#endif

  return true;
}

inline
void
EXPDES::_sample_select
( size_t const start, size_t const inc, std::set<std::pair<size_t,size_t>>& parsel,
  std::vector<double> const& E0, std::ostream& os )
{
  std::vector<std::pair<size_t,double>> BRall( _vOUTSAM.size()-1 );
  std::vector<std::pair<size_t,double>> BRtop( std::round(-options.UNCREDUC) );
  int DISPFREQ = (_vOUTSAM.size()-start)/20;

  for( size_t j=start; j<_vOUTSAM.size(); j+=inc ){

    if( options.DISPLEVEL && !(j%DISPFREQ) )
      os << "=" << std::flush;

    for( size_t k=0, kk=0; k<_vOUTSAM.size(); ++k )
      if( k != j ) BRall[kk++] = std::make_pair( k, FFDOEBase::atom_BR( _vOUTSAM.at(j), _vOUTSAM.at(k), E0 ) );
    auto BRgt = []( std::pair<size_t,double> const& a, std::pair<size_t,double> const& b )
                  { return a.second > b.second; };
    assert( std::partial_sort_copy( BRall.begin(), BRall.end(), BRtop.begin(), BRtop.end(), BRgt ) == BRtop.end() );

    //std::cerr << "BRtop[" << j << "]: ";
    for( auto const& [k,val] : BRtop ){
      //std::cerr << "(" << k << "," << val << ") ";
      parsel.insert( j<k? std::make_pair(j,k): std::make_pair(k,j) );
    }
    //std::cerr << std::endl;
  }

  if( options.DISPLEVEL )
    os << "|" << std::flush;
}

inline
void
EXPDES::_sample_rank
( std::set<std::pair<size_t,size_t>>& parsel, std::vector<double> const& E0, std::ostream& os )
{
  // Select the pair that contribute (1-options.UNCREDUC)*100% of the total Bayes Risk
  size_t NUNC = _vPARVAL.size();
  std::vector<std::pair<std::pair<size_t,size_t>,double>> BRall( NUNC*(NUNC+1)/2 );
  double BRtot = 0., BRsum = 0.;
  
  for( size_t j=0, kk=0; j<NUNC; ++j ){
    for( size_t k=j+1; k<NUNC; ++k, ++kk ){
      BRall[kk] = std::make_pair( std::make_pair(j,k), FFDOEBase::atom_BR( _vOUTSAM.at(j), _vOUTSAM.at(k), E0 ) );
      BRtot += BRall[kk].second;
    }
  }
  
  auto BRgt = []( std::pair<std::pair<size_t,size_t>,double> const& a, std::pair<std::pair<size_t,size_t>,double> const& b )
                { return a.second > b.second; };
  std::sort( BRall.begin(), BRall.end(), BRgt );

  for( auto const& [jk,val] : BRall ){
    BRsum += val;
    parsel.insert( jk );
    //std::cerr << "(" << jk.first << "," << jk.second << "," << val << ") " << BRsum/BRtot*1e2 << "%\n";
    if( BRsum >= BRtot*(1-options.UNCREDUC) ) break;
  }
  //std::cerr << parsel.size() << " pairs\n";
}

inline
bool
EXPDES::_append_out
( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
  std::vector<std::vector<double>>& Output, std::vector<std::vector<arma::vec>>& Response,
  std::ostream& os )
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  try{
    if( _nc ) _dag->veval( _sgOUT, _wkOUT, _vOUT, Output, _vPAR, Parameter, _vCON, Control, _vCST, _vCSTVAL );
    else      _dag->veval( _sgOUT, _wkOUT, _vOUT, Output, _vPAR, Parameter, _vCON, Control );
  }
  catch(...){
    return false;
  }

  for( size_t k=0; k<Output.size(); ++k ){
    arma::vec vOut( Output[k] );
    if( Response[k].size() && arma::size( Response[k].back() ) != arma::size( vOut ) )
      throw Exceptions( Exceptions::BADSIZE );
    Response[k].push_back( vOut );
#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
    std::cout << "OUT[" << k << "][" << Response[k].size() << "]:" << std::endl << arma::trans(vOut);
#endif
  }

  return true;
}

inline
bool
EXPDES::_sample_fim
( size_t const NSAM, std::ostream& os )
{
  auto&& tstart = stats.start();
  int DISPFREQ = (_ne0+NSAM)/20;
  if( options.DISPLEVEL )
    os << "** GENERATING SUPPORT SAMPLES     |" << std::flush;

  // Reset and resize FIM/intermediate containers
  size_t NUNC = _vPARVAL.size();
  _vFIMSAM.clear();
  _vFIMSAM.resize( NUNC );
  _dFIM.resize( NUNC );

  // Compute FIMs for every a priori control and uncertainty scenarios
  for( size_t s=0; s<_ne0; ++s ){
    if( !_append_fim( _vCONAP[s], _vPARVAL, _dFIM, _vFIMSAM, os ) )
      return false;
    if( options.DISPLEVEL  && !(s%DISPFREQ) )
      os << "=" << std::flush;
  }

  // Compute FIMs at every control samples and uncertainty scenarios
  for( size_t s=0; s<NSAM; ++s ){
    if( !_append_fim( _vCONSAM[s], _vPARVAL, _dFIM, _vFIMSAM, os ) )
      return false;
    if( options.DISPLEVEL  && !((s+_ne0)%DISPFREQ) )
      os << "=" << std::flush;
  }
  if( options.DISPLEVEL )
    os << "| " << NUNC*(_ne0+NSAM)
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.lapse( tstart ) ) << " SEC" << std::flush;

#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
  size_t s=0;
  for( auto const& FIMk : _vFIMSAM )
    std::cout << "_vFIMSAM[" << s++ << "]: " << FIMk.size() << std::endl;
#endif

  if( options.FIMSTOL <= 0e0 )
    return true;

  tstart = stats.start();
  if( options.DISPLEVEL )
    os << std::endl
       << "** CHECKING FIM REGULARITY        | " << std::flush;
  arma::mat FIM( _np, _np, arma::fill::zeros ); // average FIM
  for( size_t s=0; s<NUNC; ++s ){
    arma::mat FIMs( _np, _np, arma::fill::zeros ); // average FIM
    size_t e=0;
    for( auto const& eff : _vEFFAP ) // Atomic FIM of prior experiment
      FIMs += eff * _vFIMSAM.at(s).at(e++);
    for( size_t i=0; i<NSAM; ++i ) // Atomic FIM of new experiment
      FIMs += _vFIMSAM.at(s).at(e++) / NSAM;
#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
    std::cout << "FIM[" << s << "]: rank = " << arma::rank( FIMs ) << std::endl << FIMs;
#endif
    if( _vPARWEI.size() == NUNC ) FIMs *= _vPARWEI[s]; // uncertainty scenario probability
    FIM += FIMs;
  }
  if( _mPARSCA.n_elem ) FIM = arma::trans(_mPARSCA) * FIM * _mPARSCA; // parameter scaling factors
#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
  std::cout << "Average FIM: rank = " << arma::rank( FIM ) << std::endl << FIM;
#endif

  arma::mat PROJ = arma::orth( FIM, options.FIMSTOL );
#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
  std::cout << "\nFIM image space: " << std::endl << PROJ;
#endif
  if( PROJ.n_rows > PROJ.n_cols )
    if( _mPARSCA.n_elem ) _mPARSCA *= PROJ;
    else                  _mPARSCA  = PROJ;

  if( options.DISPLEVEL )
    os << PROJ.n_rows-PROJ.n_cols << " SINGULAR VALUES BELOW THRESHOLD ("
       << std::scientific << options.FIMSTOL << ")"
       << std::right << std::fixed << std::setprecision(2)
       << std::setw(10) << stats.to_time( stats.lapse( tstart ) ) << " SEC" << std::flush;

// USE ARMADILLO FUNCTIONS NULL AND ORTH: https://arma.sourceforge.net/docs.html#orth
// os << "  FIM is regular on the experimental space discretization" << std::endl; 
// os << "  FIM is rank-defficient on the experimental space discretization - Rank = " << std::endl; 
// os << "  x relationships between parameters... " << std::endl; 

  return true;
}

inline
bool
EXPDES::_append_fim
( std::vector<double> const& Control, std::vector<std::vector<double>> const& Parameter,
  std::vector<std::vector<double>>& FIM, std::vector<std::vector<arma::mat>>& Response,
  std::ostream& os )
{
  if( !_ny ) 
    throw Exceptions( Exceptions::NOMODEL );

  try{
    if( _nc ) _dag->veval( _sgFIM, _wkFIM, _vFIM, FIM, _vPAR, Parameter, _vCON, Control, _vCST, _vCSTVAL );
    else      _dag->veval( _sgFIM, _wkFIM, _vFIM, FIM, _vPAR, Parameter, _vCON, Control );
  }
  catch(...){
    return false;
  }

  for( size_t k=0; k<FIM.size(); ++k ){
    arma::mat mFIM( _np, _np, arma::fill::none );
    for( size_t i=0, l=0; i<_np; ++i )
      for( size_t j=i; j<_np; ++j, ++l )
        if( i == j ) mFIM(i,i) = FIM[k][l]; 
        else         mFIM(i,j) = mFIM(j,i) = FIM[k][l];
    if( Response[k].size() && arma::size( Response[k].back() ) != arma::size( mFIM ) )
      throw Exceptions( Exceptions::BADSIZE );
    Response[k].push_back( mFIM );
#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
    std::cout << "FIM[" << k << "][" << Response[k].size() << "]:" << std::endl << mFIM;
#endif
  }

  return true;
}

inline
size_t
EXPDES::_redundant_support
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
EXPDES::_mae_support
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
#ifdef MAGNUS__EXPDES_SAMPLE_DEBUG
  std::cout << "mae: " << std::scientific << std::setprecision(7) << mae << std::endl;
#endif
  return mae;
}

inline
bool
EXPDES::_update_supports
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
      case ODIST:
        if( !_append_out( _vCONSAM[s], _vPARVAL, _dOUT, _vOUTSAM, os ) )
          return false;
        break;
      
      case AOPT:
      case DOPT:
      case EOPT:
      default:
        if( !_append_fim( _vCONSAM[s], _vPARVAL, _dFIM, _vFIMSAM, os ) )
          return false;
        break;
    }
    if( options.DISPLEVEL > 1 )
      os << "." << std::flush;
  }
  if( options.DISPLEVEL > 1 )
    os << std::endl;

  stats.walltime_samgen += stats.lapse( t_samgen );
  stats.walltime_all    += stats.lapse( t_samgen );
  return true;
}

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

inline
int
EXPDES::combined_solve
( size_t const NEXP, std::vector<double> const& vcst, bool const exact,
  std::map<size_t,double> const& EIni, std::ostream& os )
{
  _EOpt = EIni;
  double VLast = 0.;
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
EXPDES::_effort_set_BRisk
( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF )
const
{
  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF, 1. ) - NEXP );  

  FFBREff OpBRCrit;
  doe.set_obj( BASE_OPT::MIN, OpBRCrit( EFF, &_vOUTSAM, &_vEFFAP ) );
}

inline
void
EXPDES::_effort_set_ODist
( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF,
  std::vector<double>& E0, std::vector<I>& EBND )
const
{
  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF, 1. ) - NEXP );  

  size_t const NSUPP = _vCONSAM.size();
  //std::vector<FFVar> SEL( NSUPP*(NSUPP-1)/2 );
  std::vector<FFVar> SEL( NSUPP );
  for( auto& Sk : SEL )
    Sk.set( _dagdoe );
  doe.add_var( SEL, 0e0, 1e0 ); // auxiliary continuous
  E0.insert( E0.end(), NSUPP, 0e0 );
  if( !EBND.empty() ) EBND.insert( EBND.end(), NSUPP, I(0e0,1e0) );

  FFVar OD( _dagdoe );
  doe.add_var( OD, 0e0 );
  E0.insert( E0.end(), 1, 0e0 );
  if( !EBND.empty() ) EBND.insert( EBND.end(), 1, I(0e0,BASE_OPT::INF) );

  FFODISTEff OpODISTEff;
  //std::vector<FFVar> SELi( NSUPP );
  for( size_t i=0; i<NSUPP; i++ ){
    //for( size_t j=0; j<i; j++ )
    //  SELi[j] = SEL[i*(i-1)/2+j];
    //SELi[i] = 0.;
    //for( size_t j=i+1; j<NSUPP; j++ )
    //  SELi[j] = SEL[j*(j-1)/2+i];
    //FFVar vODISTEff = OpODISTEff( SELi, NEXP, i, options.IDWTOL, &_vOUTSAM, &_vEFFAP );
    FFVar vODISTEff = OpODISTEff( SEL, NEXP, i, options.IDWTOL, &_vOUTSAM, &_vEFFAP );
    doe.add_ctr( BASE_OPT::GE, OD - vODISTEff + (1-SEL[i]) / options.IDWTOL );
    doe.add_ctr( BASE_OPT::GE, SEL[i] - EFF[i] );
    //doe.add_ctr( BASE_OPT::GE, OD - vODISTEff );
    //for( size_t j=0; j<i; j++ )
    //  doe.add_ctr( BASE_OPT::GE, SELi[j] - EFF[i] - EFF[j] + 1 );
  }

  doe.set_obj( BASE_OPT::MIN, OD );

}

inline
void
EXPDES::_effort_set_FIMNeutral
( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF )
const
{
  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF, 1. ) - NEXP );

  FFFIMEff OpFIMEff;
  FFVar const*const* ppFIMEff = OpFIMEff( EFF, &_vFIMSAM, &_vEFFAP );

  size_t const NUNC  = _vPARVAL.size();
  if( NUNC == 1 ){
    doe.set_obj( BASE_OPT::MAX, *(ppFIMEff[0]) );
    return;
  }

  doe.set_obj( BASE_OPT::MAX, Sum( NUNC, OpFIMEff( EFF, &_vFIMSAM, &_vEFFAP ), _vPARWEI.data() ) );
}

inline
void
EXPDES::_effort_set_FIMAverse
( MINLP& doe, FFVar const& NEXP, std::vector<FFVar> const& EFF,
  std::vector<double>& E0, std::vector<I>& EBND )
const
{
  FFLin<I> Sum;
  doe.add_ctr( BASE_OPT::EQ, Sum( EFF, 1. ) - NEXP );

  FFFIMEff OpFIMEff;
  FFVar const*const* ppFIMEff = OpFIMEff( EFF, &_vFIMSAM, &_vEFFAP );

  size_t const NUNC  = _vPARVAL.size();
  if( NUNC == 1 ){
    doe.set_obj( BASE_OPT::MAX, *(ppFIMEff[0]) );
    return;
  }

  FFVar VaR( _dagdoe );
  doe.add_var( VaR );//, -1e2, 1e2 );
  E0.insert( E0.end(), 1, 0e0 );
  if( !EBND.empty() ) EBND.insert( EBND.end(), 1, I(-BASE_OPT::INF,BASE_OPT::INF) );

  std::vector<FFVar> DELTA( NUNC );
  for( auto& Dk : DELTA )
    Dk.set( _dagdoe );
  doe.add_var( NUNC, DELTA.data(), 0e0 );//, 1e2 );
  E0.insert( E0.end(), NUNC, 0e0 );
  if( !EBND.empty() ) EBND.insert( EBND.end(), NUNC, I(0e0,BASE_OPT::INF) );

  doe.set_obj( BASE_OPT::MAX, VaR - Sum( DELTA, _vPARWEI ) / options.CVARTHRES );
  for( size_t s=0; s<NUNC; s++ )
    doe.add_ctr( BASE_OPT::LE, VaR - DELTA[s] - *(ppFIMEff[s]) );

}

inline
int
EXPDES::effort_solve
( std::vector<size_t> const& vNEXP, bool const exact, std::map<size_t,double> const& EIni,
  std::ostream& os )
{
  if( vNEXP.empty() ) throw Exceptions( Exceptions::BADNEXP );

  auto&& t_slvmip = stats.start();

  // Define MINLP DAG
  delete _dagdoe; _dagdoe = new DAG;

  FFVar NOEXP = _dagdoe->add_var( "NEXP" );
  double DNEXP = vNEXP.front();

  size_t const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF( NSUPP );
  for( auto& Ek : EFF )
    Ek.set( _dagdoe );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, DNEXP/NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [ndx,eff] : EIni )
      E0[ndx] = eff;
  }
  
  // Total experiment count and default bounds
  size_t TOTEXP = std::accumulate( std::next(vNEXP.cbegin()), vNEXP.cend(), vNEXP.front() );
  std::vector<I> EBND( NSUPP, I(0e0,TOTEXP) );

  // Update external operations
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;

  // Convex MINLP optimization
  MINLP doe;
  doe.options = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( EFF, 0e0, TOTEXP, exact );
  doe.set_par( NOEXP );

  switch( options.CRITERION ){
    case BRISK:
      _effort_set_BRisk( doe, NOEXP, EFF );
      break;

    case ODIST:
      if( TOTEXP > NSUPP ) throw Exceptions( Exceptions::BADNEXP );
      _effort_set_ODist( doe, NOEXP, EFF, E0, EBND );
      break;
      
    case AOPT:
    case DOPT:
    case EOPT:
    default:
      switch( options.RISK){
        case Options::NEUTRAL:
          _effort_set_FIMNeutral( doe, NOEXP, EFF );
          break;
        case Options::AVERSE:
          _effort_set_FIMAverse( doe, NOEXP, EFF, E0, EBND );
          break;
      }
      break;
  }
  
  std::cout << "\nSTARTING MINLP SET-UP\n";
  doe.setup();
  
  bool first = true;
  for( auto const& NEXP : vNEXP ){
    // Update bounds and initial guess
    if( !first ){
      size_t isupp = 0;
      for( auto const& Ek : doe.get_incumbent().x ){
        if( isupp >= NSUPP )
          break;
        if( Ek > 1e-3 ){
          E0[isupp]   = Ek;
          EBND[isupp] = Ek;
        }
        else{
          E0[isupp]   = (double)NEXP/(double)NSUPP;
          EBND[isupp] = I(0,NEXP);
        }
        ++isupp;
      }
      DNEXP += NEXP;

    }
    first = false;

    std::cout << "\nSTARTING MINLP OPTIMIZATION\n";
    //doe.optimize( E0.data(), EBND.data(), &DNEXP );
    //doe.optimize( E0.data(), EBND.data(), &DNEXP, effort_apportion );
    doe.optimize( E0.data(), EBND.data(), &DNEXP, effort_rounding );

    if( doe.get_status() != MINLP::SUCCESSFUL )
      break; 
  }
  
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
int
EXPDES::effort_solve
( size_t const NEXP, bool const exact, std::map<size_t,double> const& EIni,
  std::ostream& os )
{
  auto&& t_slvmip = stats.start();

  // Define MINLP DAG
  delete _dagdoe; _dagdoe = new DAG;

  FFVar NOEXP = _dagdoe->add_var( "NEXP" );
  double DNEXP = NEXP;
  
  size_t const NSUPP = _vCONSAM.size();
  std::vector<FFVar> EFF = _dagdoe->add_vars( NSUPP, "EFF" );
  std::vector<double> E0;
  if( EIni.empty() )
    E0.assign( NSUPP, (double)NEXP/(double)NSUPP );
  else{
    E0.assign( NSUPP, 0e0 );
    for( auto const& [ndx,eff] : EIni )
      E0[ndx] = eff;
  }
  std::vector<I> EBND;
    
  // Update external operations
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;

  // Convex MINLP optimization
  MINLP doe;
  doe.options = options.MINLPSLV;
  doe.set_dag( _dagdoe );
  doe.set_var( EFF, 0e0, NEXP, exact );
  doe.set_par( NOEXP );

  switch( options.CRITERION ){
    case BRISK:
      _effort_set_BRisk( doe, NOEXP, EFF );
      break;

    case ODIST:
      if( NEXP > NSUPP ) throw Exceptions( Exceptions::BADNEXP );
      _effort_set_ODist( doe, NOEXP, EFF, E0, EBND );
      break;
      
    case AOPT:
    case DOPT:
    case EOPT:
    default:
      switch( options.RISK){
        case Options::NEUTRAL:
          _effort_set_FIMNeutral( doe, NOEXP, EFF );
          break;
        case Options::AVERSE:
          _effort_set_FIMAverse( doe, NOEXP, EFF, E0, EBND );
          break;
      }
      break;
  }
  
  doe.setup();
  //doe.optimize( E0.data(), nullptr, &DNEXP );
  //doe.optimize( E0.data(), nullptr, &DNEXP, effort_apportion );
  doe.optimize( E0.data(), nullptr, &DNEXP, effort_rounding );

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
EXPDES::campaign
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
void
EXPDES::_evaluate_constraints
( std::vector<double>& DCTR, size_t const NSUP, size_t const NUNC,
  double const* DCTRSAM, std::ostream& os )
{
  if( !_ng ) return;

  DCTR.resize( NSUP );
  for( size_t e=0; e<NSUP; ++e ){
    // Return maximal constraint violation if unique scenario
    if( NUNC == 1 ){
     if( _ng == 1 )
        DCTR[e] = DCTRSAM[e]; 
      else{
#ifdef MAGNUS__EXPDES_EVAL_DEBUG
        std::cout << "DCTRSAM[" << e*_ng << ":" << (e+1)*_ng-1 << "] = " << arma::rowvec( DCTRSAM+e*_ng, _ng, false );
#endif
        auto itMax = std::max_element( DCTRSAM+e*_ng, DCTRSAM+(e+1)*_ng );
        DCTR[e] = *itMax;
      }
#ifdef MAGNUS__EXPDES_EVAL_DEBUG
      std::cout << "DCTR[" << e << "] = " << DCTR[e] << std::endl;
#endif
      continue;
    }

    // Conditional-value-at-risk
    std::multimap<double,double> ResCTR;
    for( size_t s=0; s<NUNC; ++s ){
      auto itMax = std::max_element( DCTRSAM+e*_ng*NUNC+s*_ng, DCTRSAM+e*_ng*NUNC+(s+1)*_ng );
      ResCTR.insert( { -*itMax, _vPARWEI[s] } ); // ordered by largest (negative) violation first 
    }

    double PrMass = 0., VaR = 0.;
    for( auto const& [Res,Pr] : ResCTR ){
      VaR = Res;
      if( PrMass + Pr > options.FEASTHRES ) break;
      PrMass += Pr;
    }
    DCTR[e] = -VaR;
#ifdef MAGNUS__EXPDES_EVAL_DEBUG
    std::cout << "VaR[" << e <<  "= ]" << -VaR << std::endl;
#endif

    for( auto const& [Res,Pr] : ResCTR ){
      if( Res > VaR ) break;
      DCTR[e] += ( VaR - Res ) * Pr / options.FEASTHRES;
    }
#ifdef MAGNUS__EXPDES_EVAL_DEBUG
    std::cout << "DCTR[" << e << "] = " << DCTR[e] << std::endl;
#endif
  }
}

inline
std::pair<double,std::vector<double>>
EXPDES::_evaluate_ODist
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double> const& UTOT0, std::ostream& os )
{

  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() ){
    auto DISPLEVEL  = options.DISPLEVEL;
    auto UNCREDUC   = options.UNCREDUC;
    options.UNCREDUC  = 0.;
    options.DISPLEVEL = 0;
    _sample_out( 0, os );
    options.DISPLEVEL = DISPLEVEL;
    options.UNCREDUC  = UNCREDUC;
  }

  // Define output distance criterion
  FFODISTCrit OpODCrit;
  FFVar const*const* ppODCrit = OpODCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vOUT, &_vCTR, &EOpt,
                                          &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP, options.IDWTOL );
#ifdef MAGNUS__EXPDES_EVAL_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppODCrit[0] ), " ODISTCrit" );
#endif

  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();
  std::vector<double> DODCrit( NSUP*(1+_ng*NUNC) );
  std::vector<FFVar>  FODCrit( NSUP*(1+_ng*NUNC) );
  for( size_t s=0; s<NSUP*(1+_ng*NUNC); ++s )
    FODCrit[s] = *ppODCrit[s];

  // Evaluate cost and any constraints
  _dagdoe->eval( FODCrit, DODCrit, UTOT, UTOT0 );
#ifdef MAGNUS__EXPDES_EVAL_GRADIENT
  double const TOL = 1e-3;
  for( size_t i=0; i<UTOT.size(); i++ ){
    auto UTOT0_pert = UTOT0;
    UTOT0_pert[i] = UTOT0[i] * ( 1 + TOL ) + TOL;
    std::vector<double> DODCrit_pert( NSUP*(1+_ng*NUNC) );
    _dagdoe->eval( FODCrit, DODCrit_pert, UTOT, UTOT0_pert );
    std::cout << "d_UTOT[" << i << "] = " << (UTOT0_pert[i]-UTOT0[i]) << std::endl;
    std::cout << "gradFD OD[" << i << "] =";
    for( size_t s=0; s<NSUP*(1+_ng*NUNC); ++s )
      std::cout << "  " << (DODCrit_pert[s]-DODCrit[s]) / (UTOT0_pert[i]-UTOT0[i]);
    std::cout << std::endl;
  }
  auto&& FGRADODCrit = _dagdoe->FAD( FODCrit, UTOT );
  //_dagdoe->output( _dagdoe->subgraph( FGRADODCrit ), " GradODISTCrit" );
  std::vector<double> DGRADODCrit;
  _dagdoe->eval( FGRADODCrit, DGRADODCrit, UTOT, UTOT0 );
  for( size_t i=0; i<UTOT.size(); i++ ){
    std::cout << "grad OD[" << i << "] =";
    for( size_t s=0; s<NSUP*(1+_ng*NUNC); ++s )
      std::cout << "  " << DGRADODCrit[s*UTOT.size()+i];
    std::cout << std::endl;
  }
#endif

  // Distance criterion
  double ODIST = 0.;
  for( size_t e=0; e<NSUP; ++e )
    if( ODIST < DODCrit[e] ) ODIST = DODCrit[e];
#ifdef MAGNUS__EXPDES_EVAL_DEBUG
  std::cout << "ODIST = " << ODIST << std::endl;
#endif

  // Constraint satisfaction
  std::vector<double> DCTR;
  _evaluate_constraints( DCTR, NSUP, NUNC, &DODCrit[NSUP], os );

  return std::make_pair( ODIST, DCTR );
}

inline
std::pair<double,std::vector<double>>
EXPDES::_evaluate_BRisk
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double> const& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() ){
    auto DISPLEVEL  = options.DISPLEVEL;
    auto UNCREDUC   = options.UNCREDUC;
    options.UNCREDUC  = 0.;
    options.DISPLEVEL = 0;
    _sample_out( 0, os );
    options.DISPLEVEL = DISPLEVEL;
    options.UNCREDUC  = UNCREDUC;
  }

  // Define Bayes risk criterion
  FFBRCrit OpBRCrit;
  FFVar const*const* ppBRCrit = OpBRCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vOUT, &_vCTR, &EOpt,
                                          &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppBRCrit[0] ), " BRCrit" );
#endif

  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();
  std::vector<double> DBRCrit( 1+NSUP*_ng*NUNC );
  std::vector<FFVar>  FBRCrit( 1+NSUP*_ng*NUNC );
  for( size_t s=0; s<1+NSUP*_ng*NUNC; ++s )
    FBRCrit[s] = *ppBRCrit[s];

  // Evaluate Bayes risk criterion and any constraints
  _dagdoe->eval( FBRCrit, DBRCrit, UTOT, UTOT0 );
#ifdef MAGNUS__EXPDES_EVAL_GRADIENT
  double const TOL = 1e-3;
  for( size_t i=0; i<UTOT.size(); i++ ){
    auto UTOT0_pert = UTOT0;
    UTOT0_pert[i] = UTOT0[i] * ( 1 + TOL ) + TOL;
    std::vector<double> DBRCrit_pert( 1+NSUP*_ng*NUNC );
    _dagdoe->eval( FBRCrit, DBRCrit_pert, UTOT, UTOT0_pert );
    std::cout << "d_UTOT[" << i << "] = " << (UTOT0_pert[i]-UTOT0[i]) << std::endl;
    std::cout << "gradFD BR[" << i << "] =";
    for( size_t s=0; s<1+NSUP*_ng*NUNC; ++s )
      std::cout << "  " << (DBRCrit_pert[s]-DBRCrit[s]) / (UTOT0_pert[i]-UTOT0[i]);
    std::cout << std::endl;
  }
  auto&& FGRADBRCrit = _dagdoe->FAD( FBRCrit, UTOT );
  //_dagdoe->output( _dagdoe->subgraph( FGRADBRCrit ), " GradBRCrit" );
  std::vector<double> DGRADBRCrit;
  _dagdoe->eval( FGRADBRCrit, DGRADBRCrit, UTOT, UTOT0 );
  for( size_t i=0; i<UTOT.size(); i++ ){
    std::cout << "grad BR[" << i << "] =";
    for( size_t s=0; s<1+NSUP*_ng*NUNC; ++s )
      std::cout << "  " << DGRADBRCrit[s*UTOT.size()+i];
    std::cout << std::endl;
  }
#endif

  // Constraint satisfaction
  std::vector<double> DCTR;
  _evaluate_constraints( DCTR, NSUP, NUNC, &DBRCrit[1], os );

  return std::make_pair( DBRCrit[0], DCTR );
}

inline
std::pair<double,std::vector<double>>
EXPDES::_evaluate_FIMNeutral
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double> const& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() ){
    auto DISPLEVEL = options.DISPLEVEL;
    auto FIMSTOL   = options.FIMSTOL;
    options.FIMSTOL   = 0.;
    options.DISPLEVEL = 0;
    _sample_fim( 0, os );
    options.DISPLEVEL = DISPLEVEL;
    options.FIMSTOL   = FIMSTOL;
  }

  // Define FIM criterion
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vFIM, &_vCTR, &EOpt,
                                            &_vPARVAL, &_vCSTVAL, &_vFIMSAM, &_vEFFAP );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppFIMCrit[0] ), " FIMCrit" );
#endif

  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();
  std::vector<double> DFIMCrit( 1+NSUP*_ng*NUNC );
  std::vector<FFVar>  FFIMCrit( 1+NSUP*_ng*NUNC );
  FFLin<I> Sum;
  FFIMCrit[0] = Sum( NUNC, ppFIMCrit, _vPARWEI.data() );
  for( size_t s=0; s<NSUP*_ng*NUNC; ++s )
    FFIMCrit[1+s] = *ppFIMCrit[NUNC+s];

  // Evaluate FIM-based criterion and any constraints
  _dagdoe->eval( FFIMCrit, DFIMCrit, UTOT, UTOT0 );
#ifdef MAGNUS__EXPDES_EVAL_GRADIENT
  double const TOL = 1e-3;
  for( size_t i=0; i<UTOT.size(); i++ ){
    auto UTOT0_pert = UTOT0;
    UTOT0_pert[i] = UTOT0[i] * ( 1 + TOL ) + TOL;
    std::vector<double> DFIMCrit_pert( 1+NSUP*_ng*NUNC );
    _dagdoe->eval( FFIMCrit, DFIMCrit_pert, UTOT, UTOT0_pert );
    std::cout << "d_UTOT[" << i << "] = " << (UTOT0_pert[i]-UTOT0[i]) << std::endl;
    std::cout << "gradFD FIM[" << i << "] =";
    for( size_t s=0; s<1+NSUP*_ng*NUNC; ++s )
      std::cout << "  " << (DFIMCrit_pert[s]-DFIMCrit[s]) / (UTOT0_pert[i]-UTOT0[i]);
    std::cout << std::endl;
  }
  auto&& FGRADFIMCrit = _dagdoe->FAD( FFIMCrit, UTOT );
  //_dagdoe->output( _dagdoe->subgraph( FGRADFIMCrit ), " GradFIMCrit" );
  std::vector<double> DGRADFIMCrit;
  _dagdoe->eval( FGRADFIMCrit, DGRADFIMCrit, UTOT, UTOT0 );
  for( size_t i=0; i<UTOT.size(); i++ ){
    for( size_t s=0; s<1+NSUP*_ng*NUNC; ++s )
      std::cout << "  " << DGRADFIMCrit[s*UTOT.size()+i];
    std::cout << std::endl;
  }
#endif

  // Constraint satisfaction
  std::vector<double> DCTR;
  _evaluate_constraints( DCTR, NSUP, NUNC, &DFIMCrit[1], os );

  return std::make_pair( DFIMCrit[0], DCTR );
}

inline
std::pair<double,std::vector<double>>
EXPDES::_evaluate_FIMAverse
( std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double> const& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() ){
    auto DISPLEVEL = options.DISPLEVEL;
    auto FIMSTOL   = options.FIMSTOL;
    options.FIMSTOL   = 0.;
    options.DISPLEVEL = 0;
    _sample_fim( 0, os );
    options.DISPLEVEL = DISPLEVEL;
    options.FIMSTOL   = FIMSTOL;
  }

  // Define FIM criterion
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vFIM, &_vCTR, &EOpt,
                                            &_vPARVAL, &_vCSTVAL, &_vFIMSAM, &_vEFFAP );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
  _dagdoe->output( _dagdoe->subgraph( 1, ppFIMCrit[0] ), " FIMCrit" );
#endif

  size_t const NSUP = EOpt.size();
  size_t const NUNC = _vPARVAL.size();
  std::vector<double> DFIMCrit( (1+NSUP*_ng)*NUNC );
  std::vector<FFVar>  FFIMCrit( (1+NSUP*_ng)*NUNC );
  for( size_t s=0; s<(1+NSUP*_ng)*NUNC; ++s )
    FFIMCrit[s] = *ppFIMCrit[s];

  // Evaluate FIM-based criterion and any constraints
  _dagdoe->eval( FFIMCrit, DFIMCrit, UTOT, UTOT0 );
#ifdef MAGNUS__EXPDES_EVAL_GRADIENT
  double const TOL = 1e-3;
  for( size_t i=0; i<UTOT.size(); i++ ){
    auto UTOT0_pert = UTOT0;
    UTOT0_pert[i] = UTOT0[i] * ( 1 + TOL ) + TOL;
    std::vector<double> DFIMCrit_pert( (1+NSUP*_ng)*NUNC );
    _dagdoe->eval( FFIMCrit, DFIMCrit_pert, UTOT, UTOT0_pert );
    std::cout << "d_UTOT[" << i << "] = " << (UTOT0_pert[i]-UTOT0[i]) << std::endl;
    std::cout << "gradFD FIM[" << i << "] =";
    for( size_t s=0; s<1+NSUP*_ng*NUNC; ++s )
      std::cout << "  " << (DFIMCrit_pert[s]-DFIMCrit[s]) / (UTOT0_pert[i]-UTOT0[i]);
    std::cout << std::endl;
  }
  auto&& FGRADFIMCrit = _dagdoe->FAD( FFIMCrit, UTOT );
  //_dagdoe->output( _dagdoe->subgraph( FGRADFIMCrit ), " GradFIMCrit" );
  std::vector<double> DGRADFIMCrit;
  _dagdoe->eval( FGRADFIMCrit, DGRADFIMCrit, UTOT, UTOT0 );
  for( size_t i=0; i<UTOT.size(); i++ ){
    for( size_t s=0; s<(1+NSUP*_ng)*NUNC; ++s )
      std::cout << "  " << DGRADFIMCrit[s*UTOT.size()+i];
    std::cout << std::endl;
  }
#endif

  std::multimap<double,double> SA;
  for( size_t s=0; s<NUNC; ++s )
    SA.insert( { DFIMCrit[s], _vPARWEI[s] } );

  double prsum = 0., VaR = 0.;
  for( auto const& [crit,pr] : SA ){
    VaR = crit;
    if( prsum + pr > options.CVARTHRES ) break;
    prsum += pr;
  }
  //std::cout << "VaR = " << VaR << std::endl;

  double DFIM = VaR;
  for( auto const& [crit,pr] : SA ){
    if( crit > VaR ) break;
    DFIM -= ( VaR - crit ) * pr / options.CVARTHRES;
  }
  //std::cout << "CVaR = " << DFIM << std::endl;

  // Constraint satisfaction
  std::vector<double> DCTR;
  _evaluate_constraints( DCTR, NSUP, NUNC, &DFIMCrit[NUNC], os );

  return std::make_pair( DFIM, DCTR );
}

inline
std::tuple<double,std::vector<double>,bool>
EXPDES::evaluate_design
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
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;

  // Evaluate design
  std::string header( type.empty()? "DESIGN PERFORMANCE": type + " DESIGN PERFORMANCE" );
  std::pair<double,std::vector<double>> crit{ 0./0., std::vector<double>() };

  try{
    switch( options.CRITERION ){
      case BRISK:
        crit = _evaluate_BRisk( EOpt, UTOT, UTOT0, os );
        break;

      case ODIST:
        crit = _evaluate_ODist( EOpt, UTOT, UTOT0, os );
        break;
      
      case AOPT:
      case DOPT:
      case EOPT:
      default:
        switch( options.RISK){
          case Options::NEUTRAL:
            crit = _evaluate_FIMNeutral( EOpt, UTOT, UTOT0, os );
            break;
          case Options::AVERSE:
            crit = _evaluate_FIMAverse( EOpt, UTOT, UTOT0, os );
            break;
        }
        break;
    }
  }

  catch(...){
    if( options.DISPLEVEL )
      _display_design( header, crit.first, std::list<std::pair<double,std::vector<double>>>(), os ); 
    return std::make_tuple( crit.first, crit.second, false ); // NaN
  }

  if( options.DISPLEVEL )
    _display_design( header, crit.first, Campaign, os ); 
  return std::make_tuple( crit.first, crit.second, true );
}

inline
void
EXPDES::_refine_set_constraints
( NLP& doeref, size_t const NE, FFVar const*const* ppCTRSAM,
  std::vector<double>& UTOT0, std::ostream& os )
{
  if( !_ng ) return;
  size_t const NG = _vCTR.size();

  // Single scenario case
  size_t const NS = _vPARVAL.size();
  if( NS == 1 ){
    for( size_t e=0; e<NE; ++e )
      for( size_t g=0; g<NG; ++g )
        doeref.add_ctr( BASE_OPT::LE, *ppCTRSAM[e*NG+g] );
    return;    
  }

  // Multiple scenario case: Define CVaR constraints
  FFLin<I> Sum;
  for( size_t e=0; e<NE; ++e ){

    // Same VaR and delta vector shared between constraints since CVaR is on the max violation
    std::vector<FFVar> Dg = _dagdoe->add_vars( NS, "Dg" + std::to_string(e) );
    FFVar VaRg = _dagdoe->add_var( "VaRg" + std::to_string(e) );
    doeref.add_var( Dg, 0e0 );//, 1e2 );
    doeref.add_var( VaRg );//, -1e2, 1e2 );

    // Append initial point for VaR and delta vector
    //if( _ROpt.size() == NS+1 )
    //  for( auto const& r0 : _ROpt ) UTOT0.push_back( r0 ); 
    //else
    UTOT0.resize( UTOT0.size()+NS+1, 0e0 );

    // Add CVaR linear constraints to DAG
    for( size_t g=0; g<NG; ++g ){
      doeref.add_ctr( BASE_OPT::LE, VaRg + Sum( NS, Dg.data(), _vPARWEI.data() ) / options.FEASTHRES );  // enforce CVaR constraint
      for( size_t s=0; s<NS; s++ )
        doeref.add_ctr( BASE_OPT::GE, VaRg + Dg[s] - *ppCTRSAM[(e*NS+s)*NG+g] );
    }
  }
}

inline
void
EXPDES::_refine_set_ODist
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double>& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() )
    _sample_out( 0, os );

  // Define output distance objective
  FFVar OD =_dagdoe->add_var( "OD" );
  doeref.add_var( OD );
  doeref.set_obj( BASE_OPT::MIN, OD );

  size_t const NC = _vCONSAM.size();
  if( _ROpt.size() == NC+1 )
    UTOT0.push_back( _ROpt.back() ); // last element recorded from effort_solve is OD
  else
    UTOT0.push_back( 0e0 );

  size_t const NE = EOpt.size();
  FFODISTCrit OpODCrit;
  FFVar const*const* ppODCrit = OpODCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vOUT, &_vCTR, &EOpt,
                                          &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP, options.IDWTOL );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
  auto sgODCrit = _dagdoe->subgraph( NE, ppODCrit );
  std::vector<FFExpr> exCTR = FFExpr::subgraph( _dagdoe, sgODCrit ); 
  for( size_t i=0; i<NE; ++i )
    std::cout << "ODISTCrit[" << i << "] = " << exCTR[i] << std::endl;
  { std::cout << "Enter <1> to continue"; int dum; std::cin >> dum;}
#endif

  for( size_t e=0; e<NE; ++e )
    doeref.add_ctr( BASE_OPT::LE, *ppODCrit[e] - OD );

  // Define constraints
  _refine_set_constraints( doeref, NE, ppODCrit+NE, UTOT0, os );
}

inline
void
EXPDES::_refine_set_BRisk
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double>& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vOUTSAM.empty() )
    _sample_out( 0, os );

  // Define Bayes risk objective
  size_t const NE = EOpt.size();
  FFBRCrit OpBRCrit;
  FFVar const*const* ppBRCrit = OpBRCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vOUT, &_vCTR, &EOpt,
                                          &_vPARVAL, &_vCSTVAL, &_vOUTSAM, &_vEFFAP );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
  auto sgBRCrit = _dagdoe->subgraph( NE, ppBRCrit );
  std::vector<FFExpr> exBRCrit = FFExpr::subgraph( _dagdoe, sgBRCrit ); 
  for( size_t i=0; i<NE; ++i )
    std::cout << "BRCrit[" << i << "] = " << exBRCrit[i] << std::endl;
  { std::cout << "Enter <1> to continue"; int dum; std::cin >> dum;}
#endif
  doeref.set_obj( BASE_OPT::MIN, *ppBRCrit[0] ); // minimize Bayesian risk

  // Define constraints
  _refine_set_constraints( doeref, NE, ppBRCrit+1, UTOT0, os );
}

inline
void
EXPDES::_refine_set_FIMNeutral
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double>& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() )
    _sample_fim( 0, os );

  // Define FIM criterion
  size_t const NE = EOpt.size();
  size_t const NS = _vPARVAL.size();
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vFIM, &_vCTR, &EOpt,
                                            &_vPARVAL, &_vCSTVAL, &_vFIMSAM, &_vEFFAP );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
  auto sgFIMCrit = _dagdoe->subgraph( NE, ppFIMCrit );
  std::vector<FFExpr> exFIMCrit = FFExpr::subgraph( _dagdoe, sgFIMCrit ); 
  for( size_t i=0; i<NS; ++i )
    std::cout << "FIMCrit[" << i << "] = " << exFIMCrit[i] << std::endl;
  { std::cout << "Enter <1> to continue"; int dum; std::cin >> dum;}
#endif
  FFLin<I> Sum;
  FFVar FFIM = Sum( NS, ppFIMCrit, _vPARWEI.data() );
  doeref.set_obj( BASE_OPT::MAX, FFIM ); // maximize average FIM-based criterion

  // Define constraints
  _refine_set_constraints( doeref, NE, ppFIMCrit+NS, UTOT0, os );
}

inline
void
EXPDES::_refine_set_FIMAverse
( NLP& doeref, std::map<size_t,double> const& EOpt, std::vector<FFVar> const& UTOT,
  std::vector<double>& UTOT0, std::ostream& os )
{
  // Check supports for a priori experiments
  if( !_vEFFAP.empty() && _vFIMSAM.empty() )
    _sample_fim( 0, os );

  // Define FIM criterion
  size_t const NE = EOpt.size();
  size_t const NS = _vPARVAL.size();
  FFFIMCrit OpFIMCrit;
  FFVar const*const* ppFIMCrit = OpFIMCrit( UTOT.data(), _dag, &_vPAR, &_vCST, &_vCON, &_vFIM, &_vCTR, &EOpt,
                                            &_vPARVAL, &_vCSTVAL, &_vFIMSAM, &_vEFFAP );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
  auto sgFIMCrit = _dagdoe->subgraph( NE, ppFIMCrit );
  std::vector<FFExpr> exFIMCrit = FFExpr::subgraph( _dagdoe, sgFIMCrit ); 
  for( size_t i=0; i<NS; ++i )
    std::cout << "FIMCrit[" << i << "] = " << exFIMCrit[i] << std::endl;
  { std::cout << "Enter <1> to continue"; int dum; std::cin >> dum;}
#endif
  if( NS == 1 ){
    doeref.set_obj( BASE_OPT::MAX, *ppFIMCrit[0] );
  }
  else{
    std::vector<FFVar> DELTA( NS );
    FFVar VaR( _dagdoe );
    for( auto& Dk : DELTA )
      Dk.set( _dagdoe );
    if( _ROpt.size() == NS+1 )
      for( auto const& r0 : _ROpt ) UTOT0.push_back( r0 ); 
    else
      UTOT0.resize( UTOT.size()+NS+1, 0e0 );
    doeref.add_var( DELTA, 0e0 );//, 1e2 );
    doeref.add_var( VaR );//, -1e2, 1e2 );

    FFLin<I> Sum;
    doeref.set_obj( BASE_OPT::MAX, VaR - Sum( NS, DELTA.data(), _vPARWEI.data() ) / options.CVARTHRES );  // maximize risk-averse FIM-based criterion
    for( size_t s=0; s<NS; s++ )
      doeref.add_ctr( BASE_OPT::LE, VaR - DELTA[s] - *ppFIMCrit[s] );
  }

  // Define constraints
  _refine_set_constraints( doeref, NE, ppFIMCrit+NS, UTOT0, os );
}

inline
int
EXPDES::gradient_solve
( std::list<std::pair<double,std::vector<double>>> const& Campaign,
  std::vector<double> const& vcst, bool const update, std::ostream& os )
{
  // Initial guess and control bounds
  size_t const NSUP  = Campaign.size();
  size_t const NUTOT = _nu * NSUP;
  std::vector<double> UTOT0, UTOTLB, UTOTUB;
  UTOT0.reserve(NUTOT);
  UTOTLB.reserve(NUTOT);
  UTOTUB.reserve(NUTOT);
  std::map<size_t,double> EOpt;
  size_t ieff = 0;
  for( auto const& [eff,supp] : Campaign ){
    EOpt[ieff++] = eff;
    UTOT0.insert( UTOT0.end(), supp.cbegin(), supp.cend() );
    UTOTLB.insert( UTOTLB.end(), _vCONLB.cbegin(), _vCONLB.cend() );
    UTOTUB.insert( UTOTUB.end(), _vCONUB.cbegin(), _vCONUB.cend() );
  }

  // Perform optimisation and update
  return _gradient_solve( EOpt, UTOT0, UTOTLB, UTOTUB, vcst, update, os );
}

inline
int
EXPDES::gradient_solve
( std::map<size_t,double> const& EOpt, std::vector<double> const& vcst,
  bool const update, std::ostream& os )
{
  // Initial guess and control bounds 
  //_EOpt = EOpt;
  size_t const NSUP = EOpt.size();
  size_t const NUTOT = _nu * NSUP;
  std::vector<double> UTOT0, UTOTLB, UTOTUB;
  UTOT0.reserve(NUTOT);
  UTOTLB.reserve(NUTOT);
  UTOTUB.reserve(NUTOT);
  for( auto const& [ndx,eff] : _EOpt ){
    UTOT0.insert( UTOT0.end(), _vCONSAM[ndx].cbegin(), _vCONSAM[ndx].cend() );
#ifdef MAGNUS__EXPDES_SOLVE_DEBUG
    std::cout << "U[" << ndx << "] = ";
    for( size_t i=0; i<_nu; i++ )
      std::cout << _vCONSAM[ndx][i] << "  ";
    std::cout << std::endl;
#endif
    UTOTLB.insert( UTOTLB.end(), _vCONLB.cbegin(), _vCONLB.cend() );
    UTOTUB.insert( UTOTUB.end(), _vCONUB.cbegin(), _vCONUB.cend() );
  }

  // Perform optimisation and update
  return _gradient_solve( EOpt, UTOT0, UTOTLB, UTOTUB, vcst, update, os );
}

inline
int
EXPDES::_gradient_solve
( std::map<size_t,double> const& EOpt, std::vector<double>& UTOT0,
  std::vector<double> const& UTOTLB, std::vector<double> const& UTOTUB,
  std::vector<double> const& vcst, bool const update, std::ostream& os )
{
  auto&& t_slvnlp = stats.start();

  // Define NLP DAG
  delete _dagdoe; _dagdoe = new DAG;
  
  //_EOpt = EOpt;
  size_t const NSUP = EOpt.size();
  size_t const NUTOT = _nu * NSUP;
  std::vector<FFVar> UTOT = _dagdoe->add_vars(NUTOT,"C");  // Concatenated experimental controls

  // Update constants
  if( vcst.size() && vcst.size() == _nc ) _vCSTVAL = vcst;
  if( _nc && _vCSTVAL.empty() ) throw Exceptions( Exceptions::BADCONST );

  // Update external operations
  FFDOEBase::set_weighting( _vPARWEI );
  FFDOEBase::set_scaling( _mPARSCA );
  FFDOEBase::set_noise( _vOUTVAR );
  FFDOEBase::parsubset = &_sPARSEL;
  FFDOEBase::type = options.CRITERION;

  // Local NLP optimization
  NLP doeref;
  doeref.options = options.NLPSLV;
  doeref.set_dag( _dagdoe );
  doeref.add_var( UTOT, UTOTLB, UTOTUB );

  switch( options.CRITERION ){
    case BRISK:
      _refine_set_BRisk( doeref, EOpt, UTOT, UTOT0, os );
      break;
      
    case ODIST:
      _refine_set_ODist( doeref, EOpt, UTOT, UTOT0, os );
      break;
      
    case AOPT:
    case DOPT:
    case EOPT:
    default:
      switch( options.RISK){
        case Options::NEUTRAL:
          _refine_set_FIMNeutral( doeref, EOpt, UTOT, UTOT0, os );
          break;
        case Options::AVERSE:
          _refine_set_FIMAverse( doeref, EOpt, UTOT, UTOT0, os );
          break;
      }
      break;
  }

  doeref.setup();
  doeref.solve( UTOT0.data() );

  if( options.DISPLEVEL > 1 )
    os << "#  FEASIBLE:   " << doeref.is_feasible( 1e-6 )   << std::endl
       << "#  STATIONARY: " << doeref.is_stationary( 1e-6 ) << std::endl
       << std::endl;

  //if( update ){
    if( doeref.get_status() == NLP::SUCCESSFUL
     || doeref.get_status() == NLP::FAILURE
     || doeref.get_status() == NLP::INTERRUPTED ){
      std::map<size_t,std::vector<double>> SOpt;
      double const* dC = doeref.solution().x.data();
      for( auto const& [ndx,eff] : EOpt ){
        SOpt[ndx] = std::vector<double>( dC, dC+_nu );
        dC += _nu;
      }
      _update_supports( EOpt, SOpt, os );
      _VOpt = doeref.solution().f[0];
      size_t const NEXTRA = doeref.solution().x.size() - NUTOT;
      if( NEXTRA > 0 )
        _ROpt.assign( dC, dC+NEXTRA );
    }
    else{
      _EOpt.clear();
      _SOpt.clear();
      _VOpt = 0./0.;
    }
  //}

  if( options.DISPLEVEL )
    _display_design( "GRADIENT-BASED REFINED DESIGN", _VOpt, _EOpt, _SOpt, os ); 
  //os << doeref.solution();

  stats.walltime_slvnlp += stats.lapse( t_slvnlp );
  stats.walltime_all    += stats.lapse( t_slvnlp );

  return doeref.get_status();
}

inline
void
EXPDES::_display_design
( std::string const& title, double const& crit, std::map<size_t,double> const& eff,
  std::map<size_t,std::vector<double>> const& supp, std::ostream& os )
const
{
  os << "** " << title << ": ";

  if( eff.empty() || supp.empty() ){
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
EXPDES::_display_design
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

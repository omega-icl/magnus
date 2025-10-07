// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MAGNUS__FFNSAMP_HPP
#define MAGNUS__FFNSAMP_HPP

#include <fstream>
#include <iomanip>
#include <armadillo>

#include "base_mbfa.hpp"
#include "ffdep.hpp"

namespace mc
{

////////////////////////////////////////////////////////////////////////
// EXTERNAL OPERATIONS
////////////////////////////////////////////////////////////////////////

class FFNSamp
: public FFOp
{
private:

  // number of parameters
  size_t _np;
  // number of constants
  size_t _nc;
  // number of controls
  size_t _nu;
  // number of constraints
  size_t _ng;
  // number of likelihood
  size_t _nl;
  // number of scenarios
  size_t _ns;

  // DAG of constraints
  mutable FFGraph* _DAG;
  // parameters
  std::vector<FFVar> _FPAR;
  // constants
  std::vector<FFVar> _FCST;
  // controls
  std::vector<FFVar> _FCON;
  // functions
  std::vector<FFVar> _FFCT;
  
  // parameter scenarios: _ns x _np
  std::vector<std::vector<double>> const* _DPAR;
  // parameter scenario weights: _np
  std::vector<double> const* _WPAR;
  // constant values: _nc
  std::vector<double> const* _DCST;

  // control values
  mutable std::vector<double> _DCON;
  // function values
  mutable std::vector<std::vector<double>> _DFCT; // _ns x #functions
  // subgraph
  mutable FFSubgraph _sgFCT;
  // work storage
  mutable std::vector<double> _wkD;
  // thread storage
  mutable std::vector<FFGraph::Worker<double>> _wkThd;

  // Evaluation of feasibility criterion from constraint values
  void _feasval
    ( double* CRIT, std::vector<std::vector<double>>& DFCT )
    const;
  // Evaluation of likelihood criterion
  void _lkhdval
    ( double* CRIT, std::vector<std::vector<double>>& DFCT )
    const;
    
public:

  // Feasibility confidence threshold for constraints
  static double ConfCTR;

  // Feasibility confidence threshold for likelihood
  static double ConfLKH;

  // Set variables, parameters and scenarios
  void set
    ( FFGraph* dag, std::vector<FFVar> const* con, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* cst, std::vector<FFVar> const* ctr, FFVar const* lkh,
      std::vector<std::vector<double>> const* vpar, std::vector<double> const* wpar,
      std::vector<double> const* vcst )
    {
#ifdef MAGNUS__FFNSAMP_CHECK
      assert( dag && con->size() && (ctr || lkh) );
#endif
      _np = par? par->size(): 0;
      _nc = cst? cst->size(): 0;
      _nu = con->size();
      _ng = ctr? ctr->size(): 0;
      _nl = lkh? 1: 0;

      if( _DAG ) delete _DAG;
      _DAG  = new FFGraph;
      _DAG->options = dag->options;

      _DAG->insert( dag, *con, _FCON );
      if( _np ) _DAG->insert( dag, *par, _FPAR );
      if( _nc ) _DAG->insert( dag, *cst, _FCST );

      _FFCT.clear();
      _FFCT.reserve( _ng+_nl );
      if( _ng ) _DAG->insert( dag, *ctr, _FFCT );
      if( _nl ){
        FFVar FLKH;
        _DAG->insert( dag, 1, lkh, &FLKH );
        _FFCT.push_back( FLKH );
      }

#ifdef MAGNUS__FFNSAMP_CHECK
      assert( !vcst || vcst->size() == _nc );
#endif
      _DCST = vcst;

#ifdef MAGNUS__FFNSAMP_CHECK
      assert( !vpar || vpar->at(0).size() == _np );
#endif
      _DPAR = vpar;
      _WPAR = wpar;
      _ns = _DPAR? _DPAR->size(): 1;
    }

  // Default constructor
  FFNSamp
    ( double const& ConfCTR_=0.10, double const& ConfLKH_=0.10 )
    : FFOp   ( EXTERN ),
      _DAG   ( nullptr )
    {
      ConfCTR = ConfCTR_;
      ConfLKH = ConfLKH_;
    }

  // Copy constructor
  FFNSamp
    ( FFNSamp const& Op )
    : FFOp   ( Op ),
      _np    ( Op._np ),
      _nc    ( Op._nc ),
      _nu    ( Op._nu ),
      _ng    ( Op._ng ),
      _nl    ( Op._nl ),
      _ns    ( Op._ns ),
      _DPAR  ( Op._DPAR ),
      _WPAR  ( Op._WPAR ),
      _DCST  ( Op._DCST )
    {
      _DAG  = new FFGraph;
      _DAG->options = Op._DAG->options;

      _DAG->insert( Op._DAG, Op._FCON, _FCON );
      if( _np ) _DAG->insert( Op._DAG, Op._FPAR, _FPAR );
      if( _nc ) _DAG->insert( Op._DAG, Op._FCST, _FCST );
      _DAG->insert( Op._DAG, Op._FFCT, _FFCT );
    }

  // Destructor
  virtual ~FFNSamp
    ()
    {
      delete _DAG;
    }

  // Define operation
  // Return values:
  //  [0] feasibility probability
  //  [1] max constraint value-at-risk
  //  [2] max constraint conditional-value-at-risk
  //  [3] likelihood value-at-risk
  //  [4] likelihood conditional value-at-risk
  // VaR and CVaR are in reference to confidence threshold in static public members mc::FFNSamp::ConfCTR and mc::FFNSamp::confLKH
  // For a single parameter scenario or no parameter dependence, VaR and CVaR are equal to the constraint/likelihood value 
  FFVar** operator()
    ( FFGraph* dag, std::vector<FFVar> const* con, std::vector<FFVar> const* par,
      std::vector<FFVar> const* cst, std::vector<FFVar> const* ctr, FFVar const* lkh,
      std::vector<std::vector<double>> const* vpar, std::vector<double> const* wpar,
      std::vector<double> const* vcst )

    {
      set( dag, con, par, cst, ctr, lkh, vpar, wpar, vcst );
      return insert_external_operation( *this, 5, _nu, con->data() );
    }

  FFVar& operator()
    ( unsigned const idep, FFGraph* dag, std::vector<FFVar> const* con, std::vector<FFVar> const* par,
      std::vector<FFVar> const* cst, std::vector<FFVar> const* ctr, FFVar const* lkh,
      std::vector<std::vector<double>> const* vpar, std::vector<double> const* wpar,
      std::vector<double> const* vcst )
    {
#ifdef MAGNUS__FFNSAMP_CHECK
      assert( idep < 5 );
#endif
      set( dag, con, par, cst, ctr, lkh, vpar, wpar, vcst );
      return *(insert_external_operation( *this, 5, _nu, con->data() )[idep]);
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );

      throw std::runtime_error( "FFNSamp::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
    }

  void eval
    ( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    {
      return "Likelihood";
    }
    
  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }
};

inline double FFNSamp::ConfCTR = 0.1;
inline double FFNSamp::ConfLKH = 0.1;

inline void
FFNSamp::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFNSAMP_TRACE
  std::cout << "FFNSamp::eval: FFVar\n"; 
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( size_t j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
}

inline void
FFNSamp::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFNSAMP_TRACE
  std::cout << "FFNSamp::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}

inline void
FFNSamp::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFNSAMP_TRACE
  std::cout << "FFNSamp::eval: double\n"; 
#endif
#ifdef MC__FFNSAMP_CHECK
  assert( nRes == 5 && nVar == _nu );
#endif
  if( _nc && ( !_DCST || _DCST->size() < _nc ) ) 
    throw std::runtime_error( "FFNSamp::eval ** Constant values missing for constraint evaluation" );
  if( _np && ( !_DPAR || !_WPAR || _DPAR->at(0).size() < _np || _WPAR->size() < _ns ) ) 
    throw std::runtime_error( "FFNSamp::eval ** Parameter values missing for constraint evaluation" );

  // Get constraints for each scenario and controls vVar 
  _DFCT.assign( _ns?_ns:1, std::vector<double>( _FFCT.size(), 0. ) );
  _DCON.assign( vVar, vVar+_nu );
#ifdef MC__FFNSAMP_DEBUG
  std::cout << "CON = " << arma::vec( _DCON.data(), _nu, false ).t();
#endif
  if( !_np ){
    if( !_nc ) _DAG->eval( _sgFCT, _wkD, _FFCT, _DFCT[0], _FCON, _DCON );
    else       _DAG->eval( _sgFCT, _wkD, _FFCT, _DFCT[0], _FCON, _DCON, _FCST, *_DCST );
  }
  else if( _ns == 1 ){
    if( !_nc ) _DAG->eval( _sgFCT, _wkD, _FFCT, _DFCT[0], _FPAR, _DPAR->at(0), _FCON, _DCON );
    else       _DAG->eval( _sgFCT, _wkD, _FFCT, _DFCT[0], _FPAR, _DPAR->at(0), _FCON, _DCON, _FCST, *_DCST );
  }
  else{
    //std::cout << "using veval in FFNSamp\n";
    if( !_nc ) _DAG->veval( _sgFCT, _wkD, _wkThd, _FFCT, _DFCT, _FPAR, *_DPAR, _FCON, _DCON );
    else       _DAG->veval( _sgFCT, _wkD, _wkThd, _FFCT, _DFCT, _FPAR, *_DPAR, _FCON, _DCON, _FCST, *_DCST );
  }
#ifdef MC__FFNSAMP_DEBUG
  for( size_t s=0; s<(_ns?_ns:1); ++s )
    std::cout << "FCT[" << s << "] = " << arma::rowvec( _DFCT[s].data(), _DFCT[s].size(), false );
#endif

  // Calculate Bayes risk-based criterion
  for( size_t i=0; i<nRes; ++i ) vRes[i] = 0;
  if( _ng ) _feasval( vRes, _DFCT );
  if( _nl ) _lkhdval( vRes, _DFCT );
#ifdef MC__FFNSAMP_DEBUG
  std::cout << name() << " = " << arma::rowvec( vRes, nRes, false );
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFNSamp::_feasval
( double* vRes, std::vector<std::vector<double>>& DFCT )
const
{
#ifdef MC__FFNSAMP_TRACE
  std::cout << "FFNSamp::_feasval\n";
#endif

  // Return maximal constraint violation if unique scenario
  if( _ns <= 1 ){
    auto it1 = DFCT[0].cbegin(), it2 = it1;
    std::advance( it2, _ng );
    auto itMax = std::max_element( it1, it2 );
    vRes[0] = *itMax<=0? 1.: 0.;
    vRes[1] = vRes[2] = *itMax;
    return;
  }

  // Order scenarios in order of maximal constraint violation
  std::multimap<double,double> ResCTR;
  for( size_t s=0; s<_ns; ++s ){
    auto it1 = DFCT[s].cbegin(), it2 = it1;
    std::advance( it2, _ng );
    auto itMax = std::max_element( it1, it2 );
    ResCTR.insert( { -*itMax, _WPAR->at(s) } ); // ordered by largest (negative) violation first 
  }

  // Feasibility probability
  vRes[0] = 1.;
  for( auto const& [Res,Pr] : ResCTR ){
    if( Res >= 0. ) break;
    vRes[0] -= Pr;
  }

  // Value-at-risk
  double PrMass = 0., VaR = 0.;
  for( auto const& [Res,Pr] : ResCTR ){
    VaR = Res;
    if( PrMass + Pr > ConfCTR ) break;
    PrMass += Pr;
  }
  vRes[1] = vRes[2] = -VaR;

  // Conditional-value-at-risk
  for( auto const& [Res,Pr] : ResCTR ){
    if( Res > VaR ) break;
    vRes[2] += ( VaR - Res ) * Pr / ConfCTR;
  }

#ifdef MC__FFNSAMP_DEBUG
  std::cout << "  Feas Pr = "   << std::fixed << std::setprecision(1) << std::setw(5) << vRes[0]*1e2
            << "  Feas VaR = "  << std::scientific << std::setprecision(4) << std::setw(11) << vRes[1]
            << "  Feas CVaR = " << std::setw(11) << vRes[2] << std::endl;
#endif
}

inline void
FFNSamp::_lkhdval
( double* vRes, std::vector<std::vector<double>>& DFCT )
const
{
#ifdef MC__FFNSAMP_TRACE
  std::cout << "FFNSamp::_lkhdval\n";
#endif

  // Return likelihood if unique scenario
  if( _ns <= 1 ){
    vRes[3] = vRes[4] = DFCT[0][_ng];
    return;
  }

  // Order scenarios in order of likelihood value
  std::multimap<double,double> ResLKH;
  for( size_t s=0; s<_ns; ++s ){
    ResLKH.insert( { DFCT[s][_ng], _WPAR->at(s) } ); // ordered by smallest likelihood first 
  }

  // Value-at-risk
  double PrMass = 0., VaR = 0.;
  for( auto const& [Res,Pr] : ResLKH ){
    VaR = Res;
    if( PrMass + Pr > ConfLKH ) break;
    PrMass += Pr;
  }
  vRes[3] = vRes[4] = VaR;

  // Conditional-value-at-risk
  for( auto const& [Res,Pr] : ResLKH ){
    if( Res > VaR ) break;
    vRes[4] -= ( VaR - Res ) * Pr / ConfLKH;
  }

#ifdef MC__FFNSAMP_DEBUG
  std::cout << std::scientific << std::setprecision(4)
            << "  Lkhd VaR = "  << std::setw(11) << vRes[3]
            << "  Lkhd CVaR = " << std::setw(11) << vRes[4] << std::endl;
#endif
}

} // end namespace mc

#endif

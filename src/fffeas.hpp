// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef MAGNUS__FFFEAS_HPP
#define MAGNUS__FFFEAS_HPP

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

class FFFeas
: public FFOp
{
private:

  // DAG of constraints
  mutable FFGraph* _DAG;
  // parameters
  std::vector<FFVar> const* _FPAR;
  // constants
  std::vector<FFVar> const* _FCST;
  // controls
  std::vector<FFVar> const* _FCON;
  // constraints
  std::vector<FFVar> const* _FCTR;
  
  // parameter scenarios
  std::vector<std::vector<double>> const* _DPAR;
  // parameter scenario weights
  std::vector<double> const* _WPAR;
  // constant values: _nc
  std::vector<double> const* _DCST;

  // Number of parameters
  size_t _np;
  // Number of constants
  size_t _nc;
  // Number of controls
  size_t _nu;
  // Number of constraints
  size_t _ng;
  // Number of scenarios
  size_t _ns;

  // control values
  mutable std::vector<double> _DCON;
  // constraint values
  mutable std::vector<std::vector<double>> _DCTR; // _ns x _ng

  // Subgraph
  mutable FFSubgraph _sgCTR;
  // Work storage
  mutable std::vector<double> _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<double>> _wkThd;

  // Evaluation of feasibility criterion from constraint values
  void _feasval
    ( double* CRIT, std::vector<std::vector<double>>& DCTR )
    const;
    
public:

  // Feasibility confidence threshold
  static double Confidence;

  // Set variables, parameters and scenarios
  void set
    ( FFGraph* dag, std::vector<FFVar> const* con, std::vector<FFVar> const* par, 
      std::vector<FFVar> const* cst, std::vector<FFVar> const* ctr,
      std::vector<std::vector<double>> const* vpar, std::vector<double> const* wpar,
      std::vector<double> const* vcst )
    {
#ifdef MAGNUS__FFFEAS_CHECK
      assert( dag && con->size() && ctr->size() );
#endif
      _DAG  = dag;
      _FPAR = par;
      _FCST = cst;
      _FCON = con;
      _FCTR = ctr;

      _np = _FPAR? _FPAR->size(): 0;
      _nc = _FCST? _FCST->size(): 0;
      _nu = _FCON->size();
      _ng = _FCTR->size();
      
#ifdef MAGNUS__FFFEAS_CHECK
      assert( !vcst || vcst->size() == _nc );
#endif
      _DCST = vcst;

#ifdef MAGNUS__FFFEAS_CHECK
      assert( !vpar || vpar->at(0).size() == _nc );
#endif
      _DPAR = vpar;
      _WPAR = wpar;
      _ns = _DPAR? _DPAR->size(): 1;
    }

  // Default constructor
  FFFeas
    ( double const& Conf=0.90 )
    : FFOp( EXTERN )
    { Confidence = Conf; }
    
  // Copy constructor
  FFFeas
    ( FFFeas const& Op )
    : FFOp   ( Op ),
      _DAG   ( Op._DAG ),
      _FPAR  ( Op._FPAR ),
      _FCST  ( Op._FCST ),
      _FCON  ( Op._FCON ),
      _FCTR  ( Op._FCTR ),
      _DPAR  ( Op._DPAR ),
      _WPAR  ( Op._WPAR ),
      _DCST  ( Op._DCST ),
      _np    ( Op._np ),
      _nc    ( Op._nc ),
      _nu    ( Op._nu ),
      _ng    ( Op._ng ),
      _ns    ( Op._ns )
    {}

  // Define operation
  // Return values: [0] feasibility probability; [1] value-at-risk; [2] conditional-value-at-risk
  // VaR and CVaR are in reference to confidence threshold in static public member mc::FFFeas::Confidence
  FFVar** operator()
    ( FFGraph* dag, std::vector<FFVar> const* con, std::vector<FFVar> const* par,
      std::vector<FFVar> const* cst, std::vector<FFVar> const* ctr,
      std::vector<std::vector<double>> const* vpar, std::vector<double> const* wpar,
      std::vector<double> const* vcst )

    {
      set( dag, con, par, cst, ctr, vpar, wpar, vcst );
      return insert_external_operation( *this, 3, _nu, con->data() );
    }

  FFVar& operator()
    ( unsigned const idep, FFGraph* dag, std::vector<FFVar> const* con, std::vector<FFVar> const* par,
      std::vector<FFVar> const* cst, std::vector<FFVar> const* ctr,
      std::vector<std::vector<double>> const* vpar, std::vector<double> const* wpar,
      std::vector<double> const* vcst )
    {
#ifdef MAGNUS__FFFEAS_CHECK
      assert( idep < 3 );
#endif
      set( dag, con, par, cst, ctr, vpar, wpar, vcst );
      return *(insert_external_operation( *this, 3, _nu, con->data() )[idep]);
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

      throw std::runtime_error( "FFFeas::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
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

inline double FFFeas::Confidence = 0.9;

inline void
FFFeas::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFEAS_TRACE
  std::cout << "FFFeas::eval: FFVar\n"; 
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( size_t j=0; j<nRes; ++j ) vRes[j] = *(ppRes[j]);
}

inline void
FFFeas::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFEAS_TRACE
  std::cout << "FFFeas::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}

inline void
FFFeas::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFFEAS_TRACE
  std::cout << "FFFeas::eval: double\n"; 
#endif
#ifdef MC__FFFEAS_CHECK
  assert( nRes == 3 && nVar == _nu );
#endif
  if( _nc && ( !_DCST || _DCST->size() < _nc ) ) 
    throw std::runtime_error( "FFFeas::eval ** Constant values missing for constraint evaluation" );
  if( _np && ( !_DPAR || !_WPAR || _DPAR->at(0).size() < _np || _WPAR->size() < _ns ) ) 
    throw std::runtime_error( "FFFeas::eval ** Parameter values missing for constraint evaluation" );

  // Get constraints for each scenario and controls vVar 
  _DCTR.assign( _ns?_ns:1, std::vector<double>( _ng, 0. ) );
  _DCON.assign( vVar, vVar+_nu );
#ifdef MC__FFFEAS_DEBUG
  std::cout << "CON = " << arma::vec( _DCON.data(), _nu, false ).t();
#endif
  if( !_np ){
    if( !_nc ) _DAG->eval( _sgCTR, _wkD, *_FCTR, _DCTR[0], *_FCON, _DCON );
    else       _DAG->eval( _sgCTR, _wkD, *_FCTR, _DCTR[0], *_FCON, _DCON, *_FCST, *_DCST );
  }
  else if( _ns == 1 ){
    if( !_nc ) _DAG->eval( _sgCTR, _wkD, *_FCTR, _DCTR[0], *_FPAR, _DPAR->at(0), *_FCON, _DCON );
    else       _DAG->eval( _sgCTR, _wkD, *_FCTR, _DCTR[0], *_FPAR, _DPAR->at(0), *_FCON, _DCON, *_FCST, *_DCST );
  }
  else{
    if( !_nc ) _DAG->veval( _sgCTR, _wkD, _wkThd, *_FCTR, _DCTR, *_FPAR, *_DPAR, *_FCON, _DCON );
    else       _DAG->veval( _sgCTR, _wkD, _wkThd, *_FCTR, _DCTR, *_FPAR, *_DPAR, *_FCON, _DCON, *_FCST, *_DCST );
  }
#ifdef MC__FFFEAS_DEBUG
  for( size_t s=0; s<(_ns?_ns:1); ++s )
    std::cout << "CTR[" << s << "] = " << arma::vec( _DCTR[s].data(), _ng, false ).t();
#endif

  // Calculate Bayes risk-based criterion
  _feasval( vRes, _DCTR );
#ifdef MC__FFBRISKCRIT_DEBUG
  std::cout << name() << " = " << vRes[0] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFFeas::_feasval
( double* vRes, std::vector<std::vector<double>>& DCTR )
const
{
#ifdef MC__FFFEAS_TRACE
  std::cout << "FFFeas::_feasval\n";
#endif

  // Return maximal constraint violation if unique scenario
  if( _ns <= 1 ){
    auto itMax = std::max_element( DCTR[0].cbegin(), DCTR[0].cend() );
    vRes[0] = *itMax<=0? 1.: 0.;
    vRes[1] = vRes[2] = *itMax;
    return;
  }

  // Order scenarios in order of maximal constraint violation
  std::multimap<double,double> ResCTR;
  for( size_t s=0; s<_ns; ++s ){
    auto itMax = std::max_element( DCTR[s].cbegin(), DCTR[s].cend() );
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
    if( PrMass + Pr > Confidence ) break;
    PrMass += Pr;
  }
  vRes[1] = vRes[2] = -VaR;

  // Conditional-value-at-risk
  for( auto const& [Res,Pr] : ResCTR ){
    if( Res > VaR ) break;
    vRes[2] += ( VaR - Res ) * Pr / Confidence;
  }

#ifdef MC__FFFEAS_DEBUG
  std::cout << "  Pr = "   << std::fixed << std::setprecision(1) << std::setw(5) << vRes[0]*1e2
            << "  VaR = "  << std::scientific << std::setprecision(4) << std::setw(11) << vRes[1]
            << "  CVaR = " << std::setw(11) << vRes[2] << std::endl;
#endif
}

} // end namespace mc

#endif

// Copyright (C) Benoit Chachuat, Imperial College London.
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef CANON__FFEST_HPP
#define CANON__FFEST_HPP

#include <fstream>
#include <iomanip>
#include <armadillo>

#include "base_parest.hpp"

#define MC__FFBRCRIT_LOG
#undef  MC__FFDCRIT_EIG
#define MC__FFFIMCrit_CHECK
#undef  MC__FFODISTEFF_USEGRAD

namespace mc
{

////////////////////////////////////////////////////////////////////////
// EXTERNAL OPERATIONS
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////

class FFBaseMLE
: public FFOp
{
protected:

  // Number of parameters
  size_t _np;
  // Number of constants
  size_t _nc;
  // Number of controls
  size_t _nu;
  // Number of outputs
  size_t _ny;
  // Number of experiments
  size_t _ne;

  // DAG of outputs
  mutable FFGraph*     _DAG;
  // parameters
  std::vector<FFVar>   _PAR;
  // constants
  std::vector<FFVar>   _CST;
  // controls
  std::vector<FFVar>   _CON;
  // outputs
  std::vector<FFVar>   _OUT;

  // data
  std::vector<BASE_PAREST::Experiment> const*  _DAT;
  // constant values: _nc
  std::vector<double> const*                   _DCST;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const& par, std::vector<FFVar> const& cst,
      std::vector<FFVar> const& con, std::vector<FFVar> const& out, std::vector<double> const* cstval,
      std::vector<BASE_PAREST::Experiment> const* dat )
    {
      _np = par.size();
      _nc = cst.size();
      _nu = con.size();
      _ny = out.size();
      _ne = dat->size();
#ifdef MC__FFMLE_CHECK
      assert( dag && _np && _ny && _ne );
#endif
 
      // Deep copy of DAG variable for safe multi-threading
      delete _DAG;
      _DAG = new FFGraph;
      _DAG->options = dag->options;

      _PAR.resize( _np );
      _DAG->insert( dag, _np, par.data(), _PAR.data() );
      _CST.resize( _nc );
      _DAG->insert( dag, _nc, cst.data(), _CST.data() );
      _CON.resize( _nu );
      if( _nu ) _DAG->insert( dag, _nu, con.data(), _CON.data() );
      _OUT.resize( _ny );
      _DAG->insert( dag, _ny, out.data(), _OUT.data() );

      _DAT    = dat;
      _DCST   = cstval;
    }

  // Default constructor
  FFBaseMLE
    ()
    : FFOp( EXTERN ),
      _DAG( nullptr )
    {}

  // Destructor
  virtual ~FFBaseMLE
    ()
    {
      delete _DAG;
    }
    
  // Copy constructor
  FFBaseMLE
    ( FFBaseMLE const& Op )
    : FFOp( Op ),
      _DAG( nullptr )
    {
      set( Op._DAG, Op._PAR, Op._CST, Op._CON, Op._OUT, Op._DCST, Op._DAT );
    }
};

class FFMLE
: public FFBaseMLE
{
private:

  // control values: _ne x _nu
  std::vector<std::vector<double>>             _DCON;
  // parameter values: _np
  mutable std::vector<double>                  _DPAR;
  // output values: _ne x _ny
  mutable std::vector<std::vector<double>>     _DOUT;

  // Subgraph
  mutable FFSubgraph                           _sgOUT;
  // Work storage
  mutable std::vector<double>                  _wkD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<double>> _wkThd;

  // Evaluation of MLE criterion from output values
  void _MLEval
    ( double& MLE )
    const;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const& par, std::vector<FFVar> const& cst,
      std::vector<FFVar> const& con, std::vector<FFVar> const& out, std::vector<double> const* cstval,
      std::vector<BASE_PAREST::Experiment> const* dat )
    {
      FFBaseMLE::set( dag, par, cst, con, out, cstval, dat );
      
      _DPAR.reserve( _np );
      _DCON.clear();
      _DCON.reserve( _ne );
      for( auto const& EXP : *_DAT )
        _DCON.push_back( EXP.control );
    }

  // Default constructor
  FFMLE
    ()
    : FFBaseMLE()
    {}

  // Destructor
  virtual ~FFMLE
    ()
    {}
    
  // Copy constructor
  FFMLE
    ( FFMLE const& Op )
    : FFBaseMLE( Op ),
      _DCON( Op._DCON ),
      _DPAR( Op._DPAR )
    {}

  // Define operation
  FFVar& operator()
    ( FFVar const* var, FFGraph* dag, std::vector<FFVar> const& par, std::vector<FFVar> const& cst,
      std::vector<FFVar> const& con, std::vector<FFVar> const& out, std::vector<double> const* cstval,
      std::vector<BASE_PAREST::Experiment> const* dat )
    {
      set( dag, par, cst, con, out, cstval, dat );
      return **insert_external_operation( *this, 1, _np, var );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<FFVar> ) )
        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
      else if( idU == typeid( fadbad::F<double> ) )
        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
      else if( idU == typeid( SLiftVar ) )
        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
      else if( idU == typeid( FFExpr ) )
        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFBRCrit::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
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

  void eval
    ( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
      unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
      unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, SLiftVar* vRes, unsigned const nVar, SLiftVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFExpr* vRes, unsigned const nVar, FFExpr const* vVar, unsigned const* mVar )
    const;

  void deriv
    ( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
    const;

  // Properties
  std::string name
    ()
    const
    { 
      std::ostringstream odat; odat << _DAT;
      return "MLE[" + odat.str() + "]";
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }

  // Ordering
  bool lt
    ( FFOp const* op )
    const
    {
#ifdef MC__FFMLE_TRACE
      std::cout << "FFMLE::lt\n";
#endif
      return( _DAT < dynamic_cast<FFMLE const*>(op)->_DAT );
    }

  std::vector<std::vector<double>> const& dOUT
    ()
    const
    {
      return _DOUT;
    }
};

class FFGradMLE
: public FFBaseMLE
{
private:

  // control fadbad::F values: _ne x _nu
  std::vector<std::vector<fadbad::F<double>>>             _FDCON;
  // parameter values: _np
  mutable std::vector<fadbad::F<double>>                  _FDPAR;
  // output values: _ne x _ny
  mutable std::vector<std::vector<fadbad::F<double>>>     _FDOUT;
  // constant values: _nc
  std::vector<fadbad::F<double>>                          _FDCST;

  // Subgraph
  mutable FFSubgraph                                      _sgOUT;
  // Work storage
  mutable std::vector<fadbad::F<double>>                  _wkFD;
  // Thread storage
  mutable std::vector<FFGraph::Worker<fadbad::F<double>>> _wkFThd;

  // Evaluation of MLE derivatives from output values
  void _MLEder
    ( double* gradMLE )
    const;

public:

  void set
    ( FFGraph* dag, std::vector<FFVar> const& par, std::vector<FFVar> const& cst,
      std::vector<FFVar> const& con, std::vector<FFVar> const& out, std::vector<double> const* cstval,
      std::vector<BASE_PAREST::Experiment> const* dat )
    {
      FFBaseMLE::set( dag, par, cst, con, out, cstval, dat );
      
      _FDPAR.reserve( _np );
      _FDPAR.resize( _np );
      for( size_t p=0; p<_np; ++p ){
        _FDPAR[p].diff( p, _np );
#ifdef MC__FFGRADMLE_DEBUG
        std::cout << "_FDPAR[" << p << "].diff(" << p << "," << _np << ")\n";
#endif
      }

      _FDCON.clear();
      _FDCON.resize( _ne );
      auto itFDCON = _FDCON.begin();
      for( auto const& EXP : *_DAT ){
        itFDCON->assign( EXP.control.cbegin(), EXP.control.cend() );
#ifdef MC__FFGRADMLE_DEBUG
        for( size_t c=0; c<_nu; ++c )
          std::cout << "_FDCON[" << c << "] =" << (*itFDCON)[c].val() << std::endl;
#endif
        ++itFDCON;
      }

      _FDCST.clear();
      _FDCST.resize( _nc );
      for( size_t c=0; c<_nc; ++c )
        _FDCST[c] = cstval->at(c);
    }

  // Default constructor
  FFGradMLE
    ()
    : FFBaseMLE()
    {}
    
  // Copy constructor
  FFGradMLE
    ( FFGradMLE const& Op )
    : FFBaseMLE( Op ),
      _FDCON( Op._FDCON ),
      _FDPAR( Op._FDPAR ),
      _FDCST( Op._FDCST )
    {}

  // Define operation
  FFVar& operator()
    ( size_t const idep, FFVar const* var, FFGraph* dag, std::vector<FFVar> const& par, std::vector<FFVar> const& cst,
      std::vector<FFVar> const& con, std::vector<FFVar> const& out, std::vector<double> const* cstval,
      std::vector<BASE_PAREST::Experiment> const* dat )
    {
#ifdef MC__FFGRADMLE_CHECK
      assert( idep < _np );
#endif
      set( dag, par, cst, con, out, cstval, dat );
      return *(insert_external_operation( *this, _np, _np, var )[idep]);
    }

  FFVar** operator()
    ( FFVar const* var, FFGraph* dag, std::vector<FFVar> const& par, std::vector<FFVar> const& cst,
      std::vector<FFVar> const& con, std::vector<FFVar> const& out, std::vector<double> const* cstval,
      std::vector<BASE_PAREST::Experiment> const* dat )
    {
      set( dag, par, cst, con, out, cstval, dat );
      return insert_external_operation( *this, _np, _np, var );
    }

  // Evaluation overloads
  virtual void feval
    ( std::type_info const& idU, unsigned const nRes, void* vRes, unsigned const nVar,
      void const* vVar, unsigned const* mVar )
    const
    {
      if( idU == typeid( FFVar ) )
        return eval( nRes, static_cast<FFVar*>(vRes), nVar, static_cast<FFVar const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<FFVar> ) )
//        return eval( nRes, static_cast<fadbad::F<FFVar>*>(vRes), nVar, static_cast<fadbad::F<FFVar> const*>(vVar), mVar );
      else if( idU == typeid( FFDep ) )
        return eval( nRes, static_cast<FFDep*>(vRes), nVar, static_cast<FFDep const*>(vVar), mVar );
      else if( idU == typeid( double ) )
        return eval( nRes, static_cast<double*>(vRes), nVar, static_cast<double const*>(vVar), mVar );
//      else if( idU == typeid( fadbad::F<double> ) )
//        return eval( nRes, static_cast<fadbad::F<double>*>(vRes), nVar, static_cast<fadbad::F<double> const*>(vVar), mVar );
      else if( idU == typeid( SLiftVar ) )
        return eval( nRes, static_cast<SLiftVar*>(vRes), nVar, static_cast<SLiftVar const*>(vVar), mVar );
      else if( idU == typeid( FFExpr ) )
        return eval( nRes, static_cast<FFExpr*>(vRes), nVar, static_cast<FFExpr const*>(vVar), mVar );

      throw std::runtime_error( "FFGradMLE::feval ** No evaluation method for type"+std::string(idU.name())+"\n" );
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

  void eval
    ( unsigned const nRes, SLiftVar* vRes, unsigned const nVar, SLiftVar const* vVar, unsigned const* mVar )
    const;

  void eval
    ( unsigned const nRes, FFExpr* vRes, unsigned const nVar, FFExpr const* vVar, unsigned const* mVar )
    const;

  // Properties
  std::string name
    ()
    const
    { 
      std::ostringstream odat; odat << _DAT;
      return "Grad MLE[" + odat.str() + "]";
    }

  //! @brief Return whether or not operation is commutative
  bool commutative
    ()
    const
    { return false; }

  // Ordering
  bool lt
    ( FFOp const* op )
    const
    {
#ifdef MC__FFGRADMLE_TRACE
      std::cout << "FFGradMLE::lt\n";
#endif
      return( _DAT < dynamic_cast<FFGradMLE const*>(op)->_DAT );
    }
};

inline void
FFMLE::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::eval: FFVar\n"; 
#endif

  vRes[0] = **insert_external_operation( *this, nRes, nVar, vVar );
}

inline void
FFMLE::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
}

inline void
FFMLE::eval
( unsigned const nRes, FFExpr* vRes, unsigned const nVar, FFExpr const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::eval: FFExpr\n";
#endif
#ifdef MC__FFMLE_CHECK
  assert( nRes == 1 && nVar = _np );
#endif

  switch( FFExpr::options.LANG ){
   case FFExpr::Options::DAG:
    for( unsigned j=0; j<nRes; ++j ){
      std::ostringstream os; os << name() << "[" << j << "]";
      vRes[j] = FFExpr::compose( os.str(), nVar, vVar );
    }
    break;
   case FFExpr::Options::GAMS:
   default:
    throw typename FFExpr::Exceptions( FFExpr::Exceptions::UNDEF );
  }
}

inline void
FFMLE::eval
( unsigned const nRes, SLiftVar* vRes, unsigned const nVar, SLiftVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::eval: SLiftVar\n";
#endif
#ifdef MC__FFMLE_CHECK
  assert( nRes == 1 && nVar == _np );
#endif

  vVar->env()->lift( nRes, vRes, nVar, vVar );
}

inline void
FFGradMLE::eval
( unsigned const nRes, FFVar* vRes, unsigned const nVar, FFVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADMLE_TRACE
  std::cout << "FFGradMLE::eval: FFVar\n"; 
#endif

  FFVar** ppRes = insert_external_operation( *this, nRes, nVar, vVar );
  for( size_t j=0; j<nRes; ++j )
    vRes[j] = *(ppRes[j]);
}

inline void
FFGradMLE::eval
( unsigned const nRes, FFDep* vRes, unsigned const nVar, FFDep const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADMLE_TRACE
  std::cout << "FFGradMLE::eval: FFDep\n"; 
#endif

  vRes[0] = 0;
  for( size_t i=0; i<nVar; ++i ) vRes[0] += vVar[i];
  vRes[0].update( FFDep::TYPE::N );
  for( size_t j=1; j<nRes; ++j ) vRes[j] = vRes[0];
}

inline void
FFGradMLE::eval
( unsigned const nRes, FFExpr* vRes, unsigned const nVar, FFExpr const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADMLE_TRACE
  std::cout << "FFGradMLE::eval: FFExpr\n";
#endif
#ifdef MC__FFGRADMLE_CHECK
  assert( nRes == nVar && nVar = _np );
#endif

  switch( FFExpr::options.LANG ){
   case FFExpr::Options::DAG:
    for( unsigned j=0; j<nRes; ++j ){
      std::ostringstream os; os << name() << "[" << j << "]";
      vRes[j] = FFExpr::compose( os.str(), nVar, vVar );
    }
    break;
   case FFExpr::Options::GAMS:
   default:
    throw typename FFExpr::Exceptions( FFExpr::Exceptions::UNDEF );
  }
}

inline void
FFGradMLE::eval
( unsigned const nRes, SLiftVar* vRes, unsigned const nVar, SLiftVar const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADMLE_TRACE
  std::cout << "FFGradMLE::eval: SLiftVar\n";
#endif
#ifdef MC__FFGRADMLE_CHECK
  assert( nRes == nVar && nVar = _np );
#endif

  vVar->env()->lift( nRes, vRes, nVar, vVar );
}

inline void
FFMLE::_MLEval
( double& MLE )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::_MLEval\n";
#endif
#ifdef MC__FFMLE_CHECK
  assert( _DAT && !_DAT->empty() && _DOUT.size() == _ne && _DOUT.front().size() == _ny );
#endif

  MLE = 0.;
  size_t e = 0;
  for( auto const& EXP : *_DAT ){
    for( auto const& [ k, RECk ] : EXP.output )
      for( auto const& YMk : RECk.measurement ){
        MLE += sqr( YMk - _DOUT[e][k] ) / RECk.variance;
#ifdef MC__FFBRCRIT_DEBUG
        std::cout << "Exp " << e << " Rec " << k << ": " << YMk << " " << _DOUT[e][k] << std::endl;
#endif
      }
    ++e;
  }
  MLE *= 5e-1;

#ifdef MC__FFBRCRIT_DEBUG
  std::cout << name() << MLE << std::endl;
  //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFMLE::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::eval: double\n"; 
#endif
#ifdef MC__FFMLE_CHECK
  assert( nRes == 1 && nVar == _np );
#endif

  // Get outputs for each experiment
  _DOUT.assign( _ne, std::vector<double>( _ny, 0. ) );
  _DPAR.assign( vVar, vVar+_np );
#ifdef MC__FFMLE_DEBUG
  std::cout << "p = " << arma::vec( _DPAR.data(), _np, false );
#endif

  if( _CST.empty() )
    _DAG->veval( _sgOUT, _wkD, _wkThd, _OUT, _DOUT, _CON, _DCON, _PAR, _DPAR );
  else
    _DAG->veval( _sgOUT, _wkD, _wkThd, _OUT, _DOUT, _CON, _DCON, _PAR, _DPAR, _CST, *_DCST );
#ifdef MC__FFMLE_DEBUG
  for( size_t e=0; e<_ne; ++e )
    std::cout << "y[" << e << "] = " << arma::vec( _DOUT[e].data(), _ny, false );
#endif

  // Calculate MLE criterion
  _MLEval( vRes[0] );
#ifdef MC__FFMLE_DEBUG
  std::cout << name() << " = " << vRes[0] << std::endl;
    //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFGradMLE::_MLEder
( double* gradMLE )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFGradMLE::_MLEder\n";
#endif
#ifdef MC__FFMLE_CHECK
  assert( _DAT && !_DAT->empty() && _FDOUT.size() == _ne && _FDOUT.front().size() == _ny );
#endif

  arma::vec dMLE( gradMLE, _np, false );
  dMLE.zeros();
  size_t e = 0;
  for( auto const& EXP : *_DAT ){
    for( auto const& [ k, RECk ] : EXP.output )
      for( auto const& YMk : RECk.measurement ){
        arma::vec dOUTek( &_FDOUT[e][k].d(0), _np, false );
        dMLE -= ( YMk - _FDOUT[e][k].x() ) * dOUTek / RECk.variance;
      }
    ++e;
  }
}

inline void
FFGradMLE::eval
( unsigned const nRes, double* vRes, unsigned const nVar, double const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFGRADMLE_TRACE
  std::cout << "FFGradMLE::eval: double\n"; 
#endif
#ifdef MC__FFGRADMLE_CHECK
  assert( nRes == nVar && nVar = _np );
#endif

  // Get output derivatives for each experiment
  _FDOUT.assign( _ne, std::vector<fadbad::F<double>>( _ny, 0. ) );
  for( size_t p=0; p<_np; ++p )
    _FDPAR[p].x() = vVar[p]; // does not change differential variables

  if( _CST.empty() )
    _DAG->veval( _sgOUT, _wkFD, _wkFThd, _OUT, _FDOUT, _CON, _FDCON, _PAR, _FDPAR );
  else
    _DAG->veval( _sgOUT, _wkFD, _wkFThd, _OUT, _FDOUT, _CON, _FDCON, _PAR, _FDPAR, _CST, _FDCST );
#ifdef MC__FFGRADMLE_DEBUG
  for( size_t e=0; e<_ne; ++e ){
    for( size_t k=0; k<_ny; ++k ){
      std::cout << "_FDOUT[" << e << "][" << k << "] =";
      for( size_t i=0; i<_FDOUT[e][k].size(); ++i )
        std::cout << "  " << _FDOUT[e][k].deriv(i);
      std::cout << std::endl;
    }
  }
  //{ int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif

  // Calculate MLE criterion derivatives
  _MLEder( vRes );
#ifdef MC__FFGRADMLE_DEBUG
  for( size_t i=0; i<nVar; ++i )
    std::cout << name() << "[" << i << "] = " << vRes[i] << std::endl;
  { int dum; std::cout << "Press 1"; std::cin >> dum; }
#endif
}

inline void
FFMLE::eval
( unsigned const nRes, fadbad::F<FFVar>* vRes, unsigned const nVar, fadbad::F<FFVar> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::eval: fadbad::F<FFVar>\n"; 
#endif
#ifdef MC__FFMLE_CHECK
  assert( nRes == 1 );
#endif

  std::vector<FFVar> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  FFVar ResVal = **insert_external_operation( *this, 1, nVar, vVarVal.data() );

  FFGradMLE OpResDer;
  OpResDer.set( _DAG, _PAR, _CST, _CON, _OUT, _DCST, _DAT );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVarVal.data() );
  vRes[0] = ResVal;
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );
  for( size_t j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( size_t i=0; i<nVar; ++i ){
      if( vVar[i][j].cst() && vVar[i][j].num().val() == 0. ) continue;
      vRes[0][j] += *ppResDer[i] * vVar[i][j];
    }
  }
}

inline void
FFMLE::eval
( unsigned const nRes, fadbad::F<double>* vRes, unsigned const nVar, fadbad::F<double> const* vVar,
  unsigned const* mVar )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::eval: fadbad::F<double>\n"; 
#endif
#ifdef MC__FFMLE_CHECK
  assert( nRes == 1 );
#endif

  std::vector<double> vVarVal( nVar );
  for( size_t i=0; i<nVar; ++i )
    vVarVal[i] = vVar[i].val();
  double ResVal(0.); 
  eval( 1, &ResVal, nVar, vVarVal.data(), nullptr );
  vRes[0] = ResVal;
  for( size_t i=0; i<nVar; ++i )
    vRes[0].setDepend( vVar[i] );
  
  FFGradMLE OpResDer;
  OpResDer.set( _DAG, _PAR, _CST, _CON, _OUT, _DCST, _DAT );
  std::vector<double> vResDer( nVar ); 
  OpResDer.eval( nVar, vResDer.data(), nVar, vVarVal.data(), nullptr );
  for( size_t j=0; j<vRes[0].size(); ++j ){
    vRes[0][j] = 0.;
    for( size_t i=0; i<nVar; ++i ){
      if( vVar[i][j] == 0. ) continue;
      vRes[0][j] += vResDer[i] * vVar[i][j];
    }
  }
}

inline void
FFMLE::deriv
( unsigned const nRes, FFVar const* vRes, unsigned const nVar, FFVar const* vVar, FFVar** vDer )
const
{
#ifdef MC__FFMLE_TRACE
  std::cout << "FFMLE::deriv:\n"; 
#endif
#ifdef MC__FFMLE_CHECK
  assert( nRes == 1 );
#endif

  FFGradMLE OpResDer;
  OpResDer.set( _DAG, _PAR, _CST, _CON, _OUT, _DCST, _DAT );
  FFVar const*const* ppResDer = insert_external_operation( OpResDer, nVar, nVar, vVar );
  for( size_t i=0; i<nVar; ++i )
    vDer[0][i] = *ppResDer[i];
}

} // end namespace mc

#endif

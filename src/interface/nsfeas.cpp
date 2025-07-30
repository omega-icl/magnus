#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "nsfeas.hpp"

namespace py = pybind11;

void mc_nsfeas( py::module_ &m )
{

typedef mc::FFGraph   FFGraph;
/*
typedef mc::BASE_MBFA BASE_MBFA;

py::class_<BASE_MBFA> pyMBFA( m, "BaseFeas", py::module_local() );

pyMBFA
 .def(
   py::init<>()
 )
 .def( 
   "set_dag",
   []( BASE_MBFA& self,  FFGraph& dag )
     { self.set_dag( dag ); },
   py::keep_alive<1,2>(),
   "set DAG"
 )
 .def_property_readonly(
   "dag",
   []( BASE_MBFA const& self )
     { return &self.dag(); },
   py::return_value_policy::reference_internal,
   "DAG"
 )
 .def_property_readonly(
   "ng",
   []( BASE_MBFA const& self )
     { return self.ng(); },
   "size of model constraints"
 )
 .def_property_readonly(
   "nc",
   []( BASE_MBFA const& self )
     { return self.nc(); },
   "size of model controls"
 )
 .def_property_readonly(
   "np",
   []( BASE_MBFA const& self )
     { return self.np(); },
   "size of model parameters"
 )
 .def(
   "set_parameter",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& p, std::vector<double> const& pnom )
     { self.set_parameter( p, pnom ); },
   py::arg("var"),
   py::arg("nom"),
   "set model parameters and nominal values"
 )
 .def(
   "set_parameter",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& p, std::list<std::vector<double>> const& psam )
     { self.set_parameter( p, psam ); },
   py::arg("var"),
   py::arg("sam"),
   "set model parameters and sampled values (equal weighting)"
 )
 .def(
   "set_parameter",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& p, std::list<std::pair<std::vector<double>,double>> const& psam )
     { self.set_parameter( p, psam ); },
   py::arg("var"),
   py::arg("sam"),
   "set model parameters and sampled values and weights"
 )
 .def_property_readonly(
   "var_parameter",
   []( BASE_MBFA const& self )
     { return self.var_parameter(); },
   py::return_value_policy::reference_internal,
   "model parameters"
 )
 .def(
   "set_control",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& u, std::vector<double> const& ulb,
       std::vector<double> const& uub )
     { self.set_control( u, ulb, uub ); },
   py::arg("var"),
   py::arg("lb")=std::vector<double>(),
   py::arg("ub")=std::vector<double>(),
   "set model controls and bounds"
 )
 .def_property_readonly(
   "var_control",
   []( BASE_MBFA const& self )
     { return self.var_control(); },
   py::return_value_policy::reference_internal,
   "model controls"
 )
 .def(
   "set_constraint",
   []( BASE_MBFA& self, mc::FFVar const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "set model-based constraint"
 )
 .def(
   "set_constraint",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "set model-based constraints"
 )
 .def(
   "add_constraint",
   []( BASE_MBFA& self, mc::FFVar const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "add model-based constraint"
 )
 .def(
   "add_constraint",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "add model-based constraints"
 )
 .def(
   "reset_constraint",
   []( BASE_MBFA& self )
     { self.reset_constraint(); },
   "reset model-based constraints"
 )
 .def_property_readonly(
   "var_constraint",
   []( BASE_MBFA const& self )
     { return self.var_constraint(); },
   py::return_value_policy::reference_internal,
   "model constraints"
 )
;

//typedef mc::FFGraph   FFGraph;
//typedef mc::BASE_MBFA BASE_MBFA;
typedef mc::NSFEAS    NSFEAS;

py::class_<NSFEAS, mc::BASE_MBFA> pyNSFEAS( m, "NSFeas" );
*/

typedef mc::NSFEAS    NSFEAS;

py::class_<NSFEAS> pyNSFEAS( m, "NSFeas" );

pyNSFEAS
 .def(
   py::init<>()
 )
 .def_static(
  "uniform_sampling",
   []( size_t nsam, std::vector<double> const& lb, std::vector<double> const& ub )
     { return NSFEAS::uniform_sample( nsam, lb, ub ); },
   "uniform sampling using Sobol sequences"
 )
 .def( 
   "set_dag",
   []( NSFEAS& self,  FFGraph& dag )
     { self.set_dag( dag ); },
   py::keep_alive<1,2>(),
   "set DAG"
 )
 .def_property_readonly(
   "dag",
   []( NSFEAS const& self )
     { return &self.dag(); },
   py::return_value_policy::reference_internal,
   "DAG"
 )
 .def_property_readonly(
   "ng",
   []( NSFEAS const& self )
     { return self.ng(); },
   "size of model constraints"
 )
 .def_property_readonly(
   "nc",
   []( NSFEAS const& self )
     { return self.nc(); },
   "size of model controls"
 )
 .def_property_readonly(
   "np",
   []( NSFEAS const& self )
     { return self.np(); },
   "size of model parameters"
 )
 .def(
   "set_parameter",
   []( NSFEAS& self, std::vector<mc::FFVar> const& p, std::vector<double> const& pnom )
     { self.set_parameter( p, pnom ); },
   py::arg("var"),
   py::arg("nom"),
   "set model parameters and nominal values"
 )
 .def(
   "set_parameter",
   []( NSFEAS& self, std::vector<mc::FFVar> const& p, std::list<std::vector<double>> const& psam )
     { self.set_parameter( p, psam ); },
   py::arg("var"),
   py::arg("sam"),
   "set model parameters and sampled values (equal weighting)"
 )
 .def(
   "set_parameter",
   []( NSFEAS& self, std::vector<mc::FFVar> const& p, std::list<std::pair<std::vector<double>,double>> const& psam )
     { self.set_parameter( p, psam ); },
   py::arg("var"),
   py::arg("sam"),
   "set model parameters and sampled values and weights"
 )
 .def_property_readonly(
   "var_parameter",
   []( NSFEAS const& self )
     { return self.var_parameter(); },
   py::return_value_policy::reference_internal,
   "model parameters"
 )
 .def(
   "set_control",
   []( NSFEAS& self, std::vector<mc::FFVar> const& u, std::vector<double> const& ulb,
       std::vector<double> const& uub )
     { self.set_control( u, ulb, uub ); },
   py::arg("var"),
   py::arg("lb")=std::vector<double>(),
   py::arg("ub")=std::vector<double>(),
   "set model controls and bounds"
 )
 .def_property_readonly(
   "var_control",
   []( NSFEAS const& self )
     { return self.var_control(); },
   py::return_value_policy::reference_internal,
   "model controls"
 )
 .def(
   "set_constraint",
   []( NSFEAS& self, mc::FFVar const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "set model-based constraint"
 )
 .def(
   "set_constraint",
   []( NSFEAS& self, std::vector<mc::FFVar> const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "set model-based constraints"
 )
 .def(
   "add_constraint",
   []( NSFEAS& self, mc::FFVar const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "add model-based constraint"
 )
 .def(
   "add_constraint",
   []( NSFEAS& self, std::vector<mc::FFVar> const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "add model-based constraints"
 )
 .def(
   "reset_constraint",
   []( NSFEAS& self )
     { self.reset_constraint(); },
   "reset model-based constraints"
 )
 .def_property_readonly(
   "var_constraint",
   []( NSFEAS const& self )
     { return self.var_constraint(); },
   py::return_value_policy::reference_internal,
   "model constraints"
 )
 .def_readwrite( 
   "options",
   &NSFEAS::options
 )
 .def(
   "setup",
   []( NSFEAS& self )
     { return self.setup(); },
   "setup feasibility problem before solution"
 )
 .def(
   "sample",
   []( NSFEAS& self, bool const reset ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     return self.sample( reset );
   },
   py::arg("reset") = true,
   "sample feasible domain"
 )
 .def_property_readonly(
   "live_points",
   []( NSFEAS const& self ){
     arma::mat samples( self.nc()+1, self.live_points().size() );
     size_t c=0;
     for( auto const& [lkh,pcon] : self.live_points() ){
       double* mem = samples.colptr( c++ );
       mem[0] = lkh;
       for( size_t i=0; i<self.nc(); ++i ) mem[1+i] = pcon[i]; 
     }
     //std::cout << "samples( " << samples.n_rows << "," << samples.n_cols << "):\n" << samples;
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{samples.n_rows, samples.n_cols};
     size_t strides[ndim]{elsize, samples.n_rows * elsize};
     return py::array_t<double>( shape, strides, samples.memptr() );
   },
   py::return_value_policy::reference_internal,
   "retrieve live points"
 )
 .def_property_readonly(
   "dead_points",
   []( NSFEAS const& self ){
     arma::mat samples( self.nc()+1, self.dead_points().size() );
     size_t c=0;
     for( auto const& [lkh,pcon] : self.dead_points() ){
       double* mem = samples.colptr( c++ );
       mem[0] = lkh;
       for( size_t i=0; i<self.nc(); ++i ) mem[1+i] = pcon[i]; 
     }
     //std::cout << "samples( " << samples.n_rows << "," << samples.n_cols << "):\n" << samples;
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{samples.n_rows, samples.n_cols};
     size_t strides[ndim]{elsize, samples.n_rows * elsize};
     return py::array_t<double>( shape, strides, samples.memptr() );
   },
   py::return_value_policy::reference_internal,
   "retrieve dead points"
 )
 .def_property_readonly(
   "discard_points",
   []( NSFEAS const& self ){
     arma::mat samples( self.nc()+1, self.discard_points().size() );
     size_t c=0;
     for( auto const& [lkh,pcon] : self.discard_points() ){
       double* mem = samples.colptr( c++ );
       mem[0] = lkh;
       for( size_t i=0; i<self.nc(); ++i ) mem[1+i] = pcon[i]; 
     }
     //std::cout << "samples( " << samples.n_rows << "," << samples.n_cols << "):\n" << samples;
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{samples.n_rows, samples.n_cols};
     size_t strides[ndim]{elsize, samples.n_rows * elsize};
     return py::array_t<double>( shape, strides, samples.memptr() );
   },
   py::return_value_policy::reference_internal,
   "retrieve discard points"
 )
;

py::enum_<mc::NSFEAS::STATUS>( pyNSFEAS, "STATUS" )
 .value( "NORMAL",     mc::NSFEAS::STATUS::NORMAL,     "Normal completion" )
 .value( "INTERRUPT",  mc::NSFEAS::STATUS::INTERRUPT,  "Resource limit reached" )
 .value( "FAILURE",    mc::NSFEAS::STATUS::FAILURE,    "Terminated after numerical failure" )
 .value( "ABORT",      mc::NSFEAS::STATUS::ABORT,      "Aborted after critical error" )
 .export_values()
;

py::class_<NSFEAS::Options> pyNSFEASOptions( pyNSFEAS, "Options" );

pyNSFEASOptions
 .def(
   py::init<>()
 )
 .def(
   py::init<NSFEAS::Options const&>()
 )
 .def(
   "reset",
   []( NSFEAS::Options& self )
     { self.reset(); },
   "reset options"
 )
 .def_readwrite( "FEASCRIT",  &NSFEAS::Options::FEASCRIT,  "selected feasibility criterion [Default: VAR]" )
 .def_readwrite( "FEASTHRES", &NSFEAS::Options::FEASTHRES, "percentile infeasibility threshold [Default: 0.1]" )
 .def_readwrite( "NUMLIVE",   &NSFEAS::Options::NUMLIVE,   "number of live points [Default: 256]" )
 .def_readwrite( "NUMPROP",   &NSFEAS::Options::NUMPROP,   "number of proposals [Default: 16]" )
 .def_readwrite( "ELLCONF",   &NSFEAS::Options::ELLCONF,   "chi-squared confidence in ellipsoidal nest [Default: 0.99]" )
 .def_readwrite( "ELLMAG",    &NSFEAS::Options::ELLMAG,    "initial magnification of ellipsoidal nest [Default: 0.3]" )
 .def_readwrite( "ELLRED",    &NSFEAS::Options::ELLRED,    "reduction factor of ellipsoidal nest [Default: 0.2]" )
 .def_readwrite( "MAXITER",   &NSFEAS::Options::MAXITER,   "maximal number of iterations [Default: 0]" )
 .def_readwrite( "MAXCPU",    &NSFEAS::Options::MAXCPU,    "maximal walltime [Default: 0]" )
 .def_readwrite( "DISPLEVEL", &NSFEAS::Options::DISPLEVEL, "Verbosity level [Default: 1]" )
;

py::enum_<mc::NSFEAS::Options::TYPE>( pyNSFEASOptions, "TYPE" )
 .value( "PR",   mc::NSFEAS::Options::TYPE::PR,   "infeasibility probability" )
 .value( "VAR",  mc::NSFEAS::Options::TYPE::VAR,  "value-at-risk, VaR" )
 .value( "CVAR", mc::NSFEAS::Options::TYPE::CVAR, "conditional value-at-risk, CVaR" )
 .export_values()
;

}


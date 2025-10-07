#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
//#include <pybind11/numpy.h>

#include "base_mbdoe.hpp"

namespace py = pybind11;

void mc_base( py::module_ &m )
{

typedef mc::FFGraph       FFGraph;
typedef mc::BASE_SAMPLING BASE_SAMPLING;
typedef mc::BASE_MBFA     BASE_MBFA;
typedef mc::BASE_MBDOE    BASE_MBDOE;

py::class_<BASE_SAMPLING> pyBASE( m, "BaseSampling" );

pyBASE
 .def(
   py::init<>()
 )
 .def_static(
  "uniform_sample",
   []( size_t nsam, std::vector<double> const& lb, std::vector<double> const& ub, bool const sobol )
     { return BASE_SAMPLING::uniform_sample( nsam, lb, ub, sobol ); },
   py::arg("nsam"),
   py::arg("lb"),
   py::arg("ub"),
   py::arg("sobol")=true,
   "generate uniform sample over box domain"
 )
 .def_static(
  "gaussian_sample",
   []( size_t nsam, std::vector<double> const& mean, std::vector<double> const& var, bool const sobol )
     { return BASE_SAMPLING::gaussian_sample( nsam, mean, var, sobol ); },
   py::arg("nsam"),
   py::arg("mean"),
   py::arg("var"),
   py::arg("sobol")=true,
   "generate Gaussian sample with diagonal variance"
 )
;

py::class_<BASE_MBFA, BASE_SAMPLING> pyMBFA( m, "BaseFeas", py::multiple_inheritance() );

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
   []( BASE_MBFA const& self ){ return self.nc(); },
   "size of model constants"
 )
 .def(
   "nu",
   []( BASE_MBFA const& self )
     { return self.nu(); },
   "size of model controls"
 )
 .def_property_readonly(
   "np",
   []( BASE_MBFA const& self )
     { return self.np(); },
   "size of model parameters"
 )
 .def(
   "set_constant",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& c, std::vector<double> const& cval )
     { self.set_constant( c, cval ); },
   py::arg("var"),
   py::arg("val")=std::vector<double>(),
   "set model constants"
 )
 .def(
   "reset_constant",
   []( BASE_MBFA& self )
     { self.reset_constant(); },
   "reset model constants"
 )
 .def_property_readonly(
   "var_constant",
   []( BASE_MBFA const& self )
     { return self.var_constant(); },
   py::return_value_policy::reference_internal,
   "model constants"
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
   "set model constraint"
 )
 .def(
   "set_constraint",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "set model constraints"
 )
 .def(
   "add_constraint",
   []( BASE_MBFA& self, mc::FFVar const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "add model constraint"
 )
 .def(
   "add_constraint",
   []( BASE_MBFA& self, std::vector<mc::FFVar> const& g )
     { self.set_constraint( g ); },
   py::arg("var"),
   "add model constraints"
 )
 .def(
   "reset_constraint",
   []( BASE_MBFA& self )
     { self.reset_constraint(); },
   "reset model constraints"
 )
 .def_property_readonly(
   "var_constraint",
   []( BASE_MBFA const& self )
     { return self.var_constraint(); },
   py::return_value_policy::reference_internal,
   "model constraints"
 )
 .def(
   "set_loglikelihood",
   []( BASE_MBFA& self, mc::FFVar const& llkh )
     { self.set_loglikelihood( llkh ); },
   py::arg("var"),
   "set model log-likelihood"
 )
 .def(
   "reset_loglikelihood",
   []( BASE_MBFA& self )
     { self.reset_loglikelihood(); },
   "reset model log-likelihood"
 )
 .def_property_readonly(
   "var_loglikelihood",
   []( BASE_MBFA const& self )
     { return self.var_loglikelihood(); },
   py::return_value_policy::reference_internal,
   "model log-likelihood"
 )
;

py::class_<BASE_MBDOE, BASE_MBFA> pyMBDOE( m, "BaseMBDOE", py::multiple_inheritance() );

pyMBDOE
 .def(
   py::init<>()
 )
 .def_property_readonly(
   "nm",
   []( BASE_MBDOE const& self )
     { return self.nm(); },
   "size of model candidates"
 )
 .def_property_readonly(
   "ny",
   []( BASE_MBDOE const& self ){ return self.ny(); },
   "size of model outputs"
 )
 .def(
   "set_model",
   []( BASE_MBDOE& self, std::vector<mc::FFVar> const& out, std::vector<double> const& var )
     { self.set_model( out, var ); },
   py::arg("out"),
   py::arg("var")=std::vector<double>(),
   "set model outputs (single model)"
 )
 .def(
   "set_model",
   []( BASE_MBDOE& self, std::list<std::vector<mc::FFVar>> const& out, std::vector<double> const& var )
     { self.set_model( out, var ); },
   py::arg("out"),
   py::arg("var")=std::vector<double>(),
   "set model outputs (multiple models, equal weighting)"
 )
 .def(
   "set_model",
   []( BASE_MBDOE& self, std::list<std::pair<std::vector<mc::FFVar>,double>> const& out, std::vector<double> const& var )
     { self.set_model( out, var ); },
   py::arg("out"),
   py::arg("var")=std::vector<double>(),
   "set model outputs (multiple models and weights)"
 )
 .def(
   "set_parameter",
   []( BASE_MBDOE& self, std::vector<mc::FFVar> const& p, std::vector<double> const& pnom, std::vector<double> const& psca )
     { self.set_parameter( p, pnom, psca ); },
   py::arg("var"),
   py::arg("nom"),
   py::arg("sca")=std::vector<double>(),
   "set model parameters, nominal values, and (optional) scaling factors"
 )
 .def(
   "set_parameter",
   []( BASE_MBDOE& self, std::vector<mc::FFVar> const& p, std::list<std::vector<double>> const& psam, std::vector<double> const& psca )
     { self.set_parameter( p, psam, psca ); },
   py::arg("var"),
   py::arg("sam"),
   py::arg("sca")=std::vector<double>(),
   "set model parameters, sampled values (equal weighting), and (optional) scaling factors"
 )
 .def(
   "set_parameter",
   []( BASE_MBDOE& self, std::vector<mc::FFVar> const& p, std::list<std::pair<std::vector<double>,double>> const& psam, std::vector<double> const& psca )
     { self.set_parameter( p, psam, psca ); },
   py::arg("var"),
   py::arg("sam"),
   py::arg("sca")=std::vector<double>(),
   "set model parameters, sampled values and weights, and (optional) scaling factors"
 )
 .def_property_readonly(
   "prior_campaign",
   []( BASE_MBDOE const& self )
     { return self.prior_campaign(); },
   py::return_value_policy::reference_internal,
   "prior campaign: [ {effort0, [experiment0]}, {effort1, [experiment1]}, ... ]"
 )
 .def(
   "reset_prior_campaign",
   []( BASE_MBDOE& self )
     { self.reset_prior_campaign(); },
   "reset prior campaign"
 )
 .def(
   "set_prior_campaign",
   []( BASE_MBDOE& self, std::list<std::pair<double,std::vector<double>>> const& C )
     { self.set_prior_campaign( C ); },
   "set prior campaign: [ {effort0, [experiment0]}, {effort1, [experiment1]}, ... ]"
 )
 .def(
   "add_prior_campaign",
   []( BASE_MBDOE& self, std::list<std::pair<double,std::vector<double>>> const& C )
     { self.add_prior_campaign( C ); },
   "add to prior campaign: [ {effort0, [experiment0]}, {effort1, [experiment1]}, ... ]"
 )
;

py::enum_<BASE_MBDOE::TYPE>( pyMBDOE, "TYPE" )
 .value( "AOPT",  BASE_MBDOE::TYPE::AOPT,  "A optimality" )
 .value( "DOPT",  BASE_MBDOE::TYPE::DOPT,  "D optimality" )
 .value( "EOPT",  BASE_MBDOE::TYPE::EOPT,  "E optimality" )
 .value( "BRISK", BASE_MBDOE::TYPE::BRISK, "Bayes risk" )
 .value( "ODIST", BASE_MBDOE::TYPE::ODIST, "Output spread" )
 .export_values()
;
}


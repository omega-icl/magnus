#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
//#include <pybind11/numpy.h>

#include "base_mbfa.hpp"

namespace py = pybind11;

void mc_base( py::module_ &m )
{

typedef mc::FFGraph       FFGraph;
typedef mc::BASE_SAMPLING BASE_SAMPLING;
typedef mc::BASE_MBFA     BASE_MBFA;

py::class_<BASE_SAMPLING> pyBASE( m, "Base" );

pyBASE
 .def(
   py::init<>()
 )
 .def_static(
  "uniform_sampling",
   []( size_t nsam, std::vector<double> const& lb, std::vector<double> const& ub )
     { return BASE_SAMPLING::uniform_sample( nsam, lb, ub ); },
   "uniform sampling using Sobol sequences"
);

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
   "number of model constants"
 )
 .def(
   "nu",
   []( BASE_MBFA const& self )
     { return self.nu(); },
   "number of model controls"
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

}


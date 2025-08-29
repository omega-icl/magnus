#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "modiscr.hpp"

namespace py = pybind11;

void mc_modiscr( py::module_ &m )
{

typedef mc::MODISCR MODISCR;

py::class_<MODISCR, mc::BASE_MBDOE> pyMODISCR( m, "MoDiscr", py::multiple_inheritance() );

pyMODISCR
 .def(
   py::init<>()
 )
 .def_readwrite( 
   "options",
   &MODISCR::options
 )
 .def(
   "setup",
   []( MODISCR& self )
     { return self.setup(); },
   "setup experiment design problem before solution"
 )
 .def(
   "evaluate_design",
   []( MODISCR& self, std::list<std::pair<double,std::vector<double>>> const& campaign,
       std::string const& crit, std::vector<double> const& cst )
     { return self.evaluate_design( campaign, crit, cst ); },
   py::arg("campaign"),
   py::arg("crit"),    
   py::arg("cst")=std::vector<double>(),
   "evaluate performance of experimental campaign"
 )
 .def(
   "sample_support",
   []( MODISCR& self, size_t const nsam, std::vector<double> const& cst )
      {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       return self.sample_support( nsam, cst );
      },
   py::arg("nsam"),
   py::arg("cst")=std::vector<double>(),
   "sample feasible supports within experimental domain"
 )
 .def(
   "effort_solve",
   []( MODISCR& self, size_t const nexp, bool const exact, 
     std::map<size_t,double> const& ini )
     {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       return self.effort_solve( nexp, exact, ini );
     },
   py::arg("nexp"),
   py::arg("exact")=true,
   py::arg("ini")=std::map<size_t,double>(),
   "solve effort-based experiment design for support selection"
 )
 .def(
   "gradient_solve",
   []( MODISCR& self, std::map<size_t,double> const& supp, std::vector<double> const& cst,
      bool const updt )
     {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       return self.gradient_solve( supp, cst, updt );
     },
   py::arg("supp"),
   py::arg("cst")=std::vector<double>(),
   py::arg("updt")=true,
   "solve gradient-based experiment design for support refinement"
 )
 .def(
   "combined_solve",
   []( MODISCR& self, size_t const nexp, std::vector<double> const& cst, bool const exact, 
     std::map<size_t,double> const& ini )
     {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       return self.combined_solve( nexp, cst, exact, ini );
     },
   py::arg("nexp"),
   py::arg("cst")=std::vector<double>(),
   py::arg("exact")=true,
   py::arg("ini")=std::map<size_t,double>(),
   "solve combined effort- and gradient-based experiment design"
 )
 .def(
   "file_export",
   []( MODISCR& self, std::string const& name )
     { return self.file_export( name ); },
   py::arg("name"),
   "export sampled efforts, supports and candidate model outputs to file"
 )
 .def_property_readonly(
   "effort",
   []( MODISCR& self )
     { return self.effort(); },
   py::return_value_policy::reference_internal,
   "optimized efforts: { {index0, effort0}, {index1, effort1}, ... }"
 )
 .def_property_readonly(
   "support",
   []( MODISCR& self )
     { return self.support(); },
   py::return_value_policy::reference_internal,
   "optimized supports: { {index0, [experiment0]}, {index1, [experiment1]}, ... }"
 )
 .def_property_readonly(
   "crit",
   []( MODISCR& self )
     { return self.criterion(); },
   py::return_value_policy::reference_internal,
   "optimized criterion"
 )
 .def_property_readonly(
   "campaign",
   []( MODISCR const& self )
     { return self.campaign(); },
   py::return_value_policy::reference_internal,
   "optimized campaign: [ {effort0, [experiment0]}, {effort1, [experiment1]}, ... ]"
 )
 .def_property_readonly(
   "control_sample",
   []( MODISCR const& self )
     { return self.control_sample(); },
   py::return_value_policy::reference_internal,
   "sampled controls: [ [experiment0], [experiment1], ... ]"
 )
  .def_property_readonly(
   "output_sample",
   []( MODISCR const& self ){
     constexpr size_t elsize = sizeof(double);
     std::vector<std::vector<std::vector<py::array_t<double>>>> samout( self.output_sample().size(), std::vector<std::vector<py::array_t<double>>>( self.output_sample().front().size(), std::vector<py::array_t<double>>( self.output_sample().front().front().size() ) ) );
     size_t vshape[2]{ self.ny(), 1 };
     size_t vstrides[2]{ elsize, self.ny() * elsize };
     size_t e=0;
     for( auto const& sam_e : self.output_sample() ){
       size_t s=0;
       for( auto const& sam_e_s : sam_e ){
         size_t m=0;
         for( auto const& sam_e_s_m : sam_e_s ){
           samout[e][s][m++] = py::array_t<double>( vshape, vstrides, sam_e_s_m.memptr() );
#ifdef MAGNUS__MODISCR_DEBUG
         std::cout << "samout[" << e << "][" << s << "][" << m << "] = " << sam_e_s_m.t();
#endif
         }
         s++;
       }
       e++;
     }
     return samout;
   },
   py::return_value_policy::reference_internal,
   "sampled outputs: [ [ [output0_scenario0], [output0_scenario1], ... ], [ [output1_scenario0], [output1_scenario1], ... ], ... ]"
 )

 .def_property_readonly(
   "output_sample",
   []( MODISCR const& self )
     { return self.output_sample(); },
   py::return_value_policy::reference_internal,
   "sampled outputs: [ [output0], [output1], ... ]"
 )
;

py::class_<MODISCR::Options> pyMODISCROptions( pyMODISCR, "Options" );

pyMODISCROptions
 .def(
   py::init<>()
 )
 .def(
   py::init<MODISCR::Options const&>()
 )
 .def(
   "reset",
   []( MODISCR::Options& self )
     { self.reset(); },
   "reset options"
 )
 .def_readwrite( "CRITERION", &MODISCR::Options::CRITERION, "selected DOE criterion [Default: BRISK]" )
 .def_readwrite( "RISK",      &MODISCR::Options::RISK,      "selected attitude to risk [Default: AVERSE]" )
 .def_readwrite( "CVARTHRES", &MODISCR::Options::CVARTHRES, "information lower percentile for CVaR [Default: 0.10]" )
 .def_readwrite( "MINDIST",   &MODISCR::Options::MINDIST,   "relative mean-absolute distance between support points after refinement [Default: 1e-6]" )
 .def_readwrite( "MAXITER",   &MODISCR::Options::MAXITER,   "maximum number of iterations [Default: 4]" )
 .def_readwrite( "TOLITER",   &MODISCR::Options::TOLITER,   "stopping tolerance for iterations [Default: 1e-4]" )
 .def_readwrite( "DISPLEVEL", &MODISCR::Options::DISPLEVEL, "verbosity level [Default: 1]" )
 .def_readwrite( "MINLPSLV",  &MODISCR::Options::MINLPSLV,  "effort-based MINLP solver options" )
 .def_readwrite( "NLPSLV",    &MODISCR::Options::NLPSLV,    "gradient-based NLP solver options" )
;

py::enum_<mc::MODISCR::Options::RISK_TYPE>( pyMODISCROptions, "RISK_TYPE" )
 .value( "NEUTRAL",  mc::MODISCR::Options::RISK_TYPE::NEUTRAL, "risk-neutral average design" )
 .value( "AVERSE",   mc::MODISCR::Options::RISK_TYPE::AVERSE,  "risk-averse CVaR design" )
 .export_values()
;
}


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "expdes.hpp"

namespace py = pybind11;

void mc_expdes( py::module_ &m )
{

typedef mc::EXPDES EXPDES;

py::class_<EXPDES, mc::BASE_MBDOE> pyEXPDES( m, "ExpDes", py::multiple_inheritance() );

pyEXPDES
 .def(
   py::init<>()
 )
 .def_readwrite( 
   "options",
   &EXPDES::options
 )
 .def(
   "setup",
   []( EXPDES& self, size_t const ndxmod )
     {
       try{
         return (int)self.setup( ndxmod );
       }
       catch( mc::NSFEAS::Exceptions const& ex ){
         std::cerr << "Error code = " << ex.ierr() << std::endl;
         std::cerr << ex.what() << std::endl;
         return ex.ierr();
       }
     },
   py::arg("ndxmod")=0,    
   "setup experiment design problem before solution"
 )
 .def(
   "evaluate_design",
   []( EXPDES& self, std::list<std::pair<double,std::vector<double>>> const& campaign,
       std::string const& crit, std::vector<double> const& cst )
     {
       return self.evaluate_design( campaign, crit, cst ); // catches exceptions
     },
   py::arg("campaign"),
   py::arg("crit"),    
   py::arg("cst")=std::vector<double>(),
   "evaluate performance of experimental campaign"
 )
 .def(
   "sample_support",
   []( EXPDES& self, size_t const nsam, std::vector<double> const& cst )
     { 
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       try{
         return (int)self.sample_support( nsam, cst );
       }
       catch( mc::NSFEAS::Exceptions const& ex ){
         std::cerr << "Error code = " << ex.ierr() << std::endl;
         std::cerr << ex.what() << std::endl;
         return ex.ierr();
       }
       catch( EXPDES::Exceptions const& ex ){
         std::cerr << "Error code = " << ex.ierr() << std::endl;
         std::cerr << ex.what() << std::endl;
         return ex.ierr();
       }
     },
   py::arg("nsam"),
   py::arg("cst")=std::vector<double>(),
   "sample feasible supports within experimental domain"
 )
 .def(
   "effort_solve",
   []( EXPDES& self, size_t const nexp, bool const exact, 
     std::map<size_t,double> const& ini )
     {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       try{
         return self.effort_solve( nexp, exact, ini );
       }
       catch( GRBException const& ex ){
         std::cerr << "Error code = " << ex.getErrorCode() << std::endl;
         std::cerr << ex.getMessage() << std::endl;
         return ex.getErrorCode();
       }
       catch( EXPDES::Exceptions const& ex ){
         std::cerr << "Error code = " << ex.ierr() << std::endl;
         std::cerr << ex.what() << std::endl;
         return ex.ierr();
       }
     },
   py::arg("nexp"),
   py::arg("exact")=true,
   py::arg("ini")=std::map<size_t,double>(),
   "solve effort-based experiment design for support selection"
 )
 .def(
   "gradient_solve",
   []( EXPDES& self, std::map<size_t,double> const& supp, std::vector<double> const& cst,
      bool const updt )
     {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       try{
         return self.gradient_solve( supp, cst, updt );
       }
       catch( EXPDES::Exceptions const& ex ){
         std::cerr << "Error code = " << ex.ierr() << std::endl;
         std::cerr << ex.what() << std::endl;
         return ex.ierr();
       }
     },
   py::arg("supp"),
   py::arg("cst")=std::vector<double>(),
   py::arg("updt")=true,
   "solve gradient-based experiment design for support refinement"
 )
 .def(
   "combined_solve",
   []( EXPDES& self, size_t const nexp, std::vector<double> const& cst, bool const exact, 
     std::map<size_t,double> const& ini )
     {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       try{
         return self.combined_solve( nexp, cst, exact, ini );
       }
       catch( GRBException const& ex ){
         std::cerr << "Error code = " << ex.getErrorCode() << std::endl;
         std::cerr << ex.getMessage() << std::endl;
         return ex.getErrorCode();
       }
       catch( EXPDES::Exceptions const& ex ){
         std::cerr << "Error code = " << ex.ierr() << std::endl;
         std::cerr << ex.what() << std::endl;
         return ex.ierr();
       }
     },
   py::arg("nexp"),
   py::arg("cst")=std::vector<double>(),
   py::arg("exact")=true,
   py::arg("ini")=std::map<size_t,double>(),
   "solve combined effort- and gradient-based experiment design"
 )
 .def(
   "file_export",
   []( EXPDES& self, std::string const& name )
     { return self.file_export( name ); },
   py::arg("name"),
   "export sampled efforts, supports and outputs/FIMs to file"
 )
 .def_property_readonly(
   "effort",
   []( EXPDES& self )
     { return self.effort(); },
   py::return_value_policy::reference_internal,
   "optimized efforts: { {index0, effort0}, {index1, effort1}, ... }"
 )
 .def_property_readonly(
   "support",
   []( EXPDES& self )
     { return self.support(); },
   py::return_value_policy::reference_internal,
   "optimized supports: { {index0, [experiment0]}, {index1, [experiment1]}, ... }"
 )
 .def_property_readonly(
   "crit",
   []( EXPDES& self )
     { return self.criterion(); },
   py::return_value_policy::reference_internal,
   "optimized criterion"
 )
 .def_property_readonly(
   "campaign",
   []( EXPDES const& self )
     { return self.campaign(); },
   py::return_value_policy::reference_internal,
   "optimized campaign: [ {effort0, [experiment0]}, {effort1, [experiment1]}, ... ]"
 )
 .def_property_readonly(
   "control_sample",
   []( EXPDES const& self )
     { return self.control_sample(); },
   py::return_value_policy::reference_internal,
   "sampled controls: [ [experiment0], [experiment1], ... ]"
 )
  .def_property_readonly(
   "output_sample",
   []( EXPDES const& self ){
     constexpr size_t elsize = sizeof(double);
     std::vector<std::vector<py::array_t<double>>> samout( self.output_sample().size(), std::vector<py::array_t<double>>( self.output_sample().front().size() ) );
     size_t vshape[2]{ self.ny(), 1 };
     size_t vstrides[2]{ elsize, self.ny() * elsize };
     size_t e=0;
     for( auto const& sam_e : self.output_sample() ){
       size_t s=0;
       for( auto const& sam_e_s : sam_e ){
         samout[e][s++] = py::array_t<double>( vshape, vstrides, sam_e_s.memptr() );
#ifdef MAGNUS__EXPDES_DEBUG
         std::cout << "samout[" << e << "][" << s << "] = " << sam_e_s.t();
#endif
       }
       e++;
     }
     return samout;
   },
   py::return_value_policy::reference_internal,
   "sampled outputs: [ [ [output0_scenario0], [output0_scenario1], ... ], [ [output1_scenario0], [output1_scenario1], ... ], ... ]"
 )
 .def_property_readonly(
   "fim_sample",
   []( EXPDES const& self ){
     constexpr size_t elsize = sizeof(double);
     std::vector<std::vector<py::array_t<double>>> samfim( self.output_sample().size(), std::vector<py::array_t<double>>( self.output_sample().front().size() ) );
     size_t mshape[2]{ self.np(), self.np() };
     size_t mstrides[2]{ elsize, self.np() * elsize };
     size_t e=0;
     for( auto const& sam_e : self.fim_sample() ){
       size_t s=0;
       for( auto const& sam_e_s : sam_e ){
         samfim[e][s++] = py::array_t<double>( mshape, mstrides, sam_e_s.memptr() );
#ifdef MAGNUS__EXPDES_DEBUG
         std::cout << "samfim[" << e << "][" << s << "] = " << sam_e_s.t();
#endif
       }
       e++;
     }
     return samfim;
   },
   py::return_value_policy::reference_internal,
   "sampled FIMs: [ [ [FIM0_scenario0], [FIM0_scenario1], ... ], [ [FIM1_scenario0], [FIM1_scenario1], ... ], ... ]"
 )
;

py::class_<EXPDES::Options> pyEXPDESOptions( pyEXPDES, "Options" );

pyEXPDESOptions
 .def(
   py::init<>()
 )
 .def(
   py::init<EXPDES::Options const&>()
 )
 .def(
   "reset",
   []( EXPDES::Options& self )
     { self.reset(); },
   "reset options"
 )
 .def_readwrite( "CRITERION", &EXPDES::Options::CRITERION, "selected DOE criterion [Default: DOPT]" )
 .def_readwrite( "RISK",      &EXPDES::Options::RISK,      "selected attitude to risk [Default: NEUTRAL]" )
 .def_readwrite( "CVARTHRES", &EXPDES::Options::CVARTHRES, "information lower percentile for CVaR [Default: 0.25]" )
 .def_readwrite( "UNCREDUC",  &EXPDES::Options::UNCREDUC,  "Uncertainty scenario reduction [Default: 0.01]" )
 .def_readwrite( "FIMSTOL",   &EXPDES::Options::FIMSTOL,   "tolerance for singular value in FIM [Default: 1e-7]" )
 .def_readwrite( "IDWTOL",    &EXPDES::Options::IDWTOL,    "tolerance for inverse distance weighting measure [Default: 1e-3]" )
 .def_readwrite( "FEASTHRES", &EXPDES::Options::FEASTHRES, "constraint violation percentile for CVaR [Default: 0.10]" )
 .def_readwrite( "FEASPROP",  &EXPDES::Options::FEASPROP,  "number of proposals in nested sampling iterations of feasible experiment domain [Default: 16]" )
 .def_readwrite( "MINDIST",   &EXPDES::Options::MINDIST,   "relative mean-absolute distance between support points after refinement [Default: 1e-6]" )
 .def_readwrite( "MAXITER",   &EXPDES::Options::MAXITER,   "maximum number of iterations [Default: 4]" )
 .def_readwrite( "TOLITER",   &EXPDES::Options::TOLITER,   "stopping tolerance for iterations [Default: 1e-4]" )
 .def_readwrite( "DISPLEVEL", &EXPDES::Options::DISPLEVEL, "verbosity level [Default: 1]" )
 .def_readwrite( "MINLPSLV",  &EXPDES::Options::MINLPSLV,  "effort-based MINLP solver options" )
 .def_readwrite( "NLPSLV",    &EXPDES::Options::NLPSLV,    "gradient-based NLP solver options" )
;

py::enum_<mc::EXPDES::Options::RISK_TYPE>( pyEXPDESOptions, "RISK_TYPE" )
 .value( "NEUTRAL",  mc::EXPDES::Options::RISK_TYPE::NEUTRAL, "risk-neutral average design" )
 .value( "AVERSE",   mc::EXPDES::Options::RISK_TYPE::AVERSE,  "risk-averse CVaR design" )
 .export_values()
;
}


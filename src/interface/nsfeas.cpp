#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "nsfeas.hpp"

namespace py = pybind11;

void mc_nsfeas( py::module_ &m )
{

typedef mc::NSFEAS NSFEAS;

py::class_<NSFEAS, mc::BASE_MBFA> pyNSFEAS( m, "NSFeas", py::multiple_inheritance() );

pyNSFEAS
 .def(
   py::init<>()
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
   []( NSFEAS& self, std::vector<double> vcst, bool const reset ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     return self.sample( vcst, reset );
   },
   py::arg("cst") = std::vector<double>(),
   py::arg("reset") = true,
   "sample feasible domain"
 )
 .def_property_readonly(
   "live_points",
   []( NSFEAS const& self ){
     arma::mat mpts( self.live_points().size(), self.nu() );
     arma::vec vlkh( self.live_points().size() );
     arma::vec vprb( self.live_points().size() );
     arma::vec vaux( self.live_points().size() );
     size_t r=0;
     for( auto const& [lkh,val] : self.live_points() ){
       for( size_t c=0; c<self.nu(); ++c )
         mpts(r,c) = std::get<0>(val)[c];
       vlkh(r) = lkh;
       vprb(r) = std::get<1>(val);
       vaux(r) = std::get<2>(val);
       ++r;
     }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
     std::cout << "mpts( " << samples.n_rows << "," << samples.n_cols << "):\n" << mpts;
#endif
     constexpr size_t elsize = sizeof(double);
     size_t mshape[2]{mpts.n_rows, mpts.n_cols};
     size_t mstrides[2]{elsize, mpts.n_rows * elsize};
     size_t vshape[2]{vlkh.n_elem, 1};
     size_t vstrides[2]{elsize, vlkh.n_elem * elsize};
     return std::make_tuple( py::array_t<double>( mshape, mstrides, mpts.memptr() ),
                             py::array_t<double>( vshape, vstrides, vlkh.memptr() ),
                             py::array_t<double>( vshape, vstrides, vprb.memptr() ),
                             py::array_t<double>( vshape, vstrides, vaux.memptr() ) );
   },
   py::return_value_policy::reference_internal,
   "retrieve live points"
 )
 .def_property_readonly(
   "dead_points",
   []( NSFEAS const& self ){
     arma::mat mpts( self.dead_points().size(), self.nu() );
     arma::vec vlkh( self.dead_points().size() );
     arma::vec vprb( self.dead_points().size() );
     arma::vec vaux( self.dead_points().size() );
     size_t r=0;
     for( auto const& [lkh,val] : self.dead_points() ){
       for( size_t c=0; c<self.nu(); ++c )
         mpts(r,c) = std::get<0>(val)[c];
       vlkh(r) = lkh;
       vprb(r) = std::get<1>(val);
       vaux(r) = std::get<2>(val);
       ++r;
     }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
     std::cout << "mpts( " << samples.n_rows << "," << samples.n_cols << "):\n" << mpts;
#endif
     constexpr size_t elsize = sizeof(double);
     size_t mshape[2]{mpts.n_rows, mpts.n_cols};
     size_t mstrides[2]{elsize, mpts.n_rows * elsize};
     size_t vshape[2]{vlkh.n_elem, 1};
     size_t vstrides[2]{elsize, vlkh.n_elem * elsize};
     return std::make_tuple( py::array_t<double>( mshape, mstrides, mpts.memptr() ),
                             py::array_t<double>( vshape, vstrides, vlkh.memptr() ),
                             py::array_t<double>( vshape, vstrides, vprb.memptr() ),
                             py::array_t<double>( vshape, vstrides, vaux.memptr() ) );
   },
   py::return_value_policy::reference_internal,
   "retrieve dead points"
 )
 .def_property_readonly(
   "discard_points",
   []( NSFEAS const& self ){
     arma::mat mpts( self.discard_points().size(), self.nu() );
     arma::vec vlkh( self.discard_points().size() );
     arma::vec vprb( self.discard_points().size() );
     arma::vec vaux( self.discard_points().size() );
     size_t r=0;
     for( auto const& [lkh,val] : self.discard_points() ){
       for( size_t c=0; c<self.nu(); ++c )
         mpts(r,c) = std::get<0>(val)[c];
       vlkh(r) = lkh;
       vprb(r) = std::get<1>(val);
       vaux(r) = std::get<2>(val);
       ++r;
     }
#ifdef MAGNUS__NSFEAS_SAMPLE_DEBUG
     std::cout << "mpts( " << samples.n_rows << "," << samples.n_cols << "):\n" << mpts;
#endif
     constexpr size_t elsize = sizeof(double);
     size_t mshape[2]{mpts.n_rows, mpts.n_cols};
     size_t mstrides[2]{elsize, mpts.n_rows * elsize};
     size_t vshape[2]{vlkh.n_elem, 1};
     size_t vstrides[2]{elsize, vlkh.n_elem * elsize};
     return std::make_tuple( py::array_t<double>( mshape, mstrides, mpts.memptr() ),
                             py::array_t<double>( vshape, vstrides, vlkh.memptr() ),
                             py::array_t<double>( vshape, vstrides, vprb.memptr() ),
                             py::array_t<double>( vshape, vstrides, vaux.memptr() ) );
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
 .def_readwrite( "FEASTHRES", &NSFEAS::Options::FEASTHRES, "percentile feasibility violation threshold [Default: 0.1]" )
 .def_readwrite( "LKHCRIT",   &NSFEAS::Options::LKHCRIT,   "selected likelihood criterion [Default: VAR]" )
 .def_readwrite( "LKHTHRES",  &NSFEAS::Options::LKHTHRES,  "percentile likelihood threshold [Default: 0.1]" )
 .def_readwrite( "LKHTOL",    &NSFEAS::Options::LKHTOL,    "stopping tolerance for probability mass [Default: 0.05]" )
 .def_readwrite( "NUMLIVE",   &NSFEAS::Options::NUMLIVE,   "number of live points [Default: 256]" )
 .def_readwrite( "NUMPROP",   &NSFEAS::Options::NUMPROP,   "number of proposals [Default: 16]" )
 .def_readwrite( "ELLCONF",   &NSFEAS::Options::ELLCONF,   "chi-squared confidence in ellipsoidal nest [Default: 0.99]" )
 .def_readwrite( "ELLMAG",    &NSFEAS::Options::ELLMAG,    "initial magnification of ellipsoidal nest [Default: 0.3]" )
 .def_readwrite( "ELLRED",    &NSFEAS::Options::ELLRED,    "reduction factor of ellipsoidal nest [Default: 0.2]" )
 .def_readwrite( "MAXITER",   &NSFEAS::Options::MAXITER,   "maximal number of iterations [Default: 0]" )
 .def_readwrite( "MAXERR",    &NSFEAS::Options::MAXERR,    "maximal number of failed evaluations [Default: 0]" )
 .def_readwrite( "MAXCPU",    &NSFEAS::Options::MAXCPU,    "maximal walltime [Default: 0]" )
 .def_readwrite( "DISPLEVEL", &NSFEAS::Options::DISPLEVEL, "display level [Default: 1]" )
 .def_readwrite( "DISPITER",  &NSFEAS::Options::DISPITER,  "display frequency [Default: 25]" )
;

py::enum_<mc::NSFEAS::Options::TYPE>( pyNSFEASOptions, "TYPE" )
 .value( "VAR",  mc::NSFEAS::Options::TYPE::VAR,  "value-at-risk, VaR" )
 .value( "CVAR", mc::NSFEAS::Options::TYPE::CVAR, "conditional value-at-risk, CVaR" )
 .export_values()
;

}


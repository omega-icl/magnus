#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "parest.hpp"

namespace py = pybind11;

void mc_parest( py::module_ &m )
{
typedef mc::FFGraph FFGraph;
#ifdef MC__USE_SNOPT
  typedef mc::NLPSLV_SNOPT NLPSLV;
#elif  MC__USE_IPOPT
  typedef mc::NLPSLV_IPOPT NLPSLV;
#endif
typedef mc::PAREST PAREST;

py::class_<PAREST> pyPAREST( m, "ParEst" );

pyPAREST
 .def(
   py::init<>()
 )
 .def_readwrite( 
   "options",
   &PAREST::options
 )
 .def( 
   "set_dag",
   []( PAREST& self,  FFGraph& dag ){ self.set_dag( dag ); },
   py::keep_alive<1,2>(),
   "set DAG"
 )
 .def_property_readonly(
   "dag",
   []( PAREST const& self ){ return &self.dag(); },
   py::return_value_policy::reference_internal,
   "DAG"
 )
 .def_property_readonly(
   "np",
   []( PAREST const& self ){ return self.np(); },
   "number of estimated parameters"
 )
 .def(
   "set_parameter",
   []( PAREST& self, std::vector<mc::FFVar> const& p, std::vector<double> const& plb,
       std::vector<double> const& pub ){ self.set_parameter( p, plb, pub ); },
   py::arg("var"),
   py::arg("lb")=std::vector<double>(),
   py::arg("ub")=std::vector<double>(),
   "set estimated parameters"
 )
 .def_property_readonly(
   "var_parameter",
   []( PAREST const& self ){ return self.var_parameter(); },
   py::return_value_policy::reference_internal,
   "estimated parameters"
 )
 .def_property_readonly(
   "nc",
   []( PAREST const& self ){ return self.nc(); },
   "number of model constants"
 )
 .def(
   "set_constant",
   []( PAREST& self, std::vector<mc::FFVar> const& c ){ self.set_constant( c ); },
   "set model constants"
 )
 .def_property_readonly(
   "var_constant",
   []( PAREST const& self ){ return self.var_constant(); },
   py::return_value_policy::reference_internal,
   "model constants"
 )
 .def(
   "nu",
   []( PAREST const& self, size_t const mod ){ return self.nu( mod ); },
   py::arg("mod")=0,
   "number of model controls"
 )
 .def(
   "ny",
   []( PAREST const& self, size_t const mod ){ return self.ny( mod ); },
   py::arg("mod")=0,
   "number of model outputs"
 )
 .def(
   "add_model",
   []( PAREST& self, std::vector<mc::FFVar> const& y, std::vector<mc::FFVar> const& u,
       size_t const mod ){ self.add_model( y, u, mod ); },
   py::arg("out"),
   py::arg("in")=std::vector<mc::FFVar>(),
   py::arg("mod")=0,
   "add model"
 )
 .def(
   "reset_model",
   []( PAREST& self ){ self.reset_model(); },
   "reset model"
 )
 .def(
   "var_control",
   []( PAREST const& self, size_t const mod ){ return self.var_control( mod ); },
   py::return_value_policy::reference_internal,
   py::arg("mod")=0,
   "model controls"
 )
 .def(
   "var_output",
   []( PAREST const& self, size_t const mod ){ return self.var_output( mod ); },
   py::return_value_policy::reference_internal,
   py::arg("mod")=0,
   "model outputs"
 )
 .def_property_readonly(
   "nd",
   []( PAREST const& self ){ return self.nd(); },
   "number of experiments"
 )
 .def(
   "set_data",
   []( PAREST& self, std::vector<PAREST::Experiment> const& d ){ self.set_data( d ); },
   "set experimental data"
 )
 .def(
   "add_data",
   []( PAREST& self, std::vector<PAREST::Experiment> const& d ){ self.add_data( d ); },
   "add experimental data"
 )
 .def(
   "add_data",
   []( PAREST& self, PAREST::Experiment const& d ){ self.add_data( d ); },
   "add experimental data"
 )
 .def_property_readonly(
   "data",
   []( PAREST const& self ){ return &self.get_data(); },
   py::return_value_policy::reference_internal,
   "experimental data"
 )
 .def(
   "add_regularization",
   []( PAREST& self, mc::FFVar const& r ){ self.add_regularization( r ); },
   "add regularization term to objective function"
 )
 .def(
   "reset_regularization",
   []( PAREST& self ){ self.reset_regularization(); },
   "reset regularization terms in objective function"
 )
 .def_property_readonly(
   "get_constraint",
   []( PAREST const& self ){ return self.get_constraint(); },
   py::return_value_policy::reference_internal,
   "model constraints"
 )
 .def(
   "reset_constraint",
   []( PAREST& self ){ self.reset_constraint(); },
   "reset model constraints"
 )
 .def(
   "add_constraint",
   []( PAREST& self, mc::FFVar const& lhs, NLPSLV::t_CTR const& type, mc::FFVar const& rhs ){ self.add_constraint( lhs, type, rhs ); },
   py::arg("lhs"),
   py::arg("type"),
   py::arg("rhs")=mc::FFVar(0.),
   "add constraint"
 )
 .def_static(
   "sobol_sample",
   static_cast<std::list<std::vector<double>> (*)(size_t, std::vector<double> const&, std::vector<double> const&)>(&PAREST::uniform_sample)
 )
 .def_property_readonly(
   "mle",
   []( PAREST const& self ){ return self.mle(); },
   py::return_value_policy::reference_internal,
   "maximum likelihood solution"
 )
 .def(
   "setup",
   []( PAREST& self ){ self.setup(); },
   "setup parameter estimation problem"
 )
 .def(
   "mle_solve",
   []( PAREST& self, std::vector<double> const& par, std::vector<double> const& cst )
   { py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     return self.mle_solve( par, cst ); 
   },
   py::arg("par"),
   py::arg("cst")=std::vector<double>(),
   "solve gradient-based parameter estimation problem"
 )
 .def(
   "mle_solve",
   []( PAREST& self, size_t const nsam, std::vector<double> const& cst )
   { py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     return self.mle_solve( nsam, cst );
   },
   py::arg("nsam"),
   py::arg("cst")=std::vector<double>(),
   "multistart solve gradient-based parameter estimation problem"
 )
 .def(
   "chi2_test",
   []( PAREST& self, double const& conf ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     return self.chi2_test( conf );
   },
   py::arg("conf")=0.95,
   "compute chi-squared test (goodness-of-fit)"
 )
 .def(
   "cov_bootstrap",
   []( PAREST& self, std::vector<std::vector<PAREST::Experiment>> const& data, size_t const nsam ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     arma::mat covmat = self.cov_bootstrap( data, nsam );
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{covmat.n_rows, covmat.n_cols};
     size_t strides[ndim]{elsize, covmat.n_rows * elsize};
     return py::array_t<double>( shape, strides, covmat.memptr() );
   },
   py::return_value_policy::reference_internal,
   "compute parameter covariance matrix using bootstrapping"
 )
 .def(
   "cov_bootstrap",
   []( PAREST& self, std::vector<PAREST::Experiment> const& data, size_t const nsam ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     arma::mat covmat = self.cov_bootstrap( data, nsam );
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{covmat.n_rows, covmat.n_cols};
     size_t strides[ndim]{elsize, covmat.n_rows * elsize};
     return py::array_t<double>( shape, strides, covmat.memptr() );
   },
   py::return_value_policy::reference_internal,
   "compute parameter covariance matrix using bootstrapping"
 )
 .def(
   "cov_bootstrap",
   []( PAREST& self, size_t const nsam ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     arma::mat covmat = self.cov_bootstrap( nsam );
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{covmat.n_rows, covmat.n_cols};
     size_t strides[ndim]{elsize, covmat.n_rows * elsize};
     return py::array_t<double>( shape, strides, covmat.memptr() );
   },
   py::return_value_policy::reference_internal,
   "compute parameter covariance matrix using bootstrapping"
 )
 .def(
   "cov_linearized",
   []( PAREST& self ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     // Convert arma::mat to numpy array using auxiliary memory
     arma::mat covmat = self.cov_linearized();
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{covmat.n_rows, covmat.n_cols};
     size_t strides[ndim]{elsize, covmat.n_rows * elsize};
     return py::array_t<double>( shape, strides, covmat.memptr() );
   },
   py::return_value_policy::reference_internal,
   "compute parameter covariance matrix via linearization"
 )
 .def(
   "conf_interval",
   []( PAREST& self, py::array_t<double, py::array::c_style | py::array::forcecast> covmat,
       double const& conflevel, std::string const& type ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     // Request a buffer descriptor from Python
     py::buffer_info info = covmat.request();
     // Check dimension and type
     if( info.format != py::format_descriptor<double>::format() )
       throw std::runtime_error( "Incompatible format: double array expected" );
     if( info.ndim != 2 )
       throw std::runtime_error( "Incompatible buffer dimension: 2d array expected" );
     // Convert arma::vec to numpy array using auxiliary memory
     arma::vec cint = self.conf_interval(
       arma::mat( static_cast<double*>(info.ptr), info.shape[0], info.shape[1], false, true ),
       conflevel, type );
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 1;
     size_t shape[ndim]{cint.n_elem};
     size_t strides[ndim]{elsize};
     return py::array_t<double>( shape, strides, cint.memptr() );
   },
   py::arg("cov"),
   py::arg("conf")=0.95,
   py::arg("type")="T",
   py::return_value_policy::reference_internal,
   "compute parameter confidence intervals"
 )
 .def(
   "conf_ellipsoid",
   []( PAREST& self, py::array_t<double, py::array::c_style | py::array::forcecast> covmat,
       size_t const i, size_t const j, double const& conflevel, std::string const& type,
       size_t const nsam ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     // Request a buffer descriptor from Python
     py::buffer_info info = covmat.request();
     // Check dimension and type
     if( info.format != py::format_descriptor<double>::format() )
       throw std::runtime_error( "Incompatible format: double array expected" );
     if( info.ndim != 2 )
       throw std::runtime_error( "Incompatible buffer dimension: 2d array expected" );
     // Convert arma::mat to numpy array using auxiliary memory
     arma::mat cesam = self.conf_ellipsoid(
       arma::mat( static_cast<double*>(info.ptr), info.shape[0], info.shape[1], false, true ),
       i, j, conflevel, type, nsam );
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{cesam.n_rows, cesam.n_cols};
     size_t strides[ndim]{elsize, cesam.n_rows * elsize};
     return py::array_t<double>( shape, strides, cesam.memptr() );
   },
   py::arg("cov"),
   py::arg("first"),
   py::arg("second"),
   py::arg("conf")=0.95,
   py::arg("type")="F",
   py::arg("nsam")=50,
   py::return_value_policy::reference_internal,
   "compute parameter confidence intervals"
 )
 .def_property_readonly(
   "crsam",
   []( PAREST const& self ){
     py::scoped_ostream_redirect stream(
       std::cout,                                // std::ostream&
       py::module_::import("sys").attr("stdout") // Python output
     );
     arma::mat crmat = self.crsam();
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{crmat.n_rows, crmat.n_cols};
     size_t strides[ndim]{elsize, crmat.n_rows * elsize};
     return py::array_t<double>( shape, strides, crmat.memptr() );
   },
   //py::return_value_policy::take_ownership,
   py::return_value_policy::reference_internal,
   "bootstrapped confidence region"
 )
;

// Bind arma::mat to Python
py::class_<arma::mat>( m, "Mat", py::buffer_protocol() )
 .def_buffer(
   []( arma::mat &mat ) -> py::buffer_info {
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 2;
     size_t shape[ndim]{mat.n_rows, mat.n_cols};
     size_t strides[ndim]{elsize, mat.n_rows * elsize};
     return py::buffer_info(
       mat.memptr(),                            /* Pointer to buffer */
       elsize,                                  /* Size of one scalar */
       py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
       ndim,                                    /* Number of dimensions */
       shape,                                   /* Buffer dimensions */
       strides                                  /* Strides (in bytes) for each index */
     );
   }
 )
 .def(
   py::init( 
     []( py::buffer b ){
       // Request a buffer descriptor from Python
       py::buffer_info info = b.request();
       // Check dimension and type
       if( info.format != py::format_descriptor<double>::format() )
         throw std::runtime_error( "Incompatible format: double array expected" );
       if( info.ndim != 2 )
         throw std::runtime_error( "Incompatible buffer dimension: 2d array expected" );
       // Return arma::mat using auxiliary memory
       return arma::mat( static_cast<double*>(info.ptr), info.shape[0], info.shape[1], false, true );
     }
   )
 )
;

// Bind arma::vec to Python
py::class_<arma::vec>(m, "Vec", py::buffer_protocol())
 .def_buffer(
   [](arma::vec &vec) -> py::buffer_info {
     constexpr size_t elsize = sizeof(double);
     size_t const ndim = 1;
     size_t shape[ndim]{vec.n_elem};
     size_t strides[ndim]{elsize};
     return py::buffer_info(
       vec.memptr(),                            /* Pointer to buffer */
       elsize,                                  /* Size of one scalar */
       py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
       ndim,                                    /* Number of dimensions */
       shape,                                   /* Buffer dimensions */
       strides                                  /* Strides (in bytes) for each index */
     );
   }
 )
;

py::class_<PAREST::Record> pyPARESTRec( pyPAREST, "Record" );

pyPARESTRec
 .def( py::init<>() )
 .def( py::init<std::vector<double> const&, double const&>(),
       py::arg("meas"), py::arg("var")=0. )
 .def( py::init<PAREST::Record const&>() )
 .def_property(
   "measurement",
   []( PAREST::Record& self ){ return self.measurement; },
   []( PAREST::Record& self, std::vector<double> const& measurement ){ self.measurement = measurement; },
   "measurement replicates"
 )
 .def_property(
   "variance",
   []( PAREST::Record& self ){ return self.variance; },
   []( PAREST::Record& self, double const& variance ){ self.variance = variance; },
   "measurement variance"
 )
;

py::class_<PAREST::Experiment> pyPARESTExp( pyPAREST, "Experiment" );

pyPARESTExp
 .def( py::init<>() )
 //.def( py::init<std::vector<double> const&>() )
 .def( py::init<std::vector<double> const&,std::map<size_t,PAREST::Record> const&,size_t const>(),
       py::arg("in"), py::arg("out")=std::map<size_t,PAREST::Record>(), py::arg("mod")=0 )
 .def( py::init<PAREST::Experiment const&>() )
 .def_property(
   "index",
   []( PAREST::Experiment& self ){ return self.index; },
   []( PAREST::Experiment& self, size_t const index ){ self.index = index; },
   "model index"
 )
 .def_property(
   "control",
   []( PAREST::Experiment& self ){ return self.control; },
   []( PAREST::Experiment& self, std::vector<double> const& control ){ self.control = control; },
   "control values"
 )
 .def_property(
   "output",
   []( PAREST::Experiment& self ){ return self.output; },
   []( PAREST::Experiment& self, std::map<size_t,PAREST::Record> const& output ){ self.output = output; },
   "measurement records"
 )
;

py::class_<PAREST::Options> pyPARESTOptions( pyPAREST, "Options" );

pyPARESTOptions
 .def( py::init<>() )
 .def( py::init<PAREST::Options const&>() )
 .def_readwrite( "DISPLEVEL",   &PAREST::Options::DISPLEVEL,   "Verbosity level [Default: 1]" )
 .def_readwrite( "NLPSLV",      &PAREST::Options::NLPSLV,      "Options of gradient-based NLP solver" )
;

}


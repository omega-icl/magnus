#include <pybind11/pybind11.h>

namespace py = pybind11;

void mc_parest( py::module_ & );

PYBIND11_MODULE( magnus, m )
{
  try{
    py::module_::import("pymc");
    py::module_::import("cronos");
    py::module_::import("canon");
  }
  catch( py::error_already_set const& ){
    return;
  }
    
  m.doc() = "Python interface of library MAGNUS";

  mc_parest( m );

}


#include <pybind11/pybind11.h>

#include "indices.h"

namespace py = pybind11;

PYBIND11_PLUGIN(almo_tools_cxx) {

    py::module m("almo_tools_cxx");

    m.def("make_indices_ao", &libresponse::make_indices_ao, "");
    m.def("make_indices_mo_separate", &libresponse::make_indices_mo_separate, "");
    m.def("make_indices_mo_combined", &libresponse::make_indices_mo_combined, "");
    m.def("make_indices_mo_restricted", &libresponse::make_indices_mo_restricted, "");
    m.def("make_indices_mo_restricted_local_occ_all_virt", &libresponse::make_indices_mo_restricted_local_occ_all_virt, "");

    return m.ptr();
}

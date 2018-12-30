#include <pybind11/pybind11.h>

#include "pybind.h"

namespace libresponse {

type::indices_py py_make_indices_ao(pyarr_u nbasis_frgm)
{
    const arma::uvec nbasis_frgm_arma = py_to_uvec(nbasis_frgm);
    type::indices indices_ao = make_indices_ao(nbasis_frgm_arma);
    const size_t nfrgm = indices_ao.size();
    type::indices_py v;
    for (size_t i = 0; i < nfrgm; i++)
        v.push_back(uvec_to_py(indices_ao[i]));
    return v;
}

type::pair_indices_py py_make_indices_mo_separate_py(pyarr_u nocc_frgm, pyarr_u nvirt_frgm)
{
}

type::indices_py py_make_indices_mo_combined(pyarr_u nocc_frgm, pyarr_u nvirt_frgm)
{
}

pyarr_u py_make_indices_mo_restricted(pyarr_u nocc_frgm, pyarr_u nvirt_frgm)
{
}

type::indices_py py_make_indices_mo_restricted_local_occ_all_virt(pyarr_u nocc_frgm, pyarr_u nvirt_frgm)
{
}

} // namespace libresponse

namespace py = pybind11;


PYBIND11_PLUGIN(almo_tools_cxx) {

    py::module m("almo_tools_cxx");

    m.def("make_indices_ao", &libresponse::py_make_indices_ao, "");
    // m.def("make_indices_mo_separate", &libresponse::py_make_indices_mo_separate, "");
    // m.def("make_indices_mo_combined", &libresponse::py_make_indices_mo_combined, "");
    // m.def("make_indices_mo_restricted", &libresponse::py_make_indices_mo_restricted, "");
    // m.def("make_indices_mo_restricted_local_occ_all_virt", &libresponse::py_make_indices_mo_restricted_local_occ_all_virt, "");

    return m.ptr();
}

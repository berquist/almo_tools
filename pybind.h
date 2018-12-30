#include "indices.h"
#include "pyarma.h"

namespace libresponse {
namespace type {
typedef std::vector< pyarr_u > indices_py;
typedef std::pair< indices_py, indices_py > pair_indices_py;
typedef std::pair< pyarr_u, pyarr_u > pair_arma_py;
}

type::indices_py py_make_indices_ao(pyarr_u nbasis_frgm);
type::pair_indices_py py_make_indices_mo_separate(pyarr_u nocc_frgm, pyarr_u nvirt_frgm);
type::indices_py py_make_indices_mo_combined(pyarr_u nocc_frgm, pyarr_u nvirt_frgm);
pyarr_u py_make_indices_mo_restricted(pyarr_u nocc_frgm, pyarr_u nvirt_frgm);
type::indices_py py_make_indices_mo_restricted_local_occ_all_virt(pyarr_u nocc_frgm, pyarr_u nvirt_frgm);

} // namespace libresponse

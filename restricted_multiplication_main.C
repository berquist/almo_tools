#include "utils.h"
#include "indices.h"
#include "printing.h"

using namespace libresponse;

int main()
{

    arma::mat C;
    // assume that these are the projected ALMOs
    C.load("C.dat");
    arma::mat integrals;
    integrals.load("integrals.dat");

    arma::ivec nbasis_frgm_;
    nbasis_frgm_.load("nbasis_frgm.dat");
    arma::ivec norb_frgm_;
    norb_frgm_.load("norb_frgm.dat");
    arma::ivec nocc_frgm_;
    nocc_frgm_.load("nocc_frgm.dat");

    // size consistency checks
    if (nbasis_frgm_.n_elem != norb_frgm_.n_elem)
        throw 1;
    if (nocc_frgm_.n_elem != norb_frgm_.n_elem)
        throw 1;

    const arma::uvec nbasis_frgm = arma::conv_to<arma::uvec>::from(nbasis_frgm_);
    const arma::uvec norb_frgm = arma::conv_to<arma::uvec>::from(norb_frgm_);
    const arma::uvec nocc_frgm = arma::conv_to<arma::uvec>::from(nocc_frgm_);
    const arma::uvec nvirt_frgm = norb_frgm - nocc_frgm;

    const size_t nocc_tot = arma::accu(nocc_frgm);
    const size_t nvirt_tot = arma::accu(nvirt_frgm);
    const size_t norb_tot = arma::accu(norb_frgm);
    if ((nocc_tot + nvirt_tot) != norb_tot)
        throw 1;

    std::cout << dashes << std::endl;
    nbasis_frgm.print("nbasis_frgm");
    norb_frgm.print("norb_frgm");
    nocc_frgm.print("nocc_frgm");
    nvirt_frgm.print("nvirt_frgm");

    const size_t nfrgm = nbasis_frgm.n_elem;

    const type::indices indices_ao = make_indices_ao(nbasis_frgm);
    const type::pair_indices indices_mo_separate = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    const type::indices indices_mo_combined = make_indices_mo_combined(nocc_frgm, nvirt_frgm);
    const type::indices indices_mo_occ = indices_mo_separate.first;
    const type::indices indices_mo_virt = indices_mo_separate.second;

    std::cout << dashes << std::endl;
    std::cout << "indices_ao" << std::endl;
    std::cout << indices_ao << std::endl;
    std::cout << "indices_mo_separate" << std::endl;
    std::cout << indices_mo_separate << std::endl;
    std::cout << "indices_mo_combined" << std::endl;
    std::cout << indices_mo_combined << std::endl;
    std::cout << "indices_mo_occ" << std::endl;
    std::cout << indices_mo_occ << std::endl;
    std::cout << "indices_mo_virt" << std::endl;
    std::cout << indices_mo_virt << std::endl;

    std::cout << dashes << std::endl;

    // integrals.print("integrals");
    arma::mat integrals_masked;
    make_masked_mat(integrals_masked, integrals, indices_ao);
    // integrals_masked.print("integrals (AO-masked)");

    const arma::uvec indices_mo_restricted = make_indices_mo_restricted(nocc_frgm, nvirt_frgm);

    const type::indices indices_mo_restricted_local_occ_all_virt =      \
        make_indices_mo_restricted_local_occ_all_virt(nocc_frgm, nvirt_frgm);

    std::cout << "indices_mo_restricted" << std::endl;
    std::cout << indices_mo_restricted << std::endl;
    std::cout << "indices_mo_restricted_local_occ_all_virt" << std::endl;
    std::cout << indices_mo_restricted_local_occ_all_virt << std::endl;

    std::cout << dashes << std::endl;

    // C.print("C");
    const arma::mat integrals_ov = C.cols(0, nocc_tot - 1).t() * integrals * C.cols(nocc_tot, norb_tot - 1);
    // integrals_ov.print("integrals_ov");

    arma::vec integrals_ov_vec(integrals_ov.n_elem);
    repack_matrix_to_vector(integrals_ov_vec, integrals_ov);
    // integrals_ov_vec.print("integrals_ov_vec");
    // We don't want to select only the allowed indices, but have the
    // full set of indices and zero out the disallowed indices.
    // integrals_ov_vec(indices_mo_restricted).print("integrals_ov_vec (masked)");
    arma::vec integrals_ov_vec_masked(integrals_ov_vec.n_elem, arma::fill::zeros);
    integrals_ov_vec_masked(indices_mo_restricted) = integrals_ov_vec(indices_mo_restricted);
    // integrals_ov_vec_masked.print("integrals_ov_vec (masked)");

    // // std::cout << "Testing non-contiguous view along rows, 1 selected column." << std::endl;
    // arma::imat m(nocc_tot * nvirt_tot, 4, arma::fill::zeros);
    // for (size_t r = 0; r < m.n_rows; r++)
    //     for (size_t c = 0; c < m.n_cols; c++)
    //         m(r, c) = (10 * c) + r;

    // m.print("m");
    // // well this is dumb
    // arma::uvec c(1);
    // c(0) = 2;
    // // These aren't actually masked, it's just the selected elements.
    // // m(indices_mo_restricted, c).print("m (masked, 3rd column)");
    // // m.submat(indices_mo_restricted, c).print("m (masked, 3rd column)");

    // arma::imat m_masked(m.n_rows, m.n_cols, arma::fill::zeros);
    // m_masked.submat(indices_mo_restricted, range(m_masked.n_cols)) = m.submat(indices_mo_restricted, range(m_masked.n_cols));
    // m_masked.print("m (masked, all columns)");

    return 0;

}

#include "utils.h"
#include "indices.h"
#include "printing.h"

using namespace libresponse;

int main()
{

    const int start = 7;
    const int stop = 20;
    const int step = 2;

    const arma::uvec r3 = range(start, stop, step);
    const arma::uvec r2 = range(start, stop);
    const arma::uvec r1 = range(stop);
    const arma::uvec r0 = range(3, 10, 3);

    // r3.print("r3");
    // r2.print("r2");
    // r1.print("r1");
    // r0.print("r0");

    r3.save("r3_7_20_2.dat", arma::arma_ascii);
    r2.save("r2_7_20.dat", arma::arma_ascii);
    r1.save("r1_20.dat", arma::arma_ascii);
    r0.save("r3_3_10_3.dat", arma::arma_ascii);

    arma::ivec nbasis_frgm_;
    arma::ivec norb_frgm_;
    arma::ivec nocc_frgm_;
    nbasis_frgm_.load("nbasis_frgm.dat");
    norb_frgm_.load("norb_frgm.dat");
    nocc_frgm_.load("nocc_frgm.dat");

    // size consistency checks
    if (nbasis_frgm_.n_elem != norb_frgm_.n_elem)
        throw std::runtime_error("nbasis_frgm_.n_elem != norb_frgm_.n_elem");
    if (nocc_frgm_.n_elem != norb_frgm_.n_elem)
        throw std::runtime_error("nocc_frgm_.n_elem != norb_frgm_.n_elem");

    const arma::uvec nbasis_frgm = arma::conv_to<arma::uvec>::from(nbasis_frgm_);
    const arma::uvec norb_frgm = arma::conv_to<arma::uvec>::from(norb_frgm_);
    const arma::uvec nocc_frgm = arma::conv_to<arma::uvec>::from(nocc_frgm_);
    const arma::uvec nvirt_frgm = norb_frgm - nocc_frgm;

    const size_t nocc_tot = arma::accu(nocc_frgm);
    const size_t nvirt_tot = arma::accu(nvirt_frgm);
    const size_t norb_tot = arma::accu(norb_frgm);
    if ((nocc_tot + nvirt_tot) != norb_tot)
        throw std::runtime_error("(nocc_tot + nvirt_tot) != norb_tot");

    const size_t nfrgm = nbasis_frgm.n_elem;

    const type::indices indices_ao = make_indices_ao(nbasis_frgm);
    const type::pair_indices indices_mo_separate = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    const type::indices indices_mo_combined = make_indices_mo_combined(nocc_frgm, nvirt_frgm);
    const type::indices indices_mo_occ = indices_mo_separate.first;
    const type::indices indices_mo_virt = indices_mo_separate.second;
    const arma::uvec indices_mo_restricted = make_indices_mo_restricted(nocc_frgm, nvirt_frgm);
    const type::indices indices_mo_restricted_local_occ_all_virt =      \
        make_indices_mo_restricted_local_occ_all_virt(nocc_frgm, nvirt_frgm);

    for (size_t i = 0; i < nfrgm; i++) {
        indices_ao.at(i).save("indices_ao_" + SSTR(i) + ".dat", arma::arma_ascii);
        indices_mo_combined.at(i).save("indices_mo_combined_" + SSTR(i) + ".dat", arma::arma_ascii);
        indices_mo_occ.at(i).save("indices_mo_occ_" + SSTR(i) + ".dat", arma::arma_ascii);
        indices_mo_virt.at(i).save("indices_mo_virt_" + SSTR(i) + ".dat", arma::arma_ascii);
        indices_mo_restricted_local_occ_all_virt.at(i).save("indices_mo_restricted_local_occ_all_virt_" + SSTR(i) + ".dat", arma::arma_ascii);
    }
    indices_mo_restricted.save("indices_mo_restricted_local_occ_local_virt.dat", arma::arma_ascii);

    return 0;
}

#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

#include <armadillo>

typedef std::vector< arma::ivec > indices;
typedef std::pair<indices, indices> mo_indices;

arma::ivec range(int start, int stop, int step)
{

    return arma::conv_to<arma::ivec>::from(arma::regspace(start, step, stop - 1));

}

arma::ivec range(int start, int stop)
{

    return range(start, stop, 1);

}

arma::ivec range(int stop)
{

    return range(0, stop, 1);

}

indices make_indices_ao(arma::ivec &nbasis_frgm)
{

    const size_t nfrgm = nbasis_frgm.n_elem;
    indices v;
    size_t start, stop;
    for (size_t i = 0; i < nfrgm; i++) {
        if (i == 0)
            start = 0;
        else
            start = arma::accu(nbasis_frgm.subvec(0, i - 1));
        stop = start + nbasis_frgm(i);
        v.push_back(range(start, stop));
    }

    return v;

}

mo_indices make_indices_mo_separate(arma::ivec &nocc_frgm, arma::ivec &nvirt_frgm)
{

    assert(nocc_frgm.n_elem == nvirt_frgm.n_elem);
    const size_t nocc = arma::accu(nocc_frgm);
    const size_t nfrgm = nocc_frgm.n_elem;
    indices v_occ, v_virt;
    size_t start_occ, stop_occ, start_virt, stop_virt;
    for (size_t i = 0; i < nfrgm; i++) {
        if (i == 0) {
            start_occ = 0;
            start_virt = nocc;
        } else {
            start_occ = arma::accu(nocc_frgm.subvec(0, i - 1));
            start_virt = nocc + arma::accu(nvirt_frgm.subvec(0, i - 1));
        }
        stop_occ = start_occ + nocc_frgm(i);
        stop_virt = start_virt + nvirt_frgm(i);
        v_occ.push_back(range(start_occ, stop_occ));
        v_virt.push_back(range(start_virt, stop_virt));
    }

    mo_indices p = std::make_pair(v_occ, v_virt);

    return p;

}

int main()
{

    arma::mat C;
    // assume that these are the projected ALMOs
    C.load("C.dat");
    arma::mat integrals;
    integrals.load("integrals.dat");

    arma::ivec nbasis_frgm;
    nbasis_frgm.load("nbasis_frgm.dat");
    arma::ivec norb_frgm;
    norb_frgm.load("norb_frgm.dat");
    arma::ivec nocc_frgm;
    nocc_frgm.load("nocc_frgm.dat");
    arma::ivec nvirt_frgm = norb_frgm - nocc_frgm;

    nbasis_frgm.print("nbasis_frgm");
    norb_frgm.print("norb_frgm");
    nocc_frgm.print("nocc_frgm");
    nvirt_frgm.print("nvirt_frgm");

    indices indices_ao = make_indices_ao(nbasis_frgm);
    mo_indices indices_mo_separate = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    indices indices_mo_occ = indices_mo_separate.first;
    indices indices_mo_virt = indices_mo_separate.second;

    C.print("C");

    return 0;

}

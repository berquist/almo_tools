#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <armadillo>

typedef std::vector< arma::uvec > indices;
typedef std::pair< indices, indices > pair_indices;
// typedef std::pair< std::vector<size_t>, std::vector<size_t> > pair_std;
typedef std::pair< arma::uvec, arma::uvec > pair_arma;

// maybe this should just take size_t to avoid exception handling
arma::uvec range(int start, int stop, int step)
{

    if (start < 0 || stop < 0)
        throw std::invalid_argument("negative numbers meaningless for array indexing");
    if ((stop - start) < 0)
        throw std::domain_error("no reverse ranges");
    if (step < 1)
        throw std::domain_error("???");

    return arma::conv_to<arma::uvec>::from(arma::regspace(start, step, stop - 1));

}

arma::uvec range(int start, int stop)
{

    return range(start, stop, 1);

}

arma::uvec range(int stop)
{

    return range(0, stop, 1);

}

// How to enforce that we expect T to be some kind of Armadillo
// vector?
template<typename T>
T join(const T &a1, const T &a2)
{

    const size_t l1 = a1.n_elem;
    const size_t l2 = a2.n_elem;
    const size_t l3 = l1 + l2;

    T a3(l3);

    a3.subvec(0, l1 - 1) = a1;
    a3.subvec(l1, l2 - 1) = a2;

    return a3;

}

template arma::uvec join<arma::uvec>(const arma::uvec &a1, const arma::uvec &a2);
template arma::ivec join<arma::ivec>(const arma::ivec &a1, const arma::ivec &a2);

template <typename T>
std::vector<T> set_to_ordered_vector(const std::set<T> &s)
{

    std::vector<T> v(s.begin(), s.end());
    std::sort(v.begin(), v.end());

    return v;

}

template std::vector<size_t> set_to_ordered_vector(const std::set<size_t> &s);

indices make_indices_ao(const arma::ivec &nbasis_frgm)
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

pair_indices make_indices_mo_separate(const arma::ivec &nocc_frgm, const arma::ivec &nvirt_frgm)
{

    if (nocc_frgm.n_elem != nvirt_frgm.n_elem)
        throw 1;
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

    pair_indices p = std::make_pair(v_occ, v_virt);

    return p;

}

indices make_indices_mo_combined(const arma::ivec &nocc_frgm, const arma::ivec &nvirt_frgm)
{

    indices v;
    const pair_indices p = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    const size_t nfrgm = nocc_frgm.n_elem;
    for (size_t i = 0; i < nfrgm; i++) {
        v.push_back(join(p.first[i], p.second[i]));
    }

    return v;

}

// indices make_opposite_indices()
// {

//     indices v;
//     return v;

// }

pair_arma make_indices_from_mask(const arma::umat &mask, int mask_val_for_return)
{

    // blow up to avoid weird casting tricks
    if (mask_val_for_return < 0 || mask_val_for_return > 1)
        throw 1;

    std::set<size_t> sr, sc;

    for (size_t i = 0; i < mask.n_rows; i++) {
        for (size_t j = 0; j < mask.n_cols; j++) {
            if (mask(i, j) == mask_val_for_return) {
                sr.insert(i);
                sc.insert(j);
            }
        }
    }

    // transfer the set contents to vectors, which are then sorted in
    // place
    std::vector<size_t> vr = set_to_ordered_vector(sr);
    std::vector<size_t> vc = set_to_ordered_vector(sc);

    arma::uvec ar = arma::conv_to<arma::uvec>::from(vr);
    arma::uvec ac = arma::conv_to<arma::uvec>::from(vc);

    return std::make_pair(ar, ac);

}

void make_masked_mat(arma::mat &mm, const arma::mat &m, const indices &idxs, double fill_value = 0.0)
{

    if (idxs.empty())
        throw 1;

    mm.set_size(arma::size(m));
    mm.fill(fill_value);

    const size_t nblocks = idxs.size();

    for (size_t i = 0; i < nblocks; i++) {
        mm.submat(idxs[i], idxs[i]) = m.submat(idxs[i], idxs[i]);
    }

    return;

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

    const size_t nfrgm = nbasis_frgm.n_elem;

    indices indices_ao = make_indices_ao(nbasis_frgm);
    pair_indices indices_mo_separate = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    indices indices_mo_occ = indices_mo_separate.first;
    indices indices_mo_virt = indices_mo_separate.second;

    integrals.print("integrals");
    arma::mat integrals_masked;
    make_masked_mat(integrals_masked, integrals, indices_ao);
    integrals_masked.print("integrals (AO-masked)");

    return 0;

}

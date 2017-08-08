#include <cassert>
#include <stdexcept>
#include "utils.h"

// For to_bool().
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cctype>

void repack_matrix_to_vector(arma::vec &v, const arma::mat &m)
{

    const size_t d1 = m.n_cols;
    const size_t d2 = m.n_rows;
    const size_t dv = d1 * d2;
    assert(v.n_elem == dv);

    // If not copying the memory, this is how to do it safely.
    // size_t ia;
    // for (size_t i = 0; i < d1; i++) {
    //     for (size_t a = 0; a < d2; a++) {
    //         ia = i*d2 + a;
    //         v(ia) = m(a, i);
    //     }
    // }

    memcpy(v.memptr(), m.memptr(), dv * sizeof(double));

    return;

}

void repack_vector_to_matrix(arma::mat &m, const arma::vec &v)
{

    const size_t d1 = m.n_cols;
    const size_t d2 = m.n_rows;
    const size_t dv = d1 * d2;
    assert(v.n_elem == dv);

    // If not copying the memory, this is how to do it safely.
    // size_t ia;
    // for (size_t i = 0; i < d1; i++) {
    //     for (size_t a = 0; a < d2; a++) {
    //         ia = i*d2 + a;
    //         m(a, i) = v(ia);
    //     }
    // }

    memcpy(m.memptr(), v.memptr(), dv * sizeof(double));

    return;
}

// TODO Do these need Doxygen comments, or is the template definition
// enough?
template double rmsd<arma::vec>(const arma::vec &T_new, const arma::vec &T_old);
template double rmsd<arma::mat>(const arma::mat &T_new, const arma::mat &T_old);

template bool is_close<arma::vec>(const arma::vec &X, const arma::vec &Y, double tol, double &current_norm);
template bool is_close<arma::mat>(const arma::mat &X, const arma::mat &Y, double tol, double &current_norm);

arma::cube concatenate_cubes(const std::vector<arma::cube> &v)
{

    const size_t n_cubes = v.size();

    if (n_cubes == 0) {
        throw 1;
    } else if (n_cubes == 1) {
        return v[0];
    } else {

        // Do some size checking.
        const size_t n_rows = v[0].n_rows;
        const size_t n_cols = v[0].n_cols;
        for (size_t iv = 0; iv < n_cubes; iv++) {
            if (v[iv].n_rows != n_rows) {
                std::cout << "n_rows inconsistent in concatenate_cubes" << std::endl;
                throw 1;
            }
            if (v[iv].n_cols != n_cols) {
                std::cout << "n_cols inconsistent in concatenate_cubes" << std::endl;
                throw 1;
            }
        }

        // Figure out how many slices in total are needed.
        size_t n_slices_total = 0;
        for (size_t iv = 0; iv < n_cubes; iv++) {
            n_slices_total += v[iv].n_slices;
        }

        arma::cube c(n_rows, n_cols, n_slices_total);

        // Place the original cubes in the right slices for the final
        // cube.
        size_t idx_slice_start = 0;
        for (size_t iv = 0; iv < n_cubes; iv++) {
            c.slices(idx_slice_start, idx_slice_start + v[iv].n_slices - 1) = v[iv];
            idx_slice_start += v[iv].n_slices;
        }

        return c;

    }

}

int matsym(const arma::mat &amat)
{

    const double thrzer = 1.0e-14;

    assert(amat.n_rows == amat.n_cols);

    const size_t n = amat.n_rows;

    int isym = 1;
    int iasym = 2;

    double amats, amata;

    for (size_t j = 0; j < n; j++) {
        // The +1 is so the diagonal elements are checked.
        for (size_t i = 0; i < j+1; i++) {
            amats = std::abs(amat(i, j) + amat(j, i));
            amata = std::abs(amat(i, j) - amat(j, i));
            if (amats > thrzer)
                iasym = 0;
            if (amata > thrzer)
                isym = 0;
        }
    }

    return isym + iasym;

}

void join_vector(arma::vec &vc, const arma::vec &v1, const arma::vec &v2)
{

    const size_t lc = vc.n_elem;
    const size_t l1 = v1.n_elem;
    const size_t l2 = v2.n_elem;
    assert(lc == (l1 + l2));

    vc.subvec(0, l1 - 1) = v1;
    vc.subvec(l1, lc - 1) = v2;

    return;

}

void split_vector(const arma::vec &vc, arma::vec &v1, arma::vec &v2)
{

    const size_t lc = vc.n_elem;
    const size_t l1 = v1.n_elem;
    const size_t l2 = v2.n_elem;
    assert(lc == (l1 + l2));

    v1 = vc.subvec(0, l1 - 1);
    v2 = vc.subvec(l1, lc - 1);

    return;

}

// need a version check for compile-time definition (?); this is only
// for old versions of Armadillo.
namespace arma {
    arma::vec regspace(int start, int delta, int end) {
        std::vector<double> v;
        double e = start;
        // this will *not* do everything that arma::regspace does!
        while (e <= end) {
            v.push_back(e);
            e += delta;
        }
        return arma::conv_to<arma::vec>::from(v);
    }
}

// maybe this should just take size_t to avoid exception handling
// give me my Python dangit
arma::uvec range(int start, int stop, int step)
{

    if (start < 0 || stop < 0)
        throw std::invalid_argument("negative numbers meaningless for array indexing; no wraparound available");
    if ((stop - start) < 0)
        throw std::domain_error("no reverse ranges");
    if (step < 1)
        throw std::domain_error("???");

    return arma::conv_to<arma::uvec>::from(arma::regspace(start, step, stop - 1));

}

arma::uvec range(int start, int stop) {
    return range(start, stop, 1);
}

arma::uvec range(int stop) {
    return range(0, stop, 1);
}

template arma::uvec join<arma::uvec>(const arma::uvec &a1, const arma::uvec &a2);

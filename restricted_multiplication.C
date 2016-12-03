#include <armadillo>
#include <cassert>
#include <set>
#include <vector>

arma::uvec range(int start, int stop, int step);

arma::uvec range(int start, int stop);

arma::uvec range(int stop);

template<typename T>
T join(const T &a1, const T &a2)
{

    const size_t l1 = a1.n_elem;
    const size_t l2 = a2.n_elem;
    const size_t l3 = l1 + l2;

    T a3(l3);

    a3.subvec(0, l1 - 1) = a1;
    a3.subvec(l1, l3 - 1) = a2;

    return a3;

}
// TODO write operator<< for each
typedef std::vector< arma::uvec > indices;
typedef std::pair< indices, indices > pair_indices;
// typedef std::pair< std::vector<size_t>, std::vector<size_t> > pair_std;
typedef std::pair< arma::uvec, arma::uvec > pair_arma;

indices make_indices_ao(const arma::ivec &nbasis_frgm);

pair_indices make_indices_mo_separate(const arma::ivec &nocc_frgm, const arma::ivec &nvirt_frgm);

indices make_indices_mo_combined(const arma::ivec &nocc_frgm, const arma::ivec &nvirt_frgm);

void make_masked_mat(arma::mat &mm, const arma::mat &m, const indices &idxs, double fill_value = 0.0);

void make_masked_mat(arma::cube &mc, const arma::cube &c, const indices &idxs, double fill_value = 0.0);

void make_masked_mat(arma::mat &mm, const arma::mat &m, const indices &idxs_rows, const indices &idxs_cols, double fill_value);

void make_masked_mat(arma::cube &mc, const arma::cube &c, const indices &idxs_rows, const indices &idxs_cols, double fill_value);

//////////

template arma::uvec join<arma::uvec>(const arma::uvec &a1, const arma::uvec &a2);
template arma::ivec join<arma::ivec>(const arma::ivec &a1, const arma::ivec &a2);

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

typedef std::pair<size_t, size_t> pair;
typedef std::set<pair> pairs;
typedef std::set<pair>::const_iterator pairs_iterator;

std::ostream& operator<<(std::ostream& os, const pair& p)
{

    os << "(" << p.first << ", " << p.second << ")";

    return os;

}

std::ostream& operator<<(std::ostream& os, const pairs& ps)
{

    pairs_iterator it;

    for (it = ps.begin(); it != ps.end(); ++it) {
        os << *it << std::endl;
    }

    return os;

}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {

    typename std::vector<T>::const_iterator it;

    os << "[";
    for (it = v.begin(); it != v.end(); ++it) {
        if (it == v.end() - 1)
            os << *it;
        else
            os << *it << ", ";
    }
    os << "]" << std::endl;

    return os;

}

// Don't need explicit instantiation?

arma::uvec make_indices_mo_restricted(const arma::ivec &nocc_frgm, const arma::ivec &nvirt_frgm)
{

    size_t nocc_tot = arma::accu(nocc_frgm);
    size_t nvirt_tot = arma::accu(nvirt_frgm);
    size_t norb_tot = nocc_tot + nvirt_tot;
    size_t nfrgm = nocc_frgm.n_elem;

    pair_indices p = make_indices_mo_separate(nocc_frgm, nvirt_frgm);
    pairs pairs_all, pairs_good, pairs_bad;

    // Collect all possible occ-virt excitation pairs within each
    // fragment.
    for (size_t f = 0; f < nfrgm; f++) {
        for (size_t i = 0; i < p.first[f].n_elem; i++) {
            for (size_t a = 0; a < p.second[f].n_elem; a++) {
                pairs_good.insert(std::make_pair(p.first[f](i), p.second[f](a)));
            }
        }
    }

    // Collect all possible occ-virt excitation pairs for the
    // supersystem.
    for (size_t i = 0; i < nocc_tot; i++) {
        for (size_t a = nocc_tot; a < norb_tot; a++) {
            pairs_all.insert(std::make_pair(i, a));
        }
    }

    // All the "bad" (disallowed) pairs are the difference between the
    // supersystem and the fragment-localized pairs.

    // We do this by iterating over the set of all possible pairs, and
    // ones that are not members of the restricted set are added to
    // the disallowed set.

    pairs_iterator it_all;

    for (it_all = pairs_all.begin(); it_all != pairs_all.end(); ++it_all) {
        if (pairs_good.count(*it_all) == 0)
            pairs_bad.insert(*it_all);
    }

    if ((pairs_good.size() + pairs_bad.size()) != pairs_all.size())
        throw 1;

    // Now, convert all of the pairs to compound indices.
    std::vector<size_t> v_all, v_good, v_bad;
    size_t i = 0;
    for (it_all = pairs_all.begin(); it_all != pairs_all.end(); ++it_all) {
        v_all.push_back(i);
        if (pairs_good.count(*it_all) > 0)
            v_good.push_back(i);
        if (pairs_bad.count(*it_all) > 0)
            v_bad.push_back(i);
        i++;
    }

    // std::cout << "pairs_all" << std::endl;
    // std::cout << pairs_all << std::endl;
    // std::cout << "pairs_good" << std::endl;
    // std::cout << pairs_good << std::endl;
    // std::cout << "pairs_bad" << std::endl;
    // std::cout << pairs_bad << std::endl;
    // std::cout << "v_all" << std::endl;
    // std::cout << v_all << std::endl;
    // std::cout << "v_good" << std::endl;
    // std::cout << v_good << std::endl;
    // std::cout << "v_bad" << std::endl;
    // std::cout << v_bad << std::endl;

    return arma::conv_to<arma::uvec>::from(v_good);
}

// fill then copy, rather than copy then fill
void make_masked_mat(arma::mat &mm, const arma::mat &m, const indices &idxs, double fill_value)
{

    if (idxs.empty())
        throw 1;

    mm.set_size(m.n_rows, m.n_cols);
    mm.fill(fill_value);

    const size_t nblocks = idxs.size();

    for (size_t i = 0; i < nblocks; i++) {
        // submat takes 2 uvecs and automatically forms the correct
        // outer product between them for indexing
        mm.submat(idxs[i], idxs[i]) = m.submat(idxs[i], idxs[i]);
    }

    return;

}

void make_masked_mat(arma::cube &mc, const arma::cube &c, const indices &idxs, double fill_value)
{

    mc.set_size(c.n_rows, c.n_cols, c.n_slices);

    for (size_t ns = 0; ns < c.n_slices; ns++) {
        make_masked_mat(mc.slice(ns), c.slice(ns), idxs, fill_value);
    }

    return;

}

void make_masked_mat(arma::mat &mm, const arma::mat &m, const indices &idxs_rows, const indices &idxs_cols, double fill_value)
{

    if (idxs_rows.empty() || idxs_cols.empty())
        throw 1;

    mm.set_size(m.n_rows, m.n_cols);
    mm.fill(fill_value);

    // this is an artificial constraint, there's probably a better way
    // of doing this
    if (idxs_rows.size() != idxs_cols.size())
        throw 1;

    const size_t nblocks = idxs_rows.size();

    for (size_t i = 0; i < nblocks; i++) {
        // submat takes 2 uvecs and automatically forms the correct
        // outer product between them for indexing
        mm.submat(idxs_rows[i], idxs_cols[i]) = m.submat(idxs_rows[i], idxs_cols[i]);
    }

    return;

}

void make_masked_mat(arma::cube &mc, const arma::cube &c, const indices &idxs_rows, const indices &idxs_cols, double fill_value)
{

    mc.set_size(c.n_rows, c.n_cols, c.n_slices);

    for (size_t ns = 0; ns < c.n_slices; ns++) {
        make_masked_mat(mc.slice(ns), c.slice(ns), idxs_rows, idxs_cols, fill_value);
    }

    return;

}

void repack_matrix_to_vector(arma::vec &v, const arma::mat &m) {

    size_t d1 = m.n_rows;
    size_t d2 = m.n_cols;
    size_t dv = d1 * d2;
    assert(v.n_elem == dv);

    size_t ia;
    for (size_t i = 0; i < d1; i++) {
        for (size_t a = 0; a < d2; a++) {
            ia = i*d2 + a;
            v(ia) = m(i, a);
        }
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

    const size_t nocc_tot = arma::accu(nocc_frgm);
    const size_t nvirt_tot = arma::accu(nvirt_frgm);
    const size_t norb_tot = arma::accu(norb_frgm);
    if ((nocc_tot + nvirt_tot) != norb_tot)
        throw 1;

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

    arma::uvec indices_mo_restricted = make_indices_mo_restricted(nocc_frgm, nvirt_frgm);

    C.print("C");
    arma::mat integrals_ov = C.cols(0, nocc_tot - 1).t() * integrals * C.cols(nocc_tot, norb_tot - 1);
    integrals_ov.print("integrals_ov");

    arma::vec integrals_ov_vec(integrals_ov.n_elem);
    repack_matrix_to_vector(integrals_ov_vec, integrals_ov);
    integrals_ov_vec.print("integrals_ov_vec");
    // We don't want to select only the allowed indices, but have the
    // full set of indices and zero out the disallowed indices.
    // integrals_ov_vec(indices_mo_restricted).print("integrals_ov_vec (masked)");
    arma::vec integrals_ov_vec_masked(integrals_ov_vec.n_elem, arma::fill::zeros);
    integrals_ov_vec_masked(indices_mo_restricted) = integrals_ov_vec(indices_mo_restricted);
    integrals_ov_vec_masked.print("integrals_ov_vec (masked)");

    return 0;

}

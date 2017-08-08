#ifndef LIBRESPONSE_UTILS_H_
#define LIBRESPONSE_UTILS_H_

/*!
 * @file
 *
 * @brief Utility functions.
 */

#include <armadillo>
#include <vector>

static const std::string dots(77, '.');
static const std::string dashes(77, '-');
static const std::string equals(77, '=');
static const std::string lcarats(77, '<');
static const std::string rcarats(77, '>');

/*!
 * Convert an integer to a string.
 */
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

/*!
 * Repack an Armadillo matrix of shape \f$(i,j)\f$ into a vector of
 * shape \f$(i*j)\f$, where \f$i\f$ is the fast index.
 *
 * @param[out] &v matrix packed into a vector
 * @param[in] &m matrix to be packed
 */
void repack_matrix_to_vector(arma::vec &v, const arma::mat &m);

/*!
 * Repack an Armadillo vector of shape \f$(i*j)\f$, where \f$i\f$ is
 * the fast index, into an Armadillo matrix of shape \f$(i,j)\f$.
 *
 * @param[out] &m unpacked matrix
 * @param[in] &v vector to be unpacked
 */
void repack_vector_to_matrix(arma::mat &m, const arma::vec &v);

/*!
 * Calculate the root-mean-square deviation between two
 * identically-typed and identically-sized Armadillo containers,
 * element-wise.
 *
 * @param[in] &T_new Armadillo vector, matrix, or cube
 * @param[out] &T_old Armadillo vector, matrix, or cube
 *
 * @return rmsd value
 */
template <typename T>
double rmsd(const T &T_new, const T &T_old) {
    return sqrt(arma::accu(arma::pow((T_new - T_old), 2)));
}

/*!
 * Check for elementwise closeness between two
 * indentically-typed Armadillo objects.
 *
 * @param[in] &X first Armadillo object
 * @param[in] &Y second Armadillo object
 * @param[in] tol desired tolerance
 * @param[out] &current_norm current value of the maximum norm
 *
 * @return Whether or not the two Armadillo objects are "close" within
 * the given tolerance.
 */
template <typename T>
bool is_close(const T &X, const T &Y, double tol, double &current_norm)
{
    bool close(false);
    current_norm = arma::norm(X - Y, "inf");
    if (current_norm < tol)
    {
        close = true;
    }
    return close;
}

/*!
 * Concatenate or flatten a vector of Armadillo cubes into a
 * single cube.
 *
 * The matrix shape must be identical across all cubes.
 *
 * @param[in] &v vector of cubes to be concatenated
 *
 * @return A cube whose slices are in the same order as the input
 * vector of cubes.
 */
arma::cube concatenate_cubes(const std::vector<arma::cube> &v);

/*!
 * Check for matrix (anti,un)symmetry.
 *
 * This function returns
 * 1 if the matrix is symmetric to threshold THRZER
 * 2 if the matrix is antisymmetric to threshold THRZER
 * 3 if all elements are below THRZER
 * 0 otherwise (the matrix is unsymmetric about the diagonal)
 *
 * @param[in] &amat Armadillo matrix being tested for (anti)symmetry
 *
 * @return 1, 2, 3, or 0
 *
 * @sa DALTON/gp/gphjj.F/MATSYM, DALTON/include/thrzer.h
 */
int matsym(const arma::mat &amat);

/*!
 * Combine two vectors into one.
 *
 * The total length of vc must be equal to the sum of the lengths of
 * v1 and v2.
 *
 * @param[out] &vc combined vector
 * @param[in] &v1 first vector
 * @param[in] &v2 second vector
 */
void join_vector(arma::vec &vc, const arma::vec &v1, const arma::vec &v2);

/*!
 * Split one vector into two.
 *
 * The total length of vc must be equal to the sum of the lengths of
 * v1 and v2.
 *
 * @param[in] &vc combined vector
 * @param[out] &v1 first vector
 * @param[out] &v2 second vector
 */
void split_vector(const arma::vec &vc, arma::vec &v1, arma::vec &v2);

arma::uvec range(int start, int stop, int step);

arma::uvec range(int start, int stop);

arma::uvec range(int stop);

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
    a3.subvec(l1, l3 - 1) = a2;

    return a3;

}

#endif // LIBRESPONSE_UTILS_H_

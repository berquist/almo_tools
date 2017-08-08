#ifndef TYPEDEFS_H_
#define TYPEDEFS_H_

/*!
 * @brief Typedefs for indexing-related quantities.
 *
 * @file
 */

#include <armadillo>
#include <set>
#include <vector>

typedef std::vector< arma::uvec > indices;
typedef std::pair< indices, indices > pair_indices;
// typedef std::pair< std::vector<size_t>, std::vector<size_t> > pair_std;
typedef std::pair< arma::uvec, arma::uvec > pair_arma;

typedef std::pair< size_t, size_t > pair;
// The reason for using a set rather than a vector is the ability to
// iterate over a set.
typedef std::set< pair > pairs;
typedef std::set< pair >::const_iterator pairs_iterator;

#endif // TYPEDEFS_H_

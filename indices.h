#ifndef LIBRESPONSE_INDICES_H_
#define LIBRESPONSE_INDICES_H_

/*!
 * @brief
 *
 * @file
 */

#include "typedefs.h"
#include "printing.h"

namespace libresponse {

/*!
 * Given the number of AOs/basis functions per fragment, produce the
 * indices for AOs that span each fragment.
 */
type::indices make_indices_ao(const arma::uvec &nbasis_frgm);

/*!
 *
 */
type::pair_indices make_indices_mo_separate(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm);

/*!
 *
 */
type::indices make_indices_mo_combined(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm);

/*!
 *
 */
arma::uvec make_indices_mo_restricted(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm);

type::indices make_indices_mo_restricted_local_occ_all_virt(const arma::uvec &nocc_frgm, const arma::uvec &nvirt_frgm);

/*!
 *
 */
void make_masked_mat(arma::mat &mm, const arma::mat &m, const type::indices &idxs, double fill_value = 0.0);

/*!
 *
 */
void make_masked_mat(arma::cube &mc, const arma::cube &c, const type::indices &idxs, double fill_value = 0.0);

/*!
 *
 */
void make_masked_mat(arma::mat &mm, const arma::mat &m, const type::indices &idxs_rows, const type::indices &idxs_cols, double fill_value);

/*!
 *
 */
void make_masked_mat(arma::cube &mc, const arma::cube &c, const type::indices &idxs_rows, const type::indices &idxs_cols, double fill_value);

} // namespace libresponse

#endif // LIBRESPONSE_INDICES_H

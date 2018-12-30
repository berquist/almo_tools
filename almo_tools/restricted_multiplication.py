from __future__ import print_function

import os.path

import numpy as np

from functools import reduce

from scripts.read_arma_mat import read_arma_mat_ascii

from almo_tools.fragment_ia_ranges import form_indices_orbwin
from almo_tools.fragment_ia_ranges import repack_matrix_to_vector_slow as repack_matrix_to_vector


def product(l):
    """Calculate the product of all elements in an iterable.
    """
    return reduce(lambda x, y: x*y, l)


def make_indices_ao(nbasis_frgm):
    """From the number of basis functions present on each fragment, create
    a list of arrays where each array will index into a supermatrix for
    that fragment based upon the ordering of atomic orbitals (AOs).

    The order of basis functions (AOs) is one fragment after another.
    """
    l = []
    nfrgm = len(nbasis_frgm)
    for i in range(nfrgm):
        # print(sum(nbasis_frgm[:i]), sum(nbasis_frgm[:i]) + nbasis_frgm[i])
        indices = list(range(sum(nbasis_frgm[:i]),
                             sum(nbasis_frgm[:i]) + nbasis_frgm[i]))
        l.append(np.array(indices))
    return l


def make_indices_mo_separate(nocc_frgm, nvirt_frgm):
    """From the number of occupied and virtual orbitals present on each
    fragment, create two lists of arrays where each array will index
    into a supermatrix for that fragment based upon the ordering of
    molecular orbitals (MOs).

    One list is for the occupied MOs and the other is for
    unoccupied/virtual MOs.

    The order of ALMO coefficients is [o1, o2, ..., on, v1, v2, ...,
    vn]; all the occupieds come before all the virtuals.

    Returns indices like [o1, o2, ...] and [v1, v2, ...], where each
    list element is an array.

    """
    assert nocc_frgm.shape == nvirt_frgm.shape
    nocc = sum(nocc_frgm)
    l_occ = []
    l_virt = []
    nfrgm = len(nocc_frgm)
    for i in range(nfrgm):
        indices_occ = list(range(sum(nocc_frgm[:i]),
                                 sum(nocc_frgm[:i]) + nocc_frgm[i]))
        indices_virt = list(range(nocc + sum(nvirt_frgm[:i]),
                                  nocc + sum(nvirt_frgm[:i]) + nvirt_frgm[i]))
        l_occ.append(np.array(indices_occ))
        l_virt.append(np.array(indices_virt))
    return l_occ, l_virt


def make_indices_mo_combined(nocc_frgm, nvirt_frgm):
    """From the number of occupied and virtual orbitals present on each
    fragment, create one list of arrays ...

    The order of ALMO coefficients is [o1, o2, ..., on, v1, v2, ...,
    vn]; all the occupieds come before all the virtuals.

    Returns indices like [o1 + v1, o2 + v2, ...], where each list
    element is an array.

    """
    indices_mo_occ, indices_mo_virt = make_indices_mo_separate(nocc_frgm, nvirt_frgm)
    l = []
    nfrgm = len(nocc_frgm)
    for i in range(nfrgm):
        indices_combined = np.concatenate((indices_mo_occ[i], indices_mo_virt[i]))
        l.append(indices_combined)
    return l


def make_indices_mo_restricted(nocc_frgm, nvirt_frgm):
    """Form index-restricted excitations, that is, allow only
    occupied-virtual excitations within individual fragments.

    """
    assert nocc_frgm.shape == nvirt_frgm.shape
    assert len(nocc_frgm.shape) == 1
    nfrgm = nocc_frgm.shape[0]
    indices_mo_occ, indices_mo_virt = make_indices_mo_separate(nocc_frgm, nvirt_frgm)
    assert len(indices_mo_occ) == len(indices_mo_virt) == nfrgm

    nocc_sum = nocc_frgm.sum()
    nvirt_sum = nvirt_frgm.sum()
    # nov_unrestricted = nocc_sum * nvirt_sum
    # indices_mo_ov_unrestricted = np.asarray(range(nov_unrestricted))
    pairs_good = []
    for ifrg in range(nfrgm):
        pairs_good.extend([(i, a)
                           for i in indices_mo_occ[ifrg]
                           for a in indices_mo_virt[ifrg]])
    # print('pairs_good')
    # print(pairs_good)
    indices_good = [(pair[0]*nvirt_sum + pair[1] - nocc_sum)
                    for pair in pairs_good]
    # print('indices_good')
    # print(indices_good)
    # Now form the opposite (disallowed) pairs as the difference
    # between all (unrestricted) pairs and the allowed pairs.
    # indices_bad = [i for i in range(nocc_sum * nvirt_sum) if i not in indices_good]
    # print('indices_bad')
    # print(indices_bad)
    return np.asarray(indices_good)


def make_indices_mo_restricted_local_occ_all_virt(nocc_frgm, nvirt_frgm):
    assert nocc_frgm.shape == nvirt_frgm.shape
    assert len(nocc_frgm.shape) == 1
    nfrgm = nocc_frgm.shape[0]
    indices_mo_occ, indices_mo_virt = make_indices_mo_separate(nocc_frgm, nvirt_frgm)
    assert len(indices_mo_occ) == len(indices_mo_virt) == nfrgm

    nocc_sum = nocc_frgm.sum()
    nvirt_sum = nvirt_frgm.sum()
    norb = nocc_sum + nvirt_sum
    pairs_good = []
    indices_good = []
    for ifrg in range(nfrgm):
        pairs_good.append([(i, a)
                           for i in indices_mo_occ[ifrg]
                           for a in range(nocc_sum, norb)])
        indices_good.append(np.array([(pair[0]*nvirt_sum + pair[1] - nocc_sum)
                                      for pair in pairs_good[ifrg]]))
    for ifrg in range(nfrgm):
        assert len(pairs_good[ifrg]) == (nocc_frgm[ifrg] * nvirt_sum)
        assert len(indices_good[ifrg]) == (nocc_frgm[ifrg] * nvirt_sum)
    # indices_good = [(pair[0]*nvirt_sum + pair[1] - nocc_sum) for pair in pairs_good]
    # return np.asarray(indices_good)
    return indices_good


def make_index_arrays_into_block(iarray1, iarray2):
    """"""
    assert len(iarray1.shape) == len(iarray2.shape) == 1
    iblock1 = []
    iblock2 = []
    for idx1 in iarray1:
        for idx2 in iarray2:
            iblock1.append(idx1)
            iblock2.append(idx2)
    return np.array(iblock1), np.array(iblock2)


def extract_block_from_matrix(mat, iarray1, iarray2):
    """"""
    assert len(mat.shape) == 2
    if len(iarray1) == 0 or len(iarray2) == 0:
        # TODO there should be a better way of representing a
        # 0-element array
        return None
    else:
        ids = make_index_arrays_into_block(iarray1, iarray2)
        return mat[ids].reshape(len(iarray1), len(iarray2))


def form_vec_energy_differences(moene_i, moene_a):
    """"""
    assert len(moene_i.shape) == 1
    assert len(moene_a.shape) == 1
    nocc = moene_i.shape[0]
    nvirt = moene_a.shape[0]
    nov = nocc * nvirt
    ediff = np.empty(shape=nov)
    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            ediff[ia] = moene_a[a] - moene_i[i]
    return ediff


def one_electron_mn_mat_to_ia_vec(mn_mat, C_occ, C_virt):
    """"""
    assert len(mn_mat.shape) == 2
    assert mn_mat.shape[0] == mn_mat.shape[1]
    assert len(C_occ.shape) == 2
    assert len(C_virt.shape) == 2
    assert C_occ.shape[0] == C_virt.shape[0]
    assert C_occ.shape[0] == mn_mat.shape[0]
    nocc = C_occ.shape[1]
    nvirt = C_virt.shape[1]
    nov = nocc * nvirt
    # 1. transform
    # 2. repack
    ia_mat = np.dot(C_occ.T, np.dot(mn_mat, C_virt))
    ia_vec = np.empty(shape=nov)
    repack_matrix_to_vector(ia_vec, ia_mat)
    return ia_vec


def form_ediff_terms(F, S, nocc, nvirt):
    norb = nocc + nvirt
    nov = nocc * nvirt
    assert F.shape == (norb, norb)
    assert S.shape == (norb, norb)
    ediff_mat = np.empty(shape=(nov, nov))
    for i in range(0, nocc):
        for a in range(nocc, norb):
            ia = i*nvirt + a - nocc
            for j in range(0, nocc):
                for b in range(nocc, norb):
                    jb = j*nvirt + b - nocc
                    ediff_mat[ia, jb] = (F[a, b] * S[i, j]) - (F[i, j] * S[a, b])
    return ediff_mat


def form_superoverlap(S, nocc, nvirt):
    norb = nocc + nvirt
    nov = nocc * nvirt
    assert S.shape == (norb, norb)
    superoverlap = np.empty(shape=(nov, nov))
    for i in range(0, nocc):
        for a in range(nocc, norb):
            ia = i*nvirt + a - nocc
            for j in range(0, nocc):
                for b in range(nocc, norb):
                    jb = j*nvirt + b - nocc
                    superoverlap[ia, jb] = S[i, j] * S[a, b]
    return superoverlap


def restricted_multiplication(C):
    # Options:
    # 1. run over all {\mu\nu}, run over all {ia}, take fragment {ia}
    # 2. run over all {\mu\nu}, run over fragment {ia}, take fragment {ia}
    # 3. run over fragment {\mu\nu}, run over all {ia}, take fragment {ia}
    # 4. run over fragment {\mu\nu}, run over fragment {ia}, take fragment {ia}
    # 5. run over all {\mu\nu}, run over all {ia}, take all {ia}
    # 6. run over all {\mu\nu}, run over fragment {ia}, take all {ia}
    # 7. run over fragment {\mu\nu}, run over all {ia}, take all {ia}
    # 8. run over fragment {\mu\nu}, run over fragment {ia}, take all {ia}

    print(r"1. run over all {\mu\nu}, run over all {ia}, take fragment {ia}")
    print(r"2. run over all {\mu\nu}, run over fragment {ia}, take fragment {ia}")
    print(r"3. run over fragment {\mu\nu}, run over all {ia}, take fragment {ia}")
    print(r"4. run over fragment {\mu\nu}, run over fragment {ia}, take fragment {ia}")

    print(r"5. run over all {\mu\nu}, run over all {ia}, take all {ia}")
    rhs5_m = np.dot(C[:, :nocc].T, np.dot(integrals, C[:, nocc:]))
    rhs5_v = np.empty(shape=product(rhs5_m.shape))
    repack_matrix_to_vector(rhs5_v, rhs5_m)
    # print(rhs5_m)
    print(rhs5_v)

    print(r"6. run over all {\mu\nu}, run over fragment {ia}, take all {ia}")
    # this currently doesn't recombine the {ia} like it should
    for idx in ifrgm:
        rhs6_m = np.dot(C[:, indices_mo_occ[idx]].T, np.dot(integrals, C[:, indices_mo_virt[idx]]))
        rhs6_v = np.empty(shape=product(rhs6_m.shape))
        repack_matrix_to_vector(rhs6_v, rhs6_m)
        # print(rhs6_m)
        print(rhs6_v)

    print(r"7. run over fragment {\mu\nu}, run over all {ia}, take all {ia}")
    # this currently doesn't recombine the {ia} like it should
    for idx in ifrgm:
        integrals_frgm = extract_block_from_matrix(integrals, indices_ao[idx], indices_ao[idx])
        rhs7_m = np.dot(C[indices_ao[idx], :nocc].T, np.dot(integrals_frgm, C[indices_ao[idx], nocc:]))
        rhs7_v = np.empty(shape=product(rhs7_m.shape))
        repack_matrix_to_vector(rhs7_v, rhs7_m)
        # print(rhs7_m)
        print(rhs7_v)

    print(r"7a. zero non-fragment {\mu\nu}, run over all {ia}, take all {ia}")
    # Do everything at once.
    ## Calculate all the AO index blocks, then zero the opposite indices.
    integrals_frgm = np.zeros_like(integrals)
    blocks = []
    # Form fragment index blocks.
    for fragment_indices_ao in indices_ao:
        block_fragment_indices_ao = make_index_arrays_into_block(fragment_indices_ao, fragment_indices_ao)
        blocks.append(block_fragment_indices_ao)
    # Combine the blocks.
    blocks_combined = []
    for idx in ifrgm:
        b = []
        for block in blocks:
            b.append(block[idx])
        b = np.hstack(b)
        blocks_combined.append(b)
    blocks_combined = tuple(blocks_combined)
    # "Mask" the integrals, which here does a copy.
    integrals_frgm[blocks_combined] = integrals[blocks_combined]
    # print(integrals)
    # print(integrals_frgm)
    rhs7a_m = np.dot(C[:, :nocc].T, np.dot(integrals_frgm, C[:, nocc:]))
    rhs7a_v = np.empty(shape=product(rhs7a_m.shape))
    repack_matrix_to_vector(rhs7a_v, rhs7a_m)
    print(rhs7a_m)
    # print(rhs7a_v)

    print(r"8. run over fragment {\mu\nu}, run over fragment {ia}, take all {ia}")
    # this currently doesn't recombine the {ia} like it should
    for idx in ifrgm:
        integrals_frgm = extract_block_from_matrix(integrals, indices_ao[idx], indices_ao[idx])
        # extract the block of fragment AOs, all MOs -> fragment MOs
        # selected in multiplication call
        C_frgm_AO = extract_block_from_matrix(C, indices_ao[idx], np.arange(C.shape[1]))
        rhs8_m = np.dot(C_frgm_AO[:, indices_mo_occ[idx]].T, np.dot(integrals_frgm, C_frgm_AO[:, indices_mo_virt[idx]]))
        rhs8_v = np.empty(shape=product(rhs8_m.shape))
        repack_matrix_to_vector(rhs8_v, rhs8_m)
        print(rhs8_m)
        # print(rhs8_v)

    return None


if __name__ == '__main__':

    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    formatter = {
        'float_kind': lambda x: '{:14.8f}'.format(x)
    }
    np.set_printoptions(linewidth=200, formatter=formatter)

    # Read in all the test vectors/matrices from disk.

    E = read_arma_mat_ascii(os.path.join(datadir, 'E.dat'))[:, 0]
    print('E', E.shape)
    C = read_arma_mat_ascii(os.path.join(datadir, 'C.dat'))
    print('C', C.shape)
    integrals = read_arma_mat_ascii(os.path.join(datadir, 'integrals.dat'))
    print('integrals', integrals.shape)
    # rhsvec = read_arma_mat_ascii('rhsvecs_0_alph.dat')[:, 0]
    rhsvec = read_arma_mat_ascii(os.path.join(datadir, 'rhsvec.dat'))[:, 0]
    print('rhsvec', rhsvec.shape)
    # rspvec_guess = read_arma_mat_ascii('rspvecs_guess_0_alph.dat')[:, 0]
    rspvec_guess = read_arma_mat_ascii(os.path.join(datadir, 'rspvec.dat'))[:, 0]
    print('rspvec_guess', rspvec_guess.shape)
    ediff = read_arma_mat_ascii(os.path.join(datadir, 'ediff.dat'))[:, 0]
    print('ediff', ediff.shape)

    nbasis_frgm = read_arma_mat_ascii(os.path.join(datadir, 'nbasis_frgm.dat'))
    print('nbasis_frgm', nbasis_frgm.shape, len(nbasis_frgm))
    print(nbasis_frgm.reshape(len(nbasis_frgm)))
    nbasis_frgm = nbasis_frgm.reshape(len(nbasis_frgm))
    norb_frgm = read_arma_mat_ascii(os.path.join(datadir, 'norb_frgm.dat'))
    norb_frgm = norb_frgm.reshape(len(norb_frgm))
    print('norb_frgm', norb_frgm.shape)
    print(norb_frgm)
    nocc_frgm = read_arma_mat_ascii(os.path.join(datadir, 'nocc_frgm.dat'))
    nocc_frgm = nocc_frgm.reshape(len(nocc_frgm))
    print('nocc_frgm', nocc_frgm.shape)
    print(nocc_frgm)

    assert len(nbasis_frgm) == len(norb_frgm) == len(nocc_frgm)

    nvirt_frgm = norb_frgm - nocc_frgm
    print('nvirt_frgm')
    print(nvirt_frgm)

    ifrgm = list(range(len(nbasis_frgm)))

    # totals are sums over fragments
    nbasis = sum(nbasis_frgm)
    norb = sum(norb_frgm)
    nocc = sum(nocc_frgm)
    nvirt = sum(nvirt_frgm)
    nov = nocc * nvirt

    moene_i = E[:nocc]
    moene_a = E[nocc:]
    ediff_full = form_vec_energy_differences(moene_i, moene_a)

    # print('ediff')
    # print(ediff)
    # print(ediff_full)
    # assert np.testing.assert_almost_equal(ediff, ediff_full)
    rhsvec_calc = one_electron_mn_mat_to_ia_vec(integrals, C[:, :nocc], C[:, nocc:])
    # print('rhsvec (naive)')
    # print(rhsvec)
    # print(rhsvec_calc)
    # assert np.testing.assert_almost_equal(rhsvec, rhsvec_calc)
    # print('rspvec_guess')
    # print(rspvec_guess)
    # print(rhsvec / ediff_full)
    # assert np.testing.assert_almost_equal(rspvec_guess, rhsvec / ediff_full)

    print('AO indices')
    indices_ao = make_indices_ao(nbasis_frgm)
    print(indices_ao)
    print('MO indices (occ and virt together)')
    indices_mo_combined = make_indices_mo_combined(nocc_frgm, nvirt_frgm)
    print(indices_mo_combined)
    print('MO indices (occ and virt separate)')
    indices_mo_occ, indices_mo_virt = make_indices_mo_separate(nocc_frgm, nvirt_frgm)
    print(indices_mo_occ)
    print(indices_mo_virt)
    _, _ = make_indices_mo_restricted(nocc_frgm, nvirt_frgm)
    _, _ = make_indices_mo_restricted_local_occ_all_virt(nocc_frgm, nvirt_frgm)

    # print('MO energies')
    # print(E)
    # print('MO coefficients')
    # print(C)

    # print('select AOs for fragment 1')
    # print(C[indices_ao[0], :])
    # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
    # print('block AOs')
    # print(C[indices_ao, :])

    # print('select MOs for fragment 2')
    # print(C[:, indices_mo_combined[1]])

    # This doesn't do what you expect it to...
    # print('select block for fragment 2')
    # print(indices_ao[1])
    # print(indices_mo_combined[1])
    # print(C[indices_ao[1], indices_mo_combined[1]])
    # print(C[[6, 7], [2, 7]])
    # print(np.array([[C[6, 2], C[6, 7]],
    #                 [C[7, 2], C[7, 7]]]))

    # print('select block for fragment 1')
    # print(extract_block_from_matrix(C, indices_ao[0], indices_mo_combined[0]))

    # Options:
    # 1. run over all {\mu\nu}, run over all {ia}, take fragment {ia}
    # 2. run over all {\mu\nu}, run over fragment {ia}, take fragment {ia}
    # 3. run over fragment {\mu\nu}, run over all {ia}, take fragment {ia}
    # 4. run over fragment {\mu\nu}, run over fragment {ia}, take fragment {ia}
    # 5. run over all {\mu\nu}, run over all {ia}, take all {ia}
    # 6. run over all {\mu\nu}, run over fragment {ia}, take all {ia}
    # 7. run over fragment {\mu\nu}, run over all {ia}, take all {ia}
    # 8. run over fragment {\mu\nu}, run over fragment {ia}, take all {ia}

    print(r"1. run over all {\mu\nu}, run over all {ia}, take fragment {ia}")
    print(r"2. run over all {\mu\nu}, run over fragment {ia}, take fragment {ia}")
    print(r"3. run over fragment {\mu\nu}, run over all {ia}, take fragment {ia}")
    print(r"4. run over fragment {\mu\nu}, run over fragment {ia}, take fragment {ia}")

    print(r"5. run over all {\mu\nu}, run over all {ia}, take all {ia}")
    rhs5_m = np.dot(C[:, :nocc].T, np.dot(integrals, C[:, nocc:]))
    rhs5_v = np.empty(shape=product(rhs5_m.shape))
    repack_matrix_to_vector(rhs5_v, rhs5_m)
    print(rhs5_m)
    # print(rhs5_v)

    print(r"6. run over all {\mu\nu}, run over fragment {ia}, take all {ia}")
    # this currently doesn't recombine the {ia} like it should
    for idx in ifrgm:
        rhs6_m = np.dot(C[:, indices_mo_occ[idx]].T, np.dot(integrals, C[:, indices_mo_virt[idx]]))
        rhs6_v = np.empty(shape=product(rhs6_m.shape))
        repack_matrix_to_vector(rhs6_v, rhs6_m)
        print(rhs6_m)
        # print(rhs6_v)

    print(r"6a. run over all {\mu\nu}, run over fragment {i} and all {a}, take all {ia}")
    # This is the same as 5, but separated into one fragment at a
    # time.
    for idx in ifrgm:
        rhs6a_m = np.dot(C[:, indices_mo_occ[idx]].T, np.dot(integrals, C[:, nocc:]))
        rhs6a_v = np.empty(shape=product(rhs6a_m.shape))
        repack_matrix_to_vector(rhs6a_v, rhs6a_m)
        print(rhs6a_m)
        # print(rhs6a_v)

    print(r"7. run over fragment {\mu\nu}, run over all {ia}, take all {ia}")
    # this currently doesn't recombine the {ia} like it should
    for idx in ifrgm:
        integrals_frgm = extract_block_from_matrix(integrals, indices_ao[idx], indices_ao[idx])
        rhs7_m = np.dot(C[indices_ao[idx], :nocc].T, np.dot(integrals_frgm, C[indices_ao[idx], nocc:]))
        rhs7_v = np.empty(shape=product(rhs7_m.shape))
        repack_matrix_to_vector(rhs7_v, rhs7_m)
        print(rhs7_m)
        # print(rhs7_v)

    print(r"7a. zero non-fragment {\mu\nu}, run over all {ia}, take all {ia}")
    # Do everything at once.
    ## Calculate all the AO index blocks, then zero the opposite indices.
    integrals_frgm = np.zeros_like(integrals)
    blocks = []
    # Form fragment index blocks.
    for fragment_indices_ao in indices_ao:
        block_fragment_indices_ao = make_index_arrays_into_block(fragment_indices_ao, fragment_indices_ao)
        blocks.append(block_fragment_indices_ao)
    # Combine the blocks.
    blocks_combined = []
    for idx in ifrgm:
        b = []
        for block in blocks:
            b.append(block[idx])
        b = np.hstack(b)
        blocks_combined.append(b)
    blocks_combined = tuple(blocks_combined)
    # "Mask" the integrals, which here does a copy.
    integrals_frgm[blocks_combined] = integrals[blocks_combined]
    # print(integrals)
    # print(integrals_frgm)
    rhs7a_m = np.dot(C[:, :nocc].T, np.dot(integrals_frgm, C[:, nocc:]))
    rhs7a_v = np.empty(shape=product(rhs7a_m.shape))
    repack_matrix_to_vector(rhs7a_v, rhs7a_m)
    print(rhs7a_m)
    # print(rhs7a_v)

    print(r"8. run over fragment {\mu\nu}, run over fragment {ia}, take all {ia}")
    # this currently doesn't recombine the {ia} like it should
    for idx in ifrgm:
        integrals_frgm = extract_block_from_matrix(integrals, indices_ao[idx], indices_ao[idx])
        # extract the block of fragment AOs, all MOs -> fragment MOs
        # selected in multiplication call
        C_frgm_AO = extract_block_from_matrix(C, indices_ao[idx], np.arange(C.shape[1]))
        rhs8_m = np.dot(C_frgm_AO[:, indices_mo_occ[idx]].T, np.dot(integrals_frgm, C_frgm_AO[:, indices_mo_virt[idx]]))
        rhs8_v = np.empty(shape=product(rhs8_m.shape))
        repack_matrix_to_vector(rhs8_v, rhs8_m)
        print(rhs8_m)
        # print(rhs8_v)

    # eq35 = '=' * 35

    # print(eq35 + ' unprojected ' + eq35)
    # restricted_multiplication(C)
    # print(eq35 + ' projected ' + eq35)
    # restricted_multiplication(C_projected)

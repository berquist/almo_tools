from __future__ import print_function

import numpy as np


def repack_matrix_to_vector_slow(v, m):
    """Columns of the matrix are the fast index."""

    assert len(m.shape) == 2

    d1, d2 = m.shape
    dv = d1 * d2

    assert len(v.shape) == 1
    assert v.shape[0] == dv

    for i in range(d1):
        for a in range(d2):
            ia = i*d2 + a
            v[ia] = m[i, a]

    return


def repack_matrix_to_vector(mat):
    return np.reshape(mat, -1)


def repack_vector_to_matrix_slow(v, m):
    """Columns of the matrix are the fast index."""

    assert len(m.shape) == 2

    d1, d2 = m.shape
    dv = d1 * d2

    assert len(v.shape) == 1
    assert v.shape[0] == dv

    for i in range(d1):
        for a in range(d2):
            ia = i*d2 + a
            m[i, a] = v[ia]

    return


def repack_vector_to_matrix(vec, shape):
    """Columns of the matrix are the fast index."""
    assert len(shape) == 2
    return vec.reshape(shape)


def form_indices_zero(nocc, nvirt):
    return [(i, a) for i in range(nocc) for a in range(nvirt)]


def form_indices_orbwin(nocc, nvirt):
    norb = nocc + nvirt
    return [(i, a) for i in range(0, nocc) for a in range(nocc, norb)]


def loop_rhf_pairs_start_at_zero(indices_rhf, nvirt):
    """Looping constructs for compound index RHF arrays, using pairs
    generated from a list comprehension, with all ranges starting at 0.
    """
    print(' i -> slow, a -> fast')
    counter_main = -1
    counter_ia = -1
    for (i, a) in indices_rhf:
        ia = i*nvirt + a
        counter_ia += 1
        counter_main += 1
        print('idx_tot: {:2} i: {} a: {} ia: {}'.format(counter_main, i, a, ia))
        assert ia == counter_ia
    return


def loop_rhf_pairs_start_at_orbwin(indices_rhf, nocc, nvirt):
    """Looping constructs for compound index RHF arrays, using pairs
    generated from a list comprehension, with all ranges starting at the
    correct orbital window.
    """
    print(' i -> slow, a -> fast')
    counter_main = -1
    counter_ia = -1
    for (i, a) in indices_rhf:
        ia = i*nvirt + a - nocc
        counter_ia += 1
        counter_main += 1
        print('idx_tot: {:2} i: {} a: {} ia: {}'.format(counter_main, i, a, ia))
        assert ia == counter_ia
    return


if __name__ == '__main__':

    # Assume no linear dependencies.
    # nbasis_frg = np.array([3, 5, 12])
    # nocc_frg = np.array([1, 2, 3])
    nbasis_frg = np.array([6, 2])
    nocc_frg = np.array([2, 1])

    nvirt_frg = nbasis_frg - nocc_frg
    # TODO crash for negative number of virtuals at any position

    assert nbasis_frg.shape == nocc_frg.shape
    nfrg = len(nbasis_frg)

    nov_frg = nocc_frg * nvirt_frg

    nbasis_tot = sum(nbasis_frg)
    nocc_tot = sum(nocc_frg)
    nvirt_tot = sum(nvirt_frg)
    nov_tot = sum(nov_frg)

    print('nbasis')
    print(nbasis_tot, nbasis_frg)
    print('nocc')
    print(nocc_tot, nocc_frg)
    print('nvirt')
    print(nvirt_tot, nvirt_frg)
    print('nov')
    print(nov_tot, nocc_tot * nvirt_tot, nov_frg)

    # In the ALMO code, the ordering of fragment indices is
    # [o1, o2, ..., on, v1, v2, ..., vn].
    ordering = np.concatenate((nocc_frg, nvirt_frg))
    print('ALMO ordering (ooo...vvv...)')
    print(ordering)

    # Form index-unrestricted excitations, that is, all possible
    # occupied orbitals to all possible virtual orbitals, irrespective
    # of fragment.

    indices_unrestricted_zero = form_indices_zero(nocc_tot, nvirt_tot)
    assert len(indices_unrestricted_zero) == (nocc_tot * nvirt_tot)
    print('full: loop using preconstructed pairs starting at 0')
    print(indices_unrestricted_zero)
    loop_rhf_pairs_start_at_zero(indices_unrestricted_zero, nvirt_tot)

    indices_unrestricted_orbwin = form_indices_orbwin(nocc_tot, nvirt_tot)
    assert len(indices_unrestricted_orbwin) == (nocc_tot * nvirt_tot)
    print('full: loop using preconstructed pairs starting at correct orbital window')
    print(indices_unrestricted_orbwin)
    loop_rhf_pairs_start_at_orbwin(indices_unrestricted_orbwin, nocc_tot, nvirt_tot)

    # Form index-restricted excitations, that is, allow only
    # occupied-virtual excitations within individual fragments.

    # This approach doesn't make sense. TODO explain why
    # indices_restricted_zero = []
    # for n in range(nfrg):
    #     indices_restricted_zero.append(form_indices_zero(nocc_frg[n], nvirt_frg[n]))
    # print(indices_restricted_zero)
    # This approach doesn't make sense. TODO explain why
    # indices_restricted_orbwin = []
    # for n in range(nfrg):
    #     indices_restricted_orbwin.append(form_indices_orbwin(nocc_frg[n], nvirt_frg[n]))
    # print(indices_restricted_orbwin)

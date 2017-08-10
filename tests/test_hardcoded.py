import numpy as np

from almo_tools.fragment_ia_ranges import (repack_matrix_to_vector_slow, repack_matrix_to_vector,
                                           repack_vector_to_matrix_slow, repack_vector_to_matrix,
                                           form_indices_zero, form_indices_orbwin)

from almo_tools.restricted_multiplication import (product, make_indices_ao,
                                                  make_indices_mo_separate,
                                                  make_indices_mo_combined,
                                                  make_indices_mo_restricted,
                                                  make_indices_mo_restricted_local_occ_all_virt)

def test_repack():

    ref_v = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    ref_m = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)

    actual_v_slow = np.empty_like(ref_v)
    actual_m_slow = np.empty_like(ref_m)

    repack_matrix_to_vector_slow(actual_v_slow, ref_m)
    repack_vector_to_matrix_slow(ref_v, actual_m_slow)

    np.testing.assert_equal(actual_v_slow, ref_v)
    np.testing.assert_equal(actual_m_slow, ref_m)

    actual_v_fast = repack_matrix_to_vector(ref_m)
    actual_m_fast = repack_vector_to_matrix(ref_v, ref_m.shape)

    np.testing.assert_equal(actual_v_fast, ref_v)
    np.testing.assert_equal(actual_m_fast, ref_m)


def test_product():
    assert product(list(range(100))) == 0
    assert product(list(range(1, 3 + 1))) == 6
    assert product(list(range(1, 5 + 1))) == 120
    assert product([3, 4, 5]) == 60


def test_small_rhf():

    nbasis_frgm = np.array([6, 2])
    # assume no linear dependencies
    norb_frgm = nbasis_frgm.copy()
    nocc_frgm = np.array([2, 1])

    nfrgm = len(nocc_frgm)
    nbasis_sum = nbasis_frgm.sum()
    nocc_sum = nocc_frgm.sum()
    nvirt_frgm = norb_frgm - nocc_frgm
    nvirt_sum = nvirt_frgm.sum()

    desired_nvirt_frgm = np.array([4, 1])

    desired_indices_zero = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    ]
    desired_indices_orbwin = [
        (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
    ]

    desired_indices_ao = [
        np.array([0, 1, 2, 3, 4, 5], dtype=int),
        np.array([6, 7], dtype=int),
    ]

    desired_indices_mo_occ = [
        np.array([0, 1], dtype=int),
        np.array([2], dtype=int),
    ]
    desired_indices_mo_virt = [
        np.array([3, 4, 5, 6], dtype=int),
        np.array([7], dtype=int),
    ]


    desired_indices_mo_combined = [
        np.array([0, 1, 3, 4, 5, 6], dtype=int),
        np.array([2, 7], dtype=int),
    ]

    desired_indices_mo_restricted = np.array([0, 1, 2, 3, 5, 6, 7, 8, 14], dtype=int)

    desired_indices_mo_restricted_allvirt = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int),
        np.array([10, 11, 12, 13, 14], dtype=int),
    ]

    # actual, desired
    np.testing.assert_equal(nvirt_frgm, desired_nvirt_frgm)

    actual_indices_zero = form_indices_zero(nocc_sum, nvirt_sum)
    actual_indices_orbwin = form_indices_orbwin(nocc_sum, nvirt_sum)

    assert actual_indices_zero == desired_indices_zero
    assert actual_indices_orbwin == desired_indices_orbwin

    actual_indices_ao = make_indices_ao(nbasis_frgm)

    assert len(actual_indices_ao) == len(desired_indices_ao)
    for i in range(nfrgm):
        np.testing.assert_equal(actual_indices_ao[i], desired_indices_ao[i])

    actual_indices_mo_occ, actual_indices_mo_virt = make_indices_mo_separate(nocc_frgm, nvirt_frgm)

    assert len(actual_indices_mo_occ) == len(desired_indices_mo_occ)
    for i in range(nfrgm):
        np.testing.assert_equal(actual_indices_mo_occ[i], desired_indices_mo_occ[i])
    assert len(actual_indices_mo_virt) == len(desired_indices_mo_virt)
    for i in range(nfrgm):
        np.testing.assert_equal(actual_indices_mo_virt[i], desired_indices_mo_virt[i])
    actual_indices_mo_combined = make_indices_mo_combined(nocc_frgm, nvirt_frgm)

    assert len(actual_indices_mo_combined) == len(desired_indices_mo_combined)
    for i in range(nfrgm):
        np.testing.assert_equal(actual_indices_mo_combined[i], desired_indices_mo_combined[i])

    actual_indices_mo_restricted = make_indices_mo_restricted(nocc_frgm, nvirt_frgm)

    np.testing.assert_equal(actual_indices_mo_restricted, desired_indices_mo_restricted)

    actual_indices_mo_restricted_allvirt = make_indices_mo_restricted_local_occ_all_virt(nocc_frgm, nvirt_frgm)

    assert len(actual_indices_mo_restricted_allvirt) == len(desired_indices_mo_restricted_allvirt)
    for i in range(nfrgm):
        np.testing.assert_equal(actual_indices_mo_restricted_allvirt[i], desired_indices_mo_restricted_allvirt[i])

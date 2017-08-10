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


def test_small_rhf_reversed():

    nbasis_frgm = np.array([2, 6])
    # assume no linear dependencies
    norb_frgm = nbasis_frgm.copy()
    nocc_frgm = np.array([1, 2])

    nfrgm = len(nocc_frgm)
    nbasis_sum = nbasis_frgm.sum()
    nocc_sum = nocc_frgm.sum()
    nvirt_frgm = norb_frgm - nocc_frgm
    nvirt_sum = nvirt_frgm.sum()

    desired_nvirt_frgm = np.array([1, 4])

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
        np.array([0, 1], dtype=int),
        np.array([2, 3, 4, 5, 6, 7], dtype=int),
    ]

    desired_indices_mo_occ = [
        np.array([0], dtype=int),
        np.array([1, 2], dtype=int),
    ]
    desired_indices_mo_virt = [
        np.array([3], dtype=int),
        np.array([4, 5, 6, 7], dtype=int),
    ]

    desired_indices_mo_combined = [
        np.array([0, 3], dtype=int),
        np.array([1, 2, 4, 5, 6, 7], dtype=int),
    ]

    desired_indices_mo_restricted = np.array([0, 6, 7, 8, 9, 11, 12, 13, 14], dtype=int)

    desired_indices_mo_restricted_allvirt = [
        np.array([0, 1, 2, 3, 4], dtype=int),
        np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=int),
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


def test_medium_rhf():

    nbasis_frgm = np.array([27, 12])
    # assume no linear dependencies
    norb_frgm = nbasis_frgm.copy()
    nocc_frgm = np.array([9, 1])

    nfrgm = len(nocc_frgm)
    nbasis_sum = nbasis_frgm.sum()
    nocc_sum = nocc_frgm.sum()
    nvirt_frgm = norb_frgm - nocc_frgm
    nvirt_sum = nvirt_frgm.sum()

    desired_nvirt_frgm = np.array([18, 11])

    # print(list(itertools.product(range(0, 10), range(0, 29))))
    desired_indices_zero = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (2, 26), (2, 27), (2, 28),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (3, 27), (3, 28),
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28),
        (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28),
        (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26), (7, 27), (7, 28),
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28),
        (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28)
    ]
    # print(list(itertools.product(range(0, 10), range(10, 39))))
    desired_indices_orbwin = [
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31), (0, 32), (0, 33), (0, 34), (0, 35), (0, 36), (0, 37), (0, 38),
        (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38),
        (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (2, 26), (2, 27), (2, 28), (2, 29), (2, 30), (2, 31), (2, 32), (2, 33), (2, 34), (2, 35), (2, 36), (2, 37), (2, 38),
        (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (3, 27), (3, 28), (3, 29), (3, 30), (3, 31), (3, 32), (3, 33), (3, 34), (3, 35), (3, 36), (3, 37), (3, 38),
        (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19), (4, 20), (4, 21), (4, 22), (4, 23), (4, 24), (4, 25), (4, 26), (4, 27), (4, 28), (4, 29), (4, 30), (4, 31), (4, 32), (4, 33), (4, 34), (4, 35), (4, 36), (4, 37), (4, 38),
        (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31), (5, 32), (5, 33), (5, 34), (5, 35), (5, 36), (5, 37), (5, 38),
        (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20), (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (6, 30), (6, 31), (6, 32), (6, 33), (6, 34), (6, 35), (6, 36), (6, 37), (6, 38),
        (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (7, 20), (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (7, 30), (7, 31), (7, 32), (7, 33), (7, 34), (7, 35), (7, 36), (7, 37), (7, 38),
        (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (8, 30), (8, 31), (8, 32), (8, 33), (8, 34), (8, 35), (8, 36), (8, 37), (8, 38),
        (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30), (9, 31), (9, 32), (9, 33), (9, 34), (9, 35), (9, 36), (9, 37), (9, 38)
    ]

    # print(list(range(0, 27)))
    # print(list(range(27, 39)))
    desired_indices_ao = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], dtype=int),
        np.array([27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], dtype=int),
    ]

    desired_indices_mo_occ = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int),
        np.array([9], dtype=int),
    ]
    desired_indices_mo_virt = [
        np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=int),
        np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], dtype=int),
    ]

    desired_indices_mo_combined = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=int),
        np.array([9, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], dtype=int),
    ]

    # desired_indices_mo_restricted = np.array([0, 1, 2, 3, 5, 6, 7, 8, 14], dtype=int)

    # desired_indices_mo_restricted_allvirt = [
    #     np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int),
    #     np.array([10, 11, 12, 13, 14], dtype=int),
    # ]

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

    # np.testing.assert_equal(actual_indices_mo_restricted, desired_indices_mo_restricted)

    actual_indices_mo_restricted_allvirt = make_indices_mo_restricted_local_occ_all_virt(nocc_frgm, nvirt_frgm)

    # assert len(actual_indices_mo_restricted_allvirt) == len(desired_indices_mo_restricted_allvirt)
    # for i in range(nfrgm):
    #     np.testing.assert_equal(actual_indices_mo_restricted_allvirt[i], desired_indices_mo_restricted_allvirt[i])

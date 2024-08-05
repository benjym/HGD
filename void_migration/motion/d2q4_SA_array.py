import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import maximum_filter
import params
import operators
import random


def find_intersection(A, B):
    return np.array([x for x in A if x.tolist() in B.tolist()])


def delete_element(arr, element):
    return np.array([x for x in arr if not np.array_equal(x, element)])


def move_voids(
    u: ArrayLike,
    v: ArrayLike,
    s: ArrayLike,
    p,
    diag: int = 0,
    c: None | ArrayLike = None,
    T: None | ArrayLike = None,
    N_swap: None | ArrayLike = None,
    last_swap: None | ArrayLike = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, None | ArrayLike, None | ArrayLike]:
    """
    Function to move voids each timestep.

    Args:
        u: Storage container for counting how many voids moved horizontally
        v: Storage container for counting how many voids moved vertically
        s: 3D array containing the local sizes everywhere. `NaN`s represent voids. Other values represent the grain size. The first two dimensions represent real space, the third dimension represents the micro-structural coordinate.
        diag: Should the voids swap horizontally (von neumnann neighbourhood, `diag=0`) or diagonally upwards (moore neighbourhood, `diag=1`). Default value `0`.
        c: If ArrayLike, a storage container for tracking motion of differently labelled particles. If `None`, do nothing.
        T: If ArrayLike, the temperature field. If `None`, do nothing.
        boundary: If ArrayLike, a descriptor of cells which voids cannot move into (i.e. boundaries). If `internal_boundary` is defined in the params file, allow for reduced movement rates rather than zero. If `None`, do nothing.

    Returns:
        u: The updated horizontal velocity
        v: The updated vertical velocity
        s: The new locations of the grains
        c: The updated concentration field
        T: The updated temperature field
    """

    nu = 1.0 - np.mean(np.isnan(s[:, :, :]), axis=2)
    kernel = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    )  ## there is still a potential to speed up the code using this in array formulation, I will try this
    # nu_max = maximum_filter(nu, footprint=kernel)  # , mode='constant', cval=0.0)
    # import matplotlib.pyplot as plt

    s_bar = operators.get_average(s)
    s_inv_bar = operators.get_hyperbolic_average(s)

    scale_ang = 1.0 / ((1 / p.mu) + 1)

    ###converting mean values to i,j,k format
    nu_req = np.repeat(nu[:, :], p.nm, axis=1).reshape(p.nx, p.ny, p.nm)
    s_inv_bar_req = np.repeat(s_inv_bar[:, :], p.nm, axis=1).reshape(p.nx, p.ny, p.nm)
    s_bar_req = np.repeat(s_bar[:, :], p.nm, axis=1).reshape(p.nx, p.ny, p.nm)

    ### Recalculating Reference probabilities at each time step ###
    # p.P_u_ref = s_inv_bar_req / s
    # p.P_lr_ref = s_bar_req * p.alpha * p.P_u_ref / p.dy

    P_initial = np.random.rand(p.nx, p.ny, p.nm)

    #########################################################################################################
    ##### UP

    P_ups = p.P_u_ref * (s_inv_bar_req[:, 1:, :] / s[:, 1:, :])

    ############################### Left #####################################################

    P_ls = p.P_lr_ref * (
        np.concatenate((s[[0]], s[0:-1, :, :]))[:, 0:-1, :]
        / np.concatenate((s_bar_req[[0]], s_bar_req[0:-1, :, :]))[:, 0:-1, :]
    )

    ###################################################### Right #####################################################

    P_rs = p.P_lr_ref * (
        np.concatenate((s[1:, :, :], s[[-1]]))[:, 0:-1, :]
        / np.concatenate((s_bar_req[1:, :, :], s_bar_req[[-1]]))[:, 0:-1, :]
    )

    ##############################################################################################################

    ids_up = np.where((nu_req[:, 0:-1, :] < p.nu_cs) & (np.isnan(s[:, 0:-1, :])) & (np.isnan(s[:, 1:, :])))
    if len(ids_up[0]) != 0:
        P_ups[ids_up] = 0  ## make probabilities 0 at satisfied condition if probability is NaN

    ids_left = np.where(
        (nu_req[:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[:, 0:-1, :]))
        & (np.isnan(np.concatenate((s[[0]], s[0:-1, :, :]))[:, 0:-1, :]))
    )
    if len(ids_left[0]) != 0:
        P_ls[ids_left] = 0  ## make probabilities 0 at satisfied condition if probability is NaN

    ids_right = np.where(
        (nu_req[:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[:, 0:-1, :]))
        & (np.isnan(np.concatenate((s[1:, :, :], s[[-1]]))[:, 0:-1, :]))
    )
    if len(ids_right[0]) != 0:
        P_rs[ids_right] = 0  ## make probabilities 0 at satisfied condition if probability is NaN

    P_ts = P_ups + P_ls + P_rs

    ids_up = np.where(
        (nu_req[:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[:, 0:-1, :]))
        & (~np.isnan(np.isnan(s[:, 1:, :])))
        & (P_initial[:, 0:-1, :] < (P_ups))
        & (P_ups > 0)
        & (P_ts > 0)
    )

    ids_swap_up = ids_up[0], ids_up[1] + 1, ids_up[2]

    ids_left = np.where(
        (nu_req[1:, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[1:, 0:-1, :]))
        & (
            (~np.isnan(np.concatenate((s[[0]], s[0:-1, :, :]))[1:, 0:-1, :]))
            & (np.invert((((nu_req[0:-1, 0:-1, :] - nu_req[1:, 0:-1, :]) <= scale_ang * p.nu_cs))))
        )
        & (P_initial[1:, 0:-1, :] < (P_ls[1:, :, :] + P_ups[1:, :, :]))
        & (P_ts[1:, :, :] > 0)
        & (P_ls[1:, :, :] > 0)
        & (P_initial[1:, 0:-1, :] >= P_ups[1:, :, :])
    )

    ids_swap_l = ids_left[0] + 1, ids_left[1], ids_left[2]

    # dx_r = nu_req[1:,0:-1,:] - nu_req[0:-1,0:-1,:]
    # dy_r = nu_req[0:-1,1:,:] - nu_req[0:-1,0:-1,:]

    # angle_r = np.abs(np.degrees(np.arctan(dx_r/dy_r)))
    # mag_r = np.sqrt(dx_r**2 + dy_r**2)

    ids_right = np.where(
        (nu_req[0:-1, 0:-1, :] < p.nu_cs)
        & (np.isnan(s[0:-1, 0:-1, :]))
        & (
            (~np.isnan(np.concatenate((s[1:, :, :], s[[-1]]))[0:-1, 0:-1, :]))
            & (np.invert((((nu_req[1:, 0:-1, :] - nu_req[0:-1, 0:-1, :]) <= scale_ang * p.nu_cs))))
        )
        & (P_initial[0:-1, 0:-1, :] < (P_rs[0:-1, :, :] + P_ups[0:-1, :, :] + P_ls[0:-1, :, :]))
        & (P_ts[0:-1, :, :] > 0)
        & (P_rs[0:-1, :, :] > 0)
        & (P_initial[0:-1, 0:-1, :] >= (P_ups[0:-1, :, :] + P_ls[0:-1, :, :]))
    )

    ids_swap_r = ids_right[0] + 1, ids_right[1], ids_right[2]

    ## Destinations
    A = np.transpose(ids_swap_up)
    B = np.transpose(ids_left)
    C = np.transpose(ids_swap_r)

    # Handle A intersection B intersection C
    intersection = find_intersection(find_intersection(A, B), C)
    for selected_element in intersection:
        selected_array = random.choice([A, B, C])
        if np.array_equal(selected_array, A):
            B = delete_element(B, selected_element)
            C = delete_element(C, selected_element)
        elif np.array_equal(selected_array, B):
            A = delete_element(A, selected_element)
            C = delete_element(C, selected_element)
        else:
            A = delete_element(A, selected_element)
            B = delete_element(B, selected_element)

    # Handle A intersection B
    intersection = find_intersection(A, B)
    for selected_element in intersection:
        selected_array = random.choice([A, B])
        if np.array_equal(selected_array, A):
            B = delete_element(B, selected_element)
        else:
            A = delete_element(A, selected_element)

    # Handle B intersection C
    intersection = find_intersection(B, C)
    for selected_element in intersection:
        selected_array = random.choice([B, C])
        if np.array_equal(selected_array, B):
            C = delete_element(C, selected_element)
        else:
            B = delete_element(B, selected_element)

    # Handle A intersection C
    intersection = find_intersection(A, C)
    for selected_element in intersection:
        selected_array = random.choice([A, C])
        if np.array_equal(selected_array, A):
            C = delete_element(C, selected_element)
        else:
            A = delete_element(A, selected_element)

    A_ori = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    B_ori = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    C_ori = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if len(A) > 0:
        A_ori = tuple(np.transpose(A))
        A_ori = A_ori[0], A_ori[1] - 1, A_ori[2]  # Source
        A = tuple(np.transpose(A))  # Destination
    else:
        A = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if len(B) > 0:
        B_ori = tuple(np.transpose(B))
        B_ori = B_ori[0] + 1, B_ori[1], B_ori[2]  # Source
        B = tuple(np.transpose(B))  # Destination
    else:
        B = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if len(C) > 0:
        C_ori = tuple(np.transpose(C))
        C_ori = C_ori[0] - 1, C_ori[1], C_ori[2]  # Source
        C = tuple(np.transpose(C))  # Destination
    else:
        C = (np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    all_ids = (
        np.hstack((A_ori[0], B_ori[0], C_ori[0])),
        np.hstack((A_ori[1], B_ori[1], C_ori[1])),
        np.hstack((A_ori[2], B_ori[2], C_ori[2])),
    )
    all_swap_ids = (
        np.hstack((A[0], B[0], C[0])),
        np.hstack((A[1], B[1], C[1])),
        np.hstack((A[2], B[2], C[2])),
    )

    s[all_ids], s[all_swap_ids] = s[all_swap_ids], s[all_ids]
    if c is not None:
        c[all_ids], c[all_swap_ids] = c[all_swap_ids], c[all_ids]

    # T[ids_left],T[ids_swap] = T[ids_swap],T[ids_left]

    nu_req[all_ids] += 1 / p.nm
    nu_req[all_swap_ids] -= 1 / p.nm

    nu = nu_req[:, :, 0]
    print("lfjghflkg")
    return u, v, s, c, T, N_swap, last_swap

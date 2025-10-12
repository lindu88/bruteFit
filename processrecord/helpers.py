import numpy as np
import pandas as pd


def kk_arbspace(omega: np.ndarray, imchi: np.ndarray, alpha: int) -> np.ndarray:
    # we ought to probably look this thing up and make sure we have it implemented correctly.
    omega = np.array(omega)
    imchi = np.array(imchi)

    if omega.ndim == 1:
        omega = omega[np.newaxis, :]
    if imchi.ndim == 1:
        imchi = imchi[np.newaxis, :]

    g = omega.shape[1]
    rechi = np.zeros_like(imchi)
    a = np.zeros_like(imchi)
    b = np.zeros_like(imchi)

    #TODO: CHECK UNUSED
    deltaomega = omega[0, 1] - omega[0, 0]

    for j in range(g):
        alpha1, beta1 = 0, 0
        if j > 0:
            for k in range(j):
                a[0, j] = alpha1 + (omega[0, k + 1] - omega[0, k]) * (
                    imchi[0, k]
                    * omega[0, k] ** (2 * alpha + 1)
                    / (omega[0, k] ** 2 - omega[0, j] ** 2)
                )
                alpha1 = a[0, j]
        for k in range(j + 1, g):
            b[0, j] = beta1 + (omega[0, k] - omega[0, k - 1]) * (
                imchi[0, k]
                * omega[0, k] ** (2 * alpha + 1)
                / (omega[0, k] ** 2 - omega[0, j] ** 2)
            )
            beta1 = b[0, j]
        rechi[0, j] = 2 / np.pi * (a[0, j] + b[0, j]) * omega[0, j] ** (-2 * alpha)

    return rechi.flatten()

def calculate_differences(positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> tuple:
    # This is a way to account for a baseline that is introduce by optical abberations of the setup, incl. linear dichroisms(?)
    # we do this parametrically ( by X and by Y)
    x_diff = (positive_df["x_pos"] - negative_df["x_neg"]) / 2
    y_diff = (positive_df["y_pos"] - negative_df["y_neg"]) / 2
    x_stdev = np.sqrt(2 * ((positive_df["std_dev_x"] ** 2) + (negative_df["std_dev_x"] ** 2)))
    # this is just a stats formula, but I better cite this.
    y_stdev = np.sqrt(2 * ((positive_df["std_dev_y"] ** 2) + (negative_df["std_dev_y"] ** 2)))
    R = np.sqrt(x_diff**2 + y_diff**2) #pythagorus
    # cite all these.
    R_stdev = np.sqrt(((x_diff * x_stdev / R) ** 2) + ((y_diff * y_stdev / R) ** 2))
    R_signed = R * np.sign(y_diff) #TODO: why do we multiply by the sign of y and not x? idk it I think its bc it looked better. figure it out.
    return x_diff, y_diff, x_stdev, y_stdev, R_signed, R_stdev

#TODO: norm to field
def convert_abs_to_extinction(abs_inp: list, concentration_MOL_L: float, pathlength_CM: float) -> list:
    extinciton = []
    for abs_value in abs_inp:
        extinciton.append(abs_value / (concentration_MOL_L * pathlength_CM))
    return extinciton
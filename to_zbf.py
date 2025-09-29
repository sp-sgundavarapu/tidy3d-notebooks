
from tidy3d.constants import C_0, EPSILON_0, ETA_0, MICROMETER, UnitScaling
import numpy as np
import struct
import tidy3d as td

def to_zbf(
        self,
        fname,
        units = "mm",
        background_refractive_index = 1,
        n_x = None,
        n_y = None,
        freq = None,
        mode_index = None,
        r_x = 0,
        r_y = 0,
        z_x = 0,
        z_y = 0,
        rec_efficiency = 0,
        sys_efficiency = 0,
        flatten_order = "F",
        ):
       
    # Check that appropriate units are used
    if units not in ["mm", "cm", "in", "m"]:
        raise ValueError("'units' must be either 'mm', 'cm', 'in', or 'm'.")

    # Mode area calculation ensures all E components are present
    mode_area = self.mode_area
    dim1, dim2 = self._tangential_dims

    # Using file-local coordinates x, y for the tangential components
    e_x = self._tangential_fields["E" + dim1]
    e_y = self._tangential_fields["E" + dim2]
    x = e_x.coords[dim1].values
    y = e_x.coords[dim2].values

    # Use the mean frequency if freq is not specified
    if freq is None:
        freq = np.mean(e_x.coords["f"].values)
    else:
        freq = np.array(freq)

    if freq.size > 1:
        raise ("'freq' must be a single value, not an array.")
    else:
        freq = freq.item()

    # If the data has just one frequency, avoid Nans at the interpolation
    if len(e_x.f) > 1:
        mode_area = mode_area.interp(f=freq)
        e_x = e_x.interp(f=freq)
        e_y = e_y.interp(f=freq)

    # If the data is ModeData, choose one of the modes to save
    if "mode_index" in e_x.coords:
        if mode_index is None:
            raise ValueError("'mode_index' is required for 'ModeData.to_zbf()'")
        mode_area = mode_area.isel(mode_index=mode_index, drop=True)
        e_x = e_x.isel(mode_index=mode_index, drop=True)
        e_y = e_y.isel(mode_index=mode_index, drop=True)

    # Header info
    version = 1
    polarized = 1
    unit_mapping = {"mm": 0, "cm": 1, "in": 2, "m": 3}
    unit_key = unit_mapping[units]
    unit_scaling = UnitScaling[units]
    lda = C_0 / freq * unit_scaling

    # Pilot (reference) beam waist: use the mode area to approximate the expected value
    w_x = (mode_area.item() / np.pi) ** 0.5 * unit_scaling
    w_y = w_x

    # Pilot beam Rayleigh distance (ignored on input)
    r_x *= unit_scaling
    r_y *= unit_scaling

    # Pilot beam z position w.r.t. the waist
    z_x *= unit_scaling
    z_y *= unit_scaling

    # defaults for n_x and n_y
    if n_x is None:
        n_x = 2 ** min(13, max(5, int(np.log2(x.size) + 1)))

    if n_y is None:
        n_y = 2 ** min(13, max(5, int(np.log2(y.size) + 1)))


    # Check that requirements are met for n_x and n_y
    # n_x and n_y must be powers of 2
    if (n_x & (n_x - 1)) != 0:
        raise ValueError("'n_x' must be a power of 2.")
    if (n_y & (n_y - 1)) != 0:
        raise ValueError("'n_y' must be a power of 2.")
    # 32 <= n_x and n_y <= 2^13
    if n_x < 32 or n_x > 2**13:
        raise ValueError("'n_x' must be between 2^5 and 2^13, inclusive.")
    if n_y < 32 or n_y > 2**13:
        raise ValueError("'n_y' must be between 2^5 and 2^13, inclusive.")

    # Interpolating coordinates
    x = np.linspace(x.min(), x.max(), n_x)
    y = np.linspace(y.min(), y.max(), n_y)

    # Interpolate fields
    coords = {dim1: x, dim2: y}
    e_x = e_x.interp(coords, assume_sorted=True)
    e_y = e_y.interp(coords, assume_sorted=True)

    # Sampling distance
    d_x = np.mean(np.diff(x)) * unit_scaling
    d_y = np.mean(np.diff(y)) * unit_scaling

    with open(fname, "wb") as fout:
        fout.write(struct.pack("<5I", version, n_x, n_y, polarized, unit_key))
        fout.write(struct.pack("<4I", 0, 0, 0, 0))  # unused values
        fout.write(struct.pack("<8d", d_x, d_y, z_x, r_x, w_x, z_y, r_y, w_y))
        fout.write(
            struct.pack("<4d", lda, background_refractive_index, rec_efficiency, sys_efficiency)
        )
        fout.write(struct.pack("<8d", 0, 0, 0, 0, 0, 0, 0, 0))  # unused values
        for e in (e_x, e_y):
            e_flat = e.values.flatten(order=flatten_order)
            # Interweave real and imaginary parts
            e_values = np.ravel(np.column_stack((e_flat.real, e_flat.imag)))
            fout.write(struct.pack(f"<{2 * n_x * n_y}d", *e_values))

    return e_x, e_y

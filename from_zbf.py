
from tidy3d.constants import C_0, EPSILON_0, ETA_0, MICROMETER, UnitScaling
import numpy as np
import struct
import tidy3d as td

from typing import Any, Callable, Optional, Union, get_args

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import Axis, xyz

from struct import unpack


class ZBFData(Tidy3dBaseModel):
    """
    Contains data read in from a ``.zbf`` file
    """

    version: int = pd.Field(title="Version", description="File format version number.")
    nx: int = pd.Field(title="Samples in X", description="Number of samples in the x direction.")
    ny: int = pd.Field(title="Samples in Y", description="Number of samples in the y direction.")
    ispol: bool = pd.Field(
        title="Is Polarized",
        description="``True`` if the beam is polarized, ``False`` otherwise.",
    )
    unit: str = pd.Field(
        title="Spatial Units", description="Spatial units, either 'mm', 'cm', 'in', or 'm'."
    )
    dx: float = pd.Field(title="Grid Spacing, X", description="Grid spacing in x.")
    dy: float = pd.Field(title="Grid Spacing, Y", description="Grid spacing in y.")
    zposition_x: float = pd.Field(
        title="Z Position, X Direction",
        description="The pilot beam z position with respect to the pilot beam waist, x direction.",
    )
    zposition_y: float = pd.Field(
        title="Z Position, Y Direction",
        description="The pilot beam z position with respect to the pilot beam waist, y direction.",
    )
    rayleigh_x: float = pd.Field(
        title="Rayleigh Distance, X Direction",
        description="The pilot beam Rayleigh distance in the x direction.",
    )
    rayleigh_y: float = pd.Field(
        title="Rayleigh Distance, Y Direction",
        description="The pilot beam Rayleigh distance in the y direction.",
    )
    waist_x: float = pd.Field(
        title="Beam Waist, X", description="The pilot beam waist in the x direction."
    )
    waist_y: float = pd.Field(
        title="Beam Waist, Y", description="The pilot beam waist in the y direction."
    )
    wavelength: float = pd.Field(..., title="Wavelength", description="The wavelength of the beam.")
    background_refractive_index: float = pd.Field(
        title="Background Refractive Index",
        description="The index of refraction in the current medium.",
    )
    receiver_eff: float = pd.Field(
        title="Receiver Efficiency",
        description="The receiver efficiency. Zero if fiber coupling is not computed.",
    )
    system_eff: float = pd.Field(
        title="System Efficiency",
        description="The system efficiency. Zero if fiber coupling is not computed.",
    )
    Ex: np.ndarray = pd.Field(
        title="Electric Field, X Component",
        description="Complex-valued electric field, x component.",
    )
    Ey: np.ndarray = pd.Field(
        title="Electric Field, Y Component",
        description="Complex-valued electric field, y component.",
    )

    def read_zbf(filename: str):
        """Reads a Zemax Beam File (``.zbf``)

        Parameters
        ----------
        filename : str
            The file name of the ``.zbf`` file to read.

        Returns
        -------
        :class:`.ZBFData`
        """

        # Read the zbf file
        with open(filename, "rb") as f:
            # Load the header
            version, nx, ny, ispol, units = unpack("<5I", f.read(20))
            f.read(16)  # unused values
            (
                dx,
                dy,
                zposition_x,
                rayleigh_x,
                waist_x,
                zposition_y,
                rayleigh_y,
                waist_y,
                wavelength,
                background_refractive_index,
                receiver_eff,
                system_eff,
            ) = unpack("<12d", f.read(96))
            f.read(64)  # unused values

            # read E field
            nsamps = 2 * nx * ny
            rawx = list(unpack(f"<{nsamps}d", f.read(8 * nsamps)))
            if ispol:
                rawy = list(unpack(f"<{nsamps}d", f.read(8 * nsamps)))

        # convert unit key to unit string
        map_units = {0: "mm", 1: "cm", 2: "in", 3: "m"}
        try:
            unit = map_units[units]
        except KeyError:
            raise KeyError(
                f"Invalid units specified in the zbf file (expected '0', '1', '2', or '3', got '{units}')."
            ) from None

        # load E field
        Ex_real = np.asarray(rawx[0::2]).reshape(nx, ny, order="F")
        Ex_imag = np.asarray(rawx[1::2]).reshape(nx, ny, order="F")
        if ispol:
            Ey_real = np.asarray(rawy[0::2]).reshape(nx, ny, order="F")
            Ey_imag = np.asarray(rawy[1::2]).reshape(nx, ny, order="F")
        else:
            Ey_real = np.zeros((nx, ny))
            Ey_imag = np.zeros((nx, ny))

        Ex = Ex_real + 1j * Ex_imag
        Ey = Ey_real + 1j * Ey_imag

        return ZBFData(
            version=version,
            nx=nx,
            ny=ny,
            ispol=ispol,
            unit=unit,
            dx=dx,
            dy=dy,
            zposition_x=zposition_x,
            zposition_y=zposition_y,
            rayleigh_x=rayleigh_x,
            rayleigh_y=rayleigh_y,
            waist_x=waist_x,
            waist_y=waist_y,
            wavelength=wavelength,
            background_refractive_index=background_refractive_index,
            receiver_eff=receiver_eff,
            system_eff=system_eff,
            Ex=Ex,
            Ey=Ey,
        )



def from_zbf(filename: str, dim1, dim2):

    # get the third dimension
    dim3 = list(set(get_args(xyz)) - {dim1, dim2})[0]
    dims = {"x": 0, "y": 1, "z": 2}
    dim2expand = dims[dim3]  # this is for expanding E field arrays

    # load zbf data
    zbfdata = ZBFData.read_zbf(filename)

    # Grab E fields, dimensions, wavelength
    edim1 = zbfdata.Ex
    edim2 = zbfdata.Ey
    n1 = zbfdata.nx
    n2 = zbfdata.ny
    d1 = zbfdata.dx / UnitScaling[zbfdata.unit]
    d2 = zbfdata.dy / UnitScaling[zbfdata.unit]
    wavelength = zbfdata.wavelength / UnitScaling[zbfdata.unit]

    # make scalar field data arrays
    len1 = d1 * (n1 - 1)
    len2 = d2 * (n2 - 1)
    coords1 = np.linspace(-len1 / 2, len1 / 2, n1)
    coords2 = np.linspace(-len2 / 2, len2 / 2, n2)
    f = [C_0 / wavelength]
    Edim1 = td.ScalarFieldDataArray(
        np.expand_dims(edim1, axis=(dim2expand, 3)),
        coords={
            dim1: coords1,
            dim2: coords2,
            dim3: [0],
            "f": f,
        },
    )
    Edim2 = td.ScalarFieldDataArray(
        np.expand_dims(edim2, axis=(dim2expand, 3)),
        coords={
            dim1: coords1,
            dim2: coords2,
            dim3: [0],
            "f": f,
        },
    )

    return td.FieldDataset(
        **{
            f"E{dim1}": Edim1,
            f"E{dim2}": Edim2,
        }
    )


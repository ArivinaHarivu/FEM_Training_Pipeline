"""Material properties — single isotropic linear-elastic material.

For v1: constant across dataset (Domex 420MC structural steel).
Material properties are loaded from config, never hardcoded in
pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Material:
    """Isotropic linear-elastic material.

    Attributes
    ----------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    sigma_yield : float
        Yield strength [Pa].
    UTS : float
        Ultimate tensile strength [Pa].
    """

    E: float
    nu: float
    sigma_yield: float
    UTS: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Material:
        """Create a Material from the config.yaml material block.

        Parameters
        ----------
        config : dict[str, Any]
            The ``material`` section of config.yaml.

        Returns
        -------
        Material
            Validated material instance.
        """
        return cls(
            E=float(config["E"]),
            nu=float(config["nu"]),
            sigma_yield=float(config["sigma_yield"]),
            UTS=float(config["UTS"]),
        )

    @property
    def lame_lambda(self) -> float:
        """First Lamé parameter λ = Eν / ((1+ν)(1-2ν))."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def lame_mu(self) -> float:
        """Second Lamé parameter μ = shear modulus = E / (2(1+ν))."""
        return self.E / (2 * (1 + self.nu))

    def constitutive_matrix(self) -> np.ndarray:
        """Full 3D constitutive matrix (6×6) for isotropic linear elasticity.

        Relates stress to strain in Voigt notation:
            σ = C · ε
        where both σ and ε use the project-wide Voigt order
        [xx, yy, zz, xy, yz, xz].

        Uses engineering shear strain convention (γ = 2ε_ij) for
        off-diagonal components, so the factor of 2μ appears for
        shear terms (not μ).

        Returns
        -------
        np.ndarray
            Shape (6, 6) constitutive matrix.
        """
        lam = self.lame_lambda
        mu = self.lame_mu

        C = np.array([
            [lam + 2*mu, lam,        lam,        0,  0,  0],
            [lam,        lam + 2*mu, lam,        0,  0,  0],
            [lam,        lam,        lam + 2*mu, 0,  0,  0],
            [0,          0,          0,          mu, 0,  0],
            [0,          0,          0,          0,  mu, 0],
            [0,          0,          0,          0,  0,  mu],
        ], dtype=np.float64)

        return C

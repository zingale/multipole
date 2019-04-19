# compute the potential of a MacLaurin spheroid and compare to the
# analytic solution

import numpy as np
from multipole import Grid, Multipole
import matplotlib.pyplot as plt
from scipy.optimize import brentq

class MacLaurinSpheroid():

    def __init__(self, grid, a1, a3):
        """ spheroid where a1 = a2 > a3"""

        assert a1 > a3

        self.g = grid

        self.a1 = a1
        self.a3 = a3

        self.rho = 1.0

        # eccentricity
        self.e = np.sqrt(1 - (a3/a1)**2)

        self.A1 = np.sqrt(1 - self.e**2)/self.e**3 * np.arcsin(self.e) - (1 - self.e**2)/self.e**2
        self.A3 = 2/self.e**2 - (2*np.sqrt(1 - self.e**2))/self.e**3 * np.arcsin(self.e)

    def _get_inside_mask(self):
        return self.g.r2d**2/self.a1**2 + self.g.z2d**2/self.a3**2 <= 1

    def density(self):
        """ take the axisymmetric r, z coordinates and return the density of the spheroid"""

        inside = self._get_inside_mask()

        dens = self.g.scratch_array()
        dens[inside] = self.rho

        return dens

    def phi(self):
        """return the potential for our spheroid"""

        inside = self._get_inside_mask()
        outside = np.logical_not(inside)

        phi = self.g.scratch_array()

        # outside the spheroid
        phi[outside] = np.pi * self.rho * (2*self.A1*self.a1**2 - self.A1*self.g.r2d[outside]**2 + self.A3*(self.a3**2 - self.g.z2d[outside]**2))

        # inside the spheroid
        lambda_const = self.g.scratch_array()

        for i in range(self.g.nr):
            for j in range(self.g.nz):
                if not inside[i,j]:
                    continue

                # we need the positive root of the lambda equation
                lambda_const[i,j] = brentq(lambda q: self.g.r[i]**2/(self.a1**2 + q) + self.g.z[j]**2/(self.a3**2 + q) - 1.0,
                                           0.0, self.a1)

        h = self.g.scratch_array()

        h[inside] =  self.a1*self.e/np.sqrt(self.a3**2 + self.lambda_const[inside])

        phi[inside] = 2*self.a3/self.e**2 * np.pi * self.rho * (
            self.a1*self.e*self.arctan(h[inside]) - 0.5*(self.r2d[inside]**2 * (np.arctan(h[inside]) - h[inside]/(1 + h[inside]**2)) + 2*self.z2d[inside]**2 * (h[inside] - np.arctan(h[inside]))) )

        return phi

if __name__ == "__main__":

    g = Grid(128, 256, rlim=(0, 0.5), zlim=(-0.5, 0.5))

    ms = MacLaurinSpheroid(g, 0.25, 0.1)

    dens = ms.density()

    plt.imshow(np.transpose(dens), origin="lower",
               interpolation="nearest",
               extent=[g.rlim[0], g.rlim[1],
                       g.zlim[0], g.zlim[1]])

    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig("dens.png")


    phi = ms.phi()

    plt.imshow(np.log10(np.abs(np.transpose(phi))), origin="lower",
               interpolation="nearest",
               extent=[g.rlim[0], g.rlim[1],
                       g.zlim[0], g.zlim[1]])

    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig("phi.png")


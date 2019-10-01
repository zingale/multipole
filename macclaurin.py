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

        # inside the spheroid (Couch et al. Eq. 21)
        phi[inside] = np.pi * self.rho * (2*self.A1*self.a1**2 - self.A1*self.g.r2d[inside]**2 + self.A3*(self.a3**2 - self.g.z2d[inside]**2))

        # outside the spheroid (Couch et al. Eq. 25-27)
        lambda_const = self.g.scratch_array()

        lambda_const = 0.5*(-self.a1**2 - self.a3**2 + self.g.r2d**2 + self.g.z2d**2) + \
            0.5*np.sqrt(self.a1**4 - 2*self.a1**2*self.a3**2 - 2*self.a1**2*self.g.r2d**2 + 2*self.a1**2*self.g.z2d**2 +
                        self.a3**4 + 2*self.a3**2*self.g.r2d**2 - 2*self.a3**2*self.g.z2d**2 +
                        self.g.r2d**4 + 2*self.g.r2d**2*self.g.z2d**2 + self.g.z2d**4)

        h = self.g.scratch_array()

        h[outside] = self.a1*self.e/np.sqrt(self.a3**2 + lambda_const[outside])

        phi[outside] = 2*self.a3/self.e**2 * np.pi * self.rho * (
            self.a1*self.e*np.arctan(h[outside]) - 0.5*(self.g.r2d[outside]**2 * (np.arctan(h[outside]) - h[outside]/(1 + h[outside]**2)) + 2*self.g.z2d[outside]**2 * (h[outside] - np.arctan(h[outside]))) )

        return phi

if __name__ == "__main__":

    # setup the grid
    g = Grid(128, 256, rlim=(0, 0.5), zlim=(-0.5, 0.5))

    # create a MacLaurin spheriod on the grid
    ms = MacLaurinSpheroid(g, 0.25, 0.1)

    # plot the density
    dens = ms.density()

    plt.clf()
    plt.imshow(np.transpose(dens), origin="lower",
               interpolation="nearest",
               extent=[g.rlim[0], g.rlim[1],
                       g.zlim[0], g.zlim[1]])

    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig("dens.png")

    # plot the analytic potential
    phi_analytic = ms.phi()

    plt.clf()
    plt.imshow(np.log10(np.abs(np.transpose(phi_analytic))), origin="lower",
               interpolation="nearest",
               extent=[g.rlim[0], g.rlim[1],
                       g.zlim[0], g.zlim[1]])

    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig("phi_analytic.png")

    # compute the multipole expansion and sample the potential
    center = (0.0, 0.0)
    m = Multipole(g, 12, 2*g.dr, center=center)
    m.compute_expansion(dens)

    phi = g.scratch_array()

    for i in range(g.nr):
        for j in range(g.nz):
            phi[i,j] = m.phi(g.r[i], g.z[j])

    # plot the potential
    plt.clf()
    plt.imshow(np.log10(np.abs(np.transpose(phi))), origin="lower",
               interpolation="nearest",
               extent=[g.rlim[0], g.rlim[1],
                       g.zlim[0], g.zlim[1]])

    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig("phi.png")


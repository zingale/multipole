import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

class Grid():
    def __init__(self, nr, nz, rlim=(0.0, 1.0), zlim=(0.0, 1.0)):
        """an axisymmetric grid"""

        self.nr = nr
        self.nz = nz

        self.rlim = rlim
        self.zlim = zlim

        self.dr = (self.rlim[1] - self.rlim[0])/self.nr
        self.dz = (self.zlim[1] - self.zlim[0])/self.nz

        self.r = (np.arange(self.nr) + 0.5)*self.dr + self.rlim[0]
        self.rl = (np.arange(self.nr))*self.dr + self.rlim[0]
        self.rr = (np.arange(self.nr + 1.0))*self.dr + self.rlim[0]

        self.r2d = np.repeat(self.r, self.nz).reshape(self.nr, self.nz)
        #self.rl2d = np.repeat(self.rl, self.nz).reshape(self.nr, self.nz)
        #self.rr2d = np.repeat(self.rr, self.nz).reshape(self.nr, self.nz)

        self.z = (np.arange(self.nz) + 0.5)*self.dz + self.zlim[0]
        self.z2d = np.transpose(np.repeat(self.z, self.nr).reshape(self.nz, self.nr))

        # pi (r_r^2 - r_l^2) dz
        self.vol = np.pi*2.0*self.r2d*self.dr*self.dz

    def scratch_array(self):
        return np.zeros((nr, nz), dtype=np.float64)

class Multipole():
    def __init__(self, grid, n_moments, dr, center=(0.0, 0.0)):

        self.g = grid
        self.n_moments = n_moments
        self.dr_mp = dr
        self.center = center

        # compute the bins
        r_max = max(abs(self.g.rlim[0] - center[0]), abs(self.g.rlim[1] - center[0]))
        z_max = max(abs(self.g.zlim[0] - center[1]), abs(self.g.zlim[1] - center[1]))

        dmax = np.sqrt(r_max**2 + z_max**2)

        self.n_bins = int(dmax/dr)

        # bin boundaries
        self.r_bin = np.linspace(0.0, dmax, n_bins)

        # storage for the inner and outer multipole moment functions
        # we'll index the list by multipole moment l
        self.m_r = []
        self.m_i = []
        for l in range(self.n_moments):
            self.m_r.append(np.zeros((self.n_bins), dtype=np.float64))
            self.m_i.append(np.zeros((self.n_bins), dtype=np.float64))

    def compute_expansion(self, rho):
        # rho is density that lives on a grid self.g

        # loop over cells
        for i in range(self.g.nr):
            for j in range(self.g.nz):

                # for each cell, i,j, compute r and theta (polar angle from z)
                # and determine which shell we are in
                r = np.sqrt((self.g.r[i] - center[0])**2 +
                            (self.g.z[j] - center[1])**2)

                theta = np.atan(self.g.z[j], self.g.r[i])

                # loop over the multipole moments, l (m = 0 here)
                m_zone = rho[i,j] * self.g.vol[i,j]

                for l in range(n_moments):

                    # compute Y_l^m (note: we use theta as the polar
                    # angle, scipy is opposite)
                    Y_lm = sph_harm(0, l, 0.0, theta)

                    R_lm = np.sqrt(4*np.pi/(2*l + 1)) * r**l * Y_lm
                    I_lm = np.sqrt(4*np.pi/(2*l + 1)) * Y_lm / r**(l+1)

                    # add to the all of the appropriate inner or outer
                    # moment functions
                    imask = r <= self.r_bin
                    omask = r > self.r_bin

                    self.m_r[l][imask] += R_lm * m_zone
                    self.m_i[l][omask] += I_lm * m_zone

    def sample_mtilde(self, r):
        # this returns the result of Eq. 19


    def phi(self, r, z):
        # return Phi(r), using Eq. 20


g = Grid(128, 256, xlim=(0, 0.5))

p = PointMasses(g, 10)

p.plot()


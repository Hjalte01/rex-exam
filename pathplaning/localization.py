
from pathplaning.grid import Position, Grid
# from grid import Position, Grid
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

# from types import List, Tuple

NOISE_THETA = 0.2
FACTOR_THETA = 1/np.sqrt(2*np.pi*np.power(NOISE_THETA, 2))
NOISE_DELTA = 0.05
FACTOR_DELTA = 1/np.sqrt(2*np.pi*np.power(NOISE_DELTA, 2))

class ParticleFilter(object):
    def __init__(self, n: int, grid: Grid):
        super(ParticleFilter, self).__init__()
        self.n = n
        self.grid = grid
        self.particles = np.ndarray((self.n, 3))
        self.particles[:, 0] = rnd.uniform(0, len(grid), self.n)
        self.particles[:, 1] = rnd.uniform(0, len(grid), self.n)
        self.particles[:, 2] = rnd.uniform(0, 2 * np.pi, self.n)
 
    def update(self, control: tuple[float, float], poses: list[Position]):
        if (len(self.grid.markers) < 2):
            return None, None
        
        # Predict step
        # Update where the particles are heading 
        self.particles[:, 2] += rnd.normal(control[1], NOISE_THETA, self.n)
        self.particles[:, 2] %= 2 * np.pi 
        
        # Move particles according to control
        delta = rnd.normal(control[0], NOISE_DELTA, self.n)
        self.particles[:, 0] += (delta)*np.cos(self.particles[:, 2])
        self.particles[:, 1] += (delta)*np.sin(self.particles[:, 2]) 
        weights = np.ones((self.n, 1))

            # Update step 
        for marker in poses:
            dist = np.sqrt(
                (marker.x - self.particles[:, 0])**2 + 
                (marker.y - self.particles[:, 1])**2
            )
            dist = np.array(dist).reshape(-1, 1).transpose()

            marker_vec = np.array((marker.x - self.particles[:, 0], marker.y - self.particles[:, 1]))/dist # el
            orientation_vec = np.array((np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2]))) # et
            hat_vec = np.array((-np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2]))) # eth
            
            temp1 = np.sum(marker_vec * hat_vec, axis=0).reshape(-1, 1) # el * eth 
            temp2 = np.sum(marker_vec * orientation_vec, axis=0).reshape(-1, 1)  #, el * et
            angle = np.sign(temp1) * np.arccos(temp2) # el * eth , el * et
            angle = np.array(angle)
            dist = dist.transpose()

            weights *= FACTOR_DELTA*np.exp(-(marker.delta - dist)**2/((2*NOISE_DELTA)**2)) # pose.delta - 
            weights *= FACTOR_THETA*np.exp(-(marker.rad - angle)**2/((2*NOISE_THETA)**2)) # pose.rad -
        
        weights += 1.e-300 
        weights /= sum(weights)
        
        # resample step
        indexes = self.residual_resample(weights)
        self.resample_from_index(indexes, weights)

        # estimate step
        pos = np.average(self.particles[:, :2], weights=weights, axis=0)
        orientation = np.average(self.particles[:, 2], weights=weights)

        return (pos[0]*self.grid.zone_size, pos[1]*self.grid.zone_size), orientation


    # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
    def residual_resample(self, weights):
        indexes = np.zeros(self.n, 'i')

        # take int(N*w) copies of each weight
        num_copies = (self.n*np.asarray(weights)).astype(int)
        k = 0
        for i in range(self.n):
            for _ in range(num_copies[i]): # make n copies
                indexes[k] = i
                k += 1

        # use multinormial resample on the residual to fill up the rest.
        residual = self.n * np.asarray(weights) - num_copies     # get fractional part
        residual /= sum(residual)     # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1. # ensures sum is exactly one
        indexes[k:self.n] = np.searchsorted(cumulative_sum, rnd.random(self.n-k))

        return indexes
    
    def resample_from_index(self, indexes, weights: np.ndarray):
        self.particles[:] = self.particles[indexes]
        weights.resize(len(self.particles))
        weights.fill(1.0/self.n)

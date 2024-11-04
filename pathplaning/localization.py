
from pathplaning.grid import Pose, Grid
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
    def __init__(self, grid: Grid, n: int):
        super(ParticleFilter, self).__init__()
        self.n = n
        self.grid = grid
        self.particles = np.ndarray((self.n, 3))
        self.particles[:, 0] = rnd.uniform(0, len(grid), self.n)
        self.particles[:, 1] = rnd.uniform(0, len(grid), self.n)
        self.particles[:, 2] = rnd.uniform(0, 2*np.pi, self.n)
        self.particles[:, 2] %= 2*np.pi
        # plt.ion()
        # plt.scatter(self.particles[:, 0], self.particles[:, 1],
        #     color='k', marker=',', s=1)
        # plt.draw()
        # plt.pause(0.01)
 
    def update(self, control: list[float, float], poses: list[Pose]):
        if (len(self.grid.markers) < 2):
            return None, None
        
        
        # Predict step
        # Update where the particles are heading 
        self.particles[:, 2] += rnd.normal(control[1], NOISE_THETA, self.n)
        self.particles[:, 2] %= 2 * np.pi 
        
        # Move particles according to control
        delta = rnd.normal(control[0]/self.grid.zone_size, NOISE_DELTA, self.n)
        self.particles[:, 0] += (delta)*np.cos(self.particles[:, 2])
        self.particles[:, 1] += (delta)*np.sin(self.particles[:, 2]) 
        weights = np.ones((self.n))

            # Update step 
        for marker in poses:
            deltas = np.sqrt(
                (marker[0]/self.grid.zone_size - self.particles[:, 0])**2 +
                (marker[1]/self.grid.zone_size - self.particles[:, 1])**2 
            )


            marker_vec = np.array([
                marker[0]/self.grid.zone_size - self.particles[:, 0], 
                marker[1]/self.grid.zone_size - self.particles[:, 1]]
            )/deltas # el
            orientation_vec = np.array([np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2])]) # et
            hat_vec = np.array([-np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2])]) # eth
            
            temp1 = np.sum(marker_vec*hat_vec, axis=0) # el * eth 
            temp2 = np.sum(marker_vec*orientation_vec, axis=0) #, el * et
            thetas = np.sign(temp1)*np.clip(np.arccos(temp2), -1, 1) # el * eth , el * et
            thetas %= 2*np.pi

            weights *= FACTOR_DELTA*np.exp(-(marker.delta/self.grid.zone_size - deltas)**2/((2*NOISE_DELTA)**2)) # pose.delta - 
            weights *= FACTOR_THETA*np.exp(-(marker.rad - thetas)**2/((2*NOISE_THETA)**2)) # pose.rad -
            weights += 1.e-300 
        
        weights /= sum(weights)
        
        # resample step
        indexes = self.residual_resample(weights)
        self.resample_from_index(indexes, weights)
        # plt.scatter(self.particles[:, 0], self.particles[:, 1],
        #     color='k', marker=',', s=1)
        # plt.draw()
        # plt.pause(0.01)
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

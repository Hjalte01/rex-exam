from grid import Position, Grid
import numpy as np
from numpy import random as rnd
# from types import List, Tuple

noise_dist = 0.1
noise_angle = 0.1
sigma_dist = 1/np.sqrt(2*np.pi*np.power(noise_dist, 2))
sigma_angle = 1/np.sqrt(2*np.pi*np.power(noise_angle, 2))

class ParticleFilter(object):
    def __init__(self, initial_pos: Position, n: int, grid: Grid):
        super(ParticleFilter, self).__init__()
        self.n = n
        self.grid = grid
        self.initial_pos = initial_pos
        self.particles = np.ndarray((self.n, 3))
        # TODO - test if uniform inital is better, since we don't know where we start so initial_pos.x is undefined
        self.particles[:, 0] = rnd.normal(initial_pos.x, noise_dist, self.n)
        self.particles[:, 1] = rnd.normal(initial_pos.y, noise_dist, self.n)
        self.particles[:, 2] = rnd.normal(initial_pos.rad, noise_angle, self.n)
        self.weights = np.array((self.n))
        self.weights.fill(1/self.n)
    

    def update(self, control: tuple[float, float], poses: list[Position]):
        if (len(poses) < 2):
            return None, None
        
        # Predict step
        # Update where the particles are heading 
        self.particles[:, 2] += rnd.normal(control[1], noise_angle, self.n)
        self.particles[:, 2] %= 2 * np.pi 
        
        # Move particles according to control
        delta = rnd.normal(control[0], noise_dist, self.n)
        self.particles[:, 0] += (delta)*np.cos(self.particles[:, 2])
        self.particles[:, 1] += (delta)*np.sin(self.particles[:, 2]) 
        weights = np.ones((self.n, 1))

        # Update step
        for pose in poses:
            dist = np.sqrt((self.particles[:, 0] - pose.cx)**2
                + (self.particles[:, 1] - pose.cy)**2)
            dist = np.array(dist).reshape(-1, 1)

            orientation_vec = np.array((np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2]))) # et
            marker_vec = np.array((pose.cx - self.particles[:, 0], pose.cy - self.particles[:, 1]))/dist # el
            hat_vec = (-np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2])) # eth
            angle = np.sign(np.dot(marker_vec, hat_vec))*np.arccos(np.dot(marker_vec, orientation_vec)) # el * eth , el * et
            angle = np.array(angle)

            weights *= sigma_dist * np.exp(-((dist*0.99) - dist)**2/((2*noise_dist)**2)) # pose.delta - 
            print(np.array(weights).shape, angle.shape, dist.shape)
            weights *= sigma_angle*np.exp(-((angle * 0.99) - angle)**2/((2*noise_angle)**2)) # pose.rad -

        weights += 1.e-300 
        weights /= sum(weights)
        
        # resample step
        indexes = self.residual_resample(weights)
        self.resample_from_index(indexes, weights)

        # estimate step
        pos = np.average(self.particles[:, :2], weights=weights, axis=0)
        orientation = np.average(self.particles[:, 2], weights=weights)

        return self.grid.transform_xy(pos[0], pos[1]), orientation


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

    def run_pf(self):

        
        cell, orientation = self.update((1, 0), self.grid.markers)
        print(cell, orientation)
        exit(0)

    

def main():

    grid = Grid((4, 4), 450)
    grid.create_marker(grid[4, 1].diffuse(), grid[4, 1][4, 4], 1)
    grid.create_marker(grid[2, 1].diffuse(), grid[2, 1][4, 4], 2)
    pf = ParticleFilter(Position(1, 0), 100, grid)
    pf.run_pf()

main()
        


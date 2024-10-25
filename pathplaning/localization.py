from grid import Position, Grid
import numpy as np
from numpy import random as rnd
# from types import List, Tuple

noise_dist = 150
noise_angle = 0.2
sigma_dist = 1/np.sqrt(2*np.pi*np.power(noise_dist, 2))
sigma_angle = 1/np.sqrt(2*np.pi*np.power(noise_angle, 2))

class ParticleFilter(object):
    def __init__(self, n: int, grid: Grid):
        super(ParticleFilter, self).__init__()
        self.n = n
        self.grid = grid
        self.particles = np.ndarray((self.n, 3))

        self.particles[:, 0] = rnd.uniform(0, len(grid) * grid.zone_size, self.n)
        self.particles[:, 1] = rnd.uniform(0, len(grid) * grid.zone_size, self.n)
        self.particles[:, 2] = rnd.uniform(0, 2 * np.pi, self.n)
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

        # take a photo get markers
        observed_markers = {
            1: (450*9, 0*np.pi/180),
            2: (450*9, 90*np.pi/180)
        }
        if not isinstance(observed_markers, type(None)):
            # Update step 
            for index, (id, (marker_delta, marker_rad)) in enumerate(observed_markers.items()):
                marker = next((m for m in self.grid.markers if m.id == id), None)
                # print(id, marker)
                if marker is None:
                    continue
                # print(marker.id, marker.cx, marker.cy)

                dist = np.sqrt((marker.cx - self.particles[:, 0])**2
                    + (marker.cy - self.particles[:, 1])**2)
                dist = np.array(dist).reshape(-1, 1).transpose()

                marker_vec = np.array((marker.cx - self.particles[:, 0], marker.cy - self.particles[:, 1]))/dist # el
                orientation_vec = np.array((np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2]))) # et
                hat_vec = np.array((-np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2]))) # eth
                
                temp1 = np.sum(marker_vec * hat_vec, axis=0).reshape(-1, 1) # el * eth 
                temp2 = np.sum(marker_vec * orientation_vec, axis=0).reshape(-1, 1)  #, el * et
                angle = np.sign(temp1) * np.arccos(temp2) # el * eth , el * et
                angle = np.array(angle)            
                dist = dist.transpose()
                
                weights *= sigma_dist * np.exp(-(marker_delta - dist)**2/((2*noise_dist)**2)) # pose.delta - 
                weights *= sigma_angle*np.exp(-(marker_rad - angle)**2/((2*noise_angle)**2)) # pose.rad -
            
            print(min(weights), max(weights))
            
            weights += 1.e-300 
            weights /= sum(weights)
            
            # resample step
            indexes = self.residual_resample(weights)
            self.resample_from_index(indexes, weights)
        else:
            for i in len(weights):
                weights[i] = 1/self.n
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
            for _ in range(num_copies[i][0]): # make n copies
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

        for i in range(15):        
            cell, orientation = self.update((450, 0.25*np.pi))
            print(cell, orientation*180/np.pi)

    

def main():

    grid = Grid((6, 1), 450)
    grid.create_marker(grid[0, 0].diffuse(), grid[0, 0][0, 0], 1)
    grid.create_marker(grid[0, 8].diffuse(), grid[0, 8][0, 0], 2)
    grid.create_marker(grid[8, 0].diffuse(), grid[8, 0][0, 0], 3)
    grid.create_marker(grid[8, 8].diffuse(), grid[8, 8][0, 0], 4)
    # observered postition from particles to markers


    pf = ParticleFilter(100000, grid)
    pf.run_pf()

main()
        


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

        self.particles[:, 0] = rnd.uniform(0, 20, self.n)
        self.particles[:, 1] = rnd.uniform(0, 20, self.n)
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

        if not isinstance(poses, type(None)):
            # Update step 
            for marker in poses:
                # print(id, marker)
                if marker is None:
                    continue

                dist = np.sqrt((marker.x - self.particles[:, 0])**2
                    + (marker.y - self.particles[:, 1])**2)
                dist = np.array(dist).reshape(-1, 1).transpose()

                marker_vec = np.array((marker.x - self.particles[:, 0], marker.y - self.particles[:, 1]))/dist # el
                orientation_vec = np.array((np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2]))) # et
                hat_vec = np.array((-np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2]))) # eth
                
                temp1 = np.sum(marker_vec * hat_vec, axis=0).reshape(-1, 1) # el * eth 
                temp2 = np.sum(marker_vec * orientation_vec, axis=0).reshape(-1, 1)  #, el * et
                angle = np.sign(temp1) * np.arccos(temp2) # el * eth , el * et
                angle = np.array(angle)
                dist = dist.transpose()
                
                weights *= sigma_dist * np.exp(-(marker.delta - dist)**2/((2*noise_dist)**2)) # pose.delta - 
                weights *= sigma_angle*np.exp(-(marker.rad - angle)**2/((2*noise_angle)**2)) # pose.rad -
            
            
            weights += 1.e-300 
            weights /= sum(weights)
            
            # resample step
            indexes = self.residual_resample(weights)
            self.resample_from_index(indexes, weights)
        else:
            for i in range(len(weights)):
                weights[i] = 1/self.n
        # estimate step
        pos = np.average(self.particles[:, :2], weights=weights, axis=0)
        orientation = np.average(self.particles[:, 2], weights=weights)

        # return self.grid.transform_xy(pos[0], pos[1]), orientation
        return (pos[0], pos[1]), orientation


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
        n = 5
        marker = [
            [10, 20],
            [20, 10],
            [10, 0],
            [0, 10]
        ]
        for i in range(1, n):      
            #
            dist = np.sqrt(1**2+ 1**2) # 45 cm
            theta = np.pi/4 # 45 deg
            marker_poses = []
            for mark in marker:
                dist = np.sqrt((mark[0] - i)**2 + (mark[1] - i)**2)
                marker_theta = np.arctan2(mark[0]-i, mark[1]-i) 
                # marker_theta = (marker_theta + 2*np.pi) % (2*np.pi)
                # marker_theta = marker_theta - theta
                temp = Position(dist, marker_theta)
                # temp.x = mark[0]
                # temp.y = mark[1]
                marker_poses.append(temp)
                
                
                # print(marker_theta, marker_theta*180/np.pi)
        
            cell, orientation = self.update((dist, theta), marker_poses)
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
        


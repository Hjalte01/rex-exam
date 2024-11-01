
# from pathplaning.grid import Position, Grid
from grid import Position, Grid
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

# from types import List, Tuple

noise_angle = 5
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
    

    def update(self, control: tuple[float, float], poses: list[Position], visualize: bool=False):
        if (len(poses) < 2):
            return None, None
        
        noise_dist = control[0]/10
        sigma_dist = 1/np.sqrt(2*np.pi*np.power(noise_dist, 2))
        
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
                
                weights *= sigma_dist 
                dist_delta = marker.delta - dist
                noise_term = (2*noise_dist)**2
                exp_term = -(dist_delta**2)/(noise_term)
                weights *= np.exp(exp_term) # pose.delta - 
                # weights *= sigma_dist * np.exp(-(marker.delta - dist)**2/((2*noise_dist)**2)) # pose.delta - 
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

        # visualization option
        if visualize:
            self.visualize_particles(poses)
        # return self.grid.transform_xy(pos[0], pos[1]), orientation
        print(pos[0], pos[1])
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


    def visualize_particles(self, poses: list[Position]):
        plt.clf()  # Clear the figure for the next frame
        plt.scatter(self.particles[:, 0], self.particles[:, 1], s=10, c='blue', label="Particles")
        
        # Plot the markers
        for pose in poses:
            if pose is not None:
                # show know markers in red and unknow in black
                if pose.id < 5:
                    # label just ensure that we don't see 4 markers in legend, but just one unique "marker" that represent all of them
                    plt.plot(pose.x, pose.y, 'ro', markersize=8, label="Marker" if 'Marker' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.plot(pose.x, pose.y, 'ko', markersize=8, label="Unknown Marker" if 'Unknown Marker' not in plt.gca().get_legend_handles_labels()[1] else "")
        global index
        plt.plot(index, index, 'c*', markersize=16, label="Robot pos")
        # plt.xlim(0, 20)
        # plt.ylim(0, 20)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.draw()
        plt.pause(0.5)  # Pause briefly to allow animation effect




    # global index used for plotting our location(green dot) remove this at some point af testing
    index = 1
    def run_pf(self):
        n = 9 * 450
        marker = [[n//2, n, 1], [n, n//2, 2], [n//2, 0, 3], [0, n//2, 4]]
        # we only know the dist and angle to these markers, 
        # but use in testing x and y for calculation of dist & angle
        unknown_marker = [[n//4, n//1.3, 5], [n//1.2, n//3, 6]] 
        origo_xy = (0, 0)

        for i in range(450//2, n, 450):      
            global index
            index = i
            diag = 450
            dist = np.sqrt(diag**2 + diag**2) # 45 cm
            theta = np.pi/4 # 45 deg
            marker_poses = []

            for mark in marker:
                marker_dist = np.sqrt((mark[0] - i)**2 + (mark[1] - i)**2)
                marker_theta = np.arctan2(mark[1]-i, mark[0]-i) 
                marker_theta = (marker_theta + 2*np.pi) % (2*np.pi)
                temp = Position(marker_dist, marker_theta, mark[2])
                temp.x = mark[0]
                temp.y = mark[1]
                marker_poses.append(temp)


            for mark in unknown_marker:
                marker_dist = np.sqrt((mark[0] - i)**2 + (mark[1] - i)**2)
                marker_theta = np.arctan2(mark[1]-i, mark[0]-i) 
                marker_theta = (marker_theta + 2*np.pi) % (2*np.pi)
                # marker_theta = marker_theta - theta
                temp = Position(marker_dist, marker_theta, mark[2])
                temp.x = temp.x + origo_xy[0] + np.cos(theta)*dist # from origin to x + guess pos.x + current direction * delta
                temp.y = temp.y + origo_xy[1] + np.sin(theta)*dist
                # print(f"({temp.x}, {temp.y}) - ({guess[0]}, {guess[1]}) & {marker_dist}")
                marker_poses.append(temp)
                
            origo_xy, orientation = self.update((dist, theta), marker_poses, True)
            
            # guess = (cell.zone.col * cell.zone.size + cell.col * cell.size, 
            #          cell.zone.row * cell.zone.size + cell.row * cell.size)
            # print(f"""grid[{cell.zone.col}, {cell.zone.row}][{cell.col}, {cell.row}] = ({guess[0]}, {guess[1]})\n zone_size: {cell.zone.size}, cell_size: {cell.size}\n""")
            print(origo_xy)
    

def main():

    grid = Grid((0, 0), 450)
    grid.create_marker(grid[0, 0].diffuse(), grid[0, 0][0, 0], 1)
    grid.create_marker(grid[0, 8].diffuse(), grid[0, 8][0, 0], 2)
    grid.create_marker(grid[8, 0].diffuse(), grid[8, 0][0, 0], 3)
    grid.create_marker(grid[8, 8].diffuse(), grid[8, 8][0, 0], 4)
    # observered postition from particles to markers


    pf = ParticleFilter(100000, grid)
    pf.run_pf()
if __name__ == "__main__":
    main()
        

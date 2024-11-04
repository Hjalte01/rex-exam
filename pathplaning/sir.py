from typing import List, Tuple
from pathplaning.grid import Pose, Position
# from grid import Position
import numpy as np

def create_particles_uniform(n: int, xy_max: float, theta_max: float):
    particles = np.empty((n, 3))
    particles[:, 0] = np.random.uniform(0, xy_max, n)
    particles[:, 1] = np.random.uniform(0, xy_max, n)
    particles[:, 2] = np.random.uniform(0, theta_max, n)
    particles[:, 2] %= 2*np.pi
    return particles

def create_particles_normal(n, xy, theta):
    particles = np.empty((n, 3))
    particles[:, 0] = xy[0] + (np.random.randn(n)*5)
    particles[:, 1] = xy[1] + (np.random.randn(n)*5)
    particles[:, 2] = theta + (np.random.randn(n)*0.25*np.pi)
    particles[:, 2] %= 2*np.pi
    return particles

def predict(u: Tuple[float, float], particles: np.ndarray, std: Tuple[float, float]):
    n = len(particles)

    particles[:, 2] += u[1] + np.random.randn(n)*std[1]
    particles[:, 2] %= 2*np.pi

    delta = u[0] + np.random.randn(n)*std[0]
    particles[:, 0] += delta * np.cos(particles[:, 2])
    particles[:, 1] += delta * np.sin(particles[:, 2])

    return particles

def update(particles: np.ndarray, weights: np.ndarray, zs: np.ndarray, markers: np.ndarray, std: Tuple[float, float]):
    # zs are robot measurements.
    for delta, theta, marker in zip(zs[0], zs[1], markers):
        deltas = np.linalg.norm(particles[:, 0:2] - marker, axis=1)
        weights *= 1/np.sqrt(2*np.pi*std[0]**2)*np.exp(-(delta - deltas)**2/(2*std[0]**2))

    vs = np.array([
        [marker[0] - particles[:, 0], marker[1] - particles[:, 1]], # marker
        [np.cos(particles[:, 2]), np.sin(particles[:, 2])], # unit
        [-np.sin(particles[:, 2]), np.cos(particles[:, 2])] # perpendicular
    ])
    vs[0] /= deltas
    thetas = np.sign(np.einsum('ij,ij->i', vs[0].T, vs[2].T))*np.arccos(np.clip(np.einsum('ij,ij->i', vs[0].T, vs[1].T), -1, 1))
    thetas %= 2*np.pi
    weights *= 1/np.sqrt(2*np.pi*std[1]**2)*np.exp(-(theta - thetas)**2/(2*std[1]**2))

    weights += 1.e-300
    weights /= sum(weights)
    return weights

def estimate(particles: np.ndarray, weights: np.ndarray):
    xy = np.average(particles[:, 0:2], weights=weights, axis=0)
    theta = np.average(particles[:, 2], weights=weights, axis=0)
    return xy*450, theta, (np.average((particles[:, 0:2] - xy)**2, weights=weights, axis=0), np.average((particles[:, 2] - theta)**2, weights=weights, axis=0))

def resample_index(particles: np.ndarray, weights: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    particles[:] = particles[indices]
    weights.resize(len(particles))
    weights.fill(1.0/len(weights))
    return particles, weights

def resample_systematic(weights: np.ndarray) -> np.ndarray:
    n = len(weights)
    positions = np.arange(n + np.random.random())/n
    
    indices = np.zeros(n, np.int32)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices

def residual_resample(weights):
    indexes = np.zeros(5000, 'i')

    # take int(N*w) copies of each weight
    num_copies = (5000*np.asarray(weights)).astype(int)
    k = 0
    for i in range(5000):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormial resample on the residual to fill up the rest.
    residual = 5000 * np.asarray(weights) - num_copies     # get fractional part
    residual /= sum(residual)     # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # ensures sum is exactly one
    indexes[k:5000] = np.searchsorted(cumulative_sum, np.random.random(5000-k))

    return indexes

def particle_filter_update(u: Tuple[float, float], particles: np.ndarray, poses: List[Pose]):
    # TODO: Convert to class.
    # NOTE: PF should probably not own transformation of xy to cell, so handle this elsewhere (state/task).
    zs = np.array([(p.delta/450, p.rad) for p in poses])
    markers = np.array([[p.x/450, p.y/450] for p in poses])
    weights = np.ones(len(particles))
    particles = predict((u[0]/450, u[1]), particles, (0.045, 0.2))                  # Tuple[noise_delta, noise_theta]
    weights = update(particles, weights, zs, markers, (0.1, 0.1))  # Tuple[noise_delta, noise_theta]
    particles, weights = resample_index(particles, weights, resample_systematic(weights))
    return estimate(particles, weights)

def main():
    markers = np.array([[0, 10], [20, 10], [10, 0], [10, 20]])
    particles = create_particles_uniform(5000, 9, 2*np.pi)
    robot = np.array((0.0, 0.0)) # True position.
    u = (np.sqrt(2), 0.25*np.pi) # Control.

    for _ in range(10):
        robot += (1.0, 1.0)
        zs = np.linalg.norm(markers - robot, axis=1), np.arctan2(markers[:, 1] - robot[1], markers[:, 0] - robot[0])
        zs_pos = []
        print(zs)
        for (dist, theta) in zs:
            zs_pos(Position(dist, theta))


        (x, y), heading, ((var_x, var_y), var_heading) = particle_filter_update(u, particles, zs_pos)
        # robot = np.array([x, y]) # This is tempting, but xy is a guess and not the true position.

# np.random.seed(1)
if __name__ == "__main__":
    main()

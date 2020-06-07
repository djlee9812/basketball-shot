import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from multiprocessing import Pool
from simulate import Ball
from multiprocessing import Pool
import time
import os
plt.style.use('seaborn')

def data_gen(speeds, phis, h, thetas, omegas):
    for i, v in enumerate(speeds):
        for j, phi in enumerate(phis):
            yield i, j, v, phi, h, thetas[i,j], omegas[i,j]

def compute_shot(args):
    i, j, v, phi, h, theta, omega = args
    ball = Ball(15, 0, h, v, phi, theta, omega)
    return i, j, ball.score

def noisy_data_gen(speeds, phis, h, thetas, omegas, num_trial):
    for i, v in enumerate(speeds):
        for j, phi in enumerate(phis):
            yield i, j, v, phi, h, thetas[i,j], omegas[i,j], num_trial

def noisy_compute_shot(args):
    i, j, v, phi, h, theta, omega, num_trial = args
    count = 0
    for t in range(num_trial):
        theta_i = theta + np.random.normal(0, 1.2)
        phi_i = phi + np.random.normal(0, 3)
        v_i = v + np.random.normal(0, 0.6)
        ball = Ball(15, 0, h, v_i, phi_i, theta_i, omega)
        count += ball.score
        if t > 3 and count == 0: break
    return i, j, count

def free_throws(height, noise=False, data_file="temp.npz", save=False):
    nspeed = 100
    nphi = 100
    speeds = np.linspace(23, 34, nspeed)
    phis = np.linspace(35, 70, nphi)
    thetas = np.zeros((nspeed, nphi))
    omegas = 5 * np.ones((nspeed, nphi))
    scored = np.zeros((nphi, nspeed))
    start = time.time()
    pool = Pool()
    if noise:
        num_trials = 10
        results = pool.map(noisy_compute_shot,
                           noisy_data_gen(speeds, phis, height, thetas,
                                          omegas, num_trials))
    else:
        results = pool.map(compute_shot, data_gen(speeds, phis, height, thetas,
                                                  omegas))
    for i, j, score in results:
        scored[j, i] = score
    np.savez(data_file, speeds=speeds, phis=phis, thetas=thetas,
             omegas=omegas, scored=scored)
    print(np.round((time.time() - start)/(nspeed*nphi),3), "secs per shot")
    plot_data(data_file)
    if not save:
        os.remove(data_file)

def plot_data(data_file):
    data = np.load(data_file)
    speeds, phis, thetas, omegas, scored = [data[arr] for arr in data.files]
    plt.figure()
    X, Y = np.meshgrid(speeds, phis)
    plt.pcolormesh(X, Y, scored)
    plt.xlabel("Speeds [ft/s]")
    plt.ylabel("Launch Angle [deg]")
    plt.title("Free Throws at Various Launch Speeds and Angles")
    plt.tight_layout()
    plt.show()
    data.close()



if __name__ == "__main__":
    # free_throws(6)
    plot_data("data.npz")

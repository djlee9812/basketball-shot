import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from multiprocessing import Pool
from simulate import Ball
from multiprocessing import Pool
import time
plt.style.use('seaborn')

def compute_shot(args):
    i, j, v, phi, h, theta, omega = args
    ball = Ball(15, 0, h, v, phi, theta, omega)
    return i, j, ball.score

def data_gen(speeds, phis, h, thetas, omegas):
    for i, v in enumerate(speeds):
        for j, phi in enumerate(phis):
            yield i, j, v, phi, h, thetas[i,j], omegas[i,j]

def free_throws(height, data_file, save=False):
    nspeed = 100
    nphi = 100
    speeds = np.linspace(23, 34, nspeed)
    phis = np.linspace(35, 70, nphi)
    thetas = np.zeros((nspeed, nphi))
    omegas = 5 * np.ones((nspeed, nphi))
    scored = np.zeros((nphi, nspeed))
    start = time.time()
    pool = Pool()
    results = pool.map(compute_shot, data_gen(speeds, phis, height, thetas, omegas))
    for i, j, score in results:
        scored[j, i] = score
    if save: np.savez(data_file, speeds=speeds, phis=phis, thetas=thetas,
                      omegas=omegas, scored=scored)
    print(np.round((time.time() - start)/(nspeed*nphi),3), "secs per shot")
    plot_data(data_file)

def noisy_free_throws(height, data_file, save=False):
    pass

def plot_data(data_file):
    data = np.load(data_file)
    speeds, phis, thetas, omegas, scored = [data[arr] for arr in data.files]
    plt.figure()
    X, Y = np.meshgrid(speeds, phis)
    plt.pcolormesh(X, Y, scored)
    plt.xlabel("Speeds [ft/s]")
    plt.ylabel("Launch Angle [deg]")
    plt.show()
    data.close()



if __name__ == "__main__":
    # free_throws(6, "data.npz")
    plot_data("data.npz")

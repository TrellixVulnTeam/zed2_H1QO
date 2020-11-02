from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
from pycpd import RigidRegistration,AffineRegistration
import numpy as np
import open3d


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)
def save_dt(iteration, error, X, Y):
    # plt.cla()
    # ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    # ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    # ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
    #     iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    # plt.draw()
    # plt.pause(0.001)
    print(iteration,X.shape,Y.shape)

def main():
    # fish_target = np.loadtxt('data/fish_target.txt')
    ft='C:/00_work/05_src/data/frm_t/20201015155835/pcd_extracted.ply'
    fs='C:/00_work/05_src/data/frm_t/20201015155844/pcd_extracted.ply'
    pcdt = open3d.io.read_point_cloud(ft)
    pcds = open3d.io.read_point_cloud(fs)
    voxel_size=0.01
    keypoints_src = pcds.voxel_down_sample(voxel_size)
    keypoints_tgt = pcdt.voxel_down_sample(voxel_size)
    fish_source = np.array(keypoints_src.points)
    fish_target = np.array(keypoints_tgt.points)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)
    # callback = partial(save_dt)

    # reg = DeformableRegistration(**{'X': fish_target, 'Y': fish_source})
    reg = RigidRegistration(**{'X': fish_target, 'Y': fish_source})
    # reg = AffineRegistration(**{'X': fish_target, 'Y': fish_source})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()

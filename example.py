import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from camera import Camera
import structure
import processor
import features

# Download images from http://www.robots.ox.ac.uk/~vgg/data/data-mview.html

def dino(img1_path, img2_path):
    # Dino
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    # cv2.waitKey()

    pts1, pts2 = features.find_correspondence_points(img1, img2)
    points1 = processor.cart2hom(pts1)
    points2 = processor.cart2hom(pts2)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].autoscale_view('tight')
    # ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # ax[0].plot(points1[0], points1[1], 'r.')
    # ax[1].autoscale_view('tight')
    # ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # ax[1].plot(points2[0], points2[1], 'r.')
    # fig.show()

    height, width, ch = img1.shape
    intrinsic = np.array([  # for dino
        [2360, 0, width / 2],
        [0, 2360, height / 2],
        [0, 0, 1]])

    return points1, points2, intrinsic

def process_each_pair(img1_path, img2_path):
    points1, points2, intrinsic = dino(img1_path, img2_path)

    # Calculate essential matrix with 2d points.
    # Result will be up to a scale
    # First, normalize points
    points1n = np.dot(np.linalg.inv(intrinsic), points1)
    points2n = np.dot(np.linalg.inv(intrinsic), points2)

    # print('intrinsic:', intrinsic.shape, intrinsic)
    # print('points1', points1n.shape)
    # print('points2', points2n.shape)

    E = structure.compute_essential_normalized(points1n, points2n)
    # print('Computed essential matrix:', (-E / E[0][1]))

    # Given we are at camera 1, calculate the parameters for camera 2
    # Using the essential matrix returns 4 possible camera paramters
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = structure.compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        # Find the correct camera parameters
        d1 = structure.reconstruct_one_point(
            points1n[:, 0], points2n[:, 0], P1, P2)

        # Convert P2 from camera view to world view
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    #tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
    tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)

    return tripoints3d

if __name__ == '__main__':

    is_init = False
    file_names = ['imgs/dino/'+file_name for file_name in os.listdir('imgs/dino')]
    for i in range(2):
        tripoints3d = process_each_pair(file_names[i], file_names[i+1])
        if not is_init:
            points = tripoints3d
            is_init = True
        else:
            points = np.concatenate((points, tripoints3d), axis=1)

    print('[RESULT] points', points.shape)

    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(points[0], points[1], points[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()

    cv2.destroyAllWindows()
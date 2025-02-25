# -*- coding: utf-8 -*-
"""
Created on 2 Dec 2020
Updated 17 Jan 2025
@author: chatoux, métais, mathieu, dunand
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import stereovision

def CameraCalibration(Size, path, path1image, savename):
    """
    Calibre une caméra en utilisant des images d'échiquier et rectifie une image donnée.
    Paramètres:
    Size (tuple): Le nombre de coins intérieurs par ligne et colonne de l'échiquier (lignes, colonnes).
    path (str): Le chemin vers le répertoire contenant les images d'échiquier pour la calibration.
    path1image (str): Le chemin vers l'image qui doit être rectifiée.
    savename (str): Le nom du fichier pour enregistrer l'image rectifiée.
    Retourne:
    newcameramtx (numpy.ndarray): La nouvelle matrice de la caméra après calibration.
    dst (numpy.ndarray): L'image rectifiée.
    """
    # Définition des critères pour la précision des coins de pixels
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Préparation de la liste des points d'objet basés sur la taille passée en paramètre
    objp = np.zeros((Size[0]*Size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:Size[0], 0:Size[1]].T.reshape(-1, 2)
    objp[:, :2] *= 40  # Échelle des points d'objet
    
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan de l'image
    
    # Lecture de toutes les images du chemin donné
    images = glob.glob(path)
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Trouver les coins des carrés présents sur l'échiquier
        ret, corners = cv.findChessboardCorners(gray, (Size[0], Size[1]), None)
        print(ret)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Affine la position des coins
            imgpoints.append(corners)
            
            # Dessiner et afficher les coins
            cv.drawChessboardCorners(img, Size, corners2, ret) # Dessine les coins du chess board sur l'image
            cv.namedWindow('img', 0) # Crée une fenêtre
            cv.imshow('img', img) # Affiche l'image dans cette fenêtre
            cv.waitKey(500)
    cv.destroyAllWindows()

    # Calibrer la caméra et obtenir la matrice de la caméra et les coefficients de distorsion
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('camraMatrix\n', mtx)
    print('dist\n', dist)

    # Lire l'image à rectifier
    img = cv.imread(path1image)
    h, w = img.shape[:2]
    
    # Obtenir la nouvelle matrice de la caméra optimale
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h)) # Renvoie la nouvelle matrice de calibration de la caméra
    print('newcameramtx\n', newcameramtx) # print la nouvelle matrice de calibration de la caméra

    # Rectifier la déformation de l'image
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    
    # Recadrer l'image rectifiée
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
    cv.imshow('img', dst)
    cv.waitKey(0)
    
    # Enregistrer l'image rectifiée
    cv.imwrite(savename, dst)

    # 
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) # Projette les points 3D sur le plan 2D
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2) 
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    return newcameramtx, dst # Renvoie la nouvelle matrice de calibration de la caméra et l'image rectifiée

def drawlines(img1, img2, lines, pts1, pts2):

    ''' img1 - image sur laquelle nous dessinons les épilignes pour les points dans img2
        lines - épilignes correspondantes '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR) 
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def StereoCalibrate(rect1, rect2):

    # Créer un détecteur SIFT
    sift = cv.SIFT_create()
    
    # Détecter les points clés et calculer les descripteurs
    kp1, des1 = sift.detectAndCompute(rect1, None)
    kp2, des2 = sift.detectAndCompute(rect2, None)
    
    # Paramètres du matcher basé sur FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    # Faire correspondre les descripteurs en utilisant KNN
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    
    # Appliquer le test de ratio pour trouver les bonnes correspondances
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    print(pts1.shape)

    # Dessiner les correspondances
    img3 = cv.drawMatches(rect1, kp1, rect2, kp2, good, None, flags=2)
    plt.imshow(img3)
    plt.show()

    # Calculer la matrice essentielle
    identity = np.eye(3)
    E, maskE = cv.findEssentialMat(pts1, pts2, identity, method=cv.FM_LMEDS)
    print('E\n', E)
    
    # Récupérer la pose à partir de la matrice essentielle
    retval, R, t, maskP = cv.recoverPose(E, pts1, pts2, identity, maskE) # Renvoie la matrice de rotation et le vecteur de translation
    print('R\n', R)
    print('t\n', t)

    # Calculer la matrice fondamentale
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print('F\n', F)

    return pts1, pts2, F, maskF

def EpipolarGeometry(pts1, pts2, F, maskF, img1, img2):

    r, c = img1.shape

    # Sélectionner uniquement les points inliers
    pts1F = pts1[maskF.ravel() == 1]
    pts2F = pts2[maskF.ravel() == 1]

    # Calculer et dessiner les épilignes pour la première image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1F, pts2F)
    
    # Calculer et dessiner les épilignes pour la deuxième image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    
    # Afficher les épilignes
    plt.figure('Fright')
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img6)
    plt.figure('Fleft')
    plt.subplot(121), plt.imshow(img4)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    # Calculer les transformations de rectification
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (c, r))
    print(H1)
    print(H2)
    
    # Appliquer les transformations de rectification
    im_dst1 = cv.warpPerspective(img1, H1, (c, r))
    im_dst2 = cv.warpPerspective(img2, H2, (c, r))
    cv.namedWindow('left', 0)
    cv.imshow('left', im_dst1)
    cv.namedWindow('right', 0)
    cv.imshow('right', im_dst2)
    cv.waitKey(0)
    
def update_disparity(imgL, imgR):
    
    min_disp = cv.getTrackbarPos('Min Disparity', 'Disparity')
    num_disp = cv.getTrackbarPos('Num Disparities', 'Disparity') * 16
    block_size = cv.getTrackbarPos('Block Size', 'Disparity')
    uniqueness_ratio = cv.getTrackbarPos('Uniqueness Ratio', 'Disparity')
    speckle_window_size = cv.getTrackbarPos('Speckle Window Size', 'Disparity')
    speckle_range = cv.getTrackbarPos('Speckle Range', 'Disparity')

    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=block_size,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=16,
                                  uniquenessRatio=uniqueness_ratio,
                                  speckleWindowSize=speckle_window_size,
                                  speckleRange=speckle_range)
    
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    
    cv.imshow('Disparity', (disparity - min_disp) / num_disp)

    
def DepthMapfromStereoImages(imgL, imgR):

    # Définir les paramètres pour l'algorithme de correspondance stéréo
    global window_size
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=16,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=16,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32)
    
    # Calculer la map de disparité
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    disparity = np.uint8(disparity)
    cv.namedWindow('Disparity', cv.WINDOW_NORMAL)
    cv.resizeWindow('Disparity', 1980, 1280)
    cv.createTrackbar('Min Disparity', 'Disparity', 16, 100, lambda x: update_disparity(imgL, imgR))
    cv.createTrackbar('Num Disparities', 'Disparity', 6, 20, lambda x: update_disparity(imgL, imgR))
    cv.createTrackbar('Block Size', 'Disparity', 16, 50, lambda x: update_disparity(imgL, imgR))
    cv.createTrackbar('Uniqueness Ratio', 'Disparity', 10, 50, lambda x: update_disparity(imgL, imgR))
    cv.createTrackbar('Speckle Window Size', 'Disparity', 100, 200, lambda x: update_disparity(imgL, imgR))
    cv.createTrackbar('Speckle Range', 'Disparity', 32, 100, lambda x: update_disparity(imgL, imgR))

    cv.imshow('Disparity', (disparity - min_disp) / num_disp)

    update_disparity(imgL, imgR)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
    # Afficher la map de disparité
    # plt.figure('3D')
    # plt.imshow((disparity - min_disp) / num_disp, 'gray')
    # plt.colorbar()
    # plt.show()

if __name__ == "__main__":
    # Define paths and parameters for camera calibration
    path1 = 'Images/RedCam/*.jpg'
    path2 = 'Images/BlueCam/*.jpg'
    path1image = 'Images/bouteille2_red.jpg'
    savename1 = 'Images/rougerect2.png'
    path2image = 'Images/bouteille2_blue.jpg'
    savename2 = 'Images/bleurect2.png'
    Size = [5, 8]
    
    # Calibrate cameras and rectify images
    # cameraMatrix1, rect1 = CameraCalibration(Size, path1, path1image, savename1)
    # cameraMatrix2, rect2 = CameraCalibration(Size, path2, path2image, savename2)
    
    # Read rectified images in grayscale
    imageL = cv.imread(path1image, 0)
    imageR = cv.imread(path2image, 0)
    
    # # Uncomment to perform stereo calibration and epipolar geometry
    # pts1, pts2, F, maskF = StereoCalibrate(imageL, imageR)
    # EpipolarGeometry(pts1, pts2, F, maskF, imageL, imageR)
    
    # Read downsampled stereo images for depth map computation
    imageL = cv.pyrDown(cv.imread("Images/bouteille1_red.jpg"))
    imageR = cv.pyrDown(cv.imread("Images/bouteille1_blue.jpg"))
    
    # Compute and display the depth map
    DepthMapfromStereoImages(imageL, imageR)

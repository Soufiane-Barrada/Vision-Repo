import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = []  # numpy array, [nPointsX*nPointsY, 2]

    dim_y= (img.shape[1] - border) // nPointsY
    dim_x= (img.shape[0] - border) // nPointsX
    
    for i in range(nPointsX):
        for j in range(nPointsY):
            vPoints.append((border//2 + i*dim_x, border//2 + j*dim_y))

    return np.array(vPoints)


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    angle = np.arctan2(grad_y, grad_x) 
    angle = (angle + 2 * np.pi) % (2 * np.pi)

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        center_x = round(vPoints[i, 0])
        center_y = round(vPoints[i, 1])

        desc = []
        for cell_y in range(-2, 2):
            for cell_x in range(-2, 2):
                start_y = center_y + (cell_y) * h
                end_y = center_y + (cell_y + 1) * h

                start_x = center_x + (cell_x) * w
                end_x = center_x + (cell_x + 1) * w
                
                hist = np.zeros(nBins)

                if ( not (start_x < 0 or end_x >= img.shape[1] or start_y < 0 or end_y >= img.shape[0])):
                    for y in range(start_y, end_y):
                        for x in range(start_x, end_x):
                            bin_idx = int(angle[y, x] // (2 * np.pi / nBins)) % nBins
                            hist[bin_idx] += 1

                desc.extend(hist)

        descriptors.append(desc)

    # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    descriptors = np.asarray(descriptors)
    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + \
        sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    vFeatures = []
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [h, w, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        # [100, 128]
        features = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        # all features from one image [n_vPoints, 128] (100 grid points)
        vFeatures.append(features)

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    # [n_imgs*n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])
    print('number of extracted features: ', len(vFeatures))

    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    histo = np.zeros(vCenters.shape[0])

    idx, dist=findnn(vFeatures,vCenters)

    for i in idx:
        histo[i] = histo[i] + 1

    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [h, w, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]
        feature_detection= grid_points(img, nPointsX, nPointsY, border)
        feature_description= descriptors_hog(img,feature_detection,cellWidth, cellHeight)
        vBoW.append(bow_histogram(feature_description,vCenters))
        

    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW


def bow_recognition_nearest(histogram, vBoWPos, vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    idxPos, DistPos =findnn(histogram, vBoWPos)
    idxNeg, DistNeg =findnn(histogram, vBoWNeg)

    if (DistPos[0] < DistNeg[0]):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel


if __name__ == '__main__':
    # set a fixed random seed
    np.random.seed(42)
    
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    # number of k-means clusters
    k = 2 
    # maximum iteration numbers for k-means clustering
    numiter = 50

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(
        nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(
            vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(
        nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(
            vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)


"""
if __name__ == '__main__':
    # set a fixed random seed
    np.random.seed(42)
    
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    # maximum iteration numbers for k-means clustering
    numiter = 200  

    k_values = range(1, 101) 
    acc_pos_list = []
    acc_neg_list = []

    for k in k_values:
        print(f'Running for k={k}...')

        print('creating codebook ...')
        vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

        print('creating bow histograms (pos) ...')
        vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
        print('creating bow histograms (neg) ...')
        vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

      
        print('creating bow histograms for test set (pos) ...')
        vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters) 
        result_pos = 0
        print('testing pos samples ...')
        for i in range(vBoWPos_test.shape[0]):
            cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
            result_pos += cur_label
        acc_pos = result_pos / vBoWPos_test.shape[0]
        acc_pos_list.append(acc_pos)
        print('test pos sample accuracy:', acc_pos)

        print('creating bow histograms for test set (neg) ...')
        vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)
        result_neg = 0
        print('testing neg samples ...')
        for i in range(vBoWNeg_test.shape[0]):
            cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
            result_neg += cur_label
        acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
        acc_neg_list.append(acc_neg)
        print('test neg sample accuracy:', acc_neg)


    plt.figure(figsize=(10, 6))
    plt.plot(k_values, acc_pos_list, label='Positive Sample Accuracy', marker='o')
    plt.plot(k_values, acc_neg_list, label='Negative Sample Accuracy', marker='x')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Clusters (k)')
    plt.legend()
    plt.grid(True)
    plt.show()
"""

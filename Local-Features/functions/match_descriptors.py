import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = np.linalg.norm(desc1[:, np.newaxis] - desc2, axis=2) ** 2
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        nearest_neighbor = np.argmin(distances, axis=1)
        matches = np.stack((np.arange(q1), nearest_neighbor), axis=1) 
        
    elif method == "mutual":
        neareast1_2 = np.argmin(distances, axis=1)
        neareast2_1 = np.argmin(distances, axis=0)

        mutual_matches = [(i, neareast1_2[i]) for i in range(q1) if neareast2_1[neareast1_2[i]] == i]
        matches = np.array(mutual_matches)
        
        
    elif method == "ratio":
        sorted_distances = np.sort(distances, axis=1)
        second_min_distances = sorted_distances[:, 1]
        nearest_neighbor = np.argmin(distances, axis=1) 

        ratio = sorted_distances[:, 0] / second_min_distances
        valid_matches = ratio < ratio_thresh 

        matches = np.stack((np.arange(q1)[valid_matches], nearest_neighbor[valid_matches]), axis=1)
    else:
        raise ValueError(f"Unknown matching method: {method}")
    return matches


import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(0)
random.seed(0)

def least_square(x,y):
	# return the least-squares solution
	# you can use np.linalg.lstsq
	x_b=  np.expand_dims(x,axis=1)
	#print(x_b.shape)
	x_b= np.concatenate((x_b, np.ones(x_b.shape)), axis=1)
	#print(x_b[0])
	k_b, _, _, _= np.linalg.lstsq(x_b, y)
	k= k_b[0]
	b= k_b[1]
	return k, b

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# compute the number of inliers and a mask that denotes the indices of inliers
	num = 0
	mask = np.zeros(x.shape, dtype=bool)

	y_pred= k*x + b
	dist= np.abs(y-y_pred)
	mask= dist <= thres_dist
	num= np.sum(mask)

	return num, mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	# ransac
	k_ransac = None
	b_ransac = None
	inlier_mask = None
	best_inliers = 0

	x_y= np.expand_dims(x, axis=1)
	x_y= np.concatenate((x_y, np.expand_dims(y, axis=1)), axis=1)

	for i in range(iter):
		subset= np.array(random.sample(x_y.tolist(), num_subset))
		x_subset= subset[:,0]
		y_subset= subset[:,1]

		k_subset, b_subset= least_square(x_subset, y_subset)
		inliners_count_subset, mask_subset= num_inlier(x, y, k_subset, b_subset, n_samples, thres_dist)

		if(inliners_count_subset>best_inliers):
			inlier_mask= mask_subset
			k_ransac= k_subset
			b_ransac= b_subset
	
	return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	#print(x_noisy.shape, x_noisy)
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# Test num_inlier
	#print(num_inlier(x_noisy, y_noisy, k_ls, b_ls, n_samples, 0.1))

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()
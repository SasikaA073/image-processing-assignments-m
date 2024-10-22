import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd

def compute_homography(src_pts, dst_pts):
    """
    Compute homography matrix H from corresponding points
    """
    if len(src_pts) < 4:
        return None
    
    # Construct A matrix
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0]
        u, v = dst_pts[i][0]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    A = np.array(A)
    
    # Solve using SVD
    _, _, Vt = svd(A)
    H = Vt[-1].reshape(3, 3)
    
    # Normalize
    H = H / H[2, 2]
    
    return H

def compute_homography_ransac(src_pts, dst_pts, threshold=4.0, max_iterations=2000):
    """
    Compute homography matrix using RANSAC
    """
    best_H = None
    max_inliers = 0
    best_inliers_mask = None
    num_points = len(src_pts)
    
    for _ in range(max_iterations):
        # Randomly select 4 points
        idx = np.random.choice(num_points, 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]
        
        # Compute homography for these points
        H = compute_homography(src_sample, dst_sample)
        if H is None:
            continue
            
        # Transform all points using this homography
        src_pts_homog = np.hstack((src_pts.reshape(-1, 2), np.ones((num_points, 1))))
        transformed_pts = (H @ src_pts_homog.T).T
        transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:]
        
        # Calculate distances
        distances = np.sqrt(np.sum((dst_pts.reshape(-1, 2) - transformed_pts) ** 2, axis=1))
        
        # Count inliers
        inliers_mask = distances < threshold
        num_inliers = np.sum(inliers_mask)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers_mask = inliers_mask
            best_H = H
    
    # Refine homography using all inliers
    if best_H is not None and np.sum(best_inliers_mask) > 4:
        src_inliers = src_pts[best_inliers_mask]
        dst_inliers = dst_pts[best_inliers_mask]
        best_H = compute_homography(src_inliers, dst_inliers)
    
    return best_H, best_inliers_mask

def compute_and_show_sift_homography(img1_path, img5_path):
    """
    Compute SIFT features, match them, compute homography, and show visualizations
    """
    # Read images
    img1 = cv2.imread(img1_path)
    img5 = cv2.imread(img5_path)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray5, None)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC
    H, inliers_mask = compute_homography_ransac(src_pts, dst_pts)
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    
    # Draw matches with inliers/outliers in different colors
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers_mask[i]]
    outlier_matches = [good_matches[i] for i in range(len(good_matches)) if not inliers_mask[i]]
    
    # Draw inliers in green
    img_matches = cv2.drawMatches(img1, kp1, img5, kp2, inlier_matches, None,
                                matchColor=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # Draw outliers in red
    img_matches = cv2.drawMatches(img1, kp1, img5, kp2, outlier_matches, img_matches,
                                matchColor=(0, 0, 255),
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # Display matches
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Matches: Green=Inliers ({sum(inliers_mask)}), Red=Outliers ({len(good_matches)-sum(inliers_mask)})')
    plt.axis('off')
    
    # Apply homography to show transformation
    h1, w1 = img1.shape[:2]
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    # Draw transformed boundaries
    img_transformed = img5.copy()
    cv2.polylines(img_transformed, [np.int32(transformed_corners)], True, (0, 255, 0), 3)
    
    # Display transformed image
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.title('Transformed Image Boundaries')
    plt.axis('off')
    
    # Save and show visualization
    plt.tight_layout()
    plt.savefig('homography_visualization.png', dpi=300, bbox_inches='tight')
    
    # Print statistics and homography matrix
    print(f"Total matches: {len(good_matches)}")
    print(f"Inliers: {sum(inliers_mask)}")
    print(f"Outliers: {len(good_matches)-sum(inliers_mask)}")
    print("\nComputed Homography Matrix:")
    print(H)
    
    return H, inliers_mask

if __name__ == "__main__":
    # Specify image paths
    img1_path = "graf/img1.ppm"
    img5_path = "graf/img5.ppm"
    
    try:
        # Compute and visualize SIFT features and homography
        H, inliers_mask = compute_and_show_sift_homography(img1_path, img5_path)
        print("\nVisualization saved as 'homography_visualization.png'")
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
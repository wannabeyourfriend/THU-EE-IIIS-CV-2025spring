import numpy as np
from utils import draw_save_plane_with_points, normalize


def compute_plane(points):
    """
    Compute plane parameters from 3 points
    Returns [A, B, C, D] for plane equation A*x + B*y + C*z + D = 0
    """
    p1, p2, p3 = points
    # Calculate two vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Normal vector to the plane (cross product)
    normal = np.cross(v1, v2)
    A, B, C = normal
    
    # Calculate D
    D = -np.dot(normal, p1)
    
    return np.array([A, B, C, D])

def point_to_plane_distance(point, plane_params):
    """
    Calculate the distance from a point to a plane
    """
    A, B, C, D = plane_params
    x, y, z = point
    numerator = abs(A*x + B*y + C*z + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    return numerator / denominator

def fit_plane_ls(points):
    """
    Fit a plane to points using least squares method
    Minimizes the sum of squared perpendicular distances
    """
    # Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Compute the covariance matrix
    cov = np.dot(centered_points.T, centered_points)
    
    # Find the eigenvector corresponding to the smallest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]  # Smallest eigenvalue's eigenvector
    
    # Ensure the normal vector points outward (positive z)
    if normal[2] < 0:
        normal = -normal
    
    # Calculate D
    D = -np.dot(normal, centroid)
    
    # Return [A, B, C, D]
    return np.array([normal[0], normal[1], normal[2], D])


if __name__ == "__main__":
    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("ransac_points.txt")

    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     
    
    # Calculate the minimal sample time
    # For a plane, we need at least 3 points
    # Probability of selecting an inlier = 100/130
    # Probability of all 3 points being inliers = (100/130)^3
    # Probability of at least one outlier = 1 - (100/130)^3
    # We want P(at least one good sample) > 0.999
    # 1 - (1 - (100/130)^3)^sample_time > 0.999
    # Solve for sample_time
    p_inlier = 100/130
    p_all_inliers = p_inlier**3
    p_target = 0.999
    
    sample_time = int(np.ceil(np.log(1 - p_target) / np.log(1 - p_all_inliers)))
    distance_threshold = 0.05

    # sample points group
    best_inlier_count = 0
    best_plane_params = None
    best_inliers = None
    
    # Generate all hypotheses
    for i in range(sample_time):
        # Randomly select 3 points
        sample_indices = np.random.choice(len(noise_points), 3, replace=False)
        sample_points = noise_points[sample_indices]
        
        # Estimate plane parameters
        plane_params = compute_plane(sample_points)
        
        # Count inliers
        distances = np.array([point_to_plane_distance(point, plane_params) for point in noise_points])
        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)
        
        # Update best hypothesis
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane_params = plane_params
            best_inliers = inliers
    
    # Refine with least squares using all inliers
    inlier_points = noise_points[best_inliers]
    pf = fit_plane_ls(inlier_points)
    
    # Normalize the plane parameters
    pf = normalize(pf)
    
    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points, "result/Ransac/ransac_fig.png") 
    np.savetxt("result/Ransac/ransac_plane.txt", pf)
    np.savetxt('result/Ransac/ransac_sample_time.txt', np.array([sample_time]))

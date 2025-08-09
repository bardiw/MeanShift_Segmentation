import numpy as np
import cv2
from scipy.spatial import cKDTree

# Resize and smooth the image
def resize_and_smooth(image, max_size=400, sigma=1.5):
    height, width = image.shape[:2]
    scale = 1

    # Resize if larger than max_size
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Apply Gaussian smoothing
    smoothed_image = cv2.GaussianBlur(image, (0, 0), sigma)
    return smoothed_image, scale

# Epanechnikov kernel for weights
def kernel_function(x_squared):
    return np.where(x_squared <= 1, 1 - x_squared, 0)

# One step of Mean Shift using k-d tree
def mean_shift_step_kdtree(point, tree, features, h_s, h_r, radius_factor=2.0):
    search_radius = max(h_s, h_r) * radius_factor
    indices = tree.query_ball_point(point, search_radius)

    if not indices:
        return np.zeros_like(point)

    neighbors = features[indices]
    diff = neighbors - point

    # Compute squared spatial and range distances
    spatial_dist_sq = np.sum(diff[:, :2] ** 2, axis=1) / (h_s ** 2)
    range_dist_sq = np.sum(diff[:, 2:] ** 2, axis=1) / (h_r ** 2)

    # Calculate weights
    spatial_weights = kernel_function(spatial_dist_sq)
    range_weights = kernel_function(range_dist_sq)
    weights = spatial_weights * range_weights

    total_weight = np.sum(weights)
    if total_weight > 0:
        mean_shift = np.sum(diff * weights[:, np.newaxis], axis=0) / total_weight
    else:
        mean_shift = np.zeros_like(point)

    return mean_shift

# Mean Shift image segmentation
def mean_shift_segmentation(image, h_s=10, h_r=10, epsilon=0.5,
                            max_iterations=6, max_size=400):
    original_size = image.shape[:2]

    # Resize and smooth the image
    resized_image, scale = resize_and_smooth(image, max_size)
    if scale != 1.0:
        print(f"Image resized from {original_size} to {resized_image.shape[:2]}")
        h_s *= scale

    # Convert to LAB color space
    lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    height, width = lab_image.shape[:2]

    # Create spatial and color features
    y, x = np.mgrid[0:height, 0:width]
    spatial_features = np.stack([x, y], axis=-1).reshape(-1, 2) / h_s
    color_features = lab_image.reshape(-1, 3) / h_r
    features = np.hstack([spatial_features, color_features])

    # Build k-d tree
    tree = cKDTree(features)
    shifted_features = features.copy()

    batch_size = 1000

    # Iterative Mean Shift
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        max_shift = 0

        for i in range(0, len(features), batch_size):
            batch = shifted_features[i:i + batch_size]
            shifts = np.array([mean_shift_step_kdtree(x, tree, features, h_s, h_r)
                               for x in batch])
            new_positions = batch + shifts

            max_batch_shift = np.max(np.sqrt(np.sum(shifts ** 2, axis=1)))
            max_shift = max(max_shift, max_batch_shift)

            shifted_features[i:i + batch_size] = new_positions

        if max_shift < epsilon:
            break

    # Reconstruct segmented image
    segmented_colors = shifted_features[:, 2:] * h_r
    segmented_image = segmented_colors.reshape(height, width, 3).astype(np.uint8)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)

    # Resize back to original size if needed
    if scale != 1.0:
        segmented_image = cv2.resize(segmented_image,
                                     (original_size[1], original_size[0]),
                                     interpolation=cv2.INTER_LINEAR)

    return segmented_image

if __name__ == "__main__":
    input_image_path = "image test.jpg"
    image = cv2.imread(input_image_path)

    if image is None:
        print("Error: Unable to load the image. Check the path.")
    else:
        segmented_image = mean_shift_segmentation(image, h_s=5, h_r=5, epsilon=1.0, max_iterations=10)

        cv2.imshow("Original Image", image)
        cv2.imshow("Segmented Image", segmented_image)

        output_image_path = "segmented_image.jpg"
        cv2.imwrite(output_image_path, segmented_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

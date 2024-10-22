import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def click_points(image, window_name="Select 4 points"):
    """Function to collect 4 points by clicking on an image"""
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            cv.circle(display_image, (x, y), 5, (0, 255, 0), -1)
            cv.imshow(window_name, display_image)
            points.append([x, y])
    
    display_image = image.copy()
    cv.imshow(window_name, display_image)
    cv.setMouseCallback(window_name, mouse_callback)
    
    while len(points) < 4:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.destroyAllWindows()
    return np.float32(points)

def warp_image(source_img, target_img, source_points, target_points):
    """Warp source image to target image using homography"""
    # Compute homography
    H, _ = cv.findHomography(source_points, target_points)
    
    # Get dimensions of target image
    h, w = target_img.shape[:2]
    
    # Warp source image
    warped = cv.warpPerspective(source_img, H, (w, h))
    
    # Create mask for blending
    mask = np.zeros((h, w), dtype=np.uint8)
    target_points_int = np.int32(target_points)
    cv.fillConvexPoly(mask, target_points_int, 255)
    
    # Create inverse mask
    mask_inv = cv.bitwise_not(mask)
    
    # Extract the region from target image
    background = cv.bitwise_and(target_img, target_img, mask=mask_inv)
    
    # Extract the warped region
    foreground = cv.bitwise_and(warped, warped, mask=mask)
    
    # Combine background and foreground
    result = cv.add(background, foreground)
    
    return result

def place_image(target_path, source_path):
    """Main function to place source image onto target image"""
    # Read images
    target_img = cv.imread(target_path)
    source_img = cv.imread(source_path)
    
    if target_img is None or source_img is None:
        raise ValueError("Could not read one or both images")
    
    # Get target points from user clicks
    print("Click 4 points on the target image (clockwise from top-left)")
    target_points = click_points(target_img)
    
    # Define source points (corners of source image)
    h, w = source_img.shape[:2]
    source_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Perform warping and blending
    result = warp_image(source_img, target_img, source_points, target_points)
    
    return result

# Example usage
def main():
    # Replace these paths with your image paths
    target_path = "005.jpg"  # Architectural image
    source_path = "flag.png"    # Image to be warped
    
    # Process images
    result = place_image(target_path, source_path)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv.cvtColor(cv.imread(target_path), cv.COLOR_BGR2RGB))
    plt.title('Target Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv.cvtColor(cv.imread(source_path), cv.COLOR_BGR2RGB))
    plt.title('Source Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title('Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
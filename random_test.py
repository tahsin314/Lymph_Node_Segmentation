
import cv2
import numpy as np

# Assuming gray_image is a valid 2D NumPy array
gray_image = np.clip(np.random.randn(100, 100), 0, 1)
if gray_image is not None and len(gray_image.shape) == 2:
    # Convert to uint8
    gray_image_uint8 = (gray_image * 255).astype(np.uint8)

    # Convert grayscale to RGB using cv2.cvtColor
    rgb_image = cv2.cvtColor(gray_image_uint8, cv2.COLOR_GRAY2RGB)

    # Display the resulting RGB image
    cv2.imwrite("Dummy_Image.png", rgb_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("Input array is empty or not a valid 2D array.")
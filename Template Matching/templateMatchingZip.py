import numpy as np
import cv2
import zipfile
import os
import tempfile

# Load base image in grayscale
img_base = cv2.imread('orignal/Bluethoot icon missing-class 11.png', 0)

# Path to zipped segments folder
zip_path = './templates.zip'

# Extract zip to a temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Loop through all files in the extracted folder
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                template_path = os.path.join(root, file)
                img_template = cv2.imread(template_path, 0)

                if img_template is None:
                    print(f"Could not read {file}")
                    continue

                h, w = img_template.shape

                method = cv2.TM_CCOEFF_NORMED
                result = cv2.matchTemplate(img_base, img_template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                topLeft = max_loc if method != cv2.TM_SQDIFF and method != cv2.TM_SQDIFF_NORMED else min_loc

                threshold = 0.8
                if max_val > threshold:
                    bottomRight = (topLeft[0]+w, topLeft[1]+h)
                    img_copy = img_base.copy()
                    cv2.rectangle(img_copy, topLeft, bottomRight, 255, 2)
                    cv2.imshow(f"Found: {file}", img_copy)
                    print(f"{file} found with confidence {max_val:.2f}")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(f"{file} template missing or below threshold (max_val: {max_val:.2f})")

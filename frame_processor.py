import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
import math

class FrameProcessor:
    def __init__(self):
        self.gabor_kernel1 = self.create_gabor_kernel((15, 15), 10, np.pi / 2)
        self.gabor_kernel2 = self.create_gabor_kernel((3, 3), 10, np.pi / 2)
        self.gabor_kernel3 = self.create_gabor_kernel((3, 3), 10, np.pi / 2)
        self.kernel_morph = np.ones((10, 10), np.uint8)
        self.kernel_dilate = np.ones((4, 4), np.uint8)
        self.kernel_morph2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    def create_gabor_kernel(self, kernel_size, lambd, theta):
        return cv2.getGaborKernel(kernel_size, sigma=4.5, theta=theta, lambd=lambd, gamma=0.5, psi=0, ktype=cv2.CV_32F)

    def process_dilate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, self.gabor_kernel1)
        filtered = cv2.filter2D(filtered, cv2.CV_8UC3, self.gabor_kernel2)
        filtered = cv2.filter2D(filtered, cv2.CV_8UC3, self.gabor_kernel3)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel_morph)
        dilated = cv2.dilate(closed, self.kernel_dilate, iterations=2)
        return dilated

    def process_padding(self, binary_mask, original_frame):
        pad_kernel = np.ones((20, 20), np.uint8)
        morph_kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(cv2.bitwise_not(binary_mask), cv2.MORPH_CLOSE, morph_kernel)
        padded = cv2.dilate(closed, pad_kernel, iterations=1)
        inverted = cv2.bitwise_not(padded)
        output = original_frame.copy()
        output[inverted == 255] = [255, 255, 255]
        return output

    def process_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 15, 150, 100)
        edges = cv2.Canny(blurred, 150, 200)
        return edges

    def process_skeleton(self, dilated):
        inverted = cv2.bitwise_not(dilated)
        cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, self.kernel_morph2)
        _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary == 255).astype(np.uint8) * 255
        labeled = label(skeleton)
        pruned = remove_small_objects(labeled, min_size=100)
        return (pruned > 0).astype(np.uint8) * 255
    
    def process_hough_lines(self, edges, original_frame):
        # Create a blank white image with the same size as the edge_image
        height, width = edges.shape
        blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # Create a white image
        #  Standard Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200, None, 0, 0)
        # Draw the lines
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(original_frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        return original_frame


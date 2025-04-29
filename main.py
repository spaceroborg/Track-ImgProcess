import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
import os
import math

class FrameProcessor:
    def __init__(self):
        """Initialize processing kernels."""
        self.gabor_kernel1 = self.create_gabor_kernel((10, 10), 10, 0)
        self.gabor_kernel2 = self.create_gabor_kernel((3, 3), 10, np.pi/2)
        self.gabor_kernel3 = self.create_gabor_kernel((3, 3), 10, np.pi/2)
        self.kernel_morph = np.ones((4, 4), np.uint8)
        self.kernel_dilate = np.ones((2, 2), np.uint8)
        self.kernel_morph2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    def create_gabor_kernel(self, kernel_size, lambd, theta):
        """Create a Gabor kernel for texture and edge detection."""
        return cv2.getGaborKernel(kernel_size, sigma=4.5, theta=theta, lambd=lambd, gamma=0.5, psi=0, ktype=cv2.CV_32F)

    def process_dilate(self, frame):
        """Generate a binary mask after applying Gabor filtering and morphological operations."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, self.gabor_kernel1)
        _, thresh_otsu = cv2.threshold(filtered_img, 50, 255, cv2.THRESH_OTSU)
        clean_img = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, self.kernel_morph)
        dilated = cv2.dilate(clean_img, self.kernel_dilate, iterations=0)
        return clean_img

    def process_padding(self, mask, original_frame):
        """Pad the binary mask to remove small gaps and restore to original frame."""
        inverted = cv2.bitwise_not(mask)
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))
        padded = cv2.dilate(closed, np.ones((1, 1), np.uint8), iterations=1)
        final = cv2.bitwise_not(padded)

        padding_mask = final == 255
        output = original_frame.copy()
        output[padding_mask] = [255, 255, 255]
        return output

    def process_edges(self, frame):
        """Extract edges from a frame using bilateral filter and Canny edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, d=15, sigmaColor=150, sigmaSpace=100)
        edges = cv2.Canny(blurred, 150, 100)
        return edges

    def detect_vertical_lines(self, edge_image, original_image):
        """Detect and draw vertical lines using Hough Transform."""
        height, width = edge_image.shape
        blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255

        lines = cv2.HoughLines(edge_image, 1, np.pi/180, 200, None, 0, 0)
        vertical_lines = []
        slopes = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta)
                if 0 < angle < 20 or 150 < angle < 170:
                    vertical_lines.append(line)

        image_with_lines = original_image.copy()

        for line in vertical_lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 10000 * (-b))
            y1 = int(y0 + 10000 * (a))
            x2 = int(x0 - 10000 * (-b))
            y2 = int(y0 - 10000 * (a))
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return image_with_lines, slopes

    @staticmethod
    def group_slopes(slopes, delta=0.05):
        """Group similar slopes based on a closeness threshold."""
        slopes = np.sort(np.array(slopes).flatten())
        clusters = []
        current_cluster = [slopes[0]]

        for i in range(1, len(slopes)):
            if abs(slopes[i] - slopes[i-1]) < delta:
                current_cluster.append(slopes[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [slopes[i]]
        clusters.append(current_cluster)
        return clusters

    @staticmethod
    def visualize_clusters(slopes, clusters):
        """Visualize grouped slopes with different colors."""
        plt.figure(figsize=(10, 6))
        colors = plt.cm.get_cmap('tab10', len(clusters))

        for i, cluster in enumerate(clusters):
            indices = [slopes.index(s) for s in cluster]
            plt.scatter(indices, cluster, label=f'Cluster {i+1}', color=colors(i), s=50)

        plt.xlabel('Slope Index')
        plt.ylabel('Slope Value')
        plt.title('Slope Clusters')
        plt.legend()
        plt.show()

    @staticmethod
    def get_mode_cluster(clusters):
        """Return the average slope of the largest cluster."""
        largest_cluster = max(clusters, key=len)
        return np.mean(largest_cluster)

def main():
    """Main execution function."""
    image_path = 'frame_0005.png'
    image = cv2.imread(image_path)

    processor = FrameProcessor()

    dilated_image = processor.process_dilate(image)
    output_image = processor.process_padding(dilated_image, image)
    edge_image = processor.process_edges(output_image)
    lined_image, line_slopes = processor.detect_vertical_lines(edge_image, image)

    print(f"Detected slopes: {line_slopes}")

    # Plot results
    images = {
        'Original + Lines': lined_image,
        'Edge Image': edge_image,
        'Dilated Mask': dilated_image,
        'Mask without Background': output_image,
    }

    for title, img in images.items():
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()

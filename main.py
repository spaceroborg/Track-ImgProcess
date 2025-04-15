import cv2

# For images
import matplotlib.pyplot as plt
from frame_processor import FrameProcessor

if __name__ == "__main__":
    image_path = "frame_0005.png"
    image = cv2.imread(image_path)

    processor = FrameProcessor()

    dilated = processor.process_dilate(image)
    padded = processor.process_padding(dilated, image)
    edges = processor.process_edges(padded)
    skeleton = processor.process_skeleton(dilated)
    lines = processor.process_hough_lines(edges, image)

    cv2.imwrite("output/output_dilated.jpg", dilated)
    cv2.imwrite("output/output_padded.jpg", padded)
    cv2.imwrite("output/output_edges.jpg", edges)
    cv2.imwrite("output/output_skeleton.jpg", skeleton)
    cv2.imwrite("output/output_lines.jpg", lines)

    print("All steps completed and saved.")

    # Plot the outputs
    plt.figure(1)
    plt.imshow(lines)
    plt.title('Original image with Hough lines overlay')
    plt.axis('off')  # Turn off axis

    plt.figure(2)
    plt.imshow(edges)
    plt.title('Canny edge detection')
    plt.axis('off')  # Turn off axis

    plt.figure()
    plt.imshow(dilated,cmap="gray")
    plt.title('Dilated image')
    plt.axis('off')  # Turn off axis

    plt.figure()
    plt.imshow(padded)
    plt.title('Original image with background removed')
    plt.axis('off')  # Turn off axis

    plt.figure()
    plt.imshow(skeleton, cmap="gray")
    plt.title('Skeletionized image')
    plt.axis('off')  # Turn off axis

    # Display the images side by side
    plt.show()

# # For video
# from video_processor import VideoProcessor
# if __name__ == "__main__":
#     input_video = "Original_videos/run1_compressed_split.mp4"
#     output_video = "output.avi"
#     vp = VideoProcessor(input_video, output_video)
#     vp.run()

import cv2
import numpy as np
from frame_processor import FrameProcessor

class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.processor = FrameProcessor()
        self.cap = cv2.VideoCapture(self.input_path)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.out = self.init_writer()

    def init_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_size = (self.width * 2, self.height)  # Side-by-side
        return cv2.VideoWriter(self.output_path, fourcc, self.fps, out_size, isColor=True)

    def run(self):
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            dilated_image = self.processor.process_dilate(frame)
            output_image = self.processor.process_padding(dilated_image, frame)
            edge_image = self.processor.process_edges(output_image)

            output_image = cv2.resize(output_image, (self.width, self.height))
            edge_image = cv2.resize(edge_image, (self.width, self.height))
            edge_image_3d = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

            combined_frame = np.hstack((output_image, edge_image_3d))
            self.out.write(combined_frame)

            print(f"Processed frame {frame_count}")
            frame_count += 1

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Processing complete. Output saved.")

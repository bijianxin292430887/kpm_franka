import cv2
import numpy as np
from ultralytics import FastSAM

class fastSAM():

    def __init__(self):
        self.model = FastSAM("FastSAM-x.pt")
        self.bowl_bbox_a = (310, 278)
        self.bowl_bbox_b = (553, 457)
        self.last_valid_results = None

    def get_masked_img(self,frame):
        # Run inference on the frame with a text prompt
        try:
            # Attempt to use the text prompt
            results = self.model(frame, texts="spoon")

        except RuntimeError as e:
            print(f"Detection fail . Error: {e},use last valid detections")
            results = self.last_valid_results

        self.last_valid_results = results
        # Create a blank mask for the entire frame
        mask = np.zeros_like(frame)
        if results is not None:
            # Overlay masks on the frame
            for r in results:
                masks = r.masks  # Masks object for segment masks outputs

                # If there are masks, process and overlay them
                if masks is not None:
                    for mask_data in masks:
                        # Convert the list of points into a format usable by OpenCV
                        contour = np.array(mask_data.xy, dtype=np.int32)

                        # Calculate the bounding box
                        x, y, w, h = cv2.boundingRect(contour)

                        # Draw the bounding box on the original frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box

                        # Create a mask for the bounding box
                        bbox_mask = np.zeros_like(frame)
                        bbox_mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]

                        # Add the bbox_mask to the final mask
                        mask += bbox_mask

        # Draw the bowl bounding box on the original frame
        cv2.rectangle(frame, self.bowl_bbox_a, self.bowl_bbox_b, (255, 0, 0), 2)

        # Apply the bowl bounding box mask
        bowl_mask = np.zeros_like(frame)
        bowl_mask[self.bowl_bbox_a[1]:self.bowl_bbox_b[1], self.bowl_bbox_a[0]:self.bowl_bbox_b[0]] = frame[self.bowl_bbox_a[1]:self.bowl_bbox_b[1],
                                                                                  self.bowl_bbox_a[0]:self.bowl_bbox_b[0]]
        mask += bowl_mask
        return mask


if __name__ == "__main__":
    # Define the inference source
    source = "wood_spoon/human_videos/episode_0.mp4"

    model = fastSAM()

    # Open the video using OpenCV
    cap = cv2.VideoCapture(source)

    # Create a window to display the results
    cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        # Convert the frame from BGR (OpenCV format) to RGB (expected by FastSAM)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = model.get_masked_img(frame_rgb)

        # Display only the masked parts of the frame
        cv2.imshow('Segmentation', mask)

        # Wait for a key press and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
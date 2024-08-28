from ultralytics import YOLO
import cv2
import numpy as np
import torch

class DummyBBoxTracker:
    def __init__(self):
        self.prev_bbox = None
        self.initialized = False  # Flag to check if the tracker has started

    def get_dummy_bbox(self, results):
        # Extract bounding boxes from current detection
        current_bboxes = [box.xyxy for box in results[0].boxes]

        if not self.initialized:
            if len(current_bboxes) > 0:
                # If this is the first detection, initialize with the detected bbox
                self.prev_bbox = current_bboxes[0].cpu().numpy()
                self.initialized = True
                return self.prev_bbox
            else:
                # If no detection in the first frame(s), skip processing
                return None

        if len(current_bboxes) == 0:
            # No detection, return previous bounding box
            return self.prev_bbox
        elif len(current_bboxes) == 1:
            # Only one detection, compare its size with the previous bounding box
            new_bbox = current_bboxes[0].cpu().numpy()
            if self.is_too_large(new_bbox) or self.is_too_far(new_bbox):
                print("too large or too far")
                return self.prev_bbox
            else:
                self.prev_bbox = new_bbox
                return self.prev_bbox
        else:
            # Multiple detections, find the smallest bbox among candidates
            smallest_bbox = self.get_smallest_bbox(current_bboxes)
            if self.is_too_large(smallest_bbox) or self.is_too_far(smallest_bbox):
                print("too large or too far")
                return self.prev_bbox
            else:
                self.prev_bbox = smallest_bbox
                return self.prev_bbox

    def get_smallest_bbox(self, bboxes):
        # Function to find the bounding box with the smallest area
        min_area = float('inf')
        smallest_bbox = None
        for bbox in bboxes:
            bbox = bbox.cpu().numpy()
            area = self.calculate_bbox_area(bbox)
            if area < min_area:
                min_area = area
                smallest_bbox = bbox
        return smallest_bbox

    def is_too_large(self, bbox):
        # Function to check if the current bbox is too large compared to the previous one
        if self.prev_bbox is None:
            return False  # No previous bounding box to compare with
        prev_area = self.calculate_bbox_area(self.prev_bbox)
        curr_area = self.calculate_bbox_area(bbox)
        return curr_area > 100 * prev_area

    def is_too_far(self, bbox):
        # Function to check if the current bbox is too far from the previous one
        if self.prev_bbox is None:
            return False  # No previous bounding box to compare with
        prev_center = self.get_bbox_center(self.prev_bbox)
        curr_center = self.get_bbox_center(bbox)
        distance = np.linalg.norm(np.array(prev_center) - np.array(curr_center))
        return distance >= 500

    @staticmethod
    def calculate_bbox_area(bbox):
        # Calculate the area of a bounding box
        x1, y1, x2, y2 = bbox[0]
        return (x2 - x1) * (y2 - y1)

    @staticmethod
    def get_bbox_center(bbox):
        # Calculate the center of a bounding box
        x1, y1, x2, y2 = bbox[0]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y


class yolo_v8():

    def __init__(self):


        self.model = YOLO("/home/allenbi/PycharmProjects24/mm_data/yolov8x-worldv2-spoon.pt")
        # self.model.set_classes(['spoon'])
        self.tracker = DummyBBoxTracker()

    def track_bounding_boxes_video(self,video_path, visualize=False):
        # Initialize the YOLO model and DummyBBoxTracker

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        all_bboxes = []

        # Loop through the video frames
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = self.model.track(frame, persist=True)

                # Get the dummy bounding box
                dummy_bbox = self.tracker.get_dummy_bbox(results)

                if dummy_bbox is not None:
                    # Store the bounding box
                    all_bboxes.append(dummy_bbox)

                    if visualize:
                        # Convert tensor to a list of integers
                        x1, y1, x2, y2 = dummy_bbox[0].astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # Display the annotated frame
                        cv2.imshow("YOLOv8 Tracking", frame)

                # Break the loop if 'q' is pressed
                if visualize and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        if visualize:
            cv2.destroyAllWindows()

        return all_bboxes

    def track_bounding_boxes_frame(self, frame, visualize=False):
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.model.track(frame, persist=True)

        # Get the dummy bounding box
        bbox = self.tracker.get_dummy_bbox(results)

        if bbox is not None:
            # Convert tensor to a list of integers
            x1, y1, x2, y2 = bbox[0].astype(int)

            # Print the coordinates
            print(f"BBox Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            if visualize:
                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", frame)
                cv2.waitKey(1)

        return bbox



if __name__ == "__main__":

    # yolo_model = YOLO("yolov8x-worldv2.pt")
    # yolo_model.set_classes(['spoon'])
    # yolo_model.save('yolov8x-worldv2-spoon.pt')

    yolo_model = yolo_v8()
    video_path = f'wood_spoon/human_videos/episode_0.mp4'
    # for i in range(0,30):
    #
    #     video_path = f'wood_spoon/human_videos/episode_{i}.mp4'
    #
    #     bboxes = yolo_model.track_bounding_boxes_video(video_path,visualize=True)


    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        bbox = yolo_model.track_bounding_boxes_frame(frame, visualize=True)
        print(bbox)

    cap.release()
    cv2.destroyAllWindows()
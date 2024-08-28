import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from megapose.datasets.scene_dataset import ObjectData, CameraData
from megapose.visualization.utils import make_contour_overlay
import pandas as pd
from megapose.lib3d.transform import Transform
import json
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.panda3d_renderer import Panda3dLightData
from mm_data.yolo import yolo_v8
from scipy.spatial.transform import Rotation as R



def qt_to_np(q):

    return np.array([q.x,q.y,q.z,q.w])

def transform_to_structured_array(spoon_poses):
    """
    Convert a list of Transform objects to a structured numpy array.

    Args:
        spoon_poses (list): A list of Transform objects.

    Returns:
        np.ndarray: A structured numpy array with rotation (as a quaternion) and translation.
    """
    structured_data = []
    for pose in spoon_poses:
        if pose is not None:
            rotation = pose.quaternion # Assuming this is a 4-element array
            translation = pose.translation  # Assuming this is a 3-element array
            structured_data.append(np.concatenate((rotation, translation),axis=-1))
        else:
            structured_data.append((np.zeros(7)))  # Use zeros as placeholders for None

    # Define the dtype for the structured array
    dtype = np.dtype([
        ('pose', 'f4', (7,)),  # Quaternion with 4 elements
    ])

    return np.array(structured_data, dtype=dtype)

def quaternion_difference(q1, q2):
    """
    Compute the angular difference between two quaternions.

    Args:
        q1, q2: Quaternions represented as [x, y, z, w].

    Returns:
        Angular difference in degrees.
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    diff= r2*r1.inv()
    # Extract the magnitude of the rotation (angle in radians)
    angle_rad = 2 * np.arccos(np.clip(diff.as_quat()[-1], -1.0, 1.0))

    return np.degrees(angle_rad)

def is_pose_valid(current_pose, last_valid_pose, translation_threshold=0.05, rotation_threshold=360):
    """
    Checks if the current pose is valid based on its difference from the last valid pose.

    Args:
        current_pose (dict): Current pose, containing 'TWO' key with rotation (quaternion) and translation.
        last_valid_pose (dict): Last valid pose for comparison.
        translation_threshold (float): Maximum allowed translation difference.
        rotation_threshold (float): Maximum allowed rotation difference in degrees.

    Returns:
        bool: True if the current pose is within the acceptable range, False if it is an outlier.
    """
    if last_valid_pose is None:
        # If no valid pose exists yet, accept the current one
        return True

    # Calculate translation difference
    translation_diff = np.linalg.norm(current_pose.translation - last_valid_pose.translation)

    # Calculate rotation difference using quaternion
    rotation_diff = quaternion_difference(current_pose.quaternion, last_valid_pose.quaternion)
    print(translation_diff,rotation_diff)
    # Check if the current pose is within the thresholds
    if translation_diff < translation_threshold and rotation_diff < rotation_threshold:
        return True
    else:
        return False

class PoseEstimator:
    def __init__(self, object_dataset_dir: Path):
        self.model_name = "megapose-1.0-RGB-multi-hypothesis"
        self.model_info = NAMED_MODELS[self.model_name]
        self.pose_estimator = load_named_model(self.model_name, self.load_object_dataset(object_dataset_dir)).cuda()
        self.model = yolo_v8()
        self.object_dataset = self.load_object_dataset(object_dataset_dir)
        self.camera_data = self.load_camera_data()
        self.last_valid_pose = None
        self.frame_idx = 0
    @staticmethod
    def load_object_dataset(object_dir: Path) -> RigidObjectDataset:
        rigid_objects = []
        mesh_units = "mm"
        for obj_file in object_dir.iterdir():
            if obj_file.suffix in {".obj", ".ply"}:
                rigid_objects.append(RigidObject(label=obj_file.stem, mesh_path=obj_file, mesh_units=mesh_units))
        return RigidObjectDataset(rigid_objects)

    @staticmethod
    def load_camera_data() -> CameraData:
        return CameraData(
            resolution=(480, 640),
            K=np.array([[384.765, 0.0, 310.009], [0.0, 384.765, 238.076], [0.0, 0.0, 1.0]])
        )

    def get_detections_from_yolo(self, frame: np.ndarray, visualize: bool = False) -> Tuple[Union[torch.Tensor, None], Union[np.ndarray, None]]:
        bbox = self.model.track_bounding_boxes_frame(frame, visualize)
        if bbox is None:
            print("bbox is none")
            return None, None
        return torch.Tensor(bbox), bbox

    def load_detections(self, bbox_tensor: torch.Tensor) -> DetectionsType:
        assert len(self.object_dataset.objects) == 1, "This function assumes there is only one object in the dataset."
        obj = self.object_dataset.objects[0]
        label = obj.label

        infos = [{'label': label, 'batch_im_id': 0, 'instance_id': 0, 'score': 1.0}]
        infos_df = pd.DataFrame(infos)
        detections = DetectionsType(infos=infos_df, bboxes=bbox_tensor)

        return detections.cuda()


    def run_inference_on_frame(self,frame,frame_rgb):
        bbox_tensor, bbox = self.get_detections_from_yolo(frame)
        if bbox_tensor is None:
            return None

        detections = self.load_detections(bbox_tensor)

        observation = ObservationTensor.from_numpy(frame_rgb, None, self.camera_data.K).cuda()

        pose_estimates, _ = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, **self.model_info["inference_parameters"]
        )

        current_pose = Transform(pose_estimates.poses.cpu().numpy()[0])

        if self.frame_idx <= 3:
            self.last_valid_pose = current_pose
        elif is_pose_valid(current_pose, self.last_valid_pose):
            self.last_valid_pose = current_pose
        else:
            current_pose = self.last_valid_pose
            print("outlier, use last valid pose")
        self.frame_idx += 1

        spoon_pose_array = transform_to_structured_array([current_pose])
        spoon_pose =  np.stack([pose for pose in spoon_pose_array['pose']])[0][:, 0]


        return spoon_pose

    def run_inference_on_video_example(self, video_path: str, output_dir: str) -> None:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = cap.get(cv2.CAP_PROP_FPS)

        # output_video_path = Path(output_dir) / "output_with_contour_and_bbox.mp4"
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bbox_tensor, bbox = self.get_detections_from_yolo(frame)
            if bbox_tensor is None:
                continue

            detections = self.load_detections(bbox_tensor)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            observation = ObservationTensor.from_numpy(frame_rgb, None, self.camera_data.K).cuda()

            output, _ = self.pose_estimator.run_inference_pipeline(
                observation, detections=detections, **self.model_info["inference_parameters"]
            )
            print(f"frame_idx:{frame_idx}")

            # self.save_pose_estimates(output, output_dir, frame_idx)

            # object_datas = self.load_object_data(Path(output_dir) / f"pose_estimates_frame_{frame_idx:04d}.json")
            # frame_with_overlay = self.visualize_contour_overlay(frame, self.camera_data, object_datas, bbox)
            # out.write(frame_with_overlay)

            frame_idx += 1

        cap.release()
        # out.release()
        cv2.destroyAllWindows()

    def visualize_contour_overlay(self, frame: np.ndarray, camera_data: CameraData, object_datas: List[ObjectData], bbox: np.ndarray) -> np.ndarray:
        if camera_data.TWC is None:
            camera_data.TWC = Transform(np.eye(4))

        renderer = Panda3dSceneRenderer(self.object_dataset)
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        contour_overlay = make_contour_overlay(
            frame, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]

        x1, y1, x2, y2 = bbox[0].astype(int)
        cv2.rectangle(contour_overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('Contour and Bounding Box Overlay', contour_overlay)
        cv2.waitKey(1)

        return contour_overlay

    def save_pose_estimates(self, pose_estimates: PoseEstimatesType, output_dir: str, frame_idx: int) -> None:
        output_path = Path(output_dir) / f"pose_estimates_frame_{frame_idx:04d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()

        object_data = []
        for label, pose in zip(labels, poses):
            T = Transform(pose)
            object_data.append(ObjectData(label=label, TWO=T))

        object_data_json = json.dumps([x.to_json() for x in object_data])
        # print(object_data_json)
        output_path.write_text(object_data_json)

    @staticmethod
    def load_object_data(data_path: Path) -> List[ObjectData]:
        object_data = json.loads(data_path.read_text())
        return [ObjectData.from_json(d) for d in object_data]


if __name__ == "__main__":
    video_path = "wood_spoon/human_videos/episode_0.mp4"
    output_dir = "wood_spoon/human_pose/"

    object_dataset_dir = Path("wood_spoon/meshes/wood-spoon-new/")

    pose_estimator = PoseEstimator(object_dataset_dir)

    # pose_estimator.run_inference_on_video_example(video_path, output_dir)

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        spoon_pose = pose_estimator.run_inference_on_frame(frame,frame_rgb)

        print(spoon_pose)

        pose_estimator.frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
import torch
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel
)
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
class DifferentiableFrankaPanda(DifferentiableRobotModel):
    def __init__(self, device=None):
        #rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
        #self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.urdf_path = '/home/allenbi/PycharmProjects24/mm_data/urdf/URDF/robots/panda_hand_arm_modified_tas_without_finger.urdf'
        self.learnable_rigid_body_config = None
        self.name = "differentiable_franka_panda"
        self.device=device
        super().__init__(self.urdf_path, self.name, device=device)

    def fk_solver(self,joint_angles, link="fake_target"):
        # joint_angles.append(0) ## for 7 joint-angle because of end effector
        # joint_angles.append(0)
        dummy_control = np.zeros((joint_angles.shape[0], 2))
        joint_angles = np.concatenate((joint_angles,dummy_control),axis=1)
        gt_robot_model = DifferentiableFrankaPanda(device=self.device)
        q = torch.FloatTensor(joint_angles).to(self.device)
        translations, quaternions = gt_robot_model.compute_forward_kinematics(q=q, link_name=link)
        return quaternions, translations
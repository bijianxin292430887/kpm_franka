import os
import logging
import pathlib

import hydra
import numpy as np
from pathlib import Path
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from agents.utils.sim_path import sim_framework_path
# from sam_mask import fastSAM
from mm_data.megapose_detector import PoseEstimator
from mm_data.visualize_output import visualize_robot_policy_sequence
import cv2

from pathlib import Path

current_dir = Path(__file__).resolve().parent

from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
# from robot_visualizer import RobotVisualizer

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()

video_path = current_dir/'wood_spoon/human_videos/episode_2.mp4'

# rb_visualizer = RobotVisualizer('/home/allenbi/PycharmProjects24/mm_data/urdf/URDF/robots/panda_hand_arm_modified_tas_without_finger.urdf')
def load_agent_ckpt(cfg):
    agent = hydra.utils.instantiate(cfg.agent)

    original_cwd = pathlib.Path(hydra.utils.get_original_cwd()).resolve()
    root = f'{original_cwd}/{cfg.log_dir}runs/'

    latent_policy_ckpt_path = root + 'idm_latent_policy/'
    action_decoder_ckpt_path = root + f'{cfg.agent_name}/'

    # load agent
    agent.load_model_from_ckpt(latent_policy_ckpt_path, action_decoder_ckpt_path)

    return agent


@hydra.main(config_path="configs", config_name="spoon_config_cotrain.yaml")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config
    )

    # load SAM for mask img
    # sam = fastSAM()

    crop_img_mask = [278, 458, 310, 550]

    object_path = current_dir/'wood_spoon/meshes/wood-spoon-new/'
    # load megapose and SAM
    object_dataset_dir = Path(object_path)
    spoon_estimator = PoseEstimator(object_dataset_dir)
    # camera_data = spoon_estimator.load_camera_data()

    # load agent
    agent = load_agent_ckpt(cfg)

    # for loop video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # pose estimation
        # get bbox from yolo for pose estimation

        if frame_idx%20==0:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            spoon_pose = spoon_estimator.run_inference_on_frame(frame,frame_rgb)

            if spoon_pose is None:
                print('cannot estimate spoon pose, pass to next frame')
                continue

            # masked_img = sam.get_masked_img(frame_rgb)
            masked_img = frame_rgb[crop_img_mask[0]:crop_img_mask[1],crop_img_mask[2]:crop_img_mask[3]]

            # agent prediction
            # create dummy past obs
            masked_img = np.repeat(np.expand_dims(masked_img,axis=0),repeats=2,axis=0)
            spoon_pose = np.repeat(np.expand_dims(spoon_pose,axis=0),repeats=2,axis=0)
            joint_angle,quaternions, translations = agent.predict(masked_img,spoon_pose)
            # print(frame_idx,translations)

            # rb_visualizer.visualize(translations,quaternions)
            # visualize_robot_policy_sequence(translations.cpu(), quaternions.cpu(), save_gif=True)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


    wandb.finish()


if __name__ == "__main__":

    main()
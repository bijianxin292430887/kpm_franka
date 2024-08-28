import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
import hydra
from typing import Optional
from mm_data.agent.fk_controller import DifferentiableFrankaPanda

log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Agent(nn.Module):

    def __init__(self,
                 latent_policy: DictConfig,
                 idm_model: DictConfig,
                 trainset: DictConfig,
                 device: str = 'cuda'):

        super(IDM_Agent, self).__init__()

        self.device = device
        self.latent_policy =  hydra.utils.instantiate(latent_policy)
        self.idm_model = hydra.utils.instantiate(idm_model)

        self.trainset = hydra.utils.instantiate(trainset)
        self.norm_param = self.trainset.normalization_params
        self.fk = DifferentiableFrankaPanda(self.device)

        print('init idm policy')

    def load_model_from_ckpt(self,latent_policy_ckpt_path, action_decoder_ckpt_path):

        self.latent_policy.load_pretrained_model(latent_policy_ckpt_path,"eval_best_idm.pth")
        self.idm_model.load_pretrained_model(action_decoder_ckpt_path,"eval_best_idm.pth")

        print(f'action_decoder_ckpt_path:{action_decoder_ckpt_path}')
        print(f'load idm_model from {action_decoder_ckpt_path}')


    @torch.no_grad()
    def predict(self, masked_img, past_poses) -> torch.Tensor:

        self.latent_policy.model.eval()
        self.idm_model.model.eval()

        # normalize pose input
        past_poses = self.trainset.normalize(past_poses, self.norm_param['spoon_poses_min'],
                                             self.norm_param['spoon_poses_max'])

        masked_img = torch.from_numpy(masked_img).float().to(self.device).unsqueeze(0)
        past_poses = torch.from_numpy(past_poses).float().to(self.device).unsqueeze(0)
        # diffusion planner provide future poses
        pred_poses = self.latent_policy.predict(masked_img, past_poses)
        # controller generate action sequence
        pred_action = self.idm_model.predict(torch.cat((past_poses, pred_poses), dim=1))

        joint_angle = self.trainset.inv_normalize(pred_action, self.norm_param['robot_joints_min'],
                                                  self.norm_param['robot_joints_max'])

        quaternions, translations = self.fk.fk_solver(joint_angle)

        return joint_angle, quaternions, translations


    def reset(self):
        pass

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
import numpy as np
from mm_data.agent.fk_controller import DifferentiableFrankaPanda
from mm_data.spoon_dataset import data_normalize,data_inv_normalize
log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Agent(nn.Module):

    def __init__(self,
                 latent_policy: DictConfig,
                 idm_model: DictConfig,
                 data_ratio:int = 100,
                 device: str = 'cpu'):

        super(IDM_Agent, self).__init__()

        self.device = device
        self.latent_policy =  hydra.utils.instantiate(latent_policy)
        self.idm_model = hydra.utils.instantiate(idm_model)

        self.fk = DifferentiableFrankaPanda(self.device)
        print('init idm policy')

    def load_model_from_ckpt(self,latent_policy_ckpt_path, idm_model_ckpt_path):

        self.latent_policy.load_pretrained_model(latent_policy_ckpt_path,"eval_best_idm.pth")
        self.idm_model.load_pretrained_model(idm_model_ckpt_path,"eval_best_idm.pth")
        self.norm_param = self.idm_model.norm_param
        print(f'load idm_baseline from {idm_model_ckpt_path}')


    @torch.no_grad()
    def predict(self, masked_img,past_poses) -> torch.Tensor:

        self.latent_policy.model.eval()
        self.idm_model.model.eval()

        # normalize pose input
        n_past_poses = data_normalize(past_poses,self.norm_param,'poses')

        masked_img = torch.from_numpy(masked_img).float().to(self.device).unsqueeze(0)
        n_past_poses = torch.from_numpy(n_past_poses).float().to(self.device).unsqueeze(0)
        # diffusion planner provide future poses
        n_pred_poses = self.latent_policy.predict(masked_img,n_past_poses)
        print(n_pred_poses)
        # controller generate action sequence
        n_pred_action = self.idm_model.predict(torch.cat((n_past_poses,n_pred_poses),dim=1))

        # inverse normalize action output
        joint_angle = data_inv_normalize(n_pred_action,self.norm_param,'joints')

        quaternions, translations = self.fk.fk_solver(joint_angle)

        return joint_angle,quaternions, translations

    def reset(self):
        pass

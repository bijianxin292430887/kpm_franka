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

class DF_Agent(nn.Module):

    def __init__(self,
                 df_policy: DictConfig,
                 data_ratio:int = 100,
                 device: str = 'cpu'):

        super(DF_Agent, self).__init__()

        self.device = device
        self.df_policy =  hydra.utils.instantiate(df_policy)
        self.fk = DifferentiableFrankaPanda(self.device)
        print('init df policy')

    def load_model_from_ckpt(self,latent_policy_ckpt_path, idm_model_ckpt_path):

        self.df.load_pretrained_model(latent_policy_ckpt_path,"eval_best_idm.pth")
        self.norm_param = self.idm_model.norm_param
        print(f'load df_policy from {idm_model_ckpt_path}')


    @torch.no_grad()
    def predict(self, masked_img) -> torch.Tensor:

        self.df_policy.model.eval()

        masked_img = torch.from_numpy(masked_img).float().to(self.device).unsqueeze(0)
        # diffusion planner provide future poses
        n_pred_action = self.df_policy.predict(masked_img)
        # inverse normalize action output
        joint_angle = data_inv_normalize(n_pred_action,self.norm_param,'joints')

        quaternions, translations = self.fk.fk_solver(joint_angle)

        return joint_angle,quaternions, translations

    def reset(self):
        pass

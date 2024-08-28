import logging
import os
import pathlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional

from mm_data.agent.base_agent import BaseAgent
from agents.utils.scaler import Normalizer
from agents.models.idm.ema import ExponentialMovingAverage
from agents.models.idm.util import WarmupLinearSchedule
from agents.models.idm.contrastive import Contrastive_Sampling,TripletLossSampling
from mm_data.spoon_dataset import data_normalize,data_inv_normalize


log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Cotrain_Policy(nn.Module):

    def __init__(self,
                 idm_model: DictConfig,
                 device: str = 'cuda:0'):

        super(IDM_Cotrain_Policy, self).__init__()

        self.idm_model = hydra.utils.instantiate(idm_model).to(device)

    def forward(self,nobs):
        # return both obs reconstruction and action
        out = self.idm_model.forward(nobs)

        return out

    def inference(self,nobs):
        # return only action
        act = self.idm_model.inference(nobs)

        return act

    def get_params(self):
        return self.parameters()



class IDM_Cotrain_Agent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            eval_every_n_epochs: int = 50,
            action_loss_factor:float = 0.01,
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # # Define the number of GPUs available
        # num_gpus = torch.cuda.device_count()
        #
        # # Check if multiple GPUs are available and select the appropriate device
        # if num_gpus > 1:
        #     print(f"Using {num_gpus} GPUs for training.")
        #     self.model = nn.DataParallel(self.model)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.scheduler = WarmupLinearSchedule(self.optimizer, 200, 0.1*optimization.lr, optimization.lr)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.8)

        self.eval_model_name = "eval_best_idm.pth"
        self.last_model_name = "last_idm.pth"

        if trainset is not None:
            self.norm_param = self.train_dataloader.dataset.normalization_params
            print('get norm_param from training dataset')

        self.idm_model_loaded= False

        self.action_loss_factor = action_loss_factor

        # self.contrastive= Contrastive_Sampling(num_negatives=4)
        self.contrastive = TripletLossSampling(margin=0.1)

        # print(f'init idm_model contrain agent, with action_loss_factor:{self.action_loss_factor}')

    def train_agent(self):
        best_test_mse = 1e10
        early_stop_cnt = 0
        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_total_loss = []
                test_recon_loss = []
                test_dynamics_loss = []
                test_contrastive_loss = []
                test_action_loss = []

                # total, dynamics, action, contrastive, recon
                for data in self.test_dataloader:

                    # test dataset is not normalized before creating dataloader, use norm_param from training dataset to normalize test data.
                    state = data_normalize(data['spoon_poses'], self.norm_param, 'poses').cuda()
                    act =  data_normalize(data['robot_joints'], self.norm_param, 'joints').cuda()
                    mask = data['mask'].cuda()

                    total_loss, recon_loss, dynamics_loss, contrastive_loss, action_loss, = self.evaluate(state, act, mask)

                    test_total_loss.append(total_loss)
                    test_recon_loss.append(recon_loss)
                    test_dynamics_loss.append(dynamics_loss)
                    test_contrastive_loss.append(contrastive_loss)
                    test_action_loss.append(action_loss)

                avrg_total_loss = sum(test_total_loss) / len(test_total_loss)
                avrg_dynamics_loss = sum(test_dynamics_loss) / len(test_dynamics_loss)
                avrg_action_loss = sum(test_action_loss) / len(test_action_loss)
                avrg_contrastive_loss = sum(test_contrastive_loss) / len(test_contrastive_loss)
                avrg_recon_loss = sum(test_recon_loss) / len(test_recon_loss)

                log.info("Epoch {}: TEST_ACT_loss:{}".format(num_epoch, avrg_action_loss))

                wandb.log(
                    {
                        "Test_total_loss": avrg_total_loss,
                        "Test_dyna_loss": avrg_dynamics_loss,
                        "Test_act_loss": avrg_action_loss,
                        "Test_contrastive_loss": avrg_contrastive_loss,
                        "Test_recon_loss": avrg_recon_loss,
                    }
                )

                if avrg_total_loss < best_test_mse:
                    best_test_mse = avrg_total_loss
                    early_stop_cnt = 0
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log({"best_model_epochs": num_epoch, })
                else:
                    early_stop_cnt += 1

                if early_stop_cnt >= 7:
                    log.info('Early Stop!')
                    break

            train_total_loss = []
            train_recon_loss = []
            train_dynamics_loss = []
            train_contrastive_loss = []
            train_action_loss = []

            for data in self.train_dataloader:
                state = data['spoon_poses'].cuda()
                act = data['robot_joints'].cuda()
                mask = data['mask'].cuda()

                total_loss,recon_loss,dynamics_loss,contrastive_loss, action_loss = self.train_step(state, act, mask)

                train_total_loss.append(total_loss)
                train_recon_loss.append(recon_loss)
                train_dynamics_loss.append(dynamics_loss)
                train_contrastive_loss.append(contrastive_loss)
                train_action_loss.append(action_loss)

            avrg_total_loss = sum(train_total_loss) / len(train_total_loss)
            avrg_recon_loss = sum(train_recon_loss) / len(train_recon_loss)
            avrg_dynamics_loss = sum(train_dynamics_loss) / len(train_dynamics_loss)
            avrg_contrastive_loss = sum(train_contrastive_loss) / len(train_contrastive_loss)
            avrg_action_loss = sum(train_action_loss) / len(train_action_loss)

            current_lr = float(self.scheduler.get_last_lr()[0])

            wandb.log(
                {
                    "Train_total_loss": avrg_total_loss,
                    "Train_recon_loss": avrg_recon_loss,
                    "Train_dyna_loss": avrg_dynamics_loss,
                    "Train_contrastive_loss": avrg_contrastive_loss,
                    "Train_act_loss": avrg_action_loss,
                    "learning_rate": current_lr,
                }
            )

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):

        pass


    def train_step(self, state, action, mask=None):
        """
        Executes a single training step on a mini-batch of data
        """

        self.model.train()

        x_pred, a_pred, zx_prime, pred_zx_prime,zx = self.model.forward(state)

        recon_loss = F.mse_loss(state, x_pred)

        # latent dynamics loss
        z_dynamics_loss = F.mse_loss(pred_zx_prime,zx_prime.clone())

        # contrastive loss
        # if self.model.idm_model.backbone == 'lstm':
        #     contrastive_loss = torch.zeros(1).to(recon_loss.device)
        # else:
        contrastive_loss = 1.0*self.contrastive.get_loss(zx)#torch.zeros(1).to(recon_loss.device)#self.contrastive.get_InfoNCELoss(zx)

        # action prediction loss

        act_pred_loss = F.mse_loss(a_pred, action[:, 1:-1,:], reduction='none').mean(dim=(1,2)) # mean over time and feature dimension
        act_pred_loss = act_pred_loss*(1-mask)
        act_pred_loss = self.action_loss_factor*act_pred_loss.sum()/((1-mask).sum() + 1e-8)


        loss = recon_loss + z_dynamics_loss + contrastive_loss + act_pred_loss


        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.pretrain_param, max_norm=1.0)
        self.optimizer.step()
        self.ema.update()
        self.scheduler.step()

        return loss.item(),recon_loss.item(), z_dynamics_loss.item(),contrastive_loss.item(), act_pred_loss.item()

    @torch.no_grad()
    def evaluate(self, state, action, mask=None):
        """
        Method for evaluating the model on one epoch of data
        """

        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters

        try:

            # nstate = self.Normalizer.normalize_input(state)
            # naction = self.Normalizer.normalize_output(action)
            # nstate = self.mask_act_from_state(nstate)

            x_pred, a_pred, zx_prime, pred_zx_prime,zx = self.model.forward(state)

            recon_loss = F.mse_loss(state, x_pred)
            z_dynamics_loss = F.mse_loss(pred_zx_prime,zx_prime.clone())

            # if self.model.idm_model.backbone=='lstm':
            #     contrastive_loss = torch.zeros(1).to(recon_loss.device)
            # else:
            #     contrastive_loss = self.contrastive.get_InfoNCELoss(zx)
            contrastive_loss = 1.0*self.contrastive.get_loss(zx)#torch.zeros(1).to(recon_loss.device)#self.contrastive.get_InfoNCELoss(zx)

            act_pred_loss = F.mse_loss(a_pred, action[:, 1:-1, :], reduction='none').mean(
                dim=(1, 2))  # mean over time and feature dimension
            act_pred_loss = act_pred_loss * (1 - mask)
            act_pred_loss = self.action_loss_factor * act_pred_loss.sum() / ((1 - mask).sum() + 1e-8)

            loss = act_pred_loss

        finally:
            self.ema.restore()


        return loss.item(),recon_loss.item(), z_dynamics_loss.item(),contrastive_loss.item(), act_pred_loss.item()

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:

            pred_action = self.model.inference(state)

        finally:
            self.ema.restore()

        return pred_action.detach().cpu().numpy()[0]

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name
        ckpt_path = os.path.join(weights_path, file_name)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.model.idm_model.load_state_dict(ckpt["net"])
            self.ema.load_state_dict(ckpt['ema'])
            self.norm_param = ckpt["norm_param"]
            log.info(f'Loaded pre-trained idm_model from {ckpt_path}')

        except FileNotFoundError:
            log.error(f"Checkpoint file not found: {ckpt_path}")
            raise Exception(f"Checkpoint file not found: {ckpt_path}")
        except KeyError as e:
            log.error(f"Key error in checkpoint file: {e}")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name

        state_dict = {
            "net": self.model.idm_model.state_dict(),
            "ema": self.ema.state_dict(),
            "norm_param": self.norm_param
        }

        torch.save(state_dict, os.path.join(store_path, file_name))

    def reset(self):
        pass
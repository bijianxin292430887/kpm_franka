import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional

from mm_data.agent.base_agent import BaseAgent
# from agents.utils.scaler import Normalizer
from latent_policy_from_demo.agents.models.idm.util import build_optimizer_sched

from latent_policy_from_demo.agents.models.idm.ddpm_diffusion import DDPMDiffusion
from latent_policy_from_demo.agents.models.idm.conditional_unet_1D import DiffusionConditionalUnet1D
# from torch_ema import ExponentialMovingAverage
# from agents.models.diffusion.ema import ExponentialMovingAverage
from latent_policy_from_demo.agents.models.idm.ema import ExponentialMovingAverage
from mm_data.spoon_dataset import data_normalize,data_inv_normalize
log = logging.getLogger(__name__)

OBS_HORIZON = 2


class DF_Policy(nn.Module):

    def __init__(self,
                 diffusion_opt: DictConfig,
                 input_dim,
                 obs_encoder: DictConfig,
                 visual_input: bool = False,
                 device: str = 'cuda'):
        super(DF_Policy, self).__init__()

        self.opt = diffusion_opt
        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        self.embedding_dim = self.obs_encoder.embedding_dim

        self.diffusion = DDPMDiffusion(self.opt)
        self.latent_policy = DiffusionConditionalUnet1D(input_dim=input_dim,
                                                        global_cond_dim=OBS_HORIZON * self.embedding_dim).to(device)

        combined_parameters = list(self.latent_policy.parameters()) + list(self.obs_encoder.parameters())

        self.ema = ExponentialMovingAverage(combined_parameters, decay=self.opt.ema)


    def get_obs_embedding(self,obs):
        # Step 1: Reshape to combine Batch and Seq_Length
        batch_size, seq_length, width, height, channels = obs.shape
        obs_cond_tensor = obs.reshape(batch_size * seq_length, width, height, channels)

        # Step 2: Permute the dimensions to (Batch*Seq_Length, Channels, Height, Width)
        obs_cond_tensor = obs_cond_tensor.permute(0, 3, 1, 2)
        # Step 3:
        obs_cond_embedding = self.obs_encoder(obs_cond_tensor)
        # Step 4: Reshape back to (Batch, Seq_Length, Embedding_Dim) if needed
        obs_cond_embedding = obs_cond_embedding.view(batch_size, seq_length, -1)

        return obs_cond_embedding

    def forward(self, obs, act):

        # make prediction
        obs_cond = obs[:, 0:OBS_HORIZON]
        act_future = act[:, OBS_HORIZON:OBS_HORIZON + 12]

        B = obs.shape[0]

        # diffusion latent policy

        timesteps = torch.randint(
            0, self.diffusion.noise_scheduler.config.num_train_timesteps,
            (B,), device=obs.device
        ).long()
        # sample noise to add to actions
        noisy_z, noise = self.diffusion.q_sample(timesteps, act_future)

        obs_cond_embedding = self.get_obs_embedding(obs_cond)
        past_obs_cond = obs_cond_embedding.flatten(start_dim=1)

        # predict the noise residual
        noise_pred = self.latent_policy(
            noisy_z, timesteps, global_cond=past_obs_cond)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sample(self, past_cond,act):
        self.eval()

        # diffusion latent policy
        noisy_act = torch.randn((act.size(0), 12, act.shape[-1]), device=act.device)

        pred_act = self.diffusion.ddpm_sampling(x1=noisy_act, ema=self.ema,
                                                net=self.latent_policy,
                                                cond=past_cond.flatten(start_dim=1),
                                                diffuse_step=self.opt.interval)

        return pred_act

    def get_params(self):
        return self.parameters()


class DF_Policy_Agent(BaseAgent):
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
            scale_data: bool=True,
            eval_every_n_epochs: int = 50,
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # Define the number of GPUs available
        # num_gpus = torch.cuda.device_count()

        # Check if multiple GPUs are available and select the appropriate device
        # if num_gpus > 1:
        #     print(f"Using {num_gpus} GPUs for training.")
        #     self.model = nn.DataParallel(self.model)

        # self.optimizer = hydra.utils.instantiate(optimization, params=self.model.parameters())


        self.eval_model_name = "eval_best_idm.pth"
        self.last_model_name = "last_idm.pth"

        print('init latent policy agent ')
        if trainset is not None:
            self.optimizer, self.sched = build_optimizer_sched(optimization, self.model.parameters(),
                                                               self.train_dataloader, epoch)
            self.norm_param = self.trainset.normalization_params#self.train_dataloader.dataset.normalization_params
            print('get norm_param from training dataset')

        # self.load_pretrained_model('/home/allenbi/PycharmProjects24/d3il/logs/stacking/runs/idm_latent_policy/',"eval_best_idm.pth")
        # print('load latent policy agent ')

    def train_agent(self):
        best_test_mse = 1e10
        early_stop_cnt = 0
        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch + 1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:

                    obs = data['masked_images'].cuda()
                    act = data_normalize(data['robot_joints'], self.norm_param, 'joints').cuda()
                    mean_mse = self.evaluate(obs,act)

                    test_mse.append(mean_mse)

                avrg_test_mse = sum(test_mse) / len(test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))

                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    early_stop_cnt = 0
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')
                else:
                    early_stop_cnt += 1

                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )
                if early_stop_cnt >= 7:
                    log.info('Early Stop!')
                    break

            train_loss = []
            for data in self.train_dataloader:

                obs = data['masked_images'].cuda()
                act = data['robot_joints'].cuda()

                batch_loss = self.train_step(obs,act)

                train_loss.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}, lr_rate:{}".format(num_epoch, avrg_train_loss,
                                                                             self.sched.get_last_lr()[0]))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):

        pass

    def train_step(self, state, action: Optional[torch.Tensor] = None, goal: Optional[torch.Tensor] = None):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        # state = self.Normalizer.normalize_input(state)
        loss = self.model.forward(state,action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.ema.update()

        self.sched.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, state, action: Optional[torch.Tensor] = None, goal: Optional[torch.Tensor] = None):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()

        obs_cond = state[:, 0:OBS_HORIZON]
        act_future = action[:, OBS_HORIZON:OBS_HORIZON + 12]

        obs_cond_embedding = self.model.get_obs_embedding(obs_cond)
        past_obs_cond = obs_cond_embedding.flatten(start_dim=1)

        pred_act_future = self.model.sample(past_obs_cond,act_future)

        mse = F.mse_loss(pred_act_future, act_future)

        return mse.item()

    @torch.no_grad()
    def predict(self, state, if_vision=False,k_step=12) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()

        # action = data_normalize(action.cpu(), self.norm_param, 'poses').to(self.device)

        obs_cond = state[:, 0:OBS_HORIZON]
        dummy_act_future = torch.zeros((1,12,7)).to(self.device)

        obs_cond_embedding = self.model.get_obs_embedding(obs_cond)
        past_obs_cond = obs_cond_embedding.flatten(start_dim=1)
        n_pred_act_future = self.model.sample(past_obs_cond, dummy_act_future)
        # pred_act_future = data_inv_normalize(pred_act_future,self.norm_param, 'poses')

        return n_pred_act_future

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name
        ckpt_path = os.path.join(weights_path, file_name)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.model.latent_policy.load_state_dict(ckpt["net"])
            self.model.obs_encoder.load_state_dict(ckpt["obs"])
            self.model.ema.load_state_dict(ckpt["ema"])
            self.norm_param = ckpt["norm_param"]

            log.info(f'Loaded pre-trained idm policy from {ckpt_path}')
            log.info('loaded norm param from ckpt')

        except FileNotFoundError:
            log.error(f"Checkpoint file not found: {ckpt_path}")
            raise Exception(f"Checkpoint file not found: {ckpt_path}")

        except KeyError as e:
            log.error(f"Key error in checkpoint file: {e}")
            raise Exception(f"Key error in checkpoint file: {e}")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name

        state_dict = {
            "net": self.model.latent_policy.state_dict(),
            "obs": self.model.obs_encoder.state_dict(),
            "ema": self.model.ema.state_dict(),
            "norm_param": self.trainset.normalization_params
        }

        torch.save(state_dict, os.path.join(store_path, file_name))

    def reset(self):
        pass
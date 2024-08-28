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
from agents.utils.scaler import Normalizer
from latent_policy_from_demo.agents.models.idm.util import WarmupLinearSchedule
from latent_policy_from_demo.agents.models.idm.ema import ExponentialMovingAverage
from mm_data.spoon_dataset import data_normalize,data_inv_normalize

log = logging.getLogger(__name__)

OBS_HORIZON = 2

class IDM_Baseline(nn.Module):

    def __init__(self,
                 model: DictConfig,
                 device: str = 'cuda'):

        super(IDM_Baseline, self).__init__()

        self.model = hydra.utils.instantiate(model).to(device)

    def forward(self, obs):

        pred = self.model.forward(obs)

        return pred

    def get_params(self):
        return self.parameters()


class IDM_Baseline_Agent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            # totalset:DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            eval_every_n_epochs: int = 50,
            # normalize_input = True,
            # scale_set = False,
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

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )
        self.scheduler = WarmupLinearSchedule(self.optimizer, 200, 0.1 * optimization.lr, optimization.lr)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.8)

        self.eval_model_name = "eval_best_idm.pth"
        self.last_model_name = "last_idm.pth"

        if trainset is not None:
            self.norm_param = self.train_dataloader.dataset.normalization_params
            print('get norm_param from training dataset')


    def train_agent(self):
        best_test_mse = 1e10
        early_stop_cnt = 0
        for num_epoch in tqdm(range(self.epoch)):

            if not (num_epoch+1) % self.eval_every_n_epochs:
                test_mse = []
                for data in self.test_dataloader:
                    # test dataset is not normalized before creating dataloader, use norm_param from training dataset to normalize test data.
                    state = data_normalize(data['spoon_poses'], self.norm_param, 'poses').cuda()
                    act =  data_normalize(data['robot_joints'], self.norm_param, 'joints').cuda()
                    mask = data['mask'].cuda()

                    mean_mse = self.evaluate(state, act, mask)
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

                    # log.info('New best test loss. Stored weights have been updated!')
                else:
                    early_stop_cnt +=1
                wandb.log(
                    {
                        "mean_test_loss": avrg_test_mse,
                    }
                )
                if early_stop_cnt>=7:
                    log.info('Early Stop!')
                    break


            train_loss = []
            for data in self.train_dataloader:
                state = data['spoon_poses'].cuda()
                act = data['robot_joints'].cuda()
                mask = data['mask'].cuda()

                batch_loss = self.train_step(state, act, mask)

                train_loss.append(batch_loss)

                wandb.log({"loss": batch_loss,})

            avrg_train_loss = sum(train_loss) / len(train_loss)
            log.info("Epoch {}: Average train loss is {}".format(num_epoch, avrg_train_loss))

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        print(f'save model to {self.working_dir}')
        log.info("Training done!")

    def train_vision_agent(self):

        pass


    def train_step(self, state, action: torch.Tensor, mask):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        pred_actions = self.model.forward(state)

        act_pred_loss = F.mse_loss(pred_actions, action[:, 1:-1,:], reduction='none').mean(dim=(1,2))
        act_pred_loss = act_pred_loss*(1-mask)
        loss = act_pred_loss.sum()/((1-mask).sum() + 1e-8)

        # loss = F.mse_loss(pred_actions, action[:,1:-1])

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.ema.update()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, state, action: torch.Tensor, mask):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:
            pred_actions = self.model.forward(state)

            act_pred_loss = F.mse_loss(pred_actions, action[:, 1:-1, :], reduction='none').mean(dim=(1, 2))
            act_pred_loss = act_pred_loss * (1 - mask)
            loss = act_pred_loss.sum() / ((1 - mask).sum() + 1e-8)

        finally:
            self.ema.restore()

        return loss.item()

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, if_vision=False) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()
        self.ema.store()  # Store the current model parameters
        self.ema.copy_to()  # Replace model parameters with the EMA parameters
        try:
            # normalize in idm_agent
            # state = data_normalize(state.cpu(),self.norm_param,'poses').device()
            pred_actions = self.model.forward(state)
            # pred_actions = data_normalize(pred_actions,self.norm_param,'joints')
        finally:
            self.ema.restore()

        return pred_actions.detach().cpu().numpy()[0]

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        file_name = "model_state_dict.pth" if sv_name is None else sv_name
        ckpt_path = os.path.join(weights_path, file_name)
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.model.model.load_state_dict(ckpt["net"])
            self.ema.load_state_dict(ckpt['ema'])
            self.norm_param = ckpt['norm_param']

            log.info(f'Loaded pre-trained idm dynamics model from {ckpt_path}')

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
            "net": self.model.model.state_dict(),
            "ema": self.ema.state_dict(),
            "norm_param": self.norm_param
        }

        torch.save(state_dict, os.path.join(store_path, file_name))
            
    def reset(self):
        pass


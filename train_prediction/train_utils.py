import torch
import logging
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import glob


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DrivingData(Dataset):
    def __init__(self, data_list, n_neighbors):
        self.data_list = data_list
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx],allow_pickle=True)
        ego = data['ego_agent_past']
        neighbors = data['neighbor_agents_past']
        route_lanes = data['route_lanes'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        ego_future_gt = data['ego_agent_future']
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt,neighbors_future_gt


def calc_loss(neighbors, ego, ego_regularization, scores, weights, ego_gt, neighbors_gt, neighbors_valid):
    mask = torch.ne(ego.sum(-1), 0)
    neighbors = neighbors[:, 0] * neighbors_valid 
    cmp_loss = F.smooth_l1_loss(neighbors, neighbors_gt, reduction='none')
    cmp_loss = cmp_loss * mask[:, 0, None, :, None]
    cmp_loss = cmp_loss.sum() / mask[:, 0].sum()

    regularization_loss = F.smooth_l1_loss(ego_regularization, ego_gt, reduction='none')
    regularization_loss = regularization_loss * mask[:, 0, :, None]
    regularization_loss = regularization_loss.sum() / mask[:, 0].sum()

    label = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)    
    irl_loss = F.cross_entropy(scores, label)

    weights_regularization = torch.square(weights).mean()

    loss = cmp_loss + irl_loss + 0.1 * regularization_loss + 0.01 * weights_regularization

    return loss


def calc_metrics(plan_trajectory, prediction_trajectories, scores, ego_future, neighbors_future, neighbors_future_valid):
    best_idx = torch.argmax(scores, dim=-1)
    plan_trajectory = plan_trajectory[torch.arange(plan_trajectory.shape[0]), best_idx]
    prediction_trajectories = prediction_trajectories[torch.arange(prediction_trajectories.shape[0]), best_idx]
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])

    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()


# def MFMA_loss(predictions, scores, ground_truth, weights):
#     global best_mode
#
#     predictions = predictions * weights.unsqueeze(1)
#     prediction_distance = torch.norm(predictions[:, :, :, 9::10, :2] - ground_truth[:, None, 1:, 9::10, :2], dim=-1)
#     prediction_distance = prediction_distance.mean(-1).sum(-1)
#
#     best_mode = torch.argmin(prediction_distance, dim=-1)
#     score_loss = F.cross_entropy(scores, best_mode)
#     best_mode_prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])
#     prediction = torch.cat([best_mode_prediction], dim=1)
#
#     prediction_loss: torch.tensor = 0
#     for i in range(prediction.shape[1]):
#         prediction_loss += F.smooth_l1_loss(prediction[:, i], ground_truth[:, i, :, :3])
#         prediction_loss += F.smooth_l1_loss(prediction[:, i, -1], ground_truth[:, i, -1, :3])
#
#     return 0.5 * prediction_loss + score_loss

def MFMA_loss(predictions, scores, ground_truth, weights):
    global best_mode
    # print(f'predictions shape: {predictions.shape} weights: {weights.shape}')
    # predictions = predictions * weights.unsqueeze(1)
    predictions = predictions[:, 0] * weights
    cmp_loss = F.smooth_l1_loss(predictions, ground_truth, reduction='none')
    label = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)
    irl_loss = F.cross_entropy(scores, label)

    return cmp_loss.mean() + irl_loss

def motion_metrics(prediction_trajectories, neighbors_future,scores, weights):
    best_idx = torch.argmax(scores, dim=-1)
    prediction_trajectories = prediction_trajectories[torch.arange(prediction_trajectories.shape[0]), best_idx]
    prediction_trajectories = prediction_trajectories * weights
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)

    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, weights[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, weights[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return predictorADE.item(), predictorFDE.item()

import os
import csv
import glob
import argparse
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from train_utils import *
from train_prediction.predict_module_only import Predictor


def train_epoch(data_loader, encoder, optimizer):
    epoch_loss = []
    epoch_metrics = []
    encoder.train()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data for predictor
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            neighbors_gt_future = neighbors_gt_future[:, :, -40:, :]  # 只取最后40个时间步
            ego_gt_future = ego_gt_future[:, -40:, :]  # 只取最后40个时间步
            # print('neighbors_gt_future[..., :3] shape1:', neighbors_gt_future[..., :3].shape)
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)
            # print(f'ego_gt_future: {ego_gt_future.shape}...neighbors_gt_future: {neighbors_gt_future.shape}')
            ground_truth = torch.cat([ego_gt_future.unsqueeze(1), neighbors_gt_future], dim=1)
            # print('ground_truth.shape:', ground_truth.shape)
            # encoder
            optimizer.zero_grad()  # 清空梯度
            predictions, scores = encoder( inputs['ego_agent_past'],
                                            inputs['neighbor_agents_past'],
                                            inputs['map_lanes'],
                                            inputs['map_crosswalks'])

            loss = MFMA_loss(predictions, scores, neighbors_gt_future, neighbors_future_valid)  # multi-future multi-agent loss


            # loss backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()

            # compute metrics
            metrics = motion_metrics(predictions, neighbors_gt_future, scores, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    epoch_metrics = [predictionADE, predictionFDE]
    logging.info(f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, encoder):
    epoch_loss = []
    epoch_metrics = []
    encoder.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data for predictor
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            neighbors_gt_future = neighbors_gt_future[:, :, -40:, :]  # 只取最后40个时间步
            ego_gt_future = ego_gt_future[:, -40:, :]  # 只取最后40个时间步

            # print('neighbors_gt_future[..., :3] shape1:', neighbors_gt_future[..., :3].shape)
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)
            # print(f'ego_gt_future: {ego_gt_future.shape}...neighbors_gt_future: {neighbors_gt_future.shape}')
            ground_truth = torch.cat([ego_gt_future.unsqueeze(1), neighbors_gt_future], dim=1)
            # print('ground_truth.shape:', ground_truth.shape)
            # encoder
            # predict
            with torch.no_grad():
                predictions, scores = encoder(inputs['ego_agent_past'],
                                              inputs['neighbor_agents_past'],
                                              inputs['map_lanes'],
                                              inputs['map_crosswalks'])

                loss = MFMA_loss(predictions, scores, neighbors_gt_future,
                                 neighbors_future_valid)  # multi-future multi-agent loss
            # multi-future multi-agent loss

            # compute metrics
            metrics = motion_metrics(predictions, neighbors_gt_future, scores, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    epoch_metrics = [predictionADE, predictionFDE]
    logging.info(f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics


def model_training(args):
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path + 'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up model
    encoder = Predictor(50).to(args.device)
    logging.info("Encoder Params: {}".format(sum(p.numel() for p in encoder.parameters())))

    # set up optimizer
    optimizer = optim.AdamW(encoder.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    # set up data loaders
    train_set = DrivingData(glob.glob(os.path.join(args.train_set, '*.npz')), args.num_neighbors)
    valid_set = DrivingData(glob.glob(os.path.join(args.valid_set, '*.npz')), args.num_neighbors)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))

    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch + 1}/{train_epochs}")

        train_loss, train_metrics = train_epoch(train_loader, encoder, optimizer)
        val_loss, val_metrics = valid_epoch(valid_loader, encoder)

        # save to training log
        log = {'epoch': epoch + 1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss,
               'train-predictionADE': train_metrics[0], 'train-predictionFDE': train_metrics[1],
               'val-predictionADE': val_metrics[0], 'val-predictionFDE': val_metrics[1]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        model = {'encoder': encoder.state_dict()}
        torch.save(model, f'training_log/{args.name}/model_epoch_{epoch + 1}_valADE_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name', default="prediction_training_2")
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_set', type=str, help='path to training data',default="/home/xiaoyu/PhD_Work/nuplan_prediction_planner/nuplan/processed_data/train")
    parser.add_argument('--valid_set', type=str, help='path to validation data',default="/home/xiaoyu/PhD_Work/nuplan_prediction_planner/nuplan/processed_data/valid")
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--num_candidates', type=int, help='number of max candidate trajectories', default=30)
    parser.add_argument('--variable_weights', type=bool, help='use variable cost weights', default=False)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=300)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=2e-4)
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    args = parser.parse_args()

    # Run model training
    model_training(args)
import torch
from predict_module_only import *

# 加载模型A的状态字典
model_a = Predictor(50)  # 模型A实例
state_dict_a = torch.load('/home/xiaoyu/PhD_Work/nuplan_prediction_planner/train_prediction/training_log/prediction_training_2/model_epoch_1_valADE_9.8153.pth',
                          map_location=torch.device('cpu'))  # map_location指定设备（如CPU/GPU）
model_a.load_state_dict(state_dict_a['encoder'], strict=False)  # strict=True确保参数完全匹配（推荐）


print("两个模型参数加载完成！")
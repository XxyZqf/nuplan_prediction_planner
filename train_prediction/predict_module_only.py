import torch
from torch import nn
import torch.nn.functional as F
import math

# Agent history encoder
class AgentEncoder(nn.Module):
    def __init__(self, agent_dim):
        super(AgentEncoder, self).__init__()
        # agent_dim指的input_size是x, y, vx, vy, yaw,等特征量(保证数据集中存在)，256指的hidden_size是隐藏层节点 2指的num_layers是网络个数，2层LSTM
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs) # 数据输入的是7维特征，前面两个分别是batch size 和seq_len（由数据决定，外围参数设置）,输出的是256维特征
        output = traj[:, -1] # 输出的是最后一个时间步的256维特征

        return output

# 地图位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)

"""
地图特征提取：将原始的高维地图数据压缩到统一的256维特征空间
空间关系建模：通过位置编码保留地图元素的空间关系信息
数据降维：通过分段处理（每10个点聚合成1个特征）减少计算量
掩码生成：生成有效数据的掩码，便于后续模块处理可变长度的地图数据
"""
class VectorMapEncoder(nn.Module):
    """
    参数说明：
    map_dim：输入地图点的特征维度
    map_len：地图特征序列的最大长度
    """
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    """
    map：原始地图数据
    map_encoding：经过 point_net 和位置编码处理后的特征
    """
    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape #获取输入特征的维度信息（批大小B，元素数量N_e，点数量N_p，特征维度D）
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))#使用 max_pool2d 进行池化操作，将每10个连续的点聚合为一个特征点

        #调整特征张量的维度顺序，并重新塑形
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        # 生成对应的掩码（mask），标记哪些部分是有效的地图数据
        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        # 添加位置编码
        output = self.position_encode(self.point_net(input))
        # 地图分段处理
        encoding, mask = self.segment_map(input, output)

        return encoding, mask


# Transformer modules
class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        """
        地图与轨迹的跨模态注意力
        self.cross_attention：实例化一个多头注意力机制，参数说明：
        256：输入特征的维度
        8：注意力头的数量，将特征分为8个子空间并行处理
        0.1：注意力计算中的 dropout 概率，用于防止过拟合
        batch_first=True：指定输入的第一个维度为批次维度
        """
        self.cross_attention = nn.MultiheadAttention(256, 8, 0.1, batch_first=True)

        """
        self.transformer：一个前馈神经网络模块，包含：
        层归一化 (LayerNorm)
        线性变换 (从256维到1024维)
        ReLU 激活函数
        Dropout (概率为0.1)
        线性变换 (从1024维回到256维)
        层归一化 (LayerNorm)
        """
        self.transformer = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(1024, 256), nn.LayerNorm(256))

    def forward(self, query, key, value, mask=None):
        """
        接收四个参数：
        query：查询向量，通常是目标特征（如智能体状态）
        key：键向量，通常是源特征（如地图元素）
        value：值向量，与键向量对应的数据值
        mask：可选的掩码，用于标记哪些位置的数据是有效的（非填充值）
        """
        # print(f'query1.shape: {query.shape}')
        # print(f'key.shape: {key.shape}')
        # print(f'value.shape: {value.shape}')
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.transformer(attention_output)
        # print(f'output_cross_transformer.shape: {output.shape}')
        return output

# MultiModalTransformer类是一个多模态注意力融合模块，位于自动驾驶预测规划模型中，
# 主要负责处理不同模态数据之间的交互与融合。该类使用多个并行的多头注意力机制来捕获
# 输入特征之间的复杂关系，特别适用于融合地图信息与智能体状态的场景。
class MultiModalTransformer(nn.Module):
    """
    modes：设置并行注意力头的数量，默认为3，表示模型将从多个不同角度处理输入特征
    output_dim：输出特征的维度，默认为256
    """
    def __init__(self, modes=3, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList([nn.MultiheadAttention(256, 4, 0.1, batch_first=True) for _ in range(modes)])
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(1024, output_dim), nn.LayerNorm(output_dim))

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output)

        return output


# Transformer-based encoders  使用注意力机制用于处理智能体之间的交互关系
class Agent2Agent(nn.Module):
    def __init__(self):
        super(Agent2Agent, self).__init__()
        # 关键参数：
        # d_model = 256：输入和输出特征维度为256
        # nhead = 8：使用8个注意力头，从不同角度捕获智能体间关系
        # dim_feedforward = 1024：前馈网络隐藏层维度为1024
        # batch_first = True：输入数据格式为[batch, seq, feature]
        # num_layers = 2：使用2层Transformer编码器，增加模型表达能力
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='relu',
                                                   batch_first=True)
        self.interaction_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, mask=None):
        # inputs：智能体特征序列，形状为[batch, num_agents, 256]
        # mask：可选的掩码，用于标记无效的智能体（如填充的零值）
        output = self.interaction_net(inputs, src_key_padding_mask=mask)

        return output

# 使用注意力机制，分别处理智能体与车道线、智能体与人行横道之间的交互
class Agent2Map(nn.Module):
    def __init__(self):
        super(Agent2Map, self).__init__()
        self.lane_attention = CrossTransformer()#用于处理智能体与车道线之间的交叉注意力计算
        self.crosswalk_attention = CrossTransformer() #用于处理智能体与人行横道之间的交叉注意力计算
        self.map_attention = MultiModalTransformer()#多模态注意力模块，用于融合所有地图元素特征

    def forward(self, actor, lanes, crosswalks, mask):
        """
        actor是智能体的特征向量，形状为 [batch_size, feature_dim]（例如[B, 256]）
        unsqueeze(1)操作在第1维增加一个维度，将形状变为 [B, 1, 256]
        lanes是车道线特征，形状为 [batch_size, num_lanes, feature_dim]（例如[B, 10, 256]）
        crosswalks是人行横道特征，形状为 [batch_size, num_crosswalks, feature_dim]（例如[B, 5, 256]）
        mask是可选的掩码，用于标记哪些位置的数据是有效的（非填充值）
        """
        query = actor.unsqueeze(1)

        print(f'actor.shape: {actor.shape}')
        print(f'lanes.shape: {lanes.shape}')
        print(f'crosswalks.shape: {crosswalks.shape}')
        print(f'query.shape: {query.shape}')
        print(f'lanes.shape[1]: {lanes.shape[1]}')
        lanes_actor = [self.lane_attention(query, lanes, lanes)]#for i in range(lanes.shape[1])
        crosswalks_actor = [self.crosswalk_attention(query, crosswalks, crosswalks) ]#for i inrange(crosswalks.shape[1])
        print(f'len(lanes_actor): {len(lanes_actor)}')
        map_actor = torch.cat(lanes_actor + crosswalks_actor, dim=1)
        output = self.map_attention(query, map_actor, map_actor, mask).squeeze(2)

        return map_actor, output

    # Decoders


# class AgentDecoder(nn.Module):
#     """
#     Dropout层（概率0.1）：防止过拟合
#     全连接层（512→256）：特征降维
#     ELU激活函数：引入非线性变换
#     全连接层（256→future_steps*3）：输出层，为每个预测步生成3个值（x、y、θ的变化量）
#     """
#     def __init__(self, future_steps):
#         super(AgentDecoder, self).__init__()
#         self._future_steps = future_steps
#         self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps * 3))
#
#     def transform(self, prediction, current_state):
#         x = current_state[:, 0]
#         y = current_state[:, 1]
#         theta = current_state[:, 2]
#         delta_x = prediction[:, :, 0]
#         delta_y = prediction[:, :, 1]
#         delta_theta = prediction[:, :, 2]
#         new_x = x.unsqueeze(1) + delta_x
#         new_y = y.unsqueeze(1) + delta_y
#         new_theta = theta.unsqueeze(1) + delta_theta
#         traj = torch.stack([new_x, new_y, new_theta], dim=-1)
#
#         return traj
#
#     """
#     agent_map：智能体-地图交互特征，形状为[batch, num_agents, feature_dim]
#     agent_agent：智能体-智能体交互特征，形状为[batch, num_agents, feature_dim]
#     current_state：智能体当前状态，形状为[batch, num_agents, 3]（x, y, θ）
#     agent_map: [batch_size, 10, 256] # 10个周边车辆，每个256维特征
#     agent_agent: [batch_size, 10, 256] # 智能体交互特征
#     current_state: [batch_size, 10, 3] # 10个周边车辆的当前位置和朝向
#     feature: [batch_size, 10, 512] # 10个周边车辆，每个512维特征
#     decoded: [batch_size, 10, 3, future_steps, 3] # 10个周边车辆，每个3个预测步，每个预测步3个值（x、y、θ的变化量）
#     """
#     def forward(self, agent_map, agent_agent, current_state):
#         feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1, 1)], dim=-1)
#         decoded = self.decode(feature).view(-1, 3, 10, self._future_steps, 3)
#         trajs = torch.stack(
#             [self.transform(decoded[:, i, j], current_state[:, j]) for i in range(3) for j in range(10)], dim=1)
#         trajs = torch.reshape(trajs, (-1, 3, 10, self._future_steps, 3))
#
#         return trajs

class AgentDecoder(nn.Module):
    def __init__(self, max_time, max_branch):
        super(AgentDecoder, self).__init__()
        self._max_time = max_time
        self._max_branch = max_branch
        self.traj_decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3 * 10)
        )
    def forward(self, encoding, current_state):
        encoding = torch.reshape(encoding, (encoding.shape[0], self._max_branch, self._max_time, 256))
        # print(f'encoding_AgentDecoder.shape = {encoding.shape}')
        agent_traj = self.traj_decoder(encoding).reshape(encoding.shape[0], self._max_branch, self._max_time*10, 3)
        # print(f'agent_traj.shape = {agent_traj.shape}')
        # print(f'current_state.shape = {current_state.shape}')
        agent_traj += current_state[:, None, None, :3]

        return agent_traj

class CrossAttention(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, attn_mask=mask)
        attention_output = self.norm_1(attention_output)
        linear_output = self.ffn(attention_output)
        output = attention_output + self.dropout(linear_output)
        output = self.norm_2(output)

        return output

#DIPP的评分机制
class Score(nn.Module):
    """
    self.reduce：特征降维网络，将512维输入压缩到256维，包含Dropout(0.1)防止过拟合
    self.decode：评分解码网络，将512维输入转换为1维评分，包含两层全连接和ELU激活函数
    """
    def __init__(self):
        super(Score, self).__init__()
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(256, 128), nn.ELU())
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 30))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, encoding):
        # 如果有额外维度(16,236,256)，我们需要将其压缩
        if len(encoding.shape) == 3:
            # 先通过reduce层处理最后一维
            batch_size, seq_len, feature_dim = encoding.shape
            feature = self.reduce(encoding.detach().reshape(-1, feature_dim))
            # 恢复形状并在序列维度上进行池化
            feature = feature.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
            # 使用自适应池化将序列长度维度压缩为1
            feature = self.pool(feature).squeeze(-1)
        else:
            feature = self.reduce(encoding.detach())
        scores = self.decode(feature)
        # print(f'scores.shape = {scores.shape}')
        return scores

#DTPP的评分机制
class ScoreDecoder(nn.Module):
    def __init__(self, variable_cost=False):
        super(ScoreDecoder, self).__init__()
        self._n_latent_features = 4
        self._variable_cost = variable_cost

        self.interaction_feature_encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256))
        self.interaction_feature_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(),
                                                         nn.Linear(64, self._n_latent_features), nn.Sigmoid())
        self.weights_decoder = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, self._n_latent_features + 4),
                                             nn.Softplus())

    def get_hardcoded_features(self, ego_traj, max_time):
        # ego_traj: B, M, T, 6
        # x, y, yaw, v, a, r

        speed = ego_traj[:, :, :max_time, 3]
        acceleration = ego_traj[:, :, :max_time, 4]
        jerk = torch.diff(acceleration, dim=-1) / 0.1
        jerk = torch.cat((jerk[:, :, :1], jerk), dim=-1)
        curvature = ego_traj[:, :, :max_time, 5]
        lateral_acceleration = speed ** 2 * curvature

        speed = -speed.mean(-1).clip(0, 15) / 15
        acceleration = acceleration.abs().mean(-1).clip(0, 4) / 4
        jerk = jerk.abs().mean(-1).clip(0, 6) / 6
        lateral_acceleration = lateral_acceleration.abs().mean(-1).clip(0, 5) / 5

        features = torch.stack((speed, acceleration, jerk, lateral_acceleration), dim=-1)

        return features

    def calculate_collision(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 3
        # agent_traj: B, N, T, 3
        # agents_states: B, N, 11

        agent_mask = torch.ne(agents_states.sum(-1), 0)  # B, N

        # Compute the distance between the two agents
        dist = torch.norm(ego_traj[:, None, :max_time, :2] - agent_traj[:, :, :max_time, :2], dim=-1)

        # Compute the collision cost
        cost = torch.exp(-0.2 * dist ** 2) * agent_mask[:, :, None]
        cost = cost.sum(-1).sum(-1)

        return cost

    def get_latent_interaction_features(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 6
        # agent_traj: B, N, T, 3
        # agents_states: B, N, 11

        # Get agent mask
        agent_mask = torch.ne(agents_states.sum(-1), 0)  # B, N

        # Get relative attributes of agents
        relative_yaw = agent_traj[:, :, :max_time, 2] - ego_traj[:, None, :max_time, 2]
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))
        relative_pos = agent_traj[:, :, :max_time, :2] - ego_traj[:, None, :max_time, :2]
        relative_pos = torch.stack([relative_pos[..., 0] * torch.cos(relative_yaw),
                                    relative_pos[..., 1] * torch.sin(relative_yaw)], dim=-1)
        agent_velocity = torch.diff(agent_traj[:, :, :max_time, :2], dim=-2) / 0.1
        agent_velocity = torch.cat((agent_velocity[:, :, :1, :], agent_velocity), dim=-2)
        ego_velocity_x = ego_traj[:, :max_time, 3] * torch.cos(ego_traj[:, :max_time, 2])
        ego_velocity_y = ego_traj[:, :max_time, 3] * torch.sin(ego_traj[:, :max_time, 2])
        relative_velocity = torch.stack([(agent_velocity[..., 0] - ego_velocity_x[:, None]) * torch.cos(relative_yaw),
                                         (agent_velocity[..., 1] - ego_velocity_y[:, None]) * torch.sin(relative_yaw)],
                                        dim=-1)
        relative_attributes = torch.cat((relative_pos, relative_yaw.unsqueeze(-1), relative_velocity), dim=-1)

        # Get agent attributes
        agent_attributes = agents_states[:, :, None, 6:].expand(-1, -1, relative_attributes.shape[2], -1)
        attributes = torch.cat((relative_attributes, agent_attributes), dim=-1)
        attributes = attributes * agent_mask[:, :, None, None]

        # Encode relative attributes and decode to latent interaction features
        features = self.interaction_feature_encoder(attributes)
        features = features.max(1).values.mean(1)
        features = self.interaction_feature_decoder(features)

        return features

    def forward(self, ego_traj, ego_encoding, agents_traj, agents_states, timesteps):
        ego_traj_features = self.get_hardcoded_features(ego_traj, timesteps)
        if not self._variable_cost:
            ego_encoding = torch.ones_like(ego_encoding)
        weights = self.weights_decoder(ego_encoding)
        ego_mask = torch.ne(ego_traj.sum(-1).sum(-1), 0)

        scores = []
        # print(f'agents_traj: {agents_traj.shape}')
        for i in range(agents_traj.shape[1]):
            # print(f'ego_traj_features: {ego_traj_features.shape}')
            hardcoded_features = ego_traj_features[:, i]
            interaction_features = self.get_latent_interaction_features(ego_traj[:, i], agents_traj[:, i],
                                                                        agents_states, timesteps)
            features = torch.cat((hardcoded_features, interaction_features), dim=-1)
            score = -torch.sum(features * weights, dim=-1)
            collision_feature = self.calculate_collision(ego_traj[:, i], agents_traj[:, i], agents_states, timesteps)
            score += -10 * collision_feature
            scores.append(score)

        scores = torch.stack(scores, dim=1)
        scores = torch.where(ego_mask, scores, float('-inf'))

        return scores, weights

# Build predictor
class Predictor(nn.Module):
    def __init__(self, future_steps,dim=256, layers=2, heads=8, dropout=0.1,neighbors=10, max_time=8,
                 max_branch=30, n_heads=8,  variable_cost=False):
        super(Predictor, self).__init__()
        self._future_steps = future_steps
        self._lane_len = 50
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        # agent layer
        self.ego_encoder = AgentEncoder(agent_dim=7)  # 自车仍用车辆编码器
        self.neighbor_encoder = AgentEncoder(agent_dim=11)  # 所有邻居共用一个编码器

        # map layer
        self.lane_net = VectorMapEncoder(self._lane_feature, self._lane_len)
        self.crosswalk_net = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)

        # 注意力机制encoder
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()
        # decode layers
        self._neighbors = neighbors
        self._nheads = n_heads
        self._time = max_time
        self._branch = max_branch
        self.predict = AgentDecoder(max_time, max_branch)
        self.scorer = ScoreDecoder()

    def forward(self, ego, neighbors, map_lanes, map_crosswalks,ego_traj_inputs,timesteps):
        # agents encoding
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)  # 数据拼接
        # print(f'actors.shape: {actors.shape}')
        encoded_ego = self.ego_encoder(ego)
        # print(f'encoded_ego.shape: {encoded_ego.shape}')
        # 修复：使用正确初始化的self.neighbor_encoder
        encoded_neighbors = [self.neighbor_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)
        # print(f'neighbors.shape = {neighbors.shape}...actors_mask.shape: {actors_mask.shape}...encoded_actors.shape: {encoded_actors.shape}')
        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_net(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_net(map_crosswalks)
        # print(f'encoded_map_crosswalks.shape: {encoded_map_crosswalks.shape}')
        # attention fusion encoding
        input = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        map_mask= torch.cat([lanes_mask, crosswalks_mask], dim=1)
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)
        # 将 encoding 从 16×236 补齐为 16×240
        supplement = torch.zeros(encoding.shape[0], 240 - encoding.shape[1],encoding.shape[2], device=encoding.device)
        encoding = torch.cat([encoding, supplement], dim=1)
        # print(f'encoding.shape: {encoding.shape}')
        # encoding
        # 3. 准备预测所需的当前状态
        current_states = neighbors[:, :self._neighbors, -1]
        # print(f'current_states.shape: {current_states.shape}')

        # 4. 生成预测和评分
        agents_trajecotries = []
        for i in range(self._neighbors):
            # print(f'encoding: {encoding.shape}')
            trajectory = self.predict(encoding, current_states[:, i])
            agents_trajecotries.append(trajectory)

        predictions = torch.stack(agents_trajecotries, dim=2)
        # print(f'predictions.shape: {predictions.shape}')

        # 评分
        # scores = self.score(encoding)
        scores, weights = self.scorer(ego_traj_inputs,encoding[:, 0], predictions, current_states, timesteps)

        return predictions, scores


if __name__ == "__main__":
    # set up model
    model = Predictor(50)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))

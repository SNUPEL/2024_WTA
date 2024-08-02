import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
from cfg import get_cfg
import numpy as np
from agent.GCRN import GCRN
cfg = get_cfg()
from torch.distributions import Categorical
from utils import weight_init_xavier_uniform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, n_representation_obs, layers = [20, 30 ,40]):
        """
        :param feature_size:
        :param n_representation_obs:
        :param layers:
        """
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.linears = OrderedDict()
        last_layer = self.feature_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.linears['linear{}'.format(i)]= nn.Linear(last_layer, layer)
                #self.linears['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.linears['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs)

        self.node_embedding = nn.Sequential(self.linears)
        #print(self.node_embedding)
        self.node_embedding.apply(weight_init_xavier_uniform)


    def forward(self, node_feature, missile=False):
        """
        :param node_feature:
        :param missile:
        :return:
        """
        node_representation = self.node_embedding(node_feature)
        return node_representation

class PPONetwork(nn.Module):
    def __init__(self, state_size, state_action_size, layers=[8,12]):
        """
        :param state_size:
        :param state_action_size:
        :param layers:
        """
        super(PPONetwork, self).__init__()
        self.state_size = state_size
        self.state_action_size = state_action_size
        self.NN_sequential = OrderedDict()
        self.fc_pi = nn.Linear(state_action_size, layers[0])
        self.bn_pi = nn.BatchNorm1d(layers[0])
        self.fc_v = nn.Linear(state_size, layers[0])
        self.bn_v = nn.BatchNorm1d(layers[0])
        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                #self.fcn['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            #else:
        self.forward_cal = nn.Sequential(self.fcn)
        self.output_pi = nn.Linear(last_layer, 1)
        self.output_v = nn.Linear(last_layer, 1)




    def pi(self, x, visualize = False):
        """
        :param x:
        :param visualize:
        :return:
        """
        if visualize == False:
            x = self.fc_pi(x)
            x = F.elu(x)
            x = self.forward_cal(x)
            pi = self.output_pi(x)
            return pi
        else:
            x = self.fc_pi(x)
            x = F.elu(x)
            x = self.forward_cal(x)
            pi = self.output_pi(x)
            return x

    def v(self, x):
        x = self.fc_v(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        v = self.output_v(x)
        return v





class Agent:
    def __init__(self,
                 action_size,
                 feature_size_ship,
                 feature_size_missile,
                 n_node_feature_missile,
                 node_embedding_layers_ship=list(eval(cfg.ship_layers)),
                 node_embedding_layers_missile=list(eval(cfg.missile_layers)),
                 n_representation_ship = cfg.n_representation_ship,
                 n_representation_missile = cfg.n_representation_missile,
                 n_representation_action = cfg.n_representation_action,
                 learning_rate=cfg.lr,
                 learning_rate_critic=cfg.lr_critic,
                 gamma=cfg.gamma,
                 lmbda=cfg.lmbda,
                 eps_clip = cfg.eps_clip,
                 K_epoch = cfg.K_epoch,
                 layers=list(eval(cfg.ppo_layers))
                 ):
        """
        :param action_size:
        :param feature_size_ship:
        :param feature_size_missile:
        :param n_node_feature_missile:
        :param node_embedding_layers_ship:
        :param node_embedding_layers_missile:
        :param n_representation_ship:
        :param n_representation_missile:
        :param n_representation_action:
        :param learning_rate:
        :param learning_rate_critic:
        :param gamma:
        :param lmbda:
        :param eps_clip:
        :param K_epoch:
        :param layers:
        """
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.data = []

        self.network = PPONetwork(state_size = n_representation_ship,
                               state_action_size = n_representation_ship+cfg.n_representation_action2 ,
                               layers = layers).to(device)
        self.node_representation_ship_feature = NodeEmbedding(feature_size=feature_size_ship,
                                                         n_representation_obs=n_representation_ship,
                                                         layers = node_embedding_layers_ship).to(device)  # 수정사항

        self.node_representation_wo_graph = NodeEmbedding(feature_size=feature_size_missile,
                                                              n_representation_obs=n_representation_action,
                                                              layers=node_embedding_layers_missile).to(device)  # 수정사항

        self.func_meta_path = GCRN(feature_size=feature_size_missile,
                                   embedding_size=cfg.n_representation_action,
                                   graph_embedding_size=cfg.hidden_size_meta_path,
                                   layers=node_embedding_layers_missile,
                                   num_node_cat=1,
                                   num_edge_cat=5).to(device)
        self.func_meta_path2 = GCRN(feature_size=cfg.n_representation_action,
                                    embedding_size=cfg.n_representation_action2,
                                    graph_embedding_size=cfg.hidden_size_meta_path2,
                                    layers=node_embedding_layers_missile,
                                    num_node_cat=1,
                                    num_edge_cat=5).to(device)
        self.func_meta_path3 = GCRN(feature_size=cfg.n_representation_action,
                                    embedding_size=cfg.n_representation_action2,
                                    graph_embedding_size=cfg.hidden_size_meta_path2,
                                    layers=node_embedding_layers_missile,
                                    num_node_cat=1,
                                    num_edge_cat=5).to(device)
        self.func_meta_path4 = GCRN(feature_size=cfg.n_representation_action,
                                    embedding_size=cfg.n_representation_action2,
                                    graph_embedding_size=cfg.hidden_size_meta_path2,
                                    layers=node_embedding_layers_missile,
                                    num_node_cat=1,
                                    num_edge_cat=5).to(device)
        self.func_meta_path5 = GCRN(feature_size=cfg.n_representation_action,
                                    embedding_size=cfg.n_representation_action2,
                                    graph_embedding_size=cfg.hidden_size_meta_path2,
                                    layers=node_embedding_layers_missile,
                                    num_node_cat=1,
                                    num_edge_cat=5).to(device)
        self.func_meta_path6 = GCRN(feature_size=cfg.n_representation_action,
                                    embedding_size=cfg.n_representation_action2,
                                    graph_embedding_size=cfg.hidden_size_meta_path2,
                                    layers=node_embedding_layers_missile,
                                    num_node_cat=1,
                                    num_edge_cat=5).to(device)

        if cfg.k_hop == 0:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.node_representation_ship_feature.parameters()) + \
                               list(self.node_representation_wo_graph.parameters())
        if cfg.k_hop == 2:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.node_representation_ship_feature.parameters()) + \
                               list(self.func_meta_path.parameters()) + \
                               list(self.func_meta_path2.parameters())

        if cfg.k_hop == 3:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.node_representation_ship_feature.parameters()) + \
                               list(self.func_meta_path.parameters()) + \
                               list(self.func_meta_path2.parameters()) + \
                               list(self.func_meta_path3.parameters())
        if cfg.k_hop == 4:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.node_representation_ship_feature.parameters()) + \
                               list(self.func_meta_path.parameters()) + \
                               list(self.func_meta_path2.parameters()) + \
                               list(self.func_meta_path3.parameters()) + \
                               list(self.func_meta_path4.parameters())
        if cfg.k_hop == 5:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.node_representation_ship_feature.parameters()) + \
                               list(self.func_meta_path.parameters()) + \
                               list(self.func_meta_path2.parameters()) + \
                               list(self.func_meta_path3.parameters()) + \
                               list(self.func_meta_path4.parameters()) + \
                               list(self.func_meta_path5.parameters())
        if cfg.k_hop == 6:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.node_representation_ship_feature.parameters()) + \
                               list(self.func_meta_path.parameters()) + \
                               list(self.func_meta_path2.parameters()) + \
                               list(self.func_meta_path3.parameters()) + \
                               list(self.func_meta_path4.parameters()) + \
                               list(self.func_meta_path5.parameters()) + \
                               list(self.func_meta_path6.parameters())
        if cfg.optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.eval_params, lr=learning_rate)  #
        if cfg.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.eval_params, lr=learning_rate)  #
        if cfg.optimizer == 'ADAMW':
            self.optimizer = optim.AdamW(self.eval_params, lr=learning_rate)  #
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)
        self.dummy_node = [[[0] * feature_size_missile for _ in range(i)] for i in range(n_node_feature_missile)]
        self.ship_feature_list= list()
        self.ship_feature_next_list = list()
        self.missile_node_feature_list= list()
        self.heterogeneous_edges_list= list()
        self.action_feature_list= list()
        self.action_blue_list= list()
        self.reward_list= list()
        self.prob_list= list()
        self.mask_list= list()
        self.done_list= list()
        self.avail_action_blue_list= list()
        self.a_index_list= list()
        self.dummy_ship_feature = torch.zeros((1, feature_size_ship)).to(device).float()
        # ship_feature, ship_feature_next, missile_node_feature, heterogeneous_edges, action_feature, action_blue, reward, prob, mask, done, avail_action_blue, a_index
        self.batch_store = []
    def batch_reset(self):
        self.batch_store = []

    @torch.no_grad()
    def get_td_target(self, ship_features, node_features_missile, heterogenous_edges, possible_actions, action_feature, reward, done):
        obs_next, act_graph = self.get_node_representation(ship_features,node_features_missile, heterogenous_edges,mini_batch=False)
        td_target = reward + self.gamma * self.network.v(obs_next) * (1 - done)
        return td_target.tolist()[0][0]
    @torch.no_grad()
    def sample_action_visualize(self, ship_features,node_features_missile, heterogenous_edges, possible_actions,action_feature, random):
        if random == False:
            obs, act_graph = self.get_node_representation(ship_features,node_features_missile,heterogenous_edges,mini_batch=False)
            act_graph = act_graph[0:self.action_size, :]
            obs = obs.expand([act_graph.shape[0], obs.shape[1]])
            obs_n_action = torch.cat([obs, act_graph], dim = 1)
            logit = [self.network.pi(obs_n_action[i].unsqueeze(0)) for i in range(obs_n_action.shape[0])]
            outputs = [self.network.pi(obs_n_action[i].unsqueeze(0), visualize=True) for i in
                       range(obs_n_action.shape[0])]
            logit = torch.stack(logit).view(1, -1)
            action_size = obs_n_action.shape[0]
            if action_size >= self.action_size:
                 action_size = self.action_size
            remain_action = torch.tensor([-1e8 for _ in range(self.action_size - action_size)], device=device).unsqueeze(0)
            logit = torch.cat([logit, remain_action], dim=1)
            mask = torch.tensor(possible_actions, device=device).bool()
            logit = logit.masked_fill(mask == 0, -1e8)
            prob = torch.softmax(logit, dim=-1)
            m = Categorical(prob)
            a = m.sample().item()
            a_index = a
            prob_a = prob.squeeze(0)[a]
            action_blue = action_feature[a]
        else:
            obs, act_graph = self.get_node_representation(ship_features,node_features_missile,heterogenous_edges,mini_batch=False)
            act_graph = act_graph[0:self.action_size, :]

            obs = obs.expand([act_graph.shape[0], obs.shape[1]])
            obs_n_action = torch.cat([obs, act_graph], dim = 1)
            logit = [self.network.pi(obs_n_action[i].unsqueeze(0)) for i in range(obs_n_action.shape[0])]
            outputs = [self.network.pi(obs_n_action[i].unsqueeze(0), visualize = True) for i in range(obs_n_action.shape[0])]
            logit = torch.stack(logit).view(1, -1)
            action_size = obs_n_action.shape[0]
            if action_size >= self.action_size:
                 action_size = self.action_size
            remain_action = torch.tensor([-1e8 for _ in range(self.action_size - action_size)], device=device).unsqueeze(0)
            logit = torch.cat([logit, remain_action], dim=1)
            mask = torch.tensor(possible_actions, device=device).bool()
            logit = logit.masked_fill(mask == 0, -1e8)
            logit = logit.masked_fill(mask == 1, 1)
            prob = torch.softmax(logit, dim=-1)
            m = Categorical(prob)
            a = m.sample().item()
            a_index = a
            prob_a = prob.squeeze(0)[a]
            action_blue = action_feature[a]
            #print(act_graph[a_index].tolist())
        graph_embedding = act_graph[a_index].tolist()
        node_feature = node_features_missile[a_index]

        output = outputs[a_index].tolist()[0]
        return action_blue, prob_a, mask, a_index, graph_embedding, node_feature, output

    def get_node_representation(self,
                                ship_features,
                                missile_node_feature,
                                edge_index_missile,
                                mini_batch=False):

        if mini_batch == False:

            with torch.no_grad():

                ship_features = torch.tensor(ship_features, dtype=torch.float, device=device)
                node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
                missile_node_feature = torch.tensor(missile_node_feature, dtype=torch.float,device=device).clone().detach()
                if cfg.k_hop != 0:
                    node_representation_graph = self.func_meta_path(A=edge_index_missile,X=missile_node_feature, mini_batch=mini_batch)
                    node_representation_graph = self.func_meta_path2(A=edge_index_missile, X=node_representation_graph,mini_batch=mini_batch)
                    if  cfg.k_hop == 3:
                        node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)

                    if  cfg.k_hop == 4:
                        node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)
                        node_representation_graph = self.func_meta_path4(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)



                    if  cfg.k_hop == 5:
                        node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)
                        node_representation_graph = self.func_meta_path4(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)
                        node_representation_graph = self.func_meta_path5(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)


                    if  cfg.k_hop == 6:
                        node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)
                        node_representation_graph = self.func_meta_path4(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)
                        node_representation_graph = self.func_meta_path5(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)
                        node_representation_graph = self.func_meta_path6(A=edge_index_missile, X=node_representation_graph,
                                                                         mini_batch=mini_batch)
                else:
                    node_representation_graph = self.node_representation_wo_graph(missile_node_feature)


                node_representation = torch.cat([node_embedding_ship_features], dim=1)
                return node_representation, node_representation_graph
        else:
            """ship feature 만드는 부분"""

            ship_features = torch.tensor(ship_features,dtype=torch.float).to(device).squeeze(1)
            node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
            max_len = np.max([len(mnf) for mnf in missile_node_feature])
            if max_len <= self.action_size:
                max_len = self.action_size
            temp = list()
            for mnf in missile_node_feature:
                temp.append(
                    torch.cat([torch.tensor(mnf), torch.tensor(self.dummy_node[max_len - len(mnf)])], dim=0).tolist())
            #
            missile_node_feature = torch.tensor(temp, dtype=torch.float).to(device)
            if cfg.k_hop !=0:

                node_representation_graph = self.func_meta_path(A=edge_index_missile, X=missile_node_feature, mini_batch=mini_batch)
                node_representation_graph = self.func_meta_path2(A=edge_index_missile, X=node_representation_graph, mini_batch=mini_batch)
                if cfg.k_hop == 3:
                    node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)

                if cfg.k_hop == 4:
                    node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)
                    node_representation_graph = self.func_meta_path4(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)

                if cfg.k_hop == 5:
                    node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)
                    node_representation_graph = self.func_meta_path4(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)
                    node_representation_graph = self.func_meta_path5(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)

                if cfg.k_hop == 6:
                    node_representation_graph = self.func_meta_path3(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)
                    node_representation_graph = self.func_meta_path4(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)
                    node_representation_graph = self.func_meta_path5(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)
                    node_representation_graph = self.func_meta_path6(A=edge_index_missile, X=node_representation_graph,
                                                                     mini_batch=mini_batch)
            else:
                batch_size=  missile_node_feature.shape[0]
                missile_node_size = missile_node_feature.shape[1]

                missile_node_feature = missile_node_feature.reshape(batch_size * missile_node_size, -1)
                missile_node_feature = self.node_representation_wo_graph(missile_node_feature)
                node_representation_graph = missile_node_feature.reshape(batch_size, missile_node_size, -1)
            node_representation = torch.cat([node_embedding_ship_features], dim=1)
            return node_representation, node_representation_graph


    def get_ship_representation(self, ship_features):
        """ship feature 만드는 부분"""
        ship_features = torch.tensor(ship_features,dtype=torch.float).to(device).squeeze(1)
        node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
        return node_embedding_ship_features

    def put_data(self, transition):
        # if transition[7] == True:
        self.ship_feature_list.append(transition[0])
        self.missile_node_feature_list.append(transition[1])
        self.heterogeneous_edges_list.append(transition[2])
        self.action_feature_list.append(transition[3])
        self.action_blue_list.append(transition[4])
        self.reward_list.append(transition[5])
        self.prob_list.append(transition[6])
        self.done_list.append(transition[7])
        self.avail_action_blue_list.append(transition[8])
        self.a_index_list.append(transition[9])
        if transition[7] == True:
            batch_data = (self.ship_feature_list,
                          self.missile_node_feature_list,
                          self.heterogeneous_edges_list,
                          self.action_feature_list,
                          self.action_blue_list,
                          self.reward_list,
                          self.prob_list,
                          self.done_list,
                          self.avail_action_blue_list,
                          self.a_index_list)
            self.batch_store.append(batch_data)
            self.ship_feature_list = list()
            self.ship_feature_next_list = list()
            self.missile_node_feature_list = list()
            self.heterogeneous_edges_list = list()
            self.action_feature_list = list()
            self.action_blue_list = list()
            self.reward_list = list()
            self.prob_list = list()
            self.mask_list = list()
            self.done_list = list()
            self.avail_action_blue_list = list()
            self.a_index_list = list()


            # ship_feature_list = batch_data[0]
            # missile_node_feature_list = batch_data[1]
            # heterogeneous_edges_list = batch_data[2]
            # action_feature_list = batch_data[3]
            # action_blue_list = batch_data[4]
            # reward_list = batch_data[5]
            # prob_list = batch_data[6]
            # mask_list = batch_data[7]
            # done_list = batch_data[8]
            # avail_action_blue_list = batch_data[9]
            # a_index_list = batch_data[10]



            # self.batch_store[0].append(self.ship_feature_list)
            # self.batch_store[1].append(self.missile_node_feature_list)
            # self.batch_store[2].append(self.heterogeneous_edges_list)
            # self.batch_store[3].append(self.action_feature_list)
            # self.batch_store[4].append(self.action_blue_list)
            # self.batch_store[5].append(self.reward_list)
            # self.batch_store[6].append(self.prob_list)
            # self.batch_store[7].append(self.done_list)
            # self.batch_store[8].append(self.avail_action_blue_list)
            # self.batch_store[9].append(self.a_index_list) ##
            #
            #
            # self.ship_feature_list = list()
            # self.missile_node_feature_list = list()
            # self.heterogeneous_edges_list = list()
            # self.action_feature_list = list()
            # self.action_blue_list = list()
            # self.reward_list = list()
            # self.prob_list = list()
            # self.done_list = list()
            # self.avail_action_blue_list = list()
            # self.a_index_list = list()

        # else:
        #     self.ship_feature_list.append(transition[0])
        #     self.missile_node_feature_list.append(transition[1])
        #     self.heterogeneous_edges_list.append(transition[2])
        #     self.action_feature_list.append(transition[3])
        #     self.action_blue_list.append(transition[4])
        #     self.reward_list.append(transition[5])
        #     self.prob_list.append(transition[6])
        #     self.done_list.append(transition[7])
        #     self.avail_action_blue_list.append(transition[8])
        #     self.a_index_list.append(transition[9])

    def make_batch2(self, batch_data):
        ship_feature_list = batch_data[0]
        missile_node_feature_list = batch_data[1]
        heterogeneous_edges_list = batch_data[2]
        action_feature_list = batch_data[3]
        action_blue_list = batch_data[4]
        reward_list = batch_data[5]
        prob_list= batch_data[6]
        done_list = batch_data[7]
        avail_action_blue_list= batch_data[8]
        a_index_list= batch_data[9]

        ship_feature = ship_feature_list
        missile_node_feature = missile_node_feature_list
        heterogeneous_edges = heterogeneous_edges_list
        action_feature = action_feature_list
        action_blue = action_blue_list
        reward = reward_list
        prob = prob_list
        done = done_list
        avail_action_blue = avail_action_blue_list
        a_index = a_index_list

        ship_feature = torch.tensor(ship_feature).to(device).float()
        ship_feature_next = torch.cat([ship_feature.clone().detach().squeeze(1), self.dummy_ship_feature], dim=0)[
                            1:].unsqueeze(1)
        action_blue = torch.tensor(action_blue).to(device).float()
        reward = torch.tensor(reward).to(device).float()
        prob = torch.tensor(prob).to(device).float()
        done = torch.tensor(done).to(device).float()
        avail_action_blue = torch.tensor(avail_action_blue).to(device).float()
        a_index = torch.tensor(a_index).to(device).long()
        self.ship_feature_list = list()
        self.ship_feature_next_list = list()
        self.missile_node_feature_list = list()
        self.heterogeneous_edges_list = list()
        self.action_feature_list = list()
        self.action_blue_list = list()
        self.reward_list = list()
        self.prob_list = list()
        self.mask_list = list()
        self.done_list = list()
        self.avail_action_blue_list = list()
        self.a_index_list = list()
        return ship_feature, ship_feature_next, missile_node_feature, heterogeneous_edges, action_feature, action_blue, reward, prob,  done, avail_action_blue, a_index



    def make_batch(self):


        ship_feature=self.ship_feature_list
        missile_node_feature=self.missile_node_feature_list
        heterogeneous_edges = self.heterogeneous_edges_list
        action_feature = self.action_feature_list
        action_blue = self.action_blue_list
        reward = self.reward_list
        prob = self.prob_list
        mask = self.mask_list
        done = self.done_list
        avail_action_blue = self.avail_action_blue_list
        a_index = self.a_index_list
        ship_feature = torch.tensor(ship_feature).to(device).float()
        ship_feature_next = torch.cat([ship_feature.clone().detach().squeeze(1), self.dummy_ship_feature], dim = 0)[1:].unsqueeze(1)
        action_blue = torch.tensor(action_blue).to(device).float()
        reward = torch.tensor(reward).to(device).float()
        prob = torch.tensor(prob).to(device).float()
        done = torch.tensor(done).to(device).float()
        avail_action_blue = torch.tensor(avail_action_blue).to(device).float()
        a_index = torch.tensor(a_index).to(device).long()
        self.ship_feature_list= list()
        self.ship_feature_next_list = list()
        self.missile_node_feature_list= list()
        self.heterogeneous_edges_list= list()
        self.action_feature_list= list()
        self.action_blue_list= list()
        self.reward_list= list()
        self.prob_list= list()
        self.mask_list= list()
        self.done_list= list()
        self.avail_action_blue_list= list()
        self.a_index_list= list()
        return ship_feature, ship_feature_next, missile_node_feature, heterogeneous_edges, action_feature, action_blue, reward, prob, mask, done, avail_action_blue, a_index

    @torch.no_grad()
    def sample_action(self, ship_features,node_features_missile, heterogenous_edges, possible_actions, action_feature):
        obs, act_graph = self.get_node_representation(ship_features,
                                                    node_features_missile,
                                                      heterogenous_edges,
                                                      mini_batch=False)

        act_graph = act_graph[0:self.action_size, :]
        obs = obs.expand([act_graph.shape[0], obs.shape[1]])
        obs_n_action = torch.cat([obs, act_graph], dim = 1)



        logit = [self.network.pi(obs_n_action[i].unsqueeze(0)) for i in range(obs_n_action.shape[0])]
        logit = torch.stack(logit).view(1, -1)
        action_size = obs_n_action.shape[0]
        if action_size >= self.action_size:
             action_size = self.action_size
        remain_action = torch.tensor([-1e8 for _ in range(self.action_size - action_size)], device=device).unsqueeze(0)
        logit = torch.cat([logit, remain_action], dim=1)
        #print(logit.shape, remain_action.shape)
        mask = torch.tensor(possible_actions, device=device).bool()
        #print(logit.shape, mask.shape)
        logit = logit.masked_fill(mask == 0, -1e8)
        prob = torch.softmax(logit, dim=-1)
        m = Categorical(prob)
        a = m.sample().item()


        a_index = a
        prob_a = prob.squeeze(0)[a]
        action_blue = action_feature[a]

        return action_blue, prob_a, mask, a_index

    def learn(self, cum_loss = 0):

        if cfg.algorithm == 'actor_critic':
            for l in range(len(self.batch_store)):
                batch_data = self.batch_store[l]
                ship_feature, \
                ship_feature_next, \
                missile_node_feature, \
                heterogeneous_edges, \
                action_feature, \
                action_blue, \
                reward, \
                prob, \
                done, \
                avail_action_blue, \
                a_index = self.make_batch2(batch_data)
                avg_loss = 0.0
                a_indices = a_index.unsqueeze(1)
                self.eval_check(eval=False)
                obs, act_graph = self.get_node_representation(ship_feature,missile_node_feature,heterogeneous_edges,mini_batch=True)
                obs_next = self.get_ship_representation(ship_feature_next)
                v_s = self.network.v(obs)
                td_target = reward.unsqueeze(1) + self.gamma * self.network.v(obs_next) * (1-done).unsqueeze(1)
                delta = td_target - v_s
                delta = delta.cpu().detach().numpy()
                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[ : :-1]:
                    advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
                mask = avail_action_blue
                obs_expand = obs.unsqueeze(1).expand([obs.shape[0], act_graph.shape[1], obs.shape[1]])
                obs_n_act = torch.cat([obs_expand, act_graph], dim= 2)
                logit = torch.stack([self.network.pi(obs_n_act[:, i]) for i in range(self.action_size)])
                logit = torch.einsum('ijk->jik', logit).squeeze(2)
                mask = mask.squeeze(1)
                logit = logit.masked_fill(mask == 0, -1e8)
                pi = torch.softmax(logit, dim=-1)
                pi_a = pi.gather(1, a_indices)
                log_prob = torch.log(pi_a)  # a/b == exp(log(a)-log(b))
                loss = log_prob * advantage.detach()
                if cfg.entropy == True:
                    entropy = -torch.sum(torch.exp(pi) * pi, dim=1)
                    loss = - loss.mean() + 0.5 * F.smooth_l1_loss(v_s, td_target.detach()) -0.01*entropy.mean()# 수정 + 엔트로피
                else:
                    loss = - loss.mean() + 0.5 * F.smooth_l1_loss(v_s, td_target.detach())
                if l == 0:
                    cum_loss = loss/cfg.n_data_parallelism
                else:
                    cum_loss = cum_loss+loss/cfg.n_data_parallelism

            cum_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_params, cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.optimizer.param_groups[0]['lr'] >= cfg.lr_min:
                self.scheduler.step()

        if cfg.algorithm == 'kl_penalty':
            ship_feature, \
            ship_feature_next, \
            missile_node_feature, \
            heterogeneous_edges, \
            action_feature, \
            action_blue, \
            reward, \
            prob, \
            _, \
            done, \
            avail_action_blue, \
            a_index = self.make_batch()
            avg_loss = 0.0
            a_indices = a_index.unsqueeze(1)
            self.eval_check(eval=False)

            self.adaptive_beta = 1.7
            self.kl_target =cfg.kl_target
            for i in range(self.K_epoch):
                obs, act_graph = self.get_node_representation(ship_feature,missile_node_feature,heterogeneous_edges,mini_batch=True)
                obs_next = self.get_ship_representation(ship_feature_next)
                v_s = self.network.v(obs)
                td_target = reward.unsqueeze(1) + self.gamma * self.network.v(obs_next) * (1-done).unsqueeze(1)
                delta = td_target - v_s
                delta = delta.cpu().detach().numpy()
                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[ : :-1]:
                    advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
                mask = avail_action_blue
                obs_expand = obs.unsqueeze(1).expand([obs.shape[0], act_graph.shape[1], obs.shape[1]])
                obs_n_act = torch.cat([obs_expand, act_graph], dim= 2)
                logit = torch.stack([self.network.pi(obs_n_act[:, i]) for i in range(self.action_size)])
                logit = torch.einsum('ijk->jik', logit).squeeze(2)
                mask = mask.squeeze(1)
                logit = logit.masked_fill(mask == 0, -1e8)
                pi = torch.softmax(logit, dim=-1)
                pi_a = pi.gather(1, a_indices)
                ratio = torch.exp(torch.log(pi_a.squeeze(1)) - torch.log(prob).detach())  # a/b == exp(log(a)-log(b))
                kl_penalty = torch.log(prob).detach()*(torch.log(prob).detach() - torch.log(pi_a.squeeze(1)))
                surr1 = ratio * (advantage.detach().squeeze())
                surr = surr1 - self.adaptive_beta * kl_penalty
                if cfg.entropy == True:
                    entropy = -torch.sum(torch.exp(pi) * pi, dim=1)
                    loss = - surr.mean() + 0.5 * F.smooth_l1_loss(v_s, td_target.detach()) -0.01*entropy.mean()# 수정 + 엔트로피
                else:
                    loss = - surr.mean() + 0.5 * F.smooth_l1_loss(v_s, td_target.detach())
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.eval_params, cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                if kl_penalty.mean() < self.kl_target/1.5:
                    self.adaptive_beta = self.adaptive_beta/0.5
                if kl_penalty.mean() > self.kl_target*1.5:
                    self.adaptive_beta = self.adaptive_beta*0.5

        if cfg.algorithm == 'ppo':
            if self.eps_clip >= cfg.eps_clip_min:
                self.eps_clip -=cfg.eps_clip_step

            for i in range(self.K_epoch):
                for l in range(len(self.batch_store)):
                    batch_data = self.batch_store[l]
                    ship_feature, \
                    ship_feature_next, \
                    missile_node_feature, \
                    heterogeneous_edges, \
                    action_feature, \
                    action_blue, \
                    reward, \
                    prob, \
                    done, \
                    avail_action_blue, \
                    a_index = self.make_batch2(batch_data)
                    avg_loss = 0.0
                    a_indices = a_index.unsqueeze(1)
                    self.eval_check(eval=False)
                    obs, act_graph = self.get_node_representation(ship_feature,missile_node_feature,heterogeneous_edges,mini_batch=True)
                    obs_next = self.get_ship_representation(ship_feature_next)
                    v_s = self.network.v(obs)
                    td_target = reward.unsqueeze(1) + self.gamma * self.network.v(obs_next) * (1-done).unsqueeze(1)
                    delta = td_target - v_s
                    delta = delta.cpu().detach().numpy()
                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in delta[ : :-1]:
                        advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                        advantage_lst.append([advantage])
                    advantage_lst.reverse()
                    advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
                    mask = avail_action_blue
                    obs_expand = obs.unsqueeze(1).expand([obs.shape[0], act_graph.shape[1], obs.shape[1]])
                    obs_n_act = torch.cat([obs_expand, act_graph], dim= 2)
                    logit = torch.stack([self.network.pi(obs_n_act[:, i]) for i in range(self.action_size)])
                    logit = torch.einsum('ijk->jik', logit).squeeze(2)
                    mask = mask.squeeze(1)
                    logit = logit.masked_fill(mask == 0, -1e8)
                    pi = torch.softmax(logit, dim=-1)
                    pi_a = pi.gather(1, a_indices)
                    ratio = torch.exp(torch.log(pi_a.squeeze(1)) - torch.log(prob).detach())  # a/b == exp(log(a)-log(b))
                    surr1 = ratio * (advantage.detach().squeeze())
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * (advantage.detach().squeeze())

                    if cfg.entropy == True:
                        entropy = -torch.sum(torch.exp(pi) * pi, dim=1)
                        loss = - torch.min(surr1, surr2).mean() + 0.5 * F.smooth_l1_loss(v_s, td_target.detach()) -0.01*entropy.mean()# 수정 + 엔트로피
                    else:
                        loss = - torch.min(surr1, surr2).mean() + 0.5 * F.smooth_l1_loss(v_s, td_target.detach())


                    if l == 0:
                        cum_loss = loss/cfg.n_data_parallelism
                    else:
                        cum_loss = cum_loss+loss/cfg.n_data_parallelism


                cum_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.eval_params, cfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.optimizer.param_groups[0]['lr'] >= cfg.lr_min:
                    self.scheduler.step()

        self.batch_store = list()
        return avg_loss / self.K_epoch
    def load_network(self, file_dir):
        print(file_dir)
        checkpoint = torch.load(file_dir)
        self.network.load_state_dict(checkpoint["network"])
        self.node_representation_ship_feature.load_state_dict(checkpoint["node_representation_ship_feature"])
        self.func_meta_path.load_state_dict(checkpoint["func_meta_path"])
        self.func_meta_path2.load_state_dict(checkpoint["func_meta_path2"])
        self.func_meta_path3.load_state_dict(checkpoint["func_meta_path3"])
        self.func_meta_path4.load_state_dict(checkpoint["func_meta_path4"])
        self.func_meta_path5.load_state_dict(checkpoint["func_meta_path5"])
        self.func_meta_path6.load_state_dict(checkpoint["func_meta_path6"])
        try:
            self.node_representation_wo_graph.load_state_dict(checkpoint["node_representation_wo_graph"])
        except KeyError:pass

    def eval_check(self, eval):
        if eval == True:
            self.network.eval()
            self.node_representation_ship_feature.eval()
            self.func_meta_path.eval()
            self.func_meta_path2.eval()
            self.func_meta_path3.eval()
            self.func_meta_path4.eval()
            self.func_meta_path5.eval()
            self.func_meta_path6.eval()
            self.node_representation_wo_graph.eval()

        else:
            self.network.train()
            self.node_representation_ship_feature.train()
            self.func_meta_path.train()
            self.func_meta_path2.train()
            self.func_meta_path3.train()
            self.func_meta_path4.train()
            self.func_meta_path5.train()
            self.func_meta_path6.train()
            self.node_representation_wo_graph.train()
    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "network": self.network.state_dict(),
                    "node_representation_ship_feature": self.node_representation_ship_feature.state_dict(),
                    "func_meta_path": self.func_meta_path.state_dict(),
                    "func_meta_path2": self.func_meta_path2.state_dict(),
                    "func_meta_path3": self.func_meta_path3.state_dict(),
                    "func_meta_path4": self.func_meta_path4.state_dict(),
                    "func_meta_path5": self.func_meta_path5.state_dict(),
                    "func_meta_path6": self.func_meta_path6.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "node_representation_wo_graph": self.node_representation_wo_graph.state_dict()
                    },
                   file_dir + "episode%d.pt" % e)


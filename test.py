import numpy as np
import torch, random
from cfg import get_cfg
from draw_graph import visualize_heterogeneous_graph
cfg = get_cfg()
seed = 112
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


from Components.Modeler_Component_test import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
from GDN import Agent

fix_l = 0
fix_u = 17
from scipy.stats import randint


def preprocessing(scenarios):
    scenario = scenarios
    if mode == 'txt':
        input_path = ["Data/Test/dataset{}/ship.txt".format(scenario),
                      "Data/Test/dataset{}/patrol_aircraft.txt".format(scenario),
                      "Data/Test/dataset{}/SAM.txt".format(scenario),
                      "Data/Test/dataset{}/SSM.txt".format(scenario),
                      "Data/Test/dataset{}/inception.txt".format(scenario)]
    else:
        input_path = "Data\input_data.xlsx"

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data



def evaluation(agent, env, with_noise=False):
    temp = random.uniform(fix_l, fix_u)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    eval = False
    over = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    overtime = None

    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                edge_index_ssm_to_ship = env.get_ssm_to_ship_edge_index(k = i)
                edge_index_ssm_to_ssm = env.get_ssm_to_ssm_edge_index(k = i)
                edge_index_sam_to_ssm = env.get_sam_to_ssm_edge_index(k = i)
                edge_index_ship_to_sam = env.get_ship_to_sam_edge_index(k = i)
                edge_index_ship_to_enemy = env.get_ship_to_enemy_edge_index(k = i)
                heterogeneous_edges = (edge_index_ssm_to_ship, edge_index_ssm_to_ssm, edge_index_sam_to_ssm, edge_index_ship_to_sam,edge_index_ship_to_enemy)
                ship_feature = env.get_ship_feature(k = i)
                missile_node_feature, node_cats = env.get_missile_node_feature(k = i)
                action_feature = env.get_action_feature(k = i)
                agent.eval_check(eval=True)

                node_representation, node_representation_graph = agent.get_node_representation(missile_node_feature,
                                                                                               ship_feature,
                                                                                               heterogeneous_edges,
                                                                                               n_node_feature_missile,
                                                                                               node_cats=node_cats,
                                                                                               mini_batch=False)  # 차원 : n_agents X n_representation_comm
                action_blue, u = agent.sample_action(node_representation, node_representation_graph, avail_action_blue[i],
                                                     epsilon=0, action_feature=action_feature, training=False,
                                                     with_noise=with_noise)
                actions_blue.append(action_blue)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)

            reward, win_tag, done, leakers = env.step(actions_blue, action_yellow)
            episode_reward += reward



        else:
            pass_transition = True

            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                actions_blue.append([0, 0, 0, 0, 0, 0, 0, 0])


            env.step(action_blue=actions_blue, action_yellow=enemy_action_for_transition,
                     pass_transition=pass_transition)

    return episode_reward, win_tag, leakers, overtime


if __name__ == "__main__":

    vessl_on = cfg.vessl
    if vessl_on == True:
        import vessl

        vessl.init()
        output_dir = "/output/"
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        print("시작")
        from torch.utils.tensorboard import SummaryWriter

        output_dir = "../output_susceptibility/"
        writer = SummaryWriter('./logs2')
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    import time

    """

    환경 시스템 관련 변수들

    """
    visualize = False  # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]  # 화면 size / 600, 600 pixel
    tick = 24  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False  # 고도에 의한
    num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
    mode = 'txt'  # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'  # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10,
                   20]  # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 1
    polar_chart_visualize = False
    scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    print(cfg)
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    # scenario = np.random.choice(scenarios)
    episode_polar_chart = polar_chart[0]
    records = list()

    data = preprocessing(1)
    t = 0
    env = modeler(data,
                  visualize=visualize,
                  size=size,
                  detection_by_height=detection_by_height,
                  tick=tick,
                  simtime_per_framerate=simtime_per_frame,
                  ciws_threshold=ciws_threshold,
                  action_history_step=cfg.action_history_step)

    n_node_feature_missile = env.friendlies_fixed_list[0].air_tracking_limit + env.friendlies_fixed_list[
        0].air_engagement_limit + env.friendlies_fixed_list[0].num_m_sam + 1
    n_node_feature_enemy = env.friendlies_fixed_list[0].surface_tracking_limit + 1
    agent = Agent(num_agent=1,
                  feature_size_ship=env.get_env_info()["ship_feature_shape"],
                  feature_size_enemy=env.get_env_info()["enemy_feature_shape"],
                  feature_size_missile=env.get_env_info()["missile_feature_shape"],
                  feature_size_action=env.get_env_info()["action_feature_shape"],

                  iqn_layers=list(eval(cfg.iqn_layers)),

                  node_embedding_layers_ship=list(eval(cfg.ship_layers)),
                  node_embedding_layers_missile=list(eval(cfg.missile_layers)),
                  node_embedding_layers_enemy=list(eval(cfg.enemy_layers)),
                  node_embedding_layers_action=list(eval(cfg.action_layers)),
                  hidden_size_comm=cfg.hidden_size_comm,
                  hidden_size_enemy=cfg.hidden_size_enemy,  #### 수정요망

                  n_multi_head=cfg.n_multi_head,
                  n_representation_ship=cfg.n_representation_ship,
                  n_representation_missile=cfg.n_representation_missile,
                  n_representation_enemy=cfg.n_representation_enemy,
                  n_representation_action=cfg.n_representation_action,

                  dropout=0.6,
                  action_size=env.get_env_info()["n_actions"],
                  buffer_size=cfg.buffer_size,
                  batch_size=cfg.batch_size,
                  learning_rate=cfg.lr,  # 0.0001,
                  gamma=cfg.gamma,
                  GNN='GAT',
                  teleport_probability=cfg.teleport_probability,
                  gtn_beta=0.1,

                  n_node_feature_missile=env.friendlies_fixed_list[0].air_tracking_limit +
                                         env.friendlies_fixed_list[0].air_engagement_limit +
                                         env.friendlies_fixed_list[0].num_m_sam +
                                         1,

                  n_node_feature_enemy=env.friendlies_fixed_list[0].surface_tracking_limit + 1,
                  n_step=n_step,
                  beta=cfg.per_beta,
                  per_alpha=cfg.per_alpha,
                  iqn_layer_size=cfg.iqn_layer_size,
                  iqn_N=cfg.iqn_N,
                  n_cos=cfg.n_cos,
                  num_nodes=n_node_feature_missile

                  )

    agent.load_model("2200.pt")
    anneal_episode = cfg.anneal_episode
    anneal_step = (cfg.per_beta - 1) / anneal_episode
    print("epsilon_greedy", cfg.epsilon_greedy)
    epsilon = cfg.epsilon
    min_epsilon = cfg.min_epsilon
    eval_lose_ratio = list()
    eval_win_ratio = list()
    lose_ratio = list()
    win_ratio = list()
    reward_list = list()

    eval_lose_ratio1 = list()
    eval_win_ratio1 = list()
    anneal_epsilon = (epsilon - min_epsilon) / cfg.anneal_step
    print("noise", cfg.with_noise)
    non_lose_rate = list()

    n = 500
    non_lose_rate = list()
    for j in range(n):
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step
                      )
        episode_reward, win_tag, leakers, overtime = evaluation(agent, env, with_noise=cfg.with_noise)

        if win_tag == 'draw' or win_tag == 'win':
            non_lose_rate.append(1)
        print('전', win_tag, episode_reward, env.now, overtime, np.sum(non_lose_rate)/(j+1))

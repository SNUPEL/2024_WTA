from environment.Modeler_Component import *
from environment.Adapter_Component import *
from environment.Policy import *
from cfg import get_cfg
from agent.GPO import Agent
import numpy as np
fix_l = 0
fix_u = 17

def preprocessing():

    scenario = 'train'
    if mode == 'txt':
        if vessl_on == True:
            input_path = ["/root/2024_WTA/data/{}/ship.txt".format(scenario),
                          "/root/2024_WTA/data/{}/patrol_aircraft.txt".format(scenario),
                          "/root/2024_WTA/data/{}/SAM.txt".format(scenario),
                          "/root/2024_WTA/data/{}/SSM.txt".format(scenario),
                          "/root/2024_WTA/data/{}/inception.txt".format(scenario)]

        else:
            input_path = ["data/{}/ship.txt".format(scenario),
                          "data/{}/patrol_aircraft.txt".format(scenario),
                          "data/{}/SAM.txt".format(scenario),
                          "data/{}/SSM.txt".format(scenario),
                          "data/{}/inception.txt".format(scenario)]
    else:
        input_path = "data\input_data.xlsx"

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data





def train(agent, env, t):
    temp = random.uniform(fix_l, fix_u)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    while not done:
        if env.now % (decision_timestep) <= 0.00001:

            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            edge_index_ssm_to_ship = env.get_ssm_to_ship_edge_index()
            edge_index_ssm_to_ssm = env.get_ssm_to_ssm_edge_index()
            edge_index_sam_to_ssm = env.get_sam_to_ssm_edge_index()
            edge_index_ship_to_sam = env.get_ship_to_sam_edge_index()
            edge_index_ship_to_enemy = env.get_ship_to_enemy_edge_index()
            heterogeneous_edges = (edge_index_ssm_to_ship, edge_index_ssm_to_ssm, edge_index_sam_to_ssm, edge_index_ship_to_sam,edge_index_ship_to_enemy)
            ship_feature = env.get_ship_feature()
            missile_node_feature, node_cats = env.get_missile_node_feature()
            action_feature = env.get_action_feature()

            agent.eval_check(eval=True)
            action_blue, prob, mask, a_index= agent.sample_action(ship_feature, missile_node_feature,heterogeneous_edges,avail_action_blue,action_feature)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)


            reward, win_tag, done, leakers = env.step(action_blue, action_yellow)
            t += 1
            episode_reward += reward

            transition = (ship_feature,
                          missile_node_feature, \
                          heterogeneous_edges,
                          action_feature, \
                          action_blue, \
                          reward, \
                          prob,\
                          done, \
                          avail_action_blue,
                          a_index)
            agent.put_data(transition)

        else:
            pass_transition = True
            env.step(action_blue=[0, 0, 0, 0, 0, 0, 0, 0], action_yellow=enemy_action_for_transition,pass_transition=pass_transition)

    agent.eval_check(eval=False)
    if len(agent.batch_store) % cfg.n_data_parallelism==0:
        agent.learn()

    return episode_reward, win_tag, t




def evaluation(agent, env):
    temp = random.uniform(fix_l, fix_u)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            edge_index_ssm_to_ship = env.get_ssm_to_ship_edge_index()
            edge_index_ssm_to_ssm = env.get_ssm_to_ssm_edge_index()
            edge_index_sam_to_ssm = env.get_sam_to_ssm_edge_index()
            edge_index_ship_to_sam = env.get_ship_to_sam_edge_index()
            edge_index_ship_to_enemy = env.get_ship_to_enemy_edge_index()
            heterogeneous_edges = (edge_index_ssm_to_ship, edge_index_ssm_to_ssm, edge_index_sam_to_ssm, edge_index_ship_to_sam, edge_index_ship_to_enemy)
            ship_feature = env.get_ship_feature()
            missile_node_feature, node_cats = env.get_missile_node_feature()
            action_feature = env.get_action_feature()
            agent.eval_check(eval=True)
            action_blue, prob, mask, a_index= agent.sample_action(ship_feature, missile_node_feature,heterogeneous_edges,avail_action_blue,action_feature)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leakers = env.step(action_blue, action_yellow)
            episode_reward += reward
        else:
            pass_transition = True
            env.step(action_blue=[0, 0, 0, 0, 0, 0, 0, 0], action_yellow=enemy_action_for_transition,pass_transition=pass_transition)

    return episode_reward, win_tag




if __name__ == "__main__":
    cfg = get_cfg()
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

        output_dir = "output_susceptibility/"
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
    tick = 500  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False  # 고도에 의한
    num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
    mode = 'txt'  # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'  # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10, 20]  # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 1
    polar_chart_visualize = False
    lose_ratio = list()
    remains_ratio = list()

    print(cfg)
    df_dict = {}
    episode_polar_chart = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]
    records = list()
    import torch, random

    seed = cfg.seed  # 원래 SEED 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    data = preprocessing()
    t = 0
    env = modeler(data,
                  visualize=visualize,
                  size=size,
                  detection_by_height=detection_by_height,
                  tick=tick,
                  simtime_per_framerate=simtime_per_frame,
                  ciws_threshold=ciws_threshold,
                  action_history_step=cfg.action_history_step)

    agent = Agent(action_size=env.get_env_info()["n_actions"],
                  feature_size_ship=env.get_env_info()["ship_feature_shape"],
                  feature_size_missile=env.get_env_info()["missile_feature_shape"],
                  n_node_feature_missile=env.friendlies_fixed_list[0].air_tracking_limit +
                                         env.friendlies_fixed_list[0].air_engagement_limit +
                                         env.friendlies_fixed_list[0].num_m_sam +
                                         1,
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
                 )

    reward_list = list()
    non_lose_ratio_list = list()
    for e in range(num_iteration):
        if e % 10 == 0 and e>0:
            n_eval = 20
            non_lose_ratio = 0
            for _ in range(n_eval):
                env = modeler(data,
                              visualize=visualize,
                              size=size,
                              detection_by_height=detection_by_height,
                              tick=tick,
                              simtime_per_framerate=simtime_per_frame,
                              ciws_threshold=ciws_threshold,
                              action_history_step=cfg.action_history_step)
                episode_reward, win_tag = evaluation(agent, env)
                if win_tag != 'lose':
                    non_lose_ratio += 1/n_eval
                print(episode_reward, win_tag)
            if vessl_on == True:
                vessl.log(step=e, payload={'non_lose_ratio': non_lose_ratio})
            non_lose_ratio_list.append(non_lose_ratio)
            df = pd.DataFrame(non_lose_ratio_list)
            df_reward = pd.DataFrame(reward_list)
            df.to_csv(output_dir+"non_lose_ratio.csv")
            df_reward.to_csv(output_dir+"episode_reward.csv")

        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step)
        episode_reward, win_tag, t = train(agent, env, t)
        if e % cfg.n_data_parallelism == 0:
            agent.save_network(e, output_dir)
        reward_list.append(episode_reward)
        print( "Total reward in episode {} = {}, time_step : {}, win_tag : {}, terminal_time : {}".format(e,np.round(episode_reward, 3), t, win_tag, env.now))
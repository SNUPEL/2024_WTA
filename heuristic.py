from environment.Modeler_Component import *
from environment.Adapter_Component import *
from environment.Policy import *



from collections import deque
from cfg import get_cfg
import numpy as np
import torch, random
import scipy
def preprocessing(scenarios):
    scenario = scenarios[0]
    if mode == 'txt':
        if vessl_on == True:
            input_path = ["/root/AI_SH/Data/{}/ship.txt".format(scenario),
                          "/root/AI_SH/Data/{}/patrol_aircraft.txt".format(scenario),
                          "/root/AI_SH/Data/{}/SAM.txt".format(scenario),
                          "/root/AI_SH/Data/{}/SSM.txt".format(scenario),
                          "/root/AI_SH/Data/{}/inception.txt".format(scenario)]
        else:
            input_path = ["Data/{}/ship.txt".format(scenario),
                          "Data/{}/patrol_aircraft.txt".format(scenario),
                          "Data/{}/SAM.txt".format(scenario),
                          "Data/{}/SSM.txt".format(scenario),
                          "Data/{}/inception.txt".format(scenario)]
    else:
        input_path = "Data\input_data.xlsx"

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data


def train(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_step, initializer, output_dir, vdn, n_step,
          interval_min_blue, interval_constant_blue, temperature):
    interval_min_blue = interval_min_blue
    interval_constant_blue = interval_constant_blue
    temp = random.uniform(0, 50)
    agent_blue = Policy(env, rule='rule3', temperatures=[cfg.temperature, cfg.temperature])

    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    step = 0
    losses = []
    epi_r = list()
    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    step_checker = 0
    if random.uniform(0, 1) > 0.5:
        interval_min = True
    else:
        interval_min = False
    interval_constant = random.uniform(2,4)
    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leaker = env.step(action_blue, action_yellow, rl = False)
            episode_reward += reward
            status = None
            step_checker += 1
            if e >= train_start:
                t += 1
        else:
            pass_transition = True
            env.step(action_blue=friendly_action_for_transition,
                     action_yellow=enemy_action_for_transition, pass_transition=pass_transition, rl = False)

        if done == True:
            break
    return episode_reward, epsilon, t, eval, win_tag


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
        from torch.utils.tensorboard import SummaryWriter
        output_dir = "../output_susceptibility_heuristic/"
        writer = SummaryWriter('./logs2')
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    import time
    """
    환경 시스템 관련 변수들
    """
    visualize = False# 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]  # 화면 size / 600, 600 pixel
    tick = 500  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False      # 고도에 의한
    num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
    mode = 'txt'                     # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'                   # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10, 20]           # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 0.5
    polar_chart_visualize = False
    scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    episode_polar_chart = polar_chart[0]
    records = list()

    #seed = 42

    data = preprocessing(scenarios)
    t = 0
    env = modeler(data,
                  visualize=visualize,
                  size=size,
                  detection_by_height=detection_by_height,
                  tick=tick,
                  simtime_per_framerate=simtime_per_frame,
                  ciws_threshold=ciws_threshold,
                  action_history_step=cfg.action_history_step)
    anneal_episode = cfg.anneal_episode
    anneal_step = (cfg.per_beta - 1) / anneal_episode
    epsilon = 1
    min_epsilon = 0.01
    reward_list = list()
    agent = None
    non_lose = 0
    interval_min_blue_list = list()
    interval_constant_blue_list = list()
    temperature = list()
    for e in range(num_iteration):
        interval_min_blue = cfg.interval_min_blue,
        interval_constant_blue = cfg.interval_constant_blue,
        temperature = temperature
        seed = 2 * e
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        start = time.time()
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step,
                      interval_constant_blue = [cfg.interval_constant_blue,cfg.interval_constant_blue])
        episode_reward, epsilon, t, eval, win_tag = train(agent, env, e, t, train_start=cfg.train_start, epsilon=epsilon,
                                                 min_epsilon=min_epsilon, anneal_step=anneal_step, initializer=False,
                                                 output_dir=None, vdn=True, n_step=n_step,
                                                          interval_min_blue=interval_min_blue,
                                                          interval_constant_blue=interval_constant_blue,
                                                          temperature=temperature
                                                          )
        if vessl_on == False:
            writer.add_scalar("episode", episode_reward, e)
        reward_list.append(episode_reward)
        if vessl_on == True:
            vessl.log(step=e, payload={'reward': episode_reward})
        if e % 10 == 0:
            import os
            import pandas as pd
            df = pd.DataFrame(reward_list)
            df.to_csv(output_dir + 'episode_reward_{}_{}_{}.csv'.format(cfg.temperature, cfg.interval_min_blue, cfg.interval_constant_blue))

        if win_tag != 'lose':
            non_lose += 1

        print(
            "Total reward in episode {} = {}, epsilon : {}, time_step : {}, episode_duration : {}, mean reward : {}, win_tag : {}, non_lose : {}".format(
                e,
                np.round(episode_reward, 3),
                np.round(epsilon, 3),
                t,
                np.round(time.time() - start, 3),
                np.mean(reward_list),
                win_tag,
                non_lose/(e+1)))

        # del data
        # del env
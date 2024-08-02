from environment.Modeler_Component import *
from environment.Adapter_Component import *
from environment.Policy import *
from cfg import get_cfg

import numpy as np

fix_l = 0
fix_u = 17
fit_records = []

def simulation():
    score = 0

    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    n = cfg.n_test
    print("=========평가시작===========")

    for e in range(n):
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step,
                      air_alert_distance_blue=  1,
                      interval_constant_blue = [1.2, 1.2]
                      )
        epi_reward, eval, win_tag= evaluation(env, temperature1=1,temperature2 =1, warning_distance = 200)
        if win_tag != 'lose':
            score += 1 / n

        else:
            score += 0

    return score

def fitness_func(ga_instance, solution, solution_idx):
    score = simulation(solution)
    return score

def preprocessing(scenarios):
    scenario = scenarios
    if mode == 'txt':
        input_path = ["data/test/dataset{}/ship.txt".format(scenario),
                      "data/test/dataset{}/patrol_aircraft.txt".format(scenario),
                      "data/test/dataset{}/SAM.txt".format(scenario),
                      "data/test/dataset{}/SSM.txt".format(scenario),
                      "data/test/dataset{}/inception.txt".format(scenario)]
    else:
        input_path = "data/test/dataset{}/input_data.xlsx".format(scenario)

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data



def evaluation(env,
               temperature1,
               temperature2,
               warning_distance
               ):
    temp = random.uniform(fix_l, fix_u)
    agent_blue = Policy(env, rule='rule8', temperatures=[temperature1, temperature2])
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0

    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    step_checker = 0


    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue', speed_normalizing_blue=False)
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')

            action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue, open_fire_distance = warning_distance)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leaker = env.step(action_blue, action_yellow, rl = False)
            episode_reward += reward
            status = None
            step_checker += 1
        else:
            pass_transition = True
            env.step(action_blue=friendly_action_for_transition,
                     action_yellow=enemy_action_for_transition, pass_transition=pass_transition, rl = False)

        if done == True:
            break
    return episode_reward, eval, win_tag

def save_intermediate_fitness(ga_instance):
    # 현재 세대의 평균 적합도를 저장합니다.
    fitness = np.mean(ga_instance.last_generation_fitness)
    fitness_history.append(fitness)
    print(f"Generation {ga_instance.generations_completed} - Average Fitness: {fitness}")

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

    mode = 'excel'
    vessl_on = cfg.vessl
    polar_chart_visualize = False
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    best_solution_records = dict()
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    episode_polar_chart = polar_chart[0]
    datasets = [i for i in range(1,31)]
    non_lose_ratio_list = []
    raw_data = list()
    for dataset in datasets:
        fitness_history = []
        data = preprocessing(dataset)
        visualize = False  # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
        size = [600, 600]  # 화면 size / 600, 600 pixel
        tick = 500  # 가시화 기능 사용 시 빠르기
        simtime_per_frame = cfg.simtime_per_frame
        decision_timestep = cfg.decision_timestep
        detection_by_height = False  # 고도에 의한
        num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
        rule = 'rule2'          # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
        temperature = [10, 20]  # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
        ciws_threshold = 0.5
        lose_ratio = list()
        remains_ratio = list()
        df_dict = {}
        records = list()


        score = simulation()
        non_lose_ratio_list.append(score)
        df_result = pd.DataFrame(non_lose_ratio_list)
        df_result.to_csv("interval_wo_open_fire_distance_result_angle_{}.csv".format(cfg.inception_angle))





from environment.Modeler_Component_test import *
from environment.Adapter_Component import *
from environment.Policy import *
from cfg import get_cfg

import numpy as np
fix_l = 0
fix_u = 17
fit_records = []
best_solutions = dict()
def on_start(ga_instance):
    print("on_start()")

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    fit_records.append(ga_instance.best_solution()[1])

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")


def simulation(solution, ga = True, raw_data = None):
    temperature1 = solution[0]
    interval_constant_blue1 = solution[1]
    temperature2 = solution[2]
    interval_constant_blue2 = solution[3]
    air_alert_distance = solution[4]
    enemy_ship_normalizer_blue = solution[5]
    warning_distance = solution[6]
    score = 0
    if ga == False:
        seed = cfg.seed
        np.random.seed(seed)
        random.seed(seed)
        n = cfg.n_test
        print("=========평가시작===========")
    else:
        seed=cfg.seed+220202
        np.random.seed(seed)
        random.seed(seed)
        n = cfg.n_eval_GA

    for e in range(n):
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step,
                      air_alert_distance_blue=  air_alert_distance,
                      interval_constant_blue = [interval_constant_blue1, interval_constant_blue2]
                      )
        epi_reward, eval, win_tag= evaluation(env, temperature1=temperature1,temperature2 = temperature2,
                                              warning_distance = warning_distance,
                                              enemy_ship_normalizer_blue = enemy_ship_normalizer_blue)
        if win_tag != 'lose':
            score += 1/n
            if ga==False:
                raw_data.append([str(env.random_recording), 1])
        else:
            score += 0
            if ga==False:
                raw_data.append([str(env.random_recording), 0])

    print(score, solution)
    if ga == True:
        return score
    else:
        return score, raw_data


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
               enemy_ship_normalizer_blue,
                warning_distance
               ):
    temp = random.uniform(fix_l, fix_u)
    agent_blue = Policy(env, rule='rule3', temperatures=[temperature1, temperature2])
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0

    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    step_checker = 0


    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue', speed_normalizing_blue=True, enemy_ship_normalizer_blue=enemy_ship_normalizer_blue)
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')

            action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue, open_fire_distance= warning_distance)
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
    datasets = [i for i in range(1, 31)]
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

        solution_space = [[i/10 for i in range(-200, 200)], [i/10  for i in range(0, 500)],
                          [i/10  for i in range(-200, 200)], [i/10  for i in range(0, 500)], [i/10 for i in range(0,2000)],
                          [i /10 for i in range(1, 1000)], [i/10 for i in range(0,3000)]
                          ]



        num_genes = len(solution_space)

        initial_population = []
        sol_per_pop =20
        np.random.seed(cfg.seed)
        for _ in range(sol_per_pop):
            new_solution = [np.random.choice(space) for space in solution_space]
            initial_population.append(new_solution)

        num_generations = 10 # 세대 수
        num_parents_mating = 6  # 각 세대에서 선택할 부모 수
        init_range_low = 0
        init_range_high = 20
        parent_selection_type = "sss"
        keep_parents = 2
        crossover_type = "single_point"
        mutation_type = "random"
        mutation_percent_genes = 30


        import pygad
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                                parent_selection_type = parent_selection_type,
                                keep_parents = keep_parents,
                               initial_population=initial_population,
                               gene_space = solution_space,
                               crossover_type = crossover_type,
                               mutation_type = mutation_type,
                               mutation_percent_genes = mutation_percent_genes,
                               on_start=on_start,
                               on_fitness=on_fitness,
                               on_parents=on_parents,
                               on_crossover=on_crossover,
                               on_mutation=on_mutation,
                               on_generation=on_generation,
                               on_stop=on_stop,
                               random_seed = cfg.seed
                               )

        # 최적화 실행
        ga_instance.run()
        fitness = ga_instance.best_solution()[1]
        best_solutions = ga_instance.best_solution()[0]
        #fitness_history = ga_instance.plot_fitness()


        empty_dict = dict()
        for i in range(len(best_solutions)):
            empty_dict[i] = best_solutions[i]

        best_solution_records[dataset] = empty_dict
        df_fit = pd.DataFrame(fit_records)
        #df_fit.to_csv('fitness_records_dataset{}_rule7_param2.csv'.format(dataset))
        if vessl_on == True:
            df_fit.to_csv(output_dir + 'fitness_records_dataset{}_rule7_param2_remains.csv'.format(dataset))
            for s in range(len(fit_records)):
                f = fit_records[s]
                vessl.log(step=s, payload={'fitness_records_dataset_{}'.format(dataset): f})
        else:
            df_fit.to_csv('fitness_records_dataset{}_rule7_param2_angle_{}_remains.csv'.format(dataset, cfg.inception_angle))

        fit_records = []

        print("최적해:", best_solutions)
        print("최적해의 적합도:", fitness)
        score, raw_data = simulation(best_solutions, ga = False, raw_data = raw_data)
        non_lose_ratio_list.append(score)
        df_result = pd.DataFrame(non_lose_ratio_list)
        df_raw = pd.DataFrame(raw_data)
        if vessl_on == True:
            df_result.to_csv(output_dir + "GA_result_rule7_param2_angle_{}_remains.csv".format(cfg.inception_angle))
            df_raw.to_csv(output_dir+"raw_data_rule7_angle_{}_remains.csv".format(cfg.inception_angle))
            vessl.log(step=dataset, payload={'non_lose_ratio': score})
        else:
            df_result.to_csv("GA_result_rule7_param2_angle_{}_remains.csv".format(cfg.inception_angle))
            df_raw.to_csv("raw_data_rule7_angle_{}_remains.csv".format(cfg.inception_angle))






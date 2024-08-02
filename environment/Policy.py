import numpy as np
from utils import *


class Policy:

    def __init__(self, env, rule='rule2', temperatures=[10, 20]):
        self.env = env
        self.rule = rule
        self.temperature1 = temperatures[0]
        self.temperature2 = temperatures[1]
        self.indicator = 'max'

    def get_action(self, avail_action_list, target_distance_list, air_alert, open_fire_distance=10):

        d_max = 1000000
        if self.rule == 'rule6':  # interval open_fire
            len_target_distance_list = len(target_distance_list)
            temp = []
            for l in range(len_target_distance_list):
                target_distances = list()
                for d in target_distance_list[l]:
                    if d < open_fire_distance or target_distance_list[l].index(d) == 0:
                        d = d
                    else:
                        d = d_max
                    target_distances.append(d)
                temp.append(target_distances)
            target_distance_list = temp
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                if self.indicator == 'max':
                    if np.max(target_distance_list[idx]) <d_max:
                        a = np.argmax(target_distance_list[idx])
                    else:
                        a = 0
                    self.indicator = 'mean'
                elif self.indicator == 'mean':
                    if np.min(target_distance_list[idx]) < d_max:
                        average = np.mean(target_distance_list[idx])
                        a = np.argmin(np.abs(np.array(target_distance_list[idx]) - average))
                    else:
                        a = 0
                    self.indicator = 'min'
                elif self.indicator == 'min':
                    if np.min(target_distance_list[idx]) < d_max:
                        a = np.argmin(target_distance_list[idx])
                    else:
                        a = 0
                    self.indicator = 'max'
                actions.append(avail_actions_index[a])

        if self.rule == 'rule7':  # greedy
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                a = np.argmin(target_distance_list[idx])
                actions.append(avail_actions_index[a])

        if self.rule == 'rule8':
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                if self.indicator == 'max':
                    a = np.argmax(target_distance_list[idx])
                    self.indicator = 'mean'
                elif self.indicator == 'mean':
                    average = np.mean(target_distance_list[idx])
                    a = np.argmin(np.abs(np.array(target_distance_list[idx]) - average))
                    self.indicator = 'min'
                elif self.indicator == 'min':
                    a = np.argmin(target_distance_list[idx])
                    self.indicator = 'max'
                actions.append(avail_actions_index[a])

        if self.rule == 'rule5':  # greedy open fire
            len_target_distance_list = len(target_distance_list)
            temp = []

            for l in range(len_target_distance_list):
                target_distances = list()
                for d in target_distance_list[l]:
                    if d < open_fire_distance or target_distance_list[l].index(d) == 0:
                        d = d
                    else:
                        d = d_max
                    target_distances.append(d)
                temp.append(target_distances)

            target_distance_list = temp
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                if np.min(target_distance_list[idx]) < d_max:
                    a = np.argmin(target_distance_list[idx])
                else:
                    a = 0
                actions.append(avail_actions_index[a])

        if self.rule == 'rule4':
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                a = np.argmin(target_distance_list[idx])
                actions.append(avail_actions_index[a])

        if self.rule == 'rule0':
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                actions.append(np.random.choice(avail_actions_index))

        if self.rule == 'rule1':
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                actions.append(np.random.choice(avail_actions_index, p=softmax(target_distance_list[idx], temperature=0)))

        if self.rule == 'rule2':
            actions = list()
            # print(target_distance_list)
            if air_alert == True:

                # print(self.temperature1, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature1))
                # print(self.temperature2, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature2))

                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index,
                                                    p=softmax(target_distance_list[idx], temperature=self.temperature1,
                                                              debug=True)))
            else:
                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    # print("ㅇㅇㅇㅇ", avail_actions_index)
                    # print("ㄴㄴㄴㄴㄴ", target_distance_list)
                    actions.append(np.random.choice(avail_actions_index,
                                                    p=softmax(target_distance_list[idx], temperature=self.temperature2,
                                                              debug=True)))
        if self.rule == 'rule3':
            actions = list()
            len_target_distance_list = len(target_distance_list)
            temp = []
            for l in range(len_target_distance_list):
                target_distances = list()
                for d in target_distance_list[l]:
                    if d < open_fire_distance or target_distance_list[l].index(d) == 0:
                        d = d
                    else:
                        d = d_max
                    target_distances.append(d)
                temp.append(target_distances)
            target_distance_list = temp

            # [d if d < open_fire_distance or target_distance_list[0].index(d) == 0
            #  else 1000000 for d in target_distance_list[0]]
            # print(target_distance_list)
            if air_alert == True:

                # print(self.temperature1, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature1))
                # print(self.temperature2, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature2))

                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index,
                                                    p=softmax(target_distance_list[idx], temperature=self.temperature1,
                                                              debug=True)))
            else:
                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index,
                                                    p=softmax(target_distance_list[idx], temperature=self.temperature2,
                                                              debug=True)))
        return actions




"detection probability 계산"



import numpy as np
import torch


pi = np.pi


def get_signal_to_noise(P, G, sigma, c, R, N):
    return P + 2*G + sigma + c - 4*R - N # S/N

def get_threshold_to_noise(P_fa):
    return -np.log(P_fa) # T/N

def get_probability_of_detection(S_N, T_N):
    return (1+2*T_N*S_N/(2+S_N**2))*np.exp(1)**(-2*T_N/(2+S_N))

def get_own_course(past_position_x,
                   past_position_y,
                   current_position_x,
                   current_position_y):
    theta = np.arctan2(current_position_y - past_position_y, current_position_x - past_position_x)
    if theta >= -np.pi and theta <= 0:
        theta = -theta
    else:
        theta = 2*np.pi-theta
    return theta

def get_target_bearing(own, target, noise, reverse = False):
    #if noise == True:
    if reverse == False:
        theta = np.arctan2(target.position_y+noise[1]-own.position_y, target.position_x+noise[0]-own.position_x)
    else:
        theta = np.arctan2(target.position_y  - own.position_y- noise[1], target.position_x -noise[0] - own.position_x)
    if theta >= -np.pi and theta <= 0:
        theta = -theta
    else:
        theta = 2*np.pi-theta

    return theta

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

def softmax(z, temperature, reverse = True, debug = False):
    # if debug == True:
    #     print("소맥 전", z)
    if reverse == True:
        z = np.array([-i*temperature/100 for i in z]) # 가까우면 거리가 커져
        logits = np.exp(z)
        probs = logits / np.sum(logits)
        isnan = np.isnan(probs)
    else:
        z = np.array([i / (10 * temperature) for i in z]) # 가까우면 거리가 짧아져
        logits = np.exp(z)
        probs = logits / np.sum(logits)
        isnan = np.isnan(probs)
    # if debug == True:
    #     print("소맥 후", probs)

    if True in isnan:
        probs = [1 if i == True else 0 for i in isnan]
        sum_probs = sum(probs)
        probs = [i/sum_probs for i in probs]



    return probs

def cal_distance(a, b):
    return ((a.position_x - b.position_x) ** 2 + (a.position_y - b.position_y) ** 2) ** 0.5




from utils import *
from copy import deepcopy
import pandas as pd
from math import pi
import math
import pygame
from collections import deque

import sys
sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
from cfg import get_cfg
cfg = get_cfg()

class Decoy:
    def __init__(self, env, launcher, launched_time, position_x, position_y, rcs, decoy_duration, decoy_decaying_rate):
        self.env = env
        self.cla = 'decoy'
        self.status = 'live'
        self.position_x = position_x
        self.position_y = position_y
        self.position_z = 0
        self.launcher = launcher
        self.launched_time = launched_time
        self.init_rcs = rcs
        self.sigma = rcs
        self.T = decoy_duration
        self.eps = decoy_decaying_rate

        if self.env.visualize == True:
            self.put_img('Image\decoy.png')
            self.change_size(5, 5)

    def put_img(self, address):
        self.img = self.env.pygame.image.load(address).convert_alpha()
        self.size_x, self.size_y = self.img.get_size()

    def change_size(self, sx, sy):
        self.img = self.env.pygame.transform.scale(self.img, (sx, sy))
        self.size_x, self.size_y = self.img.get_size()

    def rcs_decay(self):
        t = self.env.now - self.launched_time
        sigma = self.init_rcs * (self.eps ** (t))
        self.sigma = sigma
        if t > self.T or sigma <= 0:
            if self.launcher.side == 'blue':
                self.env.decoys_friendly.remove(self)
            else:
                self.env.decoys_enemy.remove(self)

    def show(self):
        self.env.screen.blit(self.img, (self.position_x, self.position_y))


class Bullet:
    def __init__(self, env, position_x, position_y, position_z):
        self.env = env
        self.position_x = position_x
        self.position_y = position_y
        self.position_z = position_z
        if self.env.visualize == True:
            self.put_img('Image\shell.png')
            self.change_size(2, 5)

    def put_img(self, address):
        self.img = self.env.pygame.image.load(address).convert_alpha()
        self.size_x, self.size_y = self.img.get_size()

    def change_size(self, sx, sy):
        self.img = self.env.pygame.transform.scale(self.img, (sx, sy))
        self.size_x, self.size_y = self.img.get_size()

    def show(self):
        self.env.screen.blit(self.img, (self.position_x, self.position_y))
class Dummy_Target():
    def __init__(self):
        self.dummy_variable = 0

class CIWS:
    def __init__(self, env, launcher, ciws_max_num_per_min, ciws_bullet_capacity, std_accuracy=3.94):
        self.env = env
        self.launcher = launcher
        self.max_num_per_min = ciws_max_num_per_min
        self.bullet_capacity = ciws_bullet_capacity
        self.std_accuracy = std_accuracy

        self.threshold = self.env.ciws_threshold
        self.target = None
        self.original_target = None
        self.cum_bullet = 0


    def firing(self, target, ships, flying_ssms_enemy):
        target_position_x = target.position_x
        target_position_y = target.position_y
        target_position_z = target.position_z
        distance = cal_distance(self.launcher, target)/30
        #print(distance/30)



        self.bullets = [Bullet(self.env, np.random.normal(target_position_x, self.std_accuracy*distance),
                               np.random.normal(target_position_y, self.std_accuracy*distance),
                               np.random.normal(target_position_z, 5.22))
                        for _ in range(int(self.max_num_per_min*(self.env.simtime_per_framerate/60)))]

        if self.cum_bullet <= self.bullet_capacity:
            for bullet in self.bullets:
                self.cum_bullet += 1
                if target.status != 'destroyed':
                    if ((bullet.position_x-target_position_x)**2+
                        (bullet.position_y-target_position_y)**2+
                        (bullet.position_z-target_position_z)**2)**0.5<= self.threshold:
                        target.status = 'destroyed'
                        if target in flying_ssms_enemy:
                            # 이미 기만 process에서 이미 target 개체가 remove 될 수 있기 때문에 조건문(조건문 아니면 Value error 발생)
                            flying_ssms_enemy.remove(target)
                            self.launcher.monitors['ssm_destroying_from_ciws'] += 1
                            #print('ssm_destroying_from_ciws', self.target.id)
                            for ship in ships:
                                # self.target은 yellow의 ssm -> self.target을 detecting하는 주체는 blue
                                if target in ship.ssm_detections:
                                    ship.ssm_detections.remove(target)

    def show(self):
        for bullet in self.bullets:
            bullet.show()

    def counter_attack(self):
        if self.launcher.side == 'blue':
            self.firing(self.target, self.env.friendlies, self.env.flying_ssms_enemy)
        else:
            self.firing(self.target, self.env.enemies, self.env.flying_ssms_friendly)

        self.target = None


class Missile:
    def __init__(self, env, launcher, spec):
        self.id = str(np.round(np.random.uniform(0, 10000), 0)) + "-" + str(np.round(np.random.uniform(0, 5000), 0))
        self.env = env
        self.launcher = launcher
        self.speed = spec['speed'] * self.env.mach_scaler



        self.angular_velocity = spec['angular_velocity'] * pi / 180 * self.env.simtime_per_framerate
        self.rotation_range = spec['rotation_range']

        self.length = spec['length']
        self.radius = spec['radius']

        self.last_missile_v_x = 0
        self.last_missile_v_y = 0

        self.maximum_detection_range = spec['range'] * 10
        self.attack_range = spec['attack_range'] * 10
        self.seeker_on_distance = spec['seeker_on_distance'] * 10


        self.cla = spec['cla']
        self.p_h = spec['p_h']
        self.P = 10 * math.log10(spec['radar_peak_power'])
        self.G = 10 * math.log10(spec['antenna_gain_factor'])
        self.lmbda = spec['wavelength_of_signal'] * 0.01
        self.B_n = spec['radar_receiver_bandwidth']
        self.c = 10 * math.log10((self.lmbda ** 2) / (4 * pi) ** 3)
        self.N = 10 * math.log10(env.k * env.temperature * self.B_n) + 10 * math.log10(env.F) + env.L
        self.seeker = Seeker(env=env,
                             beam_width=spec['beam_width'],
                             rotation_range=spec['rotation_range'],
                             range=spec['range']*10,
                             missile=self)
        self.sigma = -10
        self.position_z = 0
        if self.cla == 'MSAM':
            if self.env.visualize == True:
                self.put_img('Image\m_sam.png')
                self.change_size(10, 14)
        if self.cla == 'LSAM':
            if self.env.visualize == True:
                self.put_img('Image\l_sam.png')
                self.change_size(10, 14)
        if self.cla == 'SSM':
            if self.env.visualize == True:
                self.put_img('Image\missile.png')
                self.change_size(10, 14)


        self.status = 'idle'
        self.fly_mode = None

        self.target = None
        self.original_target = None

        self.angular_rotation_direction = True  # True면 +
        self.pd = 0.8
        self.estimated_hitting_point_x = None
        self.estimated_hitting_point_y = None
        self.destroyer = 'none'


        self.last_v_x = 0
        self.last_v_y = 0
        self.a_x = 0
        self.a_y = 0


    def destroying(self, enemies, ships, flying_ssms_friendly, flying_ssms_enemy, flying_sams_friendly, flying_sams_enemy):
        if self.env.temp_termination != True:
            r = cal_distance(self.target, self)
            r_est_hitting_point = self.cal_distance_estimated_hitting_point()

            if self.target.status != 'destroyed':    # target의 상태가 destroy가 아니라면 계속적으로 비행을 수행한다.
                if r <= self.env.epsilon:            # 표적과의 실제 거리가 가까워졌고,
                    if self.seeker.on == 'lock_on':  # seeker 상태가 lock_on 일때 정상적인 파괴 로직을 수행한다.
                        self.status = 'destroyed'
                        if self.target.cla == 'ship':
                            if np.random.uniform(0, 1) < self.p_h:  # probability of hitting에 따른 명중여부 판단
                                self.target.health -= self.speed/self.env.mach_scaler
                                self.launcher.ship_destroying_history += 1
                                self.original_target.missile_destroying_history += 1
                                enemies.remove(self.target)  # self.target은 yellow의 ship

                                for m in self.target.m_sam_launcher:
                                    m.status = 'destroyed'
                                    m.fly_mode = 'destroyed'
                                for m in self.target.l_sam_launcher:
                                    m.status = 'destroyed'
                                    m.fly_mode = 'destroyed'
                                for m in self.target.ssm_launcher:
                                    m.status = 'destroyed'
                                    m.fly_mode = 'destroyed'


                                self.target.monitors['enemy_flying_ssms'] = len(flying_ssms_friendly)
                                """
                                2023.2.20 업데이트
                                """
                                flying_ssms_friendly.remove(self)  # self는          blue의 ssm
                                self.target.status = 'destroyed'
                                self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.launcher.side,
                                                               "launcher_id": self.launcher.id, "missile_id": self.id,
                                                               "object": self.cla, "id": self.id, "event_type":
                                                                   "{} target / id {}".format(self.target.cla,
                                                                                              self.target.id) + "hit"})

                            else:  # 명중 안됨

                                """
                                state feature 만들기
                                """
                                self.original_target.missile_destroying_history += 1



                                self.target.monitors['ssm_mishit'] += 1
                                flying_ssms_friendly.remove(self)  # self는          blue의 ssm
                                self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.launcher.side,
                                                           "launcher_id": self.launcher.id, "missile_id": self.id,
                                                           "object": self.cla, "id": self.id,
                                                           "event_type": "{} target / id {}".format(self.target.cla,
                                                                                                    self.target.id) + "misfire"})
                            for ship in enemies:
                                if self in ship.ssm_detections:
                                    ship.ssm_detections.remove(self)  # self는          blue의 ssm -> self를 detecting하는 주체는 yellow
                        else:
                            if self.target.cla == 'decoy':
                                """
                                state feature 만들기
                                """
                                if self.launcher.side == 'yellow':
                                    self.env.bonus_reward += 1
                                self.original_target.missile_destroying_history += 1
                                flying_ssms_friendly.remove(self)  # self는          blue의 ssm
                                self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.launcher.side,
                                                           "launcher_id": self.launcher.id, "missile_id": self.id,
                                                           "object": self.cla,
                                                           "id": self.id,
                                                           "event_type": "decoy hit"})
                                self.target.status = 'live'
                                self.target.launcher.monitors['ssm_destroying_on_decoy'] += 1

                                for ship in enemies:
                                    if self in ship.ssm_detections:
                                        ship.ssm_detections.remove(
                                            self)  # self는          blue의 ssm -> self를 detecting하는 주체는 yellow
                            else:
                                if self.cla == 'LSAM':
                                    self.launcher.air_engagement_managing_list.remove([self, self.target])
                                    if np.random.uniform(0, 1) < self.p_h:  # probability of hitting에 따른 명중여부 판단
                                        """
                                        state feature 만들기
                                        """
                                        self.launcher.missile_destroying_history += 1
                                        self.target.status = 'destroyed'
                                        flying_ssms_enemy.remove(self.target)  # self.target은 yellow의 ssm
                                        self.launcher.monitors['ssm_destroying_from_lsam'] += 1
                                        flying_sams_friendly.remove(self)  # self는          blue의 sam
                                        self.env.event_log.append(
                                            {"time": self.env.now, "friend_or_foe": self.launcher.side,
                                             "launcher_id": self.launcher.id, "missile_id": self.id, "object": self.cla,
                                             "id": self.id,
                                             "event_type": "{} target / id {}".format(self.target.cla,
                                                                                      self.target.id) + "hit"})
                                        for ship in ships:  # self.target은 yellow의 ssm -> self.target을 detecting하는 주체는 blue
                                            if self.target in ship.ssm_detections:
                                                ship.ssm_detections.remove(self.target)
                                    else:
                                        self.env.event_log.append(
                                            {"time": self.env.now, "friend_or_foe": self.launcher.side,
                                             "launcher_id": self.launcher.id, "missile_id": self.id, "object": self.cla,
                                             "id": self.id,
                                             "event_type": "{} target / id {}".format(self.target.cla,
                                                                                      self.target.id) + "misfire"})
                                        flying_sams_friendly.remove(self)  # self는          blue의 sam
                                else:
                                    if np.random.uniform(0, 1) < self.p_h:  # probability of hitting에 따른 명중여부 판단

                                        """
                                        
                                        state feature 만들기
                                        
                                        """
                                        self.launcher.missile_destroying_history += 1

                                        self.target.status = 'destroyed'
                                        flying_ssms_enemy.remove(self.target)  # self.target은 yellow의 ssm
                                        self.launcher.monitors['ssm_destroying_from_msam'] += 1
                                        # print('ssm_destroying_from_msam', self.target.id)
                                        flying_sams_friendly.remove(self)  # self는          blue의 sam
                                        self.env.event_log.append(
                                            {"time": self.env.now, "friend_or_foe": self.launcher.side,
                                             "launcher_id": self.launcher.id, "missile_id": self.id, "object": self.cla,
                                             "id": self.id,
                                             "event_type": "{} target / id {}".format(self.target.cla,
                                                                                      self.target.id) + "hit"})
                                    else:
                                        flying_sams_friendly.remove(self)  # self는          blue의 sam
                                        self.env.event_log.append(
                                            {"time": self.env.now, "friend_or_foe": self.launcher.side,
                                             "launcher_id": self.launcher.id, "missile_id": self.id, "object": self.cla,
                                             "id": self.id,
                                             "event_type": "{} target / id {}".format(self.target.cla,
                                                                                      self.target.id) + "misfire"})
                                    for ship in ships:  # self.target은 yellow의 ssm -> self.target을 detecting하는 주체는 blue
                                        if self.target in ship.ssm_detections:
                                            ship.ssm_detections.remove(self.target)
                    else: # 확인완료(이상없음 / 23. 5. 3.)
                        """
                        표적과의 실제 거리는 가깝지만 lock-on 상태가 아닐 경우는 그냥 자폭해버린다.
                        self.target.cla가 'decoy'인 경우는 lock on이 아닐 수 없으므로 이부분은 생략해도 됨
                        -> 그러니까 이부분은 자폭에 관한 이야기만 적으면 됨
                        """
                        self.status = 'destroyed'      # 나(유도탄)의 상태를 파괴 상태로 전환
                        if self.target.cla == 'ship':  # 표적이 함정일 경우
                            self.original_target.missile_destroying_history += 1 # 표적(ship)의 입장에서는 하나가 격추된 느낌이다.
                            flying_ssms_friendly.remove(self)  # self는 blue의 ssm
                            self.target.monitors["ssm_mishit"] += 1
                            self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.launcher.side,
                                                       "launcher_id": self.launcher.id, "missile_id": self.id,
                                                       "object": self.cla, "id": self.id,
                                                       "event_type": "{} target / id {}".format(self.target.cla,
                                                                                                self.target.id) + "misfire"})
                            for ship in enemies:
                                if self in ship.ssm_detections:
                                    ship.ssm_detections.remove(self)  # self는 blue의 ssm -> self를 detecting하는 주체는 yellow
                        else:

                            self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.launcher.side,
                                                       "launcher_id": self.launcher.id, "missile_id": self.id,
                                                       "object": self.cla, "id": self.id,
                                                       "event_type": "{} target / id {}".format(self.target.cla,
                                                                                                self.target.id) + "misfire"})
                            flying_sams_friendly.remove(self)  # self는 blue의 sam
                            if self.cla == 'LSAM':
                                """
                                LSAM 자폭에 관해 기술
                                """
                                self.launcher.air_engagement_managing_list.remove([self, self.target])  # LSAM인 경우 지령 유도 리스트에서 공격 관리(내가 어떤 표적을 공격중이다)를 종료한다.
                            else:
                                pass

                else: # 확인완료(이상없음 / 23. 5. 3.)
                    """
                    표적과 유도탄과의 실제 거리가 먼 경우
                    """
                    if self.fly_mode == 'ccm':
                        if r_est_hitting_point <= self.env.epsilon:  # estimation은 가까운데 실제 표적 거리는 먼 상황(seeker의 작동 또는 lock on 여부와 상관없이 자폭로직을 수행)



                            self.status = 'destroyed'  # 나(유도탄)의 상태를 파괴 상태로 전환
                            if self.target.cla != 'decoy':
                                if self.target.cla == 'ship':
                                    """
                                    state feature 만들기
                                    자신은 SSM  : 자기는 파괴
                                    표적은 SHIP : 표적은 미파괴
                                    """

                                    if self.launcher.side == 'yellow' and self.original_target.status != 'destroyed':
                                        self.env.bonus_reward += 1



                                    self.original_target.missile_destroying_history += 1  # 표적(ship)의 입장에서는 하나가 격추된 느낌이다.
                                    self.target.monitors["ssm_mishit"] += 1
                                else:
                                    if self.target.cla == 'SSM':
                                        """
                                        state feature 만들기
                                        자신은 LSAM/MSAM : 자기는 파괴
                                        표적은 DECOY : 표적은 미파괴
                                        """
                            else:
                                """
                                state feature 만들기
                                자신은 SSM  : 자기는 파괴
                                표적은 DECOY : 표적은 미파괴
                                """
                                self.original_target.missile_destroying_history += 1  # 표적(ship)의 입장에서는 하나가 격추된 느낌이다.
                                self.target.launcher.monitors["ssm_mishit"] += 1

                            if self.target.cla == 'ship' or self.target.cla == 'decoy':  # 표적이 함정일 경우
                                flying_ssms_friendly.remove(self)  # self는 blue의 ssm
                                for ship in enemies:
                                    if self in ship.ssm_detections:
                                        ship.ssm_detections.remove(self)  # self는 blue의 ssm -> self를 detecting하는 주체는 yellow
                            else:
                                flying_sams_friendly.remove(self)  # self는 blue의 sam
                                if self.cla == 'LSAM':
                                    self.launcher.air_engagement_managing_list.remove(
                                        [self, self.target])  # LSAM인 경우 지령 유도 리스트에서 공격 관리(내가 어떤 표적을 공격중이다)를 종료한다.
                                else:
                                    pass
                        else:
                            pass  # estmation도 멀고 실제 표적거리도 멀면 파괴되지 않는다
                    else:
                        """
                        비행모드가 BRM인 경우
                        """
                        if self.seeker.on == 'lock_on':
                            """
                            일단 lock on을 했다면 표적으로 무조건 직행한다.
                            """
                            pass
                        else:
                            """
                            일단 lock on을 하지 않은 상태라면 (발생가능성이 매우 희박)
                            """
                            if r_est_hitting_point <= self.env.epsilon:
                                """
                                표적의 실제거리는 멀고
                                표적의 예상거리는 가까운 상황
                                그러나 lock on은 아닌 상황
                                (즉, 표적이 있을거라 생각하는 위치에 도착했는데 아무것도 없음. 또한, 그때까지 아무것도 접촉하지 않았음)
                                (재탐색은 가정하지 않음)  
                            
                                -> 일단, 나는 무조건 자폭을 해버린다.      
                                """
                                self.status = 'destroyed'  # 나(유도탄)의 상태를 파괴 상태로 전환
                                if self.target.cla != 'decoy':
                                    if self.target.cla == 'ship':

                                        if self.launcher.side == 'yellow' and self.original_target.status != 'destroyed':
                                            self.env.bonus_reward += 1


                                        self.original_target.missile_destroying_history += 1  # 표적(ship)의 입장에서는 하나가 격추된 느낌이다.
                                        self.target.monitors["ssm_mishit"] += 1
                                    if self.target.cla == 'SSM':pass
                                else:
                                    """
                                    표적의 실제거리도 멀고
                                    표적의 예상거리는 가까운 상황
                                    그러나 lock on은 아닌 상황
                                    (즉, 표적이 있을거라 생각하는 위치에 도착했는데 아무것도 없음. 또한, 그때까지 아무것도 접촉하지 않았음)
                                    (재탐색은 가정하지 않음)     
                                    그런데, 이때 표적은 decoy인 상황은 발생하지 않음(decoy가 target화 되었다는 것은 이미    
                                    """
                                    self.original_target.missile_destroying_history += 1  # 표적(ship)의 입장에서는 하나가 격추된 느낌이다.
                                    self.target.launcher.monitors["ssm_mishit"] += 1


                                if self.cla == 'SSM':
                                    "이 부분은 자폭로직에 대한 세부 절차를 기술한 부분임"
                                    flying_ssms_friendly.remove(self)  # self는 blue의 ssm
                                    for ship in enemies:
                                        if self in ship.ssm_detections:
                                            ship.ssm_detections.remove(self)  # self는 blue의 ssm -> self를 detecting하는 주체는 yellow
                                else:
                                    flying_sams_friendly.remove(self)  # self는 blue의 sam
                                    if self.cla == 'LSAM':
                                        self.launcher.air_engagement_managing_list.remove(
                                            [self, self.target])  # LSAM인 경우 지령 유도 리스트에서 공격 관리(내가 어떤 표적을 공격중이다)를 종료한다.
                                    else:
                                        pass
                            else:
                                """
                                표적의 실제거리도 멀고
                                표적의 예상거리도 멀고
                                그러면 아무일도 일어나지 않음
                                """
                                pass

            else:
                """
                target의 상태가 destroyed 상태라면 자폭로직을 수행한다.
                (더이상 표적을 추적하는 의미가 없음 -> 다른 표적을 탐지해서 공격하는 재공격 로직은 없음)
                
                """
                self.status = 'destroyed'                     # 나(SSM)의 상태를 파괴 상태로 전환
                if self.target.cla == 'ship':                 # 표적이 함정일 경우
                    flying_ssms_friendly.remove(self)         # self는 blue의 ssm
                    for ship in enemies:
                        if self in ship.ssm_detections:
                            ship.ssm_detections.remove(self)  # 나(ssm)를 탐지하고 있는 함재기의 추적목록에서 나를 제외한다.
                                                              # self는 blue의 ssm -> self를 detecting하는 주체는 yellow
                else:
                    flying_sams_friendly.remove(self)         # self는 blue의 sam
                    if self.cla == 'LSAM':                    # lsam일 경우
                        self.launcher.air_engagement_managing_list.remove([self, self.target])
                                                              # LSAM인 경우 지령 유도 리스트에서 공격 관리(내가 어떤 표적을 공격중이다)를 종료한다.
                    else:
                        pass

    def cal_distance_estimated_hitting_point(self):
        return ((self.position_x - self.estimated_hitting_point_x) ** 2 + (
                    self.position_y - self.estimated_hitting_point_y) ** 2) ** 0.5

    def flying(self):
        r_est_hitting_point = self.cal_distance_estimated_hitting_point()
        r = cal_distance(self.target, self)


        if self.status != 'destroyed' and self.target.status != 'destroyed':
            self.distance_fly = cal_distance(self, self.launcher)

            if r_est_hitting_point >= self.seeker_on_distance and self.fly_mode != 'brm':
                self.fly_mode = 'ccm'  # 비행상태를 변화시킴
                past_position_x = self.position_x
                past_position_y = self.position_y
                self.last_position_x = self.position_x
                self.last_position_y = self.position_y
                self.v_x = self.speed * (self.estimated_hitting_point_x - self.position_x) / r_est_hitting_point
                self.v_y = self.speed * (self.estimated_hitting_point_y - self.position_y) / r_est_hitting_point
                self.position_x += self.v_x
                self.position_y += self.v_y
                self.course = get_own_course(past_position_x, past_position_y, self.position_x, self.position_y)
            if r_est_hitting_point < self.seeker_on_distance:
                self.fly_mode = 'brm'
                if self.seeker.on != 'lock_on':
                    past_position_x = self.position_x
                    past_position_y = self.position_y
                    self.last_position_x = self.position_x
                    self.last_position_y = self.position_y
                    self.v_x = self.speed * (self.estimated_hitting_point_x - self.position_x) / r_est_hitting_point
                    self.v_y = self.speed * (self.estimated_hitting_point_y - self.position_y) / r_est_hitting_point
                    self.position_x += self.v_x
                    self.position_y += self.v_y
                    self.course = get_own_course(past_position_x, past_position_y, self.position_x, self.position_y)
                else:
                    past_position_x = self.position_x
                    past_position_y = self.position_y
                    self.last_position_x = self.position_x
                    self.last_position_y = self.position_y
                    self.v_x = self.speed * (self.target.position_x - self.position_x) / r
                    self.v_y = self.speed * (self.target.position_y - self.position_y) / r
                    self.position_x += self.v_x
                    self.position_y += self.v_y
                    self.course = get_own_course(past_position_x, past_position_y, self.position_x, self.position_y)
        else:
            if self.fly_mode == 'ccm':
                past_position_x = self.position_x
                past_position_y = self.position_y
                self.last_position_x = self.position_x
                self.last_position_y = self.position_y
                self.v_x = self.speed * (self.estimated_hitting_point_x - self.position_x) / r_est_hitting_point
                self.v_y = self.speed * (self.estimated_hitting_point_y - self.position_y) / r_est_hitting_point
                self.position_x += self.v_x
                self.position_y += self.v_y
            elif self.fly_mode == 'brm':
                past_position_x = self.position_x
                past_position_y = self.position_y
                self.last_position_x = self.position_x
                self.last_position_y = self.position_y
                self.v_x = self.speed * (self.target.position_x - self.position_x) / r
                self.v_y = self.speed * (self.target.position_y - self.position_y) / r
                self.position_x += self.v_x
                self.position_y += self.v_y
            else:
                past_position_x = self.launcher.position_x
                past_position_y = self.launcher.position_y
                self.last_position_x = self.launcher.position_x
                self.last_position_y = self.launcher.position_y

                self.v_x = self.speed * (self.target.position_x - self.position_x) / r
                self.v_y = self.speed * (self.target.position_y - self.position_y) / r

                self.position_x += self.v_x
                self.position_y += self.v_y
            self.course = get_own_course(past_position_x, past_position_y, self.position_x, self.position_y)
            self.fly_mode = 'destroyed'


        self.a_x = (self.v_x - self.last_v_x)/self.env.simtime_per_framerate
        self.a_y = (self.v_y - self.last_v_y) / self.env.simtime_per_framerate

        self.last_v_x = self.v_x
        self.last_v_y = self.v_y




    def seeker_operation(self):
        if self.fly_mode != 'destroyed':
            if self.fly_mode == 'brm':  # seeker on/detection
                if self.seeker.on == 'lock_on':
                    pass
                if self.seeker.on == True:
                    self.get_lock_on_target()
                if self.seeker.on == False:
                    init_angle = self.get_object_angle_noise()#self.angle_atan2(self.target)
                    self.arc_beam_angle_n = init_angle + self.seeker.beam_width
                    self.arc_beam_angle = init_angle - self.seeker.beam_width
                    self.seeker.on = True
            else:
                pass
        else:
            self.seeker.on == 'destroyed'

    def get_angle_of_incidence(self, contact):
        my_course = math.atan2(self.v_y, -self.v_x)
        target_course = math.atan2(-contact.v_y, contact.v_x)
        angle_of_incidence = np.abs(my_course - target_course)
        return angle_of_incidence

    def get_lock_on_target(self):

        in_arc_list, probabilities = self.get_in_arc_list()
        probabilities = softmax(probabilities, 1/50, reverse = False)
        if len(in_arc_list) >= 1:
            p_d = list()
            p_dc = 1
            for p_di in probabilities:
                p_d.append(p_di)
                p_dc *= 1 - p_di
            p_d.append(p_dc)
            in_arc_list.append(None)
            detection = np.random.choice(in_arc_list, p=p_d / np.sum(p_d))
            if detection == None:
                pass
            else:
                self.target = detection
                self.seeker.on = 'lock_on'
                # if self.cla == 'SSM' and self.launcher.side =='yellow':
                #     print(detection.cla, self.a_x, self.a_y)#
                if self.target.cla == 'decoy':
                    self.target.launcher.monitors['ssm_decoying'] += 1
                    self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.launcher.side,
                                               "launcher_id": self.launcher.id, "missile_id": self.id,
                                               "object": self.cla,
                                               "id": self.id,
                                               "event_type": "decoyed"})

                else:
                    self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.launcher.side,
                                               "launcher_id": self.launcher.id, "missile_id": self.id,
                                               "object": self.cla,
                                               "id": self.id,
                                               "event_type": "{} target / id {}".format(self.target.cla,
                                                                                        self.target.id) + "lock_on"})

    def get_in_arc_list(self):
        in_arc_list = []
        probabilities = []
        if self.launcher.side == 'blue':
            bearing, distance = self.get_bearing_and_distance(self.target)
            #print(self.arc_beam_angle, bearing, self.arc_beam_angle_n,distance, self.maximum_detection_range, distance)
            if (bearing <= self.arc_beam_angle_n) and (bearing >= self.arc_beam_angle) and (
                    distance <= self.maximum_detection_range):
                in_arc_list.append(self.target)
                if self.target.cla == 'ship':
                    theta = self.get_angle_of_incidence(self.target)
                    sigma = self.target.get_sigma(theta)



                    R = 10 * math.log10((cal_distance(self, self.target) / 10) * 1.852 * 10 ** 5)  # [km]
                    S_N = get_signal_to_noise(self.P, self.G, sigma, self.c, R, self.N)
                    T_N = get_threshold_to_noise(self.env.P_fa)
                    p_d = get_probability_of_detection(S_N, T_N)
                    probabilities.append(p_d)
                else:
                    sigma = self.target.sigma
                    R = 10 * math.log10((cal_distance(self, self.target) / 10) * 1.852 * 10 ** 5)  # [km]
                    S_N = get_signal_to_noise(self.P, self.G, sigma, self.c, R, self.N)
                    T_N = get_threshold_to_noise(self.env.P_fa)
                    p_d = get_probability_of_detection(S_N, T_N)
                    probabilities.append(p_d)
            else:
                pass
            if self.cla == 'SSM':
                for decoy in self.env.decoys_enemy:
                    bearing, distance = self.get_bearing_and_distance(decoy)
                    if (bearing <= self.arc_beam_angle_n) and (bearing >= self.arc_beam_angle) and (distance <= self.maximum_detection_range):
                        in_arc_list.append(decoy)
                        sigma = decoy.sigma
                        R = 10 * math.log10((cal_distance(self, decoy) / 10) * 1.852 * 10 ** 5)  # [km]
                        S_N = get_signal_to_noise(self.P, self.G, sigma, self.c, R, self.N)
                        T_N = get_threshold_to_noise(self.env.P_fa)
                        p_d = get_probability_of_detection(S_N, T_N)
                        probabilities.append(p_d)
        else:
            bearing, distance = self.get_bearing_and_distance(self.target)
            if (bearing <= self.arc_beam_angle_n) and (bearing >= self.arc_beam_angle) and (
                    distance <= self.maximum_detection_range):
                in_arc_list.append(self.target)
                if self.target.cla == 'ship':
                    theta = self.get_angle_of_incidence(self.target)
                    sigma = self.target.get_sigma(theta)
                    R = 10 * math.log10((cal_distance(self, self.target) / 10) * 1.852 * 10 ** 5)  # [km]
                    S_N = get_signal_to_noise(self.P, self.G, sigma, self.c, R, self.N)
                    T_N = get_threshold_to_noise(self.env.P_fa)
                    p_d = get_probability_of_detection(S_N, T_N)
                    probabilities.append(p_d)
                else:
                    sigma = self.target.sigma
                    R = 10 * math.log10((cal_distance(self, self.target) / 10) * 1.852 * 10 ** 5)  # [km]
                    S_N = get_signal_to_noise(self.P, self.G, sigma, self.c, R, self.N)
                    T_N = get_threshold_to_noise(self.env.P_fa)
                    p_d = get_probability_of_detection(S_N, T_N)
                    probabilities.append(p_d)

            if self.cla == 'SSM':
                for decoy in self.env.decoys_friendly:
                    bearing, distance = self.get_bearing_and_distance(decoy)
                    if (bearing <= self.arc_beam_angle_n) and (bearing >= self.arc_beam_angle) and (
                            distance <= self.maximum_detection_range):
                        in_arc_list.append(decoy)
                        sigma = decoy.sigma
                        R = 10 * math.log10((cal_distance(self, decoy) / 10) * 1.852 * 10 ** 5)  # [km]
                        S_N = get_signal_to_noise(self.P, self.G, sigma, self.c, R, self.N)
                        T_N = get_threshold_to_noise(self.env.P_fa)
                        p_d = get_probability_of_detection(S_N, T_N)
                        probabilities.append(p_d)
        return in_arc_list, probabilities

    def get_bearing_and_distance(self, object):
        bearing = self.get_object_angle(object)
        distance = cal_distance(self, object)
        return bearing, distance

    def put_img(self, address):
        self.img = self.env.pygame.image.load(address).convert_alpha()
        self.size_x, self.size_y = self.img.get_size()

    def change_size(self, sx, sy):
        self.img = self.env.pygame.transform.scale(self.img, (sx, sy))
        self.size_x, self.size_y = self.img.get_size()




    def rotate_arc_beam_angle(self):
        if self.seeker.on == True:
            arc_rotation_angle = (self.get_object_angle_noise() - self.seeker.rotation_range)
            arc_rotation_angle_n = (self.get_object_angle_noise() + self.seeker.rotation_range)
            if self.angular_rotation_direction == True:
                self.arc_beam_angle_n += self.angular_velocity
                self.arc_beam_angle += self.angular_velocity
            else:
                self.arc_beam_angle_n -= self.angular_velocity
                self.arc_beam_angle -= self.angular_velocity
            if (self.arc_beam_angle_n >= arc_rotation_angle_n) and (self.angular_rotation_direction == True) and (
                    self.arc_beam_angle > arc_rotation_angle):
                self.angular_rotation_direction = False
            if (self.arc_beam_angle <= arc_rotation_angle) and (self.angular_rotation_direction == False) and (
                    self.arc_beam_angle_n < arc_rotation_angle_n):
                self.angular_rotation_direction = True

    def angle_atan2(self, target):
        return math.atan2(target.position_y - self.position_y, target.position_x - self.position_x)

    def get_object_angle(self, object):
        return math.atan2(-(object.position_y - self.position_y), object.position_x - self.position_x)
        #2 * pi - self.angle_atan2(object)

    def get_object_angle_noise(self):
        #
        # r_est = ((self.estimated_hitting_point_y - self.position_y)**2+(self.estimated_hitting_point_x - self.position_x)**2)**0.5
        # v_y = (self.estimated_hitting_point_y - self.position_y)/r_est
        # v_x = (self.estimated_hitting_point_x - self.position_x)/r_est
        return math.atan2(-(self.estimated_hitting_point_y - self.position_y), self.estimated_hitting_point_x - self.position_x)

    def get_theta(self, object):
        angle = self.angle_atan2(object)
        return math.degrees(angle) * -1

    def missile_draw_ccm(self):
        if self.target.cla == 'decoy':
            if self.launcher.side == 'blue':
                self.relative_angle = math.atan2(self.target.position_y - self.position_y,
                                                 self.target.position_x - self.position_x)  # [rad], bullet과 target 사이각도, 동시에 움직이므로, 항상 동일, bullet 각으로 봐도 무방(TWA)
                self.apparent_speed = np.sqrt(
                    (0 ** 2) + (self.speed ** 2) + (2 * 0 * self.speed * math.cos(2 * pi - self.relative_angle)))
                self.apparent_angle = math.acos(
                    (self.speed * math.cos(self.relative_angle) + 0) / self.apparent_speed)  # 겉보기 각도, 앞지름각
                self.apparent_angle_theta = -1 * math.degrees(self.apparent_angle)
            if self.launcher.side == 'yellow':
                self.relative_angle = math.atan2(self.target.position_y - self.position_y,
                                                 self.target.position_x - self.position_x)  # [rad], bullet과 target 사이각도, 동시에 움직이므로, 항상 동일, bullet 각으로 봐도 무방(TWA)
                self.apparent_speed = np.sqrt(
                    (0 ** 2) + (self.speed ** 2) + (2 * 0 * self.speed * math.cos(self.relative_angle)))
                self.apparent_angle = math.acos(
                    (self.speed * math.cos(self.relative_angle) + 0) / self.apparent_speed)  # 겉보기 각도, 앞지름각
                self.apparent_angle_theta = math.degrees(self.apparent_angle)
        else:
            if self.launcher.side == 'blue':
                self.relative_angle = math.atan2(self.target.position_y - self.position_y,
                                                 self.target.position_x - self.position_x)  # [rad], bullet과 target 사이각도, 동시에 움직이므로, 항상 동일, bullet 각으로 봐도 무방(TWA)
                self.apparent_speed = np.sqrt((self.target.speed ** 2) + (self.speed ** 2) + (
                            2 * self.target.speed * self.speed * math.cos(2 * pi - self.relative_angle)))
                self.apparent_angle = math.acos((self.speed * math.cos(
                    self.relative_angle) + self.target.speed) / self.apparent_speed)  # 겉보기 각도, 앞지름각
                self.apparent_angle_theta = -1 * math.degrees(self.apparent_angle)
            if self.launcher.side == 'yellow':
                self.relative_angle = math.atan2(self.target.position_y - self.position_y,
                                                 self.target.position_x - self.position_x)  # [rad], bullet과 target 사이각도, 동시에 움직이므로, 항상 동일, bullet 각으로 봐도 무방(TWA)
                self.apparent_speed = np.sqrt((self.target.speed ** 2) + (self.speed ** 2) + (
                            2 * self.target.speed * self.speed * math.cos(self.relative_angle)))
                self.apparent_angle = math.acos((self.speed * math.cos(
                    self.relative_angle) + self.target.speed) / self.apparent_speed)  # 겉보기 각도, 앞지름각
                self.apparent_angle_theta = math.degrees(self.apparent_angle)

        self.image = pygame.transform.rotate(self.img, self.apparent_angle_theta)
        return self.env.screen.blit(self.image, (self.position_x, self.position_y))

    # def get_center_bearing(self):
    #     if self.seeker.on == 'lock_on':
    #
    #
    #     else:


    def rotation_range_draw(self):
        self.range = 100
        self.theta = self.get_theta(self.target)
        position_arc_x = self.position_x - 0.5 * self.range + 0.5 * self.size_x
        position_arc_y = self.position_y - 0.5 * self.range + 0.5 * self.size_y

        # arc_rotation_angle = (self.get_object_angle(self.target) - self.seeker.rotation_range)
        # arc_rotation_angle_n = (self.get_object_angle(self.target) + self.seeker.rotation_range)
        arc_rotation_angle = (self.get_object_angle_noise() - self.seeker.rotation_range)
        arc_rotation_angle_n = (self.get_object_angle_noise() + self.seeker.rotation_range)

        # if self.id == '5086.0-2619.0':
        #     print(arc_rotation_angle, arc_rotation_angle_n)

        return pygame.draw.arc(self.env.screen, (255, 0, 0),
                               [position_arc_x, position_arc_y,
                                self.range, self.range],
                               arc_rotation_angle,
                               arc_rotation_angle_n, 1)

    def arc_beam_draw(self):
        self.range = 100
        self.theta = self.get_theta(self.target)
        position_arc_x = self.position_x - 0.5 * self.range + 0.5 * self.size_x
        position_arc_y = self.position_y - 0.5 * self.range + 0.5 * self.size_y
        if self.seeker.on != 'lock_on':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        return pygame.draw.arc(self.env.screen, color,
                               [position_arc_x, position_arc_y,
                                self.range, self.range],
                               self.arc_beam_angle,
                               self.arc_beam_angle_n, 5)

    def missile_draw(self):
        return self.env.screen.blit(self.img, (self.position_x, self.position_y))

    def missile_transform(self):
        #self.image = pygame.transform.rotate(self.img, self.theta)
        return self.env.screen.blit(self.image, (self.position_x, self.position_y))

    def show(self):
        if self.fly_mode == 'ccm':  # ccm

            self.missile_draw_ccm()
            self.rotation_range_draw()

        if self.fly_mode == 'brm':  # seeker on/detection
            if self.seeker.on != 'lock_on':
                self.missile_draw_ccm()
                self.rotation_range_draw()
                self.arc_beam_draw()
            else:
                self.missile_transform()
                self.arc_beam_draw()


class Seeker:
    def __init__(self, env, beam_width, rotation_range, range, missile, on=False):
        self.env = env
        self.on = on
        self.beam_width = beam_width * (pi / 180) / 2  # [deg]
        self.rotation_range = rotation_range * (pi / 180) / 2  # [deg]
        self.range = range
        self.missile = missile


class Patrol_aircraft:

    def __init__(self,
                 env,
                 id,
                 speed,
                 course,
                 radius,
                 initial_position_x,
                 initial_position_y,
                 side):
        self.cla = 'patrol_aircraft'
        self.env = env
        self.id = id
        self.speed = speed * self.env.nautical_mile_scaler
        self.course_input = course
        self.radius = radius

        self.position_x = initial_position_x
        self.position_y = initial_position_y

        self.side = side
        if self.env.visualize == True:
            self.put_img('Image\patrol_aircraft.png')
            self.change_size(20, 20)

    def put_img(self, address):
        self.img = self.env.pygame.image.load(address).convert_alpha()
        self.size_x, self.size_y = self.img.get_size()

    def change_size(self, sx, sy):
        self.img = self.env.pygame.transform.scale(self.img, (sx, sy))
        self.size_x, self.size_y = self.img.get_size()

    def show(self):
        self.env.screen.blit(self.img, (self.position_x, self.position_y))

    def maneuvering(self):
        self.course_input += 10
        self.v_y = self.radius * math.cos(self.course_input * pi / 180)
        self.v_x = self.radius * math.sin(self.course_input * pi / 180)

        past_position_x = self.position_x
        past_position_y = self.position_y

        self.position_x += self.v_x
        self.position_y += self.v_y
        self.course = get_own_course(past_position_x, past_position_y, self.position_x, self.position_y)


class Ship:
    def __init__(self,
                 env,
                 id,
                 speed,
                 course,
                 initial_position_x,
                 initial_position_y,
                 num_m_sam,
                 type_m_sam,
                 num_l_sam,
                 type_l_sam,
                 num_ssm,
                 type_ssm,
                 length,
                 breadth,
                 height,
                 detection_range,
                 surface_tracking_limit,
                 surface_engagement_limit,
                 air_tracking_limit,
                 air_engagement_limit,
                 ciws_max_range,
                 decoy_launching_distance,
                 decoy_launching_bearing,
                 decoy_launching_interval,
                 evading_course,
                 radar_peak_power,
                 antenna_gain_factor,
                 wavelength_of_signal,
                 radar_receiver_bandwidth,
                 decoy_rcs,
                 side,
                 decoy_duration,
                 decoy_decaying_rate,
                 ciws_max_num_per_min,
                 ciws_bullet_capacity,
                 ssm_launching_duration_min,
                 ssm_launching_duration_max,
                 lsam_launching_duration_min,
                 lsam_launching_duration_max,
                 msam_launching_duration_min,
                 msam_launching_duration_max,
                 interpolating_rcs
                 ):

        self.cla = 'ship'
        self.id = id
        self.env = env
        self.speed = speed * self.env.nautical_mile_scaler
        self.course_input = course+np.random.uniform(-30,30)

        self.last_ship_v_x = 0
        self.last_ship_v_y = 0

        self.health = 3.6
        self.init_health = deepcopy(self.health)
        self.last_health = deepcopy(self.health)

        self.ship_destroying_history = 0
        self.missile_destroying_history = 0


        self.v_y = -self.speed * math.cos(self.course_input * pi / 180)
        self.v_x = self.speed * math.sin(self.course_input * pi / 180)

        self.position_x = initial_position_x
        self.position_y = initial_position_y
        self.last_position_x = initial_position_x
        self.last_position_y = initial_position_y

        self.position_z = 0

        self.length = length
        self.breadth = breadth
        self.height = height
        """
        Radar의 제원과 관련있는
        """

        self.P = 10 * math.log10(radar_peak_power)
        self.G = 10. * math.log10(antenna_gain_factor)
        self.lmbda = wavelength_of_signal / 100
        self.B_n = radar_receiver_bandwidth


        self.get_sigma = interpolating_rcs
        self.c = 10 * math.log10((self.lmbda ** 2) / (4 * pi) ** 3)
        self.N = 10 * math.log10(env.k * env.temperature * self.B_n) + 10 * math.log10(env.F) + env.L
        self.N_surface = 10 * math.log10(env.k * env.temperature * self.B_n) + 10 * math.log10(env.F_surface) + env.L


        self.ciws_max_range = ciws_max_range * 10
        self.ciws_max_num_per_min = ciws_max_num_per_min
        self.ciws_bullet_capacity = ciws_bullet_capacity
        self.CIWS = CIWS(env, launcher=self, ciws_max_num_per_min=self.ciws_max_num_per_min, ciws_bullet_capacity=self.ciws_bullet_capacity)
        if self.env.detection_by_height == False:
            self.detection_range = detection_range * 10
        else:
            self.detection_range = 1.23 * ((3.28084 * self.height)**0.5+(3.28084 * 5)**0.5)
        self.side = side
        self.decoy_rcs = decoy_rcs
        self.decoy_launching_distance = decoy_launching_distance*10
        self.decoy_launching_bearing = decoy_launching_bearing
        self.decoy_launching_interval = decoy_launching_interval
        self.decoy_duration = decoy_duration
        self.decoy_decaying_rate = decoy_decaying_rate

        self.evading_course = evading_course

        self.surface_tracking_limit = surface_tracking_limit
        self.surface_engagement_limit = surface_engagement_limit

        self.air_tracking_limit = air_tracking_limit
        self.air_engagement_limit = air_engagement_limit

        self.surface_engagement_num = 0
        self.air_engagement_num = 0
        self.ssm_detections = []
        self.type_ssm = type_ssm
        self.m_sam_launcher = \
            [Missile(env=env, launcher=self, spec=type_m_sam) for _ in range(num_m_sam)]
        self.l_sam_launcher = \
            [Missile(env=env, launcher=self, spec=type_l_sam) for _ in range(num_l_sam)]
        self.ssm_launcher = \
            [Missile(env=env, launcher=self, spec=type_ssm) for _ in range(num_ssm)]

        self.attack_range = self.ssm_launcher[0].attack_range
        self.speed_m = self.ssm_launcher[0].speed
        self.num_m_sam = num_m_sam
        self.num_l_sam = num_l_sam
        self.num_ssm = num_ssm

        self.debug_ssm_launcher = self.ssm_launcher[:]

        self.ssm_launching_duration_min = ssm_launching_duration_min
        self.ssm_launching_duration_max = ssm_launching_duration_max
        self.lsam_launching_duration_min = lsam_launching_duration_min
        self.lsam_launching_duration_max = lsam_launching_duration_max
        self.msam_launching_duration_min = msam_launching_duration_min
        self.msam_launching_duration_max = msam_launching_duration_max

        self.m_sam_max_range = self.m_sam_launcher[0].attack_range
        self.l_sam_max_range = self.l_sam_launcher[0].attack_range
        self.ssm_feature = [0, 0, 0, 0, 0, 0]

        if len(self.ssm_launcher) != 0:
            self.ssm_max_range = self.ssm_launcher[0].attack_range
        else:
            self.ssm_max_range = float('inf')
        self.status = 'searching'

        if side == 'blue':
            if self.env.visualize == True:
                self.put_img('Image\_blue.png')
                self.change_size(30, 45)
        else:
            if self.env.visualize == True:
                self.put_img('Image\yellow.png')
                self.change_size(30, 45)

        self.air_engagement_managing_list = list()
        self.air_prelaunching_managing_list = list()
        self.surface_prelaunching_managing_list = list()

        if self.side == 'blue':
            enemy_remains_init = self.env.num_enemy_ssm

        else:
            enemy_remains_init = self.env.num_friendly_ssm

        self.last_v_x = 0
        self.last_v_y = 0
        self.a_x = 0
        self.a_y = 0
        num = cfg.num_action_history
        self.action_history = deque(maxlen = num)
        for _ in range(num):
            self.action_history.append(None)
        self.monitors = {"ssm_destroying_from_lsam": 0,
                         "ssm_destroying_from_msam": 0,
                         "ssm_destroying_from_ciws": 0,
                         "ssm_destroying_on_decoy": 0,
                         "ssm_decoying": 0,
                         "ssm_mishit": 0,
                         "enemy_launching":0,
                         "lsam_launching":0,
                         "msam_launching":0,
                         "enemy_remains":enemy_remains_init,
                         "enemy_num_ssm":enemy_remains_init,
                         'enemy_flying_ssms':0}

    def put_img(self, address):
        self.img = self.env.pygame.image.load(address).convert_alpha()
        self.size_x, self.size_y = self.img.get_size()

    def change_size(self, sx, sy):
        self.img = self.env.pygame.transform.scale(self.img, (sx, sy))
        self.size_x, self.size_y = self.img.get_size()

    def show(self):
        self.env.screen.blit(self.img, (self.position_x, self.position_y))

    def get_flying_ssms_status(self):
        self.ssm_feature = [0, 0, 0, 0, 0, 0]
        if self.side == 'blue':
            for missile in self.env.flying_ssms_enemy:
                distance = cal_distance(missile, self)
                if missile.original_target == self:
                    if distance >= 500:
                        self.ssm_feature[0] += 1
                    elif distance >= 400:
                        self.ssm_feature[1] += 1
                    elif distance >= 300:
                        self.ssm_feature[2] += 1
                    elif distance >= 200:
                        self.ssm_feature[3] += 1
                    elif distance >= 100:
                        self.ssm_feature[4] += 1
                    else:
                        self.ssm_feature[5] += 1
        else:
            for missile in self.env.flying_ssms_friendly:
                distance = cal_distance(missile, self)
                if missile.original_target == self:
                    if distance >= 500:
                        self.ssm_feature[0] += 1
                    elif distance >= 400:
                        self.ssm_feature[1] += 1
                    elif distance >= 300:
                        self.ssm_feature[2] += 1
                    elif distance >= 200:
                        self.ssm_feature[3] += 1
                    elif distance >= 100:
                        self.ssm_feature[4] += 1
                    else:
                        self.ssm_feature[5] += 1

    def maneuvering(self):
        if self.position_x != None:
            past_position_x = self.position_x
            past_position_y = self.position_y
            self.last_position_x = deepcopy(self.position_x)
            self.last_position_y = deepcopy(self.position_y)

        self.v_y = -self.speed * math.cos((self.course_input ) * pi / 180)
        self.v_x = self.speed * math.sin((self.course_input ) * pi / 180)



        self.position_x += self.v_x
        self.position_y += self.v_y

        self.a_x = (self.v_x - self.last_v_x)/self.env.simtime_per_framerate
        self.a_y = (self.v_y - self.last_v_y)/self.env.simtime_per_framerate

        self.last_v_x = self.v_x
        self.last_v_y = self.v_y


        self.course = get_own_course(past_position_x, past_position_y, self.position_x, self.position_y)

    def launch_decoy(self):
        "decoy를 운영하는 로직은 현재 일정 주기에 따라 발사하는 방식임"

        if self.env.now % (self.decoy_launching_interval) <= 0.001:  # 일정 주기에 따라 decoy 발사

            for ssm in self.ssm_detections:

                launching_bearing = math.atan2(ssm.position_y - deepcopy(self.position_y),
                                               ssm.position_x - deepcopy(self.position_x)) + \
                                    self.decoy_launching_bearing * pi / 180
                #print(launching_bearing)
                launching_distance_x = self.decoy_launching_distance * math.cos(launching_bearing)
                launching_distance_y = self.decoy_launching_distance * math.sin(launching_bearing)
                launching_position_x = deepcopy(self.position_x) + launching_distance_x
                launching_position_y = deepcopy(self.position_y) + launching_distance_y
                if self.side == 'yellow':
                    self.env.decoys_enemy.append(Decoy(env=self.env,
                                                       launcher=self,
                                                       launched_time=self.env.now,
                                                       position_x=launching_position_x,
                                                       position_y=launching_position_y,
                                                       rcs=self.decoy_rcs,
                                                       decoy_duration=self.decoy_duration,
                                                       decoy_decaying_rate=self.decoy_decaying_rate))
                else:
                    self.env.decoys_friendly.append(Decoy(env=self.env,
                                                          launcher=self,
                                                          launched_time=self.env.now,
                                                          position_x=launching_position_x,
                                                          position_y=launching_position_y,
                                                          rcs=self.decoy_rcs,
                                                          decoy_duration=self.decoy_duration,
                                                          decoy_decaying_rate=self.decoy_decaying_rate))

    def get_detections(self):
        " 접촉 표적을 관리하는 원칙 "
        " 1. 접촉한 순서대로 앞에서 부터 채운다 "
        " 2. 파괴된 표적은 접촉 리스트에서 없앤다. "
        #print("전 길이", len(self.ssm_detections))
        if self.side == 'blue':
            if len(self.ssm_detections) < self.air_tracking_limit:
                for missile in self.env.flying_ssms_enemy:
                    if (missile in self.ssm_detections):
                        pass
                    else:
                        R = cal_distance(self, missile) / 10 * 1852
                        S_N = get_signal_to_noise(self.P, self.G, missile.sigma, self.c, R, self.N_surface)
                        T_N = get_threshold_to_noise(self.env.P_fa_surface)
                        p_d = get_probability_of_detection(S_N, T_N)
                        if missile.seeker.on == True:
                            if len(self.ssm_detections) < self.air_tracking_limit:
                                self.ssm_detections.append(missile)
                        else:

                            if cal_distance(self, missile) <= self.detection_range:
                                if (np.random.uniform(0,1) <= p_d):
                                    # 사거리에 대한 정보 아직 안들어감 / 한번 접촉한 표적은 지속적으로 접촉을 유지하는 것으로 가정
                                    if len(self.ssm_detections) < self.air_tracking_limit:
                                        self.ssm_detections.append(missile)

        else:
            if len(self.ssm_detections) < self.air_tracking_limit:

                for missile in self.env.flying_ssms_friendly:
                    if (missile in self.ssm_detections):
                        pass
                    else:
                        R = cal_distance(self, missile) / 10 * 1852
                        S_N = get_signal_to_noise(self.P, self.G, missile.sigma, self.c, R, self.N_surface)
                        T_N = get_threshold_to_noise(self.env.P_fa_surface)
                        p_d = get_probability_of_detection(S_N, T_N)
                        if missile.seeker.on == True:
                            if len(self.ssm_detections) < self.air_tracking_limit:
                                self.ssm_detections.append(missile)
                        else:
                            if cal_distance(self, missile) <= self.detection_range:
                                if (np.random.uniform(0,
                                                      1) <= p_d):  # 사거리에 대한 정보 아직 안들어감 / 한번 접촉한 표적은 지속적으로 접촉을 유지하는 것으로 가정
                                    if len(self.ssm_detections) < self.air_tracking_limit:
                                        self.ssm_detections.append(missile)
        self.ssm_detections = sorted(self.ssm_detections, key=lambda contact: cal_distance(self, contact))
        #print("후 길이", len(self.ssm_detections))


    def surface_prelaunching_process(self):
        surface_prelaunching_managing_list = self.surface_prelaunching_managing_list[:]
        for prelaunching_info in surface_prelaunching_managing_list:
            launching_time = prelaunching_info[0]
            ssm = prelaunching_info[1]
            target = prelaunching_info[2]
            if target.status == 'destroyed':  # target이 파괴됨
                ssm.target = None
                self.surface_prelaunching_managing_list.remove(prelaunching_info)
                ssm.status = 'idle'
            else:
                if self.env.now >= launching_time:  # 발사시간이 완료가 되면
                    ssm.status = 'flying'

                    ssm.target.monitors["enemy_launching"] += 1
                    ssm.target.monitors["enemy_remains"] -= 1
#                    print("후", self.side, ssm.target.monitors["enemy_launching"]+ssm.target.monitors["enemy_remains"], ssm.target.monitors["enemy_launching"], ssm.target.monitors["enemy_remains"])
                    self.ssm_launcher.remove(ssm)
                    self.surface_prelaunching_managing_list.remove(prelaunching_info)
                    ssm.position_x = self.position_x
                    ssm.position_y = self.position_y
                    est_x, est_y = self.get_estimated_hitting_point(ssm, target)
                    ssm.estimated_hitting_point_x_debug = est_x
                    ssm.estimated_hitting_point_y_debug = est_y
                    ssm.estimated_hitting_point_x = est_x  # target.position_x
                    ssm.estimated_hitting_point_y = est_y  # target.position_y
                    if self.side == 'blue':
                        self.env.flying_ssms_friendly.append(ssm)
                        self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.side,
                                                   "launcher_id": self.id, "missile_id": ssm.id, "object": ssm.cla,
                                                   "id": ssm.id,
                                                   "event_type": "{} target / id {}".format(ssm.target.cla,
                                                                                            ssm.target.id) + "launch"})
                    else:
                        self.env.flying_ssms_enemy.append(ssm)
                        self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.side,
                                                   "launcher_id": self.id, "missile_id": ssm.id, "object": ssm.cla,
                                                   "id": ssm.id,
                                                   "event_type": "{} target / id {}".format(ssm.target.cla,
                                                                                            ssm.target.id) + "launch"})

                else:
                    pass

    def air_prelaunching_process(self):
        air_prelaunching_managing_list = self.air_prelaunching_managing_list[:]
        for prelaunching_info in air_prelaunching_managing_list:
            launching_time = prelaunching_info[0]
            sam = prelaunching_info[1]
            target = prelaunching_info[2]
            if target.status == 'destroyed':  # target이 파괴됨
                sam.target = None
                sam.status = 'idle'
                self.air_prelaunching_managing_list.remove(prelaunching_info)
                self.air_engagement_managing_list.remove([sam, target])
            else:
                if self.env.now >= launching_time:  # 발사시간이 완료가 되면
                    sam.status = 'flying'
                    if sam.cla == 'LSAM':
                        self.monitors["lsam_launching"] += 1
                    if sam.cla == 'MSAM':
                        self.monitors["msam_launching"] += 1
                    sam.position_x = self.position_x
                    sam.position_y = self.position_y
                    est_x, est_y = self.get_estimated_hitting_point(sam, target)
                    sam.estimated_hitting_point_x_debug = est_x
                    sam.estimated_hitting_point_y_debug = est_y
                    sam.estimated_hitting_point_x = est_x  # target.position_x
                    sam.estimated_hitting_point_y = est_y  # target.position_y
                    self.air_prelaunching_managing_list.remove(prelaunching_info)
                    if sam.cla == 'MSAM':
                        self.m_sam_launcher.remove(sam)
                        #print("전", len(self.air_engagement_managing_list))
                        self.air_engagement_managing_list.remove([sam, target])
                        #print("후", len(self.air_engagement_managing_list))
                    else:
                        self.l_sam_launcher.remove(sam)
                    if self.side == 'blue':
                        self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.side,
                                                   "launcher_id": self.id, "missile_id": sam.id, "object": sam.cla,
                                                   "id": sam.id,
                                                   "event_type": "{} target / id {}".format(sam.target.cla,
                                                                                            sam.target.id) + "launch"})
                        self.env.flying_sams_friendly.append(sam)
                    else:
                        self.env.event_log.append({"time": self.env.now, "friend_or_foe": self.side,
                                                   "launcher_id": self.id, "missile_id": sam.id, "object": sam.cla,
                                                   "id": sam.id,
                                                   "event_type": "{} target / id {}".format(sam.target.cla,
                                                                                            sam.target.id) + "launch"})
                        self.env.flying_sams_enemy.append(sam)
                else:
                    pass

    def get_estimated_hitting_point(self, missile, target):
        if missile.cla == 'SSM':
            norm = 500*1/self.env.now
            noise_y = target.position_y + np.random.normal(0, norm)
            noise_x = target.position_x + np.random.normal(0, norm)
        else:
            distance = ((target.position_y-self.position_y)**2+(target.position_x-self.position_x)**2)/4000
            noise_y = target.position_y+np.random.normal(0, distance)
            noise_x = target.position_x+np.random.normal(0, distance)

        theta = math.atan2(noise_y-self.position_y, noise_x-self.position_x)
        time_to_intersection = (noise_x-self.position_x)/ (missile.speed*math.cos(theta))
        estimated_x = noise_x + target.v_x*time_to_intersection ##self.position_x + 300*math.cos(theta)#((-target.position_y+self.position_y)-(target.position_x*(-target.v_x/target.v_y)-self.position_x*math.tan(-theta)))/(math.tan(-theta)-(-target.v_x/target.v_y))
        estimated_y = noise_y + target.v_y*time_to_intersection #self.position_y + 300*math.sin(theta)#((target.position_x-self.position_y)-(target.position_y*(-target.v_y/target.v_x)-self.position_y*1/math.tan(-theta)))/(1/math.tan(-theta)-(-target.v_y/target.v_x))
        return estimated_x, estimated_y
    def target_allot_by_action_feature(self, action_feature):

        if self.side == 'blue':
            len_limit = len(self.env.enemies_fixed_list)  # 적함에 대한 표적할당
        else:
            len_limit = len(self.env.friendlies_fixed_list)  # 우군함에 대한 표적할당


        if action_feature == [0,0,0,0,0,0,0,0]:
            target_id = 0
        else:#
            for tar in self.env.enemies_fixed_list:
                if action_feature == tar.last_action_feature:
                    target_id = self.env.enemies_fixed_list.index(tar)+1
            for tar in self.ssm_detections:
                if action_feature == tar.last_action_feature:
                    target_id = self.ssm_detections.index(tar) + 1 + len_limit

        if target_id == 0:
            target = None

        if (target_id >= 1) and (target_id <= len_limit):  # 대함표적에 대한 prelaunching process logic
            target_idx = target_id - 1  # no_ops를 제외하고 index한다.
            if self.side == 'blue':
                target = self.env.enemies_fixed_list[target_idx]  # 적함에 대한 표적할당
            else:
                target = self.env.friendlies_fixed_list[target_idx]  # 우군함에 대한 표적할당
            idle_ssm_launcher = [missile for missile in self.ssm_launcher if
                                 missile.status == 'idle']  # idle한 유도탄 중에서 유도탄을 할당
            missile = idle_ssm_launcher[0]
            missile.target = target
            missile.original_target = target
            missile.status = 'target_allocated'
            launching_time = self.env.now + np.random.uniform(self.ssm_launching_duration_min,
                                                              self.ssm_launching_duration_max)  # 대함 발사 소요시간
            self.surface_prelaunching_managing_list.append([launching_time, missile, target])
        if target_id > len_limit:  # 대공표적에 대한 prelauching_process 수행
            "LSAM의 경우는 표적할당부터 해당 LSAM 파괴 시까지 capacity를 하나 가져간다"
            "MSAM의 경우는 표적할당부터 해당 MSAM 발사 시까지 capacity를 하나 가져간다"
            "각 유도탄에 대해서는 일정 시간의 발사 준비시간이 소요됨 해당 "
            "유도탄 할당 정책"
            "이 로직에 거리에 대한 sorting이 들어가야함"
            "get_avail_actions에서는 가까운 순서대로 앞 index를 가져옴"
            "surface_limit의 의미 : 표적관리를 몇개까지 할 것인가?"
            target_idx = target_id - len_limit - 1  # 대공 표적에 대한 index는 no_ops, 그다음 self.surface_limit다 채우고 시작

            target = self.ssm_detections[target_idx]
            "무장 운용 우선순위"
            "해당 거리 대에 최적화된 무장을 운용"
            "i.e. LSAM은 MSAM.max < 표적거리 <= LSAM.max 최적화"
            "     MSAM은 CIWS.max < 표적거리 <= MSAM.max 최적화"
            "     CIWS는        0 < 표적거리 <= CIWS.max 최적화"
            d = cal_distance(self, target)
            if (d <= self.l_sam_max_range) and (d > self.m_sam_max_range):
                idle_l_sam = [missile for missile in self.l_sam_launcher if missile.status == 'idle']
                if len(idle_l_sam) >= 1:
                    missile = idle_l_sam[0]
                    missile.target = target
                    missile.original_target = target
                    missile.status = 'target_allocated'
                    launching_time = self.env.now + np.random.uniform(self.lsam_launching_duration_min,
                                                                      self.lsam_launching_duration_max)  # 발사 소요시간
                    self.air_prelaunching_managing_list.append([launching_time, missile, target])
                    self.air_engagement_managing_list.append([missile, target])
            if (d <= self.m_sam_max_range) and (d > self.ciws_max_range):
                idle_m_sam = [missile for missile in self.m_sam_launcher if missile.status == 'idle']
                if len(idle_m_sam) >= 1:
                    missile = idle_m_sam[0]
                else:
                    idle_l_sam = [missile for missile in self.l_sam_launcher if missile.status == 'idle']

                    missile = idle_l_sam[0]

                missile.target = target
                missile.original_target = target

                missile.status = 'target_allocated'
                launching_time = self.env.now + np.random.uniform(self.msam_launching_duration_min,
                                                                  self.msam_launching_duration_min)  # 발사 소요시간
                self.air_prelaunching_managing_list.append([launching_time, missile, target])
                self.air_engagement_managing_list.append([missile, target])
            if (d <= self.ciws_max_range):
                self.CIWS.target = target
                self.CIWS.original_target = target
        self.action_history.append(target)
    def target_allocation_process(self, target_id):


        if self.side == 'blue':
            len_limit = len(self.env.enemies_fixed_list)  # 적함에 대한 표적할당
        else:
            len_limit = len(self.env.friendlies_fixed_list)  # 우군함에 대한 표적할당

        if target_id == 0:
            pass
        if (target_id >= 1) and (target_id <= len_limit):  # 대함표적에 대한 prelaunching process logic
            target_idx = target_id - 1  # no_ops를 제외하고 index한다.
            if self.side == 'blue':
                target = self.env.enemies_fixed_list[target_idx]     # 적함에 대한 표적할당

            else:
                target = self.env.friendlies_fixed_list[target_idx]  # 우군함에 대한 표적할당

            idle_ssm_launcher = [missile for missile in self.ssm_launcher if
                                 missile.status == 'idle']  # idle한 유도탄 중에서 유도탄을 할당
            missile = idle_ssm_launcher[0]

            missile.target = target
            missile.original_target = target


            missile.status = 'target_allocated'
            launching_time = self.env.now + np.random.uniform(self.ssm_launching_duration_min,
                                                              self.ssm_launching_duration_max)  # 대함 발사 소요시간
            self.surface_prelaunching_managing_list.append([launching_time, missile, target])
        if target_id > len_limit:  # 대공표적에 대한 prelauching_process 수행
            "LSAM의 경우는 표적할당부터 해당 LSAM 파괴 시까지 capacity를 하나 가져간다"
            "MSAM의 경우는 표적할당부터 해당 MSAM 발사 시까지 capacity를 하나 가져간다"
            "각 유도탄에 대해서는 일정 시간의 발사 준비시간이 소요됨 해당 "
            "유도탄 할당 정책"
            "이 로직에 거리에 대한 sorting이 들어가야함"
            "get_avail_actions에서는 가까운 순서대로 앞 index를 가져옴"
            "surface_limit의 의미 : 표적관리를 몇개까지 할 것인가?"
            target_idx = target_id - len_limit - 1  # 대공 표적에 대한 index는 no_ops, 그다음 self.surface_limit다 채우고 시작
            #print([cal_distance(self, tar) for tar in self.ssm_detections])
            target = self.ssm_detections[target_idx]
            "무장 운용 우선순위"
            "해당 거리 대에 최적화된 무장을 운용"
            "i.e. LSAM은 MSAM.max < 표적거리 <= LSAM.max 최적화"
            "     MSAM은 CIWS.max < 표적거리 <= MSAM.max 최적화"
            "     CIWS는        0 < 표적거리 <= CIWS.max 최적화"
            d = cal_distance(self, target)
            if (d <= self.l_sam_max_range) and (d > self.m_sam_max_range):
                idle_l_sam = [missile for missile in self.l_sam_launcher if missile.status == 'idle']
                if len(idle_l_sam) >= 1:
                    missile = idle_l_sam[0]

                    missile.target = target
                    missile.original_target = target

                    missile.status = 'target_allocated'
                    launching_time = self.env.now + np.random.uniform(self.lsam_launching_duration_min,
                                                                      self.lsam_launching_duration_max)  # 발사 소요시간
                    self.air_prelaunching_managing_list.append([launching_time, missile, target])
                    self.air_engagement_managing_list.append([missile, target])
            if (d <= self.m_sam_max_range) and (d > self.ciws_max_range):
                idle_m_sam = [missile for missile in self.m_sam_launcher if missile.status == 'idle']
                if len(idle_m_sam) >= 1:
                    missile = idle_m_sam[0]
                else:
                    idle_l_sam = [missile for missile in self.l_sam_launcher if missile.status == 'idle']

                    missile = idle_l_sam[0]

                missile.target = target
                missile.original_target = target

                missile.status = 'target_allocated'
                launching_time = self.env.now + np.random.uniform(self.msam_launching_duration_min,
                                                                  self.msam_launching_duration_min)  # 발사 소요시간
                self.air_prelaunching_managing_list.append([launching_time, missile, target])
                self.air_engagement_managing_list.append([missile, target])
            if (d <= self.ciws_max_range):
                self.CIWS.target = target
                self.CIWS.original_target = target




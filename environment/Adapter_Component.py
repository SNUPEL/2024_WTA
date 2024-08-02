import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import copy

class Adapter:
    def __init__(self, input_path, mode = 'excel', polar_chart=[33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 50], polar_chart_visualize = True):
        """
        Preprocessing the input data

        :param input_path:
        :param mode:
        :param polar_chart:
        :param polar_chart_visualize:
        """

        self.mode = mode
        self.polar_chart = polar_chart
        self.ship_data, self.SAM_data, self.SSM_data, self.patrol_aircraft_data, self.inception_data = self.preprocessing(input_path)
        self.get_rcs = self.interpolating_rcs(polar_chart_visualize)

    def preprocessing(self, input_path):
        if self.mode != 'txt':
            ship_data = pd.read_excel(input_path, sheet_name='ship')
            ship_data = ship_data.set_index('id')
            ship_data = ship_data.transpose()
            ship_dict = ship_data.to_dict(orient='index')
            for key,value in ship_dict.items():
                value['number'] = key




            inception_data = pd.read_excel(input_path, sheet_name='inception')
            inception_data = inception_data.set_index('parameter')
            inception_data = inception_data.transpose()
            inception_dict = inception_data.to_dict(orient='index')['number']



            SAM_data = pd.read_excel(input_path, sheet_name='SAM')
            SAM_data = SAM_data.set_index('type')
            SAM_data = SAM_data.transpose()
            SAM_dict = SAM_data.to_dict(orient='index')

            SSM_data = pd.read_excel(input_path, sheet_name='SSM')
            SSM_data = SSM_data.set_index('type')
            SSM_data = SSM_data.transpose()
            SSM_dict = SSM_data.to_dict(orient='index')

            patrol_aircraft_data = pd.read_excel(input_path, sheet_name='patrol_aircraft')
            patrol_aircraft_data = patrol_aircraft_data.set_index('id')
            patrol_aircraft_data = patrol_aircraft_data.transpose()
            patrol_aircraft_dict = patrol_aircraft_data.to_dict(orient='index')


        else:
            ship_data = pd.read_csv(input_path[0], delimiter='\s+', index_col=False)
            ship_data = ship_data.set_index('id')
            ship_data = ship_data.transpose()
            ship_data = ship_data.astype({'speed':'float64', 'course':'float64', 'surface_tracking_limit':'int64',
       'surface_engagement_limit':'int64', 'air_tracking_limit':'int64',
       'air_engagement_limit':'int64', 'evading_course':'float64', 'type_m_sam':'int64', 'num_m_sam':'int64',
       'type_l_sam':'int64', 'num_l_sam':'int64', 'type_ssm':'int64', 'num_ssm':'int64',
       'decoy_launching_bearing':'float64', 'decoy_launching_distance':'float64',
       'decoy_launching_interval':'float64', 'decoy_rcs':'float64', 'decoy_duration':'float64',
       'decoy_decaying_rate':'float64', 'length':'float64', 'breadth':'float64', 'height':'float64', 'detection_range':'float64',
       'radar_peak_power':'float64', 'antenna_gain_factor':'float64', 'wavelength_of_signal':'float64',
       'radar_receiver_bandwidth':'float64', 'ciws_max_range':'float64', 'ciws_max_num_per_min':'float64',
       'ciws_bullet_capacity':'float64', 'ssm_launching_duration_min':'float64',
       'ssm_launching_duration_max':'float64', 'lsam_launching_duration_min':'float64',
       'lsam_launching_duration_max':'float64', 'msam_launching_duration_min':'float64',
       'msam_launching_duration_max':'float64', 'side':'string'})
            ship_dict = ship_data.to_dict(orient='index')
            for key in list(ship_dict.keys()):
                name = int(key)
                ship_dict[name] = ship_dict.pop(key)



            patrol_aircraft_data = pd.read_csv(input_path[1], delimiter='\s+', index_col=False)
            patrol_aircraft_data = patrol_aircraft_data.set_index('id')
            patrol_aircraft_data = patrol_aircraft_data.transpose()
            patrol_aircraft_data = patrol_aircraft_data.astype({'speed':'float64', 'course':'float64', 'radius':'float64', 'side':'string'})
            patrol_aircraft_dict = patrol_aircraft_data.to_dict(orient='index')

            for key in list(patrol_aircraft_dict.keys()):
                name = int(key)
                patrol_aircraft_dict[name] = patrol_aircraft_dict.pop(key)

            SAM_data = pd.read_csv(input_path[2], delimiter='\s+', index_col=False)
            SAM_data = SAM_data.set_index('type')
            SAM_data = SAM_data.transpose()
            SAM_data = SAM_data.astype(
                {'speed':'float64', 'attack_range':'float64', 'beam_width':'float64', 'angular_velocity':'float64',
       'rotation_range':'float64', 'range':'float64', 'length':'float64', 'seeker_on_distance':'float64', 'p_h':'float64',
       'radar_peak_power':'float64', 'antenna_gain_factor':'float64', 'wavelength_of_signal':'float64',
       'radar_receiver_bandwidth':'float64', 'radius':'float64', 'cla':'string'})
            SAM_dict = SAM_data.to_dict(orient='index'
                                        )
            for key in list(SAM_dict.keys()):
                name = int(key)
                SAM_dict[name] = SAM_dict.pop(key)


            SSM_data = pd.read_csv(input_path[3], delimiter='\s+', index_col=False)
            SSM_data = SSM_data.set_index('type')
            SSM_data = SSM_data.transpose()
            SSM_data = SSM_data.astype(
                {'speed': 'float64', 'attack_range': 'float64', 'beam_width': 'float64', 'angular_velocity': 'float64',
                 'rotation_range': 'float64', 'range': 'float64', 'length': 'float64', 'seeker_on_distance': 'float64',
                 'p_h': 'float64',
                 'radar_peak_power': 'float64', 'antenna_gain_factor': 'float64', 'wavelength_of_signal': 'float64',
                 'radar_receiver_bandwidth': 'float64', 'radius': 'float64', 'cla': 'string'})
            SSM_dict = SSM_data.to_dict(orient='index')
            for key in list(SSM_dict.keys()):
                name = int(key)
                SSM_dict[name] = SSM_dict.pop(key)


            inception_data = pd.read_csv(input_path[4], delimiter='\s+')

            inception_data = inception_data.squeeze()
            inception_data = inception_data.set_index('parameter')
            inception_data = inception_data.transpose()
            inception_data = inception_data.astype(
                {'inception_distance': 'float64',
                 'inception_angle': 'float64',
                 'enemy_spacing_mean': 'float64',
                 'enemy_spacing_std': 'float64'})


            inception_dict = inception_data.to_dict(orient='index')['number']


        return ship_dict, SAM_dict, SSM_dict, patrol_aircraft_dict, inception_dict

    def interpolating_rcs(self, polar_chart_visualize):
        """
        Interpolating the RCS

        :param polar_chart_visualize:
        :return:
        """
        polar_chart = copy.deepcopy(self.polar_chart)
        for i in range(len(polar_chart) - 1, 0, -1):
            polar_chart.append(polar_chart[i])
        x = list()
        for n in range(0, len(polar_chart)):
            x.append(2 * np.pi * n / len(polar_chart))
        x = np.array(x)
        y = np.array(polar_chart)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        N = 100
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, N)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        if polar_chart_visualize == True:
            plt.plot(xx, spline(xx), 'r', label='BSpline')
            plt.grid()
            plt.legend(loc='best')
            plt.show()
        return spline
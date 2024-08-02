import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
postprocessing 2022.1월 버젼

"""
# def recording(suceptibility, win_tag, env, records):
#     records.append([suceptibility,
#                     win_tag,
#                     env.friendlies_fixed_list[0].monitors['ssm_destroying_from_lsam'],
#                     env.friendlies_fixed_list[0].monitors['ssm_destroying_from_msam'],
#                     env.friendlies_fixed_list[0].monitors['ssm_destroying_from_ciws'],
#                     env.friendlies_fixed_list[0].monitors['ssm_destroying_on_decoy'],
#                     env.friendlies_fixed_list[0].monitors['ssm_mishit'],
#                     env.friendlies_fixed_list[0].monitors['ssm_decoying'],
#                     env.friendlies_fixed_list[0].monitors['enemy_launching'],
#                     env.friendlies_fixed_list[0].monitors['lsam_launching'],
#                     env.friendlies_fixed_list[0].monitors['msam_launching']
#                     ])

# def postprocessing(records, episode):
#     df = pd.DataFrame(records)
#     df.columns = ['lose',
#                   'win_tag', 'ssm_destroying_from_lsam', 'ssm_destroying_from_msam',
#                   'ssm_destroying_from_ciws', 'ssm_destroying_on_decoy', 'ssm_mishit', 'ssm_decoying', 'enemy_launching', 'lsam_launching', 'msam_launching']
#
#     average_lose_ratio = df.loc[:, 'lose'].sum()/len(df)
#     average_reactive_fail_ratio = df.loc[:, 'lose'].sum()/(df.loc[:, 'lose'].sum()+df.loc[:, 'ssm_destroying_from_lsam'].sum()+df.loc[:, 'ssm_destroying_from_msam'].sum()+df.loc[:, 'ssm_destroying_from_ciws'].sum()+df.loc[:, 'ssm_destroying_on_decoy'].sum())
#     average_lsam_reactive_ratio = df.loc[:, 'ssm_destroying_from_lsam'].sum() / (df.loc[:, 'ssm_destroying_from_lsam'].sum() + df.loc[:,'ssm_destroying_from_msam'].sum() + df.loc[:,'ssm_destroying_from_ciws'].sum() + df.loc[:,'ssm_destroying_on_decoy'].sum())
#     average_msam_reactive_ratio = df.loc[:, 'ssm_destroying_from_msam'].sum() / (df.loc[:, 'ssm_destroying_from_lsam'].sum() + df.loc[:, 'ssm_destroying_from_msam'].sum() + df.loc[:,'ssm_destroying_from_ciws'].sum() + df.loc[:,'ssm_destroying_on_decoy'].sum())
#     average_ciws_reactive_ratio = df.loc[:, 'ssm_destroying_from_ciws'].sum() / (df.loc[:, 'ssm_destroying_from_lsam'].sum() + df.loc[:, 'ssm_destroying_from_msam'].sum() + df.loc[:,'ssm_destroying_from_ciws'].sum() + df.loc[:,'ssm_destroying_on_decoy'].sum())
#     average_decoy_reactive_ratio = df.loc[:, 'ssm_destroying_on_decoy'].sum() / (df.loc[:, 'ssm_destroying_from_lsam'].sum() + df.loc[:, 'ssm_destroying_from_msam'].sum() + df.loc[:,'ssm_destroying_from_ciws'].sum() + df.loc[:,'ssm_destroying_on_decoy'].sum())
#     average_interception_per_enemy_launching_ratio = (df.loc[:, 'ssm_destroying_from_lsam'].sum() + df.loc[:,'ssm_destroying_from_msam'].sum() + df.loc[:,'ssm_destroying_from_ciws'].sum() + df.loc[:,'ssm_destroying_on_decoy'].sum())/df.loc[:, "enemy_launching"].sum()
#     average_non_interception_per_enemy_launching_ratio = (df.loc[:, "enemy_launching"].sum()-(df.loc[:, 'ssm_destroying_from_lsam'].sum() + df.loc[:,'ssm_destroying_from_msam'].sum() + df.loc[:,'ssm_destroying_from_ciws'].sum() + df.loc[:,'ssm_destroying_on_decoy'].sum()))/df.loc[:, "enemy_launching"].sum()
#     average_lsam_success_ratio = df.loc[:, 'ssm_destroying_from_lsam'].sum() / df.loc[:, 'lsam_launching'].sum()
#     average_msam_success_ratio = df.loc[:, 'ssm_destroying_from_msam'].sum() / df.loc[:, 'msam_launching'].sum()
#
#     f = open("Result\summary_{}.txt".format(episode), 'w')
#     line1 = "lose ratio : {}\n".format(average_lose_ratio)
#     line2 = "요격 성공 ratio : {}\n".format(average_reactive_fail_ratio)
#     line3 = "요격 실패 ratio : {}\n".format(1-average_reactive_fail_ratio)
#     line4 = "성공 요격 중 LSAM의 비율 : {}\n".format(average_lsam_reactive_ratio)
#     line5 = "성공 요격 중 MSAM의 비율 : {}\n".format(average_msam_reactive_ratio)
#     line6 = "성공 요격 중 CIWS의 비율 : {}\n".format(average_ciws_reactive_ratio)
#     line7 = "성공 요격 중 DECOY의 비율 : {}\n".format(average_decoy_reactive_ratio)
#     line8 = "적 발사 탄중 대응 후 잔여율 : {}\n".format(average_non_interception_per_enemy_launching_ratio)
#     line9 = "LSAM 요격 성공률 : {}\n".format(average_lsam_success_ratio)
#     line10 = "MSAM 요격 성공률 : {}\n".format(average_msam_success_ratio)
#
#     f.write(line1)
#     f.write(line2)
#     f.write(line3)
#     f.write(line4)
#     f.write(line5)
#     f.write(line6)
#     f.write(line7)
#     f.write(line8)
#     f.write(line9)
#     f.write(line10)
#     print("===================={} reporter====================".format(episode))
#     print("lose ratio : ", average_lose_ratio)
#     print("요격 성공 ratio : ", average_reactive_fail_ratio)
#     print("요격 실패 ratio : ", 1-average_reactive_fail_ratio)
#     print("성공 요격 중 LSAM의 비율 : ", average_lsam_reactive_ratio)
#     print("성공 요격 중 MSAM의 비율 : ", average_msam_reactive_ratio)
#     print("성공 요격 중 CIWS의 비율 : ", average_ciws_reactive_ratio)
#     print("성공 요격 중 DECOY의 비율 : ", average_decoy_reactive_ratio)
#     print("적 발사 탄중 대응 후 잔여율 : ", average_non_interception_per_enemy_launching_ratio)
#     print("LSAM 요격 성공률 : ",average_lsam_success_ratio)
#     print("MSAM 요격 성공률 : ", average_msam_success_ratio)
#     print("=====================================================")
#
#     df.to_csv(r'Result\result_{}.csv'.format(episode))
#     df.to_csv(r'Result\result_{}.txt'.format(episode), header=None, index=None, sep=' ', mode='a')
#
#     return average_lose_ratio, average_reactive_fail_ratio, average_non_interception_per_enemy_launching_ratio
#
# def visualization(scenarios, lose_ratio, reactive_fail_ratio, non_interception_per_enemy_launching_ratio):
#     x = np.arange(len(scenarios))
#     width = 0.25
#     x = np.arange(len(scenarios))
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x +0.00, lose_ratio, width=width, label='lose_ratio')
#     rects2 = ax.bar(x +0.25, reactive_fail_ratio, width=width, label='reactive_fail_ratio')
#     rects3 = ax.bar(x +0.5, non_interception_per_enemy_launching_ratio, width=width, label='non_interception_per_enemy_launching_ratio')
#
#     ax.set_xticks(x, scenarios)
#     # ax.legend()
#     ax.bar_label(rects1, padding=3)
#     ax.bar_label(rects2, padding=3)
#     ax.bar_label(rects3, padding=3)
#
#     fig.tight_layout()
#     plt.show()
#

def recording(suceptibility, win_tag, env, records):
    records.append([suceptibility,
                    win_tag,
                    env.friendlies_fixed_list[0].monitors['ssm_destroying_from_lsam'],
                    env.friendlies_fixed_list[0].monitors['ssm_destroying_from_msam'],
                    env.friendlies_fixed_list[0].monitors['ssm_destroying_from_ciws'],
                    env.friendlies_fixed_list[0].monitors['ssm_destroying_on_decoy'],
                    env.friendlies_fixed_list[0].monitors['ssm_mishit'],
                    env.friendlies_fixed_list[0].monitors['enemy_flying_ssms'],
                    env.friendlies_fixed_list[0].monitors['enemy_remains'],
                    env.friendlies_fixed_list[0].monitors['enemy_num_ssm']])
                    # sum([ship.monitors['enemy_remains'] for ship in env.enemies_fixed_list]),
                    # sum([ship.monitors['enemy_num_ssm'] for ship in env.enemies_fixed_list])])

def postprocessing(records, episode):
    df = pd.DataFrame(records)
    df.columns = ['lose',
                  'win_tag',
                  'ssm_destroying_from_lsam',
                  'ssm_destroying_from_msam',
                  'ssm_destroying_from_ciws',
                  'ssm_destroying_on_decoy',
                  'ssm_mishit',
                  'enemy_flying_ssms',
                  'enemy_remains',
                  'enemy_num_ssm'
                  ]
    df.to_csv('{}.csv'.format(episode))
    average_lose_ratio = df.loc[:, 'lose'].sum()/len(df)
    average_remains_ratio = (df.loc[:, 'enemy_remains'].sum()+df.loc[:, 'enemy_flying_ssms'].sum()) / (df.loc[:,'enemy_num_ssm'].sum())



    f = open("Result\summary_{}.txt".format(episode), 'w')
    line1 = "lose ratio : {}\n".format(average_lose_ratio)
    line2 = "remain ratio : {}\n".format(average_remains_ratio)


    f.write(line1)
    f.write(line2)


    print("===================={} reporter====================".format(episode))
    print("lose ratio : ", average_lose_ratio)
    print("요격 성공 ratio : ", average_remains_ratio)
    print("=====================================================")

    df.to_csv(r'Result\result_{}.csv'.format(episode))
    df.to_csv(r'Result\result_{}.txt'.format(episode), header=None, index=None, sep=' ', mode='a')



    print(average_lose_ratio, average_remains_ratio)
    return average_lose_ratio, average_remains_ratio

def visualization(scenarios, lose_ratio, average_remains_ratio):
    x = np.arange(len(scenarios))
    width = 0.25
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots()
    rects1 = ax.bar(x +0.00, lose_ratio, width=width, label='lose_ratio')
    rects2 = ax.bar(x +0.25, average_remains_ratio, width=width, label='average_remains_ratio')
    ax.set_xticks(x, scenarios)
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.show()
#


from utils.factory import Factory

"""
Question 1.2
各园区分别配置 50kW/100kWh 储能，制定储能最优运行策略及购电计划，分析各园区运行经济性是否改善，并解释其原因；
"""


Factory_A = Factory('A', wind_power_list=[
    0 for _ in range(24)
], solar_power_list=[
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    4.3500,
    226.9500,
    451.5000,
    578.3250,
    641.6250,
    639.8250,
    588.1500,
    482.7750,
    318.1500,
    46.4250,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000
], power_demand_list=[
    275,
    275,
    277,
    310,
    310,
    293,
    293,
    380,
    375,
    281,
    447,
    447,
    447,
    405,
    404,
    403,
    268,
    313,
    287,
    288,
    284,
    287,
    277,
    275
], battery_power=50, battery_capacity=100)
print(Factory_A.calculate_cost_with_battery())


Factory_B = Factory('B', wind_power_list=[
    230.1000,
    382.8000,
    296.8000,
    444.4000,
    502.9000,
    360.9000,
    240.2000,
    47.3000,
    153.8000,
    106.8000,
    51.8000,
    216.9000,
    354.6000,
    219.4000,
    111.0000,
    218.6000,
    377.9000,
    342.1000,
    500.8000,
    464.6000,
    219.7000,
    178.3000,
    153.5000,
    0.0000
], solar_power_list=[
    0 for _ in range(24)
], power_demand_list=[
    241,
    253,
    329,
    315,
    290,
    270,
    307,
    354,
    264,
    315,
    313,
    291,
    360,
    369,
    389,
    419,
    412,
    291,
    379,
    303,
    331,
    306,
    285,
    324
], battery_power=50, battery_capacity=100)
print(Factory_B.calculate_cost_with_battery())

Factory_C = Factory('C', wind_power_list=[
    73.2000,
    108.7500,
    197.9500,
    91.5500,
    235.8000,
    310.7500,
    147.3000,
    60.7000,
    12.5000,
    151.1500,
    9.8000,
    61.2000,
    166.7500,
    132.6500,
    61.0000,
    81.6500,
    132.2500,
    170.4000,
    159.1500,
    164.9500,
    85.1500,
    82.7500,
    94.8500,
    116.1500
], solar_power_list=[
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    6.3000,
    196.8000,
    378.8400,
    476.1600,
    535.5000,
    539.9400,
    493.2600,
    400.0200,
    256.5000,
    12.9600,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000
], power_demand_list=[
    302,
    292,
    307,
    293,
    271,
    252,
    283,
    223,
    292,
    283,
    287,
    362,
    446,
    504,
    455,
    506,
    283,
    311,
    418,
    223,
    229,
    361,
    302,
    291
], battery_power=50, battery_capacity=100)
print(Factory_C.calculate_cost_with_battery())
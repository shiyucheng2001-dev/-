import numpy as np
import pandas as pd
import streamlit as st
from math import exp, factorial
from scipy.stats import nbinom 

class FootballPoissonPredictor:
    def __init__(self):
        # 确保所有联赛数据都包含overdispersion参数
        self.league_data = {
            '英超': {'home_goal_rate': 1.65, 'away_goal_rate': 1.5076, 'overdispersion': 1.25},
            '西甲': {'home_goal_rate': 1.5376, 'away_goal_rate': 1.2165, 'overdispersion': 1.30},
            '意甲': {'home_goal_rate': 1.4518, 'away_goal_rate': 1.2718, 'overdispersion': 1.28},
            '法甲': {'home_goal_rate': 1.65875, 'away_goal_rate': 1.46313, 'overdispersion': 1.35},
            '德甲': {'home_goal_rate': 1.688125, 'away_goal_rate': 1.538125, 'overdispersion': 1.20},
            '英冠': {'home_goal_rate': 1.454286, 'away_goal_rate': 1.047619, 'overdispersion': 1.40},
            '德乙': {'home_goal_rate': 1.7125, 'away_goal_rate': 1.47125, 'overdispersion': 1.45},
            '法乙': {'home_goal_rate': 1.5925, 'away_goal_rate': 1.089375, 'overdispersion': 1.50}
        }
       
        # 完整的球队数据
        self.team_data = {
            '英超': {
                '利物浦': {'home_goals': 2.21, 'home_conceded': 0.84, 'away_goals': 2.32, 'away_conceded': 1.32},
                '阿森纳': {'home_goals': 1.84, 'home_conceded': 0.89, 'away_goals': 1.79, 'away_conceded': 0.89},
                '曼城': {'home_goals': 2.26, 'home_conceded': 1.21, 'away_goals': 1.53, 'away_conceded': 1.11},
                '切尔西': {'home_goals': 1.84, 'home_conceded': 0.95, 'away_goals': 1.53, 'away_conceded': 1.32},
                '纽卡斯尔联': {'home_goals': 2.11, 'home_conceded': 1.05, 'away_goals': 1.47, 'away_conceded': 1.42},
                '阿斯顿维拉': {'home_goals': 1.79, 'home_conceded': 1.05, 'away_goals': 1.26, 'away_conceded': 1.63},
                '诺丁汉森林': {'home_goals': 1.37, 'home_conceded': 0.84, 'away_goals': 1.68, 'away_conceded': 1.58},
                '布莱顿': {'home_goals': 1.58, 'home_conceded': 1.37, 'away_goals': 1.89, 'away_conceded': 1.74},
                '伯恩茅斯': {'home_goals': 1.21, 'home_conceded': 0.84, 'away_goals': 1.84, 'away_conceded': 1.58},
                '布伦特福德': {'home_goals': 2.11, 'home_conceded': 1.84, 'away_goals': 1.37, 'away_conceded': 1.16},
                '富勒姆': {'home_goals': 1.42, 'home_conceded': 1.58, 'away_goals': 1.42, 'away_conceded': 1.26},
                '水晶宫': {'home_goals': 1.26, 'home_conceded': 1.37, 'away_goals': 1.42, 'away_conceded': 1.32},
                '埃弗顿': {'home_goals': 1.37, 'home_conceded': 1.21, 'away_goals': 0.84, 'away_conceded': 1.11},
                '西汉姆联': {'home_goals': 1.21, 'home_conceded': 1.79, 'away_goals': 1.21, 'away_conceded': 1.47},
                '曼联': {'home_goals': 1.21, 'home_conceded': 1.47, 'away_goals': 1.11, 'away_conceded': 1.37},
                '狼队': {'home_goals': 1.42, 'home_conceded': 1.68, 'away_goals': 1.42, 'away_conceded': 1.95},
                '热刺': {'home_goals': 1.84, 'home_conceded': 1.84, 'away_goals': 1.53, 'away_conceded': 1.58}
            },
            '西甲': {
                '巴塞罗那': {'home_goals': 2.74, 'home_conceded': 1.05, 'away_goals': 2.63, 'away_conceded': 1.00},
                '皇家马德里': {'home_goals': 2.37, 'home_conceded': 1.00, 'away_goals': 1.74, 'away_conceded': 1.00},
                '马德里竞技': {'home_goals': 2.21, 'home_conceded': 0.79, 'away_goals': 1.37, 'away_conceded': 0.79},
                '毕尔巴鄂竞技': {'home_goals': 1.68, 'home_conceded': 0.68, 'away_goals': 1.16, 'away_conceded': 0.84},
                '比利亚雷亚尔': {'home_goals': 2.26, 'home_conceded': 1.42, 'away_goals': 1.47, 'away_conceded': 1.26},
                '皇家贝蒂斯': {'home_goals': 1.68, 'home_conceded': 1.11, 'away_goals': 1.32, 'away_conceded': 1.53},
                '塞尔塔': {'home_goals': 1.68, 'home_conceded': 1.11, 'away_goals': 1.42, 'away_conceded': 1.89},
                '巴列卡诺': {'home_goals': 1.26, 'home_conceded': 1.37, 'away_goals': 0.89, 'away_conceded': 1.00},
                '奥萨苏纳': {'home_goals': 1.74, 'home_conceded': 1.21, 'away_goals': 0.79, 'away_conceded': 1.53},
                '马洛卡': {'home_goals': 1.05, 'home_conceded': 1.11, 'away_goals': 0.79, 'away_conceded': 1.21},
                '皇家社会': {'home_goals': 1.05, 'home_conceded': 1.05, 'away_goals': 0.79, 'away_conceded': 1.37},
                '巴伦西亚': {'home_goals': 1.37, 'home_conceded': 1.05, 'away_goals': 0.95, 'away_conceded': 1.79},
                '赫塔菲': {'home_goals': 0.79, 'home_conceded': 1.00, 'away_goals': 1.00, 'away_conceded': 1.05},
                '西班牙人': {'home_goals': 1.21, 'home_conceded': 1.05, 'away_goals': 0.89, 'away_conceded': 1.63},
                '阿拉维斯': {'home_goals': 0.74, 'home_conceded': 0.89, 'away_goals': 1.26, 'away_conceded': 1.63},
                '赫罗纳': {'home_goals': 1.42, 'home_conceded': 1.58, 'away_goals': 0.89, 'away_conceded': 1.58},
                '塞维利亚': {'home_goals': 0.89, 'home_conceded': 1.21, 'away_goals': 1.32, 'away_conceded': 1.68}
            },
            '意甲': {
                '那不勒斯': {'home_goals': 1.68, 'home_conceded': 0.68, 'away_goals': 1.42, 'away_conceded': 0.74},
                '国际米兰': {'home_goals': 2.11, 'home_conceded': 1.11, 'away_goals': 2.05, 'away_conceded': 0.74},
                '亚特兰大': {'home_goals': 1.89, 'home_conceded': 1.26, 'away_goals': 2.21, 'away_conceded': 0.68},
                '尤文图斯': {'home_goals': 1.63, 'home_conceded': 0.79, 'away_goals': 1.42, 'away_conceded': 1.05},
                '罗马': {'home_goals': 1.95, 'home_conceded': 0.79, 'away_goals': 1.00, 'away_conceded': 1.05},
                '佛罗伦萨': {'home_goals': 1.68, 'home_conceded': 0.95, 'away_goals': 1.47, 'away_conceded': 1.21},
                '拉齐奥': {'home_goals': 1.74, 'home_conceded': 1.37, 'away_goals': 1.47, 'away_conceded': 1.21},
                'AC米兰': {'home_goals': 1.58, 'home_conceded': 0.84, 'away_goals': 1.63, 'away_conceded': 1.42},
                '博洛尼亚': {'home_goals': 1.74, 'home_conceded': 0.95, 'away_goals': 1.26, 'away_conceded': 1.53},
                '科莫': {'home_goals': 1.47, 'home_conceded': 1.37, 'away_goals': 1.11, 'away_conceded': 1.37},
                '都灵': {'home_goals': 0.89, 'home_conceded': 0.95, 'away_goals': 1.16, 'away_conceded': 1.42},
                '乌迪内斯': {'home_goals': 1.16, 'home_conceded': 1.42, 'away_goals': 1.00, 'away_conceded': 1.53},
                '热那亚': {'home_goals': 1.11, 'home_conceded': 1.32, 'away_goals': 0.84, 'away_conceded': 1.26},
                '维罗纳': {'home_goals': 0.79, 'home_conceded': 1.89, 'away_goals': 1.00, 'away_conceded': 1.58},
                '卡利亚里': {'home_goals': 1.26, 'home_conceded': 1.47, 'away_goals': 0.84, 'away_conceded': 1.47},
                '帕尔马': {'home_goals': 1.32, 'home_conceded': 1.47, 'away_goals': 1.00, 'away_conceded': 1.58},
                '莱切': {'home_goals': 0.68, 'home_conceded': 1.63, 'away_goals': 0.74, 'away_conceded': 1.42}
            },
            '法甲': {
                '巴黎圣日耳曼': {'home_goals': 2.65, 'home_conceded': 0.94, 'away_goals': 2.76, 'away_conceded': 1.12},
                '马赛': {'home_goals': 2.41, 'home_conceded': 1.35, 'away_goals': 1.94, 'away_conceded': 1.41},
                '摩纳哥': {'home_goals': 2.24, 'home_conceded': 0.94, 'away_goals': 1.47, 'away_conceded': 1.47},
                '尼斯': {'home_goals': 2.24, 'home_conceded': 0.88, 'away_goals': 1.65, 'away_conceded': 1.53},
                '里尔': {'home_goals': 1.82, 'home_conceded': 1.06, 'away_goals': 1.24, 'away_conceded': 1.06},
                '里昂': {'home_goals': 2.18, 'home_conceded': 1.24, 'away_goals': 1.65, 'away_conceded': 1.47},
                '斯特拉斯堡': {'home_goals': 1.94, 'home_conceded': 1.18, 'away_goals': 1.35, 'away_conceded': 1.41},
                '朗斯': {'home_goals': 1.12, 'home_conceded': 1.29, 'away_goals': 1.35, 'away_conceded': 1.00},
                '布雷斯特': {'home_goals': 1.82, 'home_conceded': 1.24, 'away_goals': 1.24, 'away_conceded': 2.24},
                '图卢兹': {'home_goals': 1.18, 'home_conceded': 1.29, 'away_goals': 1.41, 'away_conceded': 1.24},
                '欧塞尔': {'home_goals': 1.41, 'home_conceded': 1.00, 'away_goals': 1.41, 'away_conceded': 2.00},
                '雷恩': {'home_goals': 1.47, 'home_conceded': 1.00, 'away_goals': 1.53, 'away_conceded': 1.94},
                '南特': {'home_goals': 1.12, 'home_conceded': 1.06, 'away_goals': 1.18, 'away_conceded': 2.00},
                '昂热': {'home_goals': 1.06, 'home_conceded': 1.94, 'away_goals': 0.82, 'away_conceded': 1.18},
                '勒阿弗尔': {'home_goals': 0.88, 'home_conceded': 2.41, 'away_goals': 1.47, 'away_conceded': 1.76},
                '兰斯': {'home_goals': 1.00, 'home_conceded': 1.47, 'away_goals': 0.94, 'away_conceded': 1.29}
            },
            '德甲': {
                '拜仁慕尼黑': {'home_goals': 3.12, 'home_conceded': 0.94, 'away_goals': 2.71, 'away_conceded': 0.94},
                '勒沃库森': {'home_goals': 2.12, 'home_conceded': 1.29, 'away_goals': 2.12, 'away_conceded': 1.24},
                '法兰克福': {'home_goals': 2.41, 'home_conceded': 1.29, 'away_goals': 1.59, 'away_conceded': 1.41},
                '多特蒙德': {'home_goals': 2.59, 'home_conceded': 1.12, 'away_goals': 1.59, 'away_conceded': 1.88},
                '弗莱堡': {'home_goals': 1.88, 'home_conceded': 1.53, 'away_goals': 1.00, 'away_conceded': 1.59},
                '美因茨': {'home_goals': 1.41, 'home_conceded': 1.06, 'away_goals': 1.82, 'away_conceded': 1.47},
                'RB莱比锡': {'home_goals': 1.94, 'home_conceded': 1.35, 'away_goals': 1.18, 'away_conceded': 1.47},
                '云达不莱梅': {'home_goals': 1.24, 'home_conceded': 1.53, 'away_goals': 1.94, 'away_conceded': 1.82},
                '斯图加特': {'home_goals': 2.06, 'home_conceded': 1.59, 'away_goals': 1.71, 'away_conceded': 1.53},
                '门兴格拉德巴赫': {'home_goals': 1.71, 'home_conceded': 1.53, 'away_goals': 1.53, 'away_conceded': 1.82},
                '沃尔夫斯堡': {'home_goals': 1.59, 'home_conceded': 1.76, 'away_goals': 1.71, 'away_conceded': 1.41},
                '奥格斯堡': {'home_goals': 1.06, 'home_conceded': 1.18, 'away_goals': 1.00, 'away_conceded': 1.82},
                '柏林联合': {'home_goals': 1.06, 'home_conceded': 1.35, 'away_goals': 1.00, 'away_conceded': 1.65},
                '圣保利': {'home_goals': 0.59, 'home_conceded': 1.12, 'away_goals': 1.06, 'away_conceded': 1.29},
                '霍芬海姆': {'home_goals': 1.47, 'home_conceded': 2.12, 'away_goals': 1.24, 'away_conceded': 1.88},
                '海登海姆': {'home_goals': 0.76, 'home_conceded': 1.94, 'away_goals': 1.41, 'away_conceded': 1.82}
            },
            '英冠': {
                '利兹联': {'home_goals': 2.65, 'home_conceded': 0.52, 'away_goals': 1.48, 'away_conceded': 0.78},
                '伯恩利': {'home_goals': 1.52, 'home_conceded': 0.35, 'away_goals': 1.48, 'away_conceded': 0.35},
                '谢菲尔德联队': {'home_goals': 1.43, 'home_conceded': 0.74, 'away_goals': 1.30, 'away_conceded': 0.83},
                '桑德兰': {'home_goals': 1.39, 'home_conceded': 0.78, 'away_goals': 1.13, 'away_conceded': 1.13},
                '考文垂': {'home_goals': 1.74, 'home_conceded': 1.04, 'away_goals': 1.04, 'away_conceded': 1.48},
                '布里斯托城': {'home_goals': 1.57, 'home_conceded': 0.87, 'away_goals': 1.00, 'away_conceded': 1.52},
                '布莱克本': {'home_goals': 1.48, 'home_conceded': 1.00, 'away_goals': 0.89, 'away_conceded': 1.09},
                '米尔沃尔': {'home_goals': 1.17, 'home_conceded': 0.83, 'away_goals': 0.87, 'away_conceded': 1.30},
                '西布罗姆维奇': {'home_goals': 1.43, 'home_conceded': 0.87, 'away_goals': 1.04, 'away_conceded': 1.17},
                '米德尔斯堡': {'home_goals': 1.35, 'home_conceded': 1.00, 'away_goals': 1.43, 'away_conceded': 1.43},
                '斯旺西': {'home_goals': 1.43, 'home_conceded': 1.04, 'away_goals': 0.78, 'away_conceded': 1.39},
                '谢周三': {'home_goals': 1.30, 'home_conceded': 1.39, 'away_goals': 1.30, 'away_conceded': 1.61},
                '诺维奇': {'home_goals': 2.26, 'home_conceded': 1.48, 'away_goals': 0.83, 'away_conceded': 1.48},
                '沃特福德': {'home_goals': 1.17, 'home_conceded': 0.96, 'away_goals': 1.13, 'away_conceded': 1.70},
                '女王公园巡游者': {'home_goals': 1.35, 'home_conceded': 1.48, 'away_goals': 0.96, 'away_conceded': 1.26},
                '朴茨茅斯': {'home_goals': 1.43, 'home_conceded': 0.91, 'away_goals': 1.09, 'away_conceded': 2.17},
                '牛津联队': {'home_goals': 1.35, 'home_conceded': 1.26, 'away_goals': 0.78, 'away_conceded': 1.57},
                '斯托克城': {'home_goals': 1.26, 'home_conceded': 1.09, 'away_goals': 0.70, 'away_conceded': 1.61},
                '德比郡': {'home_goals': 1.09, 'home_conceded': 0.78, 'away_goals': 1.00, 'away_conceded': 1.65},
                '普雷斯顿': {'home_goals': 1.13, 'home_conceded': 0.96, 'away_goals': 0.96, 'away_conceded': 1.61},
                '赫尔城': {'home_goals': 1.04, 'home_conceded': 1.22, 'away_goals': 0.87, 'away_conceded': 1.13}
            },
            '德乙': {
                '科隆': {'home_goals': 1.94, 'home_conceded': 1.06, 'away_goals': 1.18, 'away_conceded': 1.18},
                '汉堡': {'home_goals': 2.59, 'home_conceded': 1.18, 'away_goals': 2.00, 'away_conceded': 1.41},
                '埃弗斯堡': {'home_goals': 2.29, 'home_conceded': 1.47, 'away_goals': 1.47, 'away_conceded': 0.71},
                '帕德博恩': {'home_goals': 1.76, 'home_conceded': 1.35, 'away_goals': 1.53, 'away_conceded': 1.35},
                '马格德堡': {'home_goals': 1.65, 'home_conceded': 1.71, 'away_goals': 2.12, 'away_conceded': 1.35},
                '杜塞尔多夫': {'home_goals': 1.71, 'home_conceded': 1.59, 'away_goals': 1.65, 'away_conceded': 1.47},
                '凯泽斯劳滕': {'home_goals': 2.00, 'home_conceded': 1.29, 'away_goals': 1.29, 'away_conceded': 1.94},
                '卡尔斯鲁厄': {'home_goals': 1.82, 'home_conceded': 1.41, 'away_goals': 1.53, 'away_conceded': 1.82},
                '汉威诺96': {'home_goals': 1.35, 'home_conceded': 0.88, 'away_goals': 1.06, 'away_conceded': 1.24},
                '纽伦堡': {'home_goals': 1.76, 'home_conceded': 1.65, 'away_goals': 1.76, 'away_conceded': 1.71},
                '柏林赫塔': {'home_goals': 1.18, 'home_conceded': 1.41, 'away_goals': 1.71, 'away_conceded': 1.59},
                '达姆斯塔特': {'home_goals': 1.71, 'home_conceded': 1.18, 'away_goals': 1.59, 'away_conceded': 2.06},
                '菲尔特': {'home_goals': 1.53, 'home_conceded': 1.76, 'away_goals': 1.12, 'away_conceded': 1.71},
                '沙克尔04': {'home_goals': 1.82, 'home_conceded': 2.06, 'away_goals': 1.24, 'away_conceded': 1.59},
                '普鲁士明斯特': {'home_goals': 1.00, 'home_conceded': 1.00, 'away_goals': 1.35, 'away_conceded': 1.53},
                '布伦瑞克': {'home_goals': 1.29, 'home_conceded': 1.53, 'away_goals': 0.94, 'away_conceded': 2.24}
            },
            '法乙': {
                '梅斯': {'home_goals': 2.18, 'home_conceded': 0.88, 'away_goals': 1.59, 'away_conceded': 1.12},
                '巴黎足球会': {'home_goals': 2.00, 'home_conceded': 0.94, 'away_goals': 1.24, 'away_conceded': 1.00},
                '洛里昂': {'home_goals': 2.88, 'home_conceded': 0.71, 'away_goals': 1.12, 'away_conceded': 1.12},
                '甘冈': {'home_goals': 1.94, 'home_conceded': 1.12, 'away_goals': 1.41, 'away_conceded': 1.53},
                'USL敦刻尔克': {'home_goals': 1.65, 'home_conceded': 0.82, 'away_goals': 1.12, 'away_conceded': 1.53},
                '阿纳西': {'home_goals': 1.41, 'home_conceded': 0.94, 'away_goals': 1.06, 'away_conceded': 1.59},
                '拉瓦勒': {'home_goals': 1.41, 'home_conceded': 1.06, 'away_goals': 1.18, 'away_conceded': 1.18},
                '巴斯蒂亚': {'home_goals': 1.71, 'home_conceded': 0.82, 'away_goals': 0.82, 'away_conceded': 1.35},
                '格勒诺布尔': {'home_goals': 1.47, 'home_conceded': 1.00, 'away_goals': 1.06, 'away_conceded': 1.59},
                '特鲁瓦': {'home_goals': 1.06, 'home_conceded': 0.76, 'away_goals': 1.06, 'away_conceded': 1.24},
                '亚眠': {'home_goals': 1.53, 'home_conceded': 1.18, 'away_goals': 0.71, 'away_conceded': 1.76},
                '阿雅克肖': {'home_goals': 1.29, 'home_conceded': 1.00, 'away_goals': 0.47, 'away_conceded': 1.47},
                '波城': {'home_goals': 1.18, 'home_conceded': 1.12, 'away_goals': 1.12, 'away_conceded': 2.00},
                '罗德兹': {'home_goals': 1.65, 'home_conceded': 1.53, 'away_goals': 1.65, 'away_conceded': 1.65},
                '红星': {'home_goals': 1.12, 'home_conceded': 1.29, 'away_goals': 1.06, 'away_conceded': 1.71},
                '克莱蒙': {'home_goals': 1.00, 'home_conceded': 1.12, 'away_goals': 0.76, 'away_conceded': 1.59}
            }
        }
    def get_teams_by_league(self, league):
        """获取指定联赛的所有球队"""
        if league in self.team_data:
            return list(self.team_data[league].keys())
        return []
    
    def calculate_expected_goals(self, home_team, away_team, league):
        """计算预期进球数"""
        if league not in self.league_data:
            raise ValueError(f"不支持该联赛: {league}")
        
        if league not in self.team_data or home_team not in self.team_data[league] or away_team not in self.team_data[league]:
            raise ValueError(f"球队数据不存在，请检查球队名称是否正确")
        
        league_rates = self.league_data[league]
        home_stats = self.team_data[league][home_team]
        away_stats = self.team_data[league][away_team]
        
        # 计算进攻强度和防守强度
        home_attack_strength = home_stats['home_goals'] / league_rates['home_goal_rate']
        home_defense_strength = home_stats['home_conceded'] / league_rates['away_goal_rate']
        
        away_attack_strength = away_stats['away_goals'] / league_rates['away_goal_rate']
        away_defense_strength = away_stats['away_conceded'] / league_rates['home_goal_rate']
        
        # 计算预期进球
        home_xG = home_attack_strength * away_defense_strength * league_rates['home_goal_rate']
        away_xG = away_attack_strength * home_defense_strength * league_rates['away_goal_rate']
        
        return home_xG, away_xG

    def monte_carlo_simulation(self, home_xG, away_xG, num_simulations=10000):
        """泊松分布蒙特卡洛模拟"""
        home_goals_sim = np.random.poisson(home_xG, num_simulations)
        away_goals_sim = np.random.poisson(away_xG, num_simulations)
        total_goals_sim = home_goals_sim + away_goals_sim
        return home_goals_sim, away_goals_sim, total_goals_sim

    def monte_carlo_simulation_negative_binomial(self, home_xG, away_xG, league, num_simulations=10000):
        """负二项分布蒙特卡洛模拟（按联赛调整）"""
        # 添加防御性编程，确保联赛存在
        if league not in self.league_data:
            raise ValueError(f"联赛 '{league}' 不在支持的联赛列表中")
            
        # 获取联赛特定的过离散参数，如果不存在则使用默认值1.3
        overdispersion = self.league_data.get(league, {}).get('overdispersion', 1.3)
        
        # 计算负二项分布参数
        def get_nbinom_params(mean, overdispersion):
            variance = mean * overdispersion
            p = mean / variance
            n = mean * p / (1 - p)
            return n, p
        
        n_home, p_home = get_nbinom_params(home_xG, overdispersion)
        n_away, p_away = get_nbinom_params(away_xG, overdispersion)
        
        # 模拟进球数
        home_goals_sim = nbinom.rvs(n_home, p_home, size=num_simulations)
        away_goals_sim = nbinom.rvs(n_away, p_away, size=num_simulations)
        total_goals_sim = home_goals_sim + away_goals_sim
        
        return home_goals_sim, away_goals_sim, total_goals_sim

    def calculate_probabilities_from_simulation(self, home_goals_sim, away_goals_sim, total_goals_sim, num_simulations):
        """从模拟结果计算概率"""
        unique_goals, counts = np.unique(total_goals_sim, return_counts=True)
        goal_probabilities = counts / num_simulations
        
        prob_0_1 = np.sum((total_goals_sim <= 1)) / num_simulations
        prob_2_3 = np.sum((total_goals_sim >= 2) & (total_goals_sim <= 3)) / num_simulations
        prob_4_6 = np.sum((total_goals_sim >= 4) & (total_goals_sim <= 6)) / num_simulations
        prob_7_plus = np.sum(total_goals_sim >= 7) / num_simulations
        prob_gt_2_5 = np.sum(total_goals_sim > 2.5) / num_simulations
        prob_gt_3_5 = np.sum(total_goals_sim > 3.5) / num_simulations
        
        most_common_goals = np.argmax(np.bincount(total_goals_sim))
        
        score_counts = {}
        for i in range(len(home_goals_sim)):
            score = f"{home_goals_sim[i]}-{away_goals_sim[i]}"
            score_counts[score] = score_counts.get(score, 0) + 1
        
        most_likely_score = max(score_counts, key=score_counts.get)
        most_likely_score_prob = score_counts[most_likely_score] / num_simulations
        
        return {
            'unique_goals': unique_goals,
            'goal_probabilities': goal_probabilities,
            'prob_0_1': prob_0_1,
            'prob_2_3': prob_2_3,
            'prob_4_6': prob_4_6,
            'prob_7_plus': prob_7_plus,
            'prob_gt_2_5': prob_gt_2_5,
            'prob_gt_3_5': prob_gt_3_5,
            'most_common_goals': most_common_goals,
            'most_likely_score': most_likely_score,
            'most_likely_score_prob': most_likely_score_prob
        }

def display_results(probs, num_simulations, distribution_name, league=None, total_goals_sim=None, home_team=None, away_team=None):
    """显示预测结果"""
    st.markdown(f"### {distribution_name}预测结果")
    
    if league and "负二项" in distribution_name:
        st.info(f"当前联赛 [{league}] 使用的过离散参数: {st.session_state.predictor.league_data[league]['overdispersion']}")
    
    # 第一行：关键指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("最有可能总进球数", f"{probs['most_common_goals']}球")
    with col2:
        st.metric("最有可能比分", probs['most_likely_score'], f"{probs['most_likely_score_prob']*100:.1f}%")
    with col3:
        st.metric("模拟次数", f"{num_simulations:,}")
    
    # 第二行：概率分布（两列布局）
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**总进球数概率分布**")
        st.metric("0-1球概率", f"{probs['prob_0_1']*100:.1f}%")
        st.metric("2-3球概率", f"{probs['prob_2_3']*100:.1f}%")
        st.metric("4-6球概率", f"{probs['prob_4_6']*100:.1f}%")
        st.metric("7+球概率", f"{probs['prob_7_plus']*100:.1f}%")
    
    with col_right:
        st.markdown("**进球数超过阈值概率**")
        st.metric("大于2.5球概率", f"{probs['prob_gt_2_5']*100:.1f}%")
        st.metric("大于3.5球概率", f"{probs['prob_gt_3_5']*100:.1f}%")
    
    # 图表和详细数据
    st.markdown("---")
    st.subheader("📈📈 详细概率分布")
    
    # 处理7+球的数据
    chart_data_list = []
    for goals, prob in zip(probs['unique_goals'], probs['goal_probabilities'] * 100):
        if goals <= 6:
            chart_data_list.append({'总进球数': goals, '概率(%)': prob})
        else:
            if not any(item['总进球数'] == '7+' for item in chart_data_list):
                prob_7_plus_total = probs['prob_7_plus'] * 100
                chart_data_list.append({'总进球数': '7+', '概率(%)': prob_7_plus_total})
    
    chart_data = pd.DataFrame(chart_data_list)
    
    try:
        import altair as alt
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('总进球数:O', title='总进球数', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('概率(%):Q', title='概率(%)'),
            tooltip=['总进球数', '概率(%)']
        ).properties(
            width=600,
            height=400,
            title=f'{distribution_name}总进球数概率分布'
        )
        st.altair_chart(chart, use_container_width=True)
    except ImportError:
        st.bar_chart(chart_data.set_index('总进球数'))
    
    st.subheader("详细概率分布表")
    detail_data = []
    for goals, prob in zip(probs['unique_goals'], probs['goal_probabilities'] * 100):
        if goals <= 6:
            detail_data.append({
                '总进球数': goals,
                '概率(%)': f"{prob:.2f}%",
                '模拟次数': np.sum(total_goals_sim == goals) if total_goals_sim is not None else 0
            })
        else:
            if not any(item['总进球数'] == '7+' for item in detail_data):
                count_7_plus = np.sum(total_goals_sim >= 7) if total_goals_sim is not None else 0
                detail_data.append({
                    '总进球数': '7+',
                    '概率(%)': f"{probs['prob_7_plus']*100:.2f}%",
                    '模拟次数': count_7_plus
                })
    
    detail_df = pd.DataFrame(detail_data)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    
    # 新增的比赛结果概率分析
    st.markdown("---")
    st.subheader("比赛结果概率分析")
    
    # 计算主胜、平局、客胜概率
    if home_team and away_team and 'home_goals_sim' in probs and 'away_goals_sim' in probs:
        home_wins = np.sum(probs['home_goals_sim'] > probs['away_goals_sim']) / num_simulations
        draws = np.sum(probs['home_goals_sim'] == probs['away_goals_sim']) / num_simulations
        away_wins = np.sum(probs['home_goals_sim'] < probs['away_goals_sim']) / num_simulations
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team}胜概率", f"{home_wins*100:.1f}%")
        with col2:
            st.metric("平局概率", f"{draws*100:.1f}%")
        with col3:
            st.metric(f"{away_team}胜概率", f"{away_wins*100:.1f}%")
        
        # 计算净胜球概率
        st.subheader("净胜球概率分析")
        goal_diffs = probs['home_goals_sim'] - probs['away_goals_sim']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team}净胜1球或以上", f"{np.sum(goal_diffs >= 1)/num_simulations*100:.1f}%")
        with col2:
            st.metric(f"{home_team}净胜2球或以上", f"{np.sum(goal_diffs >= 2)/num_simulations*100:.1f}%")
        with col3:
            st.metric(f"{home_team}净胜3球或以上", f"{np.sum(goal_diffs >= 3)/num_simulations*100:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{away_team}净胜1球或以上", f"{np.sum(goal_diffs <= -1)/num_simulations*100:.1f}%")
        with col2:
            st.metric(f"{away_team}净胜2球或以上", f"{np.sum(goal_diffs <= -2)/num_simulations*100:.1f}%")
        with col3:
            st.metric(f"{away_team}净胜3球或以上", f"{np.sum(goal_diffs <= -3)/num_simulations*100:.1f}%")
def main():
    st.set_page_config(page_title="足球蒙特卡洛预测器", page_icon="⚽⚽", layout="wide")
    st.title("⚽⚽ 足球比赛进球数预测器（蒙特卡洛模拟）")
    
    # 初始化预测器
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FootballPoissonPredictor()
    
    # 用户输入部分
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            league = st.selectbox("选择联赛", list(st.session_state.predictor.league_data.keys()))
        
        with col2:
            teams = st.session_state.predictor.get_teams_by_league(league)
            home_team = st.selectbox("选择主队", teams, index=0 if teams else 0)
        
        with col3:
            away_team = st.selectbox("选择客队", teams, index=1 if len(teams) > 1 else 0)
        
        num_simulations = st.slider("模拟次数", min_value=1000, max_value=100000, 
                                   value=10000, step=1000)
        
        if st.button("开始模拟预测", type="primary"):
            try:
                home_xG, away_xG = st.session_state.predictor.calculate_expected_goals(home_team, away_team, league)
                st.session_state.update({
                    'home_xG': home_xG,
                    'away_xG': away_xG,
                    'league': league,
                    'home_team': home_team,
                    'away_team': away_team,
                    'simulation_done': True
                })
            except ValueError as e:
                st.error(str(e))
        
        if hasattr(st.session_state, 'home_xG'):
            st.info(f"**预期进球:** 主队 {st.session_state.home_xG:.3f} | 客队 {st.session_state.away_xG:.3f}")
    
    # 创建分页
    tab1, tab2 = st.tabs(["泊松分布预测", "负二项分布预测"])
    
    # 泊松分布分页
    with tab1:
        if hasattr(st.session_state, 'simulation_done') and st.session_state.simulation_done:
            home_goals_sim, away_goals_sim, total_goals_sim = st.session_state.predictor.monte_carlo_simulation(
                st.session_state.home_xG, st.session_state.away_xG, num_simulations
            )
            
            probs = st.session_state.predictor.calculate_probabilities_from_simulation(
                home_goals_sim, away_goals_sim, total_goals_sim, num_simulations
            )
            probs['home_goals_sim'] = home_goals_sim
            probs['away_goals_sim'] = away_goals_sim
            
            display_results(
                probs, num_simulations, "泊松分布", 
                total_goals_sim=total_goals_sim,
                home_team=st.session_state.home_team,
                away_team=st.session_state.away_team
            )
    
    # 负二项分布分页
    with tab2:
        if hasattr(st.session_state, 'simulation_done') and st.session_state.simulation_done:
            home_goals_sim, away_goals_sim, total_goals_sim = st.session_state.predictor.monte_carlo_simulation_negative_binomial(
                st.session_state.home_xG, st.session_state.away_xG, st.session_state.league, num_simulations
            )
            
            probs = st.session_state.predictor.calculate_probabilities_from_simulation(
                home_goals_sim, away_goals_sim, total_goals_sim, num_simulations
            )
            probs['home_goals_sim'] = home_goals_sim
            probs['away_goals_sim'] = away_goals_sim
            
            display_results(
                probs, num_simulations, "负二项分布", 
                st.session_state.league, 
                total_goals_sim,
                st.session_state.home_team,
                st.session_state.away_team
            )

if __name__ == "__main__":
    main()
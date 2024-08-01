from typing import List, Tuple, Dict, Set, Optional, Union

# 判断某节点指标的用户数据应该分为哪一类
def user_data_classify(indicator_name:str, indicator_value:Union[str,int], classify_standard) -> str:
    
    return

# 判断节点状态是否属于异常
def is_abnormal(indicator_name:str, indicator_type:str) -> bool:
    if indicator_name in {"体重", "体脂率", "睡眠时长", "深睡比例", "清醒次数",
                           "睡眠心率", "睡眠血氧", "睡眠呼吸率", "静息心率", "血氧",
                            "DBP", "SBP", "步数", "REM", "活动热量",
                            "运动心率", "血管弹性", "心血管风险", "肺功能评估", "肺部感染风险",
                            "慢阻肺风险"}:
        return indicator_type != "0"
    elif indicator_name in {"睡眠得分", "压力值", "心电图"}:
        return indicator_type not in {"0", "1"}
    elif indicator_name in {"脉搏波传导速度"}:
        return indicator_type != "1"
    elif indicator_name in {"年龄"}:
        return False
    
    return True

def is_improve(indicator_name:str, indicator_type:str) -> bool:
    improve_indicators = {"体重", "体脂率", "睡眠时长", "深睡比例", "血氧",
                          "压力值", "静息心率", "睡眠呼吸率", "活动热量", "清醒次数",
                          "步数", "运动心率"}
    return (indicator_name in improve_indicators) and is_abnormal(indicator_name, indicator_type)


if __name__ == "__main__":
    print(is_abnormal("年龄", "2"))
    print(is_abnormal("体重", "0"))
    print(is_abnormal("睡眠得分", "2"))
    print(is_improve("体脂率", "0"))
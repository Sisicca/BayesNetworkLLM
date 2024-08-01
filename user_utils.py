from typing import List, Tuple, Dict, Set, Optional, Union

# 判断某节点指标的用户数据应该分为哪一类
def user_data_classify(indicator_name:str, indicator_value:Union[str,int], classify_standard) -> str:
    
    return

# 判断节点状态是否属于异常
def is_abnormal(indicator_name:str, indicator_type:str) -> bool:
    
    return True

def is_improve(indicator_name:str, indicator_type:str) -> bool:
    improve_indicators = {"体重", "体脂率", "睡眠时长", "深睡比例", "血氧"}
    return (indicator_name in improve_indicators) and is_abnormal(indicator_name, indicator_type)
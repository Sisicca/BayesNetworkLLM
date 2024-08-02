from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from user_utils import is_abnormal, is_improve
from typing import List, Tuple, Dict, Set, Optional, Union
import numpy as np
import warnings

# 对后验概率做softmax
def _softmax(x:List[float]) -> List[float]:
    x = np.array(x)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    softmax_x = exp_x / sum_exp_x
    softmax_x = np.round(softmax_x, 5)
    return list(softmax_x)

# 列表做差
def _list_subtract(list1:list, list2:list) -> list:
    return [item for item in list1 if item not in list2]

# 字典做差
def _dict_subtract(dict1:dict, dict2:dict) -> dict:
    return {k : v for k, v in dict1.items() if k not in dict2}

# 改善指标，需要改善的就-1，不需要改善的不变
def _improve(indicator:Dict[str,str]) -> Dict[str,str]:
    improve_result = {}
    for indicator_name, indicator_type in indicator.items():
        if is_abnormal(indicator_name, indicator_type):
            improve_result[indicator_name] = str(int(indicator_type)-1)
        else:
            improve_result[indicator_name] = indicator_type
    return improve_result
            
# 改善能力评估指标一
def _improve_ratio_1(target:Dict[str,str], improve:Dict[str,str], evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork) -> float:
    """
        ratio = P(目标节点=y-1|主动节点=x-1,others)/P(目标节点=y-1|主动节点=x,others) * 100%
    """
    infer = VariableElimination(bn)
    
    new_target = _improve(target)
    new_improve = _improve(improve)
    
    others = {}
    others.update(evidence)
    others.update(user_problem_type)
    others = _dict_subtract(others, target|improve)
    
    for indicator_name, indicator_type in new_target.items():
        numerator = infer.query(variables=[indicator_name], evidence=new_improve|others, show_progress=False)
        numerator = numerator.values[int(indicator_type)]
        if np.isnan(numerator):
            warnings.warn("probability table may exist defect.")
            numerator = 0
        
        denominator = infer.query(variables=[indicator_name], evidence=improve|others, show_progress=False)
        denominator = denominator.values[int(indicator_type)]
        if np.isnan(denominator):
            warnings.warn("probability table may exist defect.")
            denominator = 0
    
    if numerator == 0 and denominator == 0:
        return 1.0
    elif numerator != 0 and denominator == 0:
        return 2.0 # 如果是inf 误差可能太大
    return numerator / denominator

# 改善能力评估指标二
def _improve_ratio_2(target:Dict[str,str], improve:Dict[str,str], evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork) -> float:
    """
        ratio = P(目标节点<y|主动节点=x-1,others)/P(目标节点>=y|主动节点=x-1,others) / P(目标节点<y|主动节点=x,others)/P(目标节点>=y|主动节点=x,others) * 100%
    """
    infer = VariableElimination(bn)
    
    new_improve = _improve(improve)
    
    others = {}
    others.update(evidence)
    others.update(user_problem_type)
    others = _dict_subtract(others, target|improve)
    
    for indicator_name, indicator_type in target.items():
        posterior1 = infer.query(variables=[indicator_name], evidence=new_improve|others, show_progress=False)
        numerator1 = sum(posterior1.values[:int(indicator_type)])
        if np.isnan(numerator1):
            warnings.warn("probability table may exist defect.")
            numerator1 = 0
        denominator1 = 1 - numerator1
        
        posterior2 = infer.query(variables=[indicator_name], evidence=improve|others, show_progress=False)
        numerator2 = sum(posterior2.values[:int(indicator_type)])
        if np.isnan(numerator2):
            warnings.warn("probability table may exist defect.")
            numerator2 = 0
        denominator2 = 1 - numerator2
    
    numerator = numerator1 * denominator2
    denominator = numerator2 * denominator1
    
    if numerator == 0 and denominator == 0:
        return 1.0
    elif numerator != 0 and denominator == 0:
        return 2.0 # 如果是inf 误差可能太大
    return numerator / denominator

# 已知异常节点排序
def posterior_sort(evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork) -> List[Tuple[str,float]]:
    """
        returns sorted abnormal evidence by posterior prob
    Args:
        evidence (dict): like {"A" : "0", "B" : "2", "C" : "1"}
        user_problem_type (dict): like {"problem1" : "1", "problem2" : "1}
        bn (BayesianNetwork): the BayesianNetwork
        
    Returns:
        list: like [("C", 0.8), ("B", 0.2)]
    """
    infer = VariableElimination(bn)
    
    query_list = [indicator_name for indicator_name, indicator_type in evidence.items() if is_abnormal(indicator_name, indicator_type)]
    result = []
    
    for mask in query_list:
        # 构建临时evidence
        temp_evidence = {indicator_name : indicator_type for indicator_name, indicator_type in evidence.items() if indicator_name!=mask}
        temp_evidence.update(user_problem_type)
        # 计算临时evidence下的后验概率
        temp_posterior = infer.query(variables=[mask], evidence=temp_evidence, show_progress=False)
        temp_prob = temp_posterior.values[int(evidence.get(mask))]
        temp_prob = 0 if np.isnan(temp_prob) else temp_prob
        
        result.append(temp_prob)
    result = _softmax(result)
    comb_result = list(zip(query_list, result))
    comb_result = sorted(comb_result, key=lambda x: x[1], reverse=True)
    
    return comb_result

# 未知异常节点预测
def abnormal_predict(evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork, max_predict:Optional[int]=None) -> List[str]:
    """
        预测未知节点里有哪些是异常的
    Args:
        evidence (Dict[str,str]): {"B" : "2", "C" : "1"} -----> A is unknown
        user_problem_type (Dict[str,str]): {"problem1" : "1", "problem2" : "1}
        bn (BayesianNetwork): the BayesianNetwork

    Returns:
        list: ["A"]
    """
    infer = VariableElimination(bn)
    
    node_list = list(bn.nodes)
    known_list = list(evidence.keys())
    problem_list = list(user_problem_type.keys())
    
    unknown_list = _list_subtract(node_list, known_list)
    unknown_list = _list_subtract(unknown_list, problem_list)
    unknown_list = unknown_list if not max_predict else unknown_list[:max_predict]
    
    all_evidence = {}
    all_evidence.update(evidence)
    all_evidence.update(user_problem_type)
    
    predict_dict = infer.map_query(variables=unknown_list, evidence=all_evidence, show_progress=False)
    abnormal_list = [indicator_name for indicator_name, indicator_type in predict_dict.items() if is_abnormal(indicator_name, indicator_type)]
    
    return abnormal_list

# 目标异常节点改善
def improve_sort(target:Dict[str,str], evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork, improve_ratio_type:str="1") -> List[Tuple[dict,float]]:
    """
        计算evidence里improve indicators的ratio 然后排序 然后排序后的列表。
    Raises:
        ValueError: ratio计算公式只有"1","2"可选    
    """
    
    if improve_ratio_type not in {"1", "2"}:
        raise ValueError("improve_ratio must be '1' or '2'!")
    _improve_ratio = _improve_ratio_1 if improve_ratio_type == "1" else _improve_ratio_2
    
    assert len(target) == 1, "One target once time!"
    
    target_name, target_type = list(target.items())[0]
    assert is_abnormal(target_name, target_type) == True, f"{target_name} is already healthy!"
    assert is_improve(target_name, target_type) == False, f"{target_name} is an improve indicator, but it should be a negtive indicator!"
    
    # 排序节点一定是主动节点
    improve_list = [{indicator_name : indicator_type}
                    for indicator_name, indicator_type in evidence.items()
                    if is_improve(indicator_name, indicator_type)]
    
    # 如果evidence里没有一个improve indicator 那就返回空列表
    if not improve_list:
        return []
    
    result = []
    
    for improve in improve_list:
        result.append(_improve_ratio(target, improve, evidence, user_problem_type, bn))
    
    result = _softmax(result)
    comb_result = list(zip(improve_list, result))
    comb_result = sorted(comb_result, key=lambda x: x[1], reverse=True)
    
    return comb_result

def abnormal_max(target:Dict[str,str], evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork, improve_ratio_type:str="1") -> Tuple[dict,float]:
    """
        求出对目标节点影响最强的异常节点
    """
    if improve_ratio_type not in {"1", "2"}:
        raise ValueError("improve_ratio must be '1' or '2'!")
    _improve_ratio = _improve_ratio_1 if improve_ratio_type == "1" else _improve_ratio_2
    
    assert len(target) == 1, "One target once time!"
    
    target_name, target_type = list(target.items())[0]
    assert is_abnormal(target_name, target_type) == True, f"{target_name} is already healthy!"
    
    # 节点一定是异常节点
    abnormal_list = [{indicator_name : indicator_type}
                     for indicator_name, indicator_type in evidence.items()
                     if (is_abnormal(indicator_name, indicator_type) and indicator_name != target_name)]
    
    # 如果evidence里 没有一个不同于target的abnormal indicator 就返回空字典
    if not abnormal_list:
        return {}, 1.0
    
    result = []
    
    for abnormal in abnormal_list:
        result.append(_improve_ratio(target, abnormal, evidence, user_problem_type, bn))
    
    comb_result = list(zip(abnormal_list, result))
    max_result = max(comb_result, key=lambda x: x[1])
    
    return max_result

def abnormal_max_recursion(target:Dict[str,str], evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork, accumulate_score:float=1.0, improve_ratio_type:str="1") -> Tuple[dict,float]:
    """
        求出对目标节点影响最大的异常节点 判断该节点是不是主动节点 是则输出 不是则继续递归
    """
    
    if not target:
        return {}, accumulate_score
    
    target_name, target_type = list(target.items())[0]
    if is_improve(target_name, target_type):
        return target, accumulate_score
    
    new_target, score = abnormal_max(target, evidence, user_problem_type, bn, improve_ratio_type)
    accumulate_score *= score
    new_evidence = _dict_subtract(evidence, target)
    new_user_problem_type = _dict_subtract(user_problem_type, target)
    
    return abnormal_max_recursion(new_target, new_evidence, new_user_problem_type, bn, accumulate_score, improve_ratio_type)

def abnormal_sort(target:Dict[str,str], evidence:Dict[str,str], user_problem_type:Dict[str,str], bn:BayesianNetwork, max_nums:int=3, improve_ratio_type:str="1") -> List[Tuple[dict,float]]:
    
    indicators_result = []
    scores_result = []
    new_evidence = _dict_subtract(evidence, {})
    count = 0
    
    while count < max_nums:
        count += 1
        
        max_indicator, max_score = abnormal_max_recursion(target, new_evidence, user_problem_type, bn, 1, improve_ratio_type)
        
        print("round: ", count)
        print("target: ", target)
        print("evidence: ", new_evidence)
        print("max_indicator: ", max_indicator)
        print(f"------------改善{max_indicator}--------------")
        
        if not max_indicator:
            break
        
        indicators_result.append(max_indicator)
        scores_result.append(max_score)
        
        new_evidence = _dict_subtract(new_evidence, max_indicator)
        improved_max_indicator = _improve(max_indicator)
        new_evidence.update(improved_max_indicator)

    if not indicators_result:
        return []
    
    scores_result = _softmax(scores_result)
    comb_result = list(zip(indicators_result, scores_result))
    comb_result = sorted(comb_result, key=lambda x: x[1], reverse=True)
    
    return comb_result


if __name__ == "__main__":
    model = BayesianNetwork.load("BayesNetwork_hw.bif")
    # print(posterior_sort(evidence={"体重":"1", "压力值":"2"}, user_problem_type={"运动表现异常":"1"}, bn=model))
    # print(abnormal_predict(evidence={"体重":"1", "压力值":"2", "睡眠得分":"1", "深睡比例":"1", "睡眠时长":"1", "年龄":"1"}, user_problem_type={"运动表现异常":"1", "睡眠异常":"1"}, bn=model))
    # print(improve_sort(target={"肺健康异常":"1"}, evidence={"肺功能评估":"1","肺部感染风险":"1","压力值":"2","深睡比例":"1","血氧":"1"}, user_problem_type={"肺健康异常":"1"}, bn=model, improve_ratio_type="1"))
    # print(abnormal_max(target={"肺健康异常":"1"}, evidence={"肺功能评估":"1","肺部感染风险":"1","深睡比例":"1","血氧":"1", "压力值":"2"}, user_problem_type={"肺健康异常":"1"}, bn=model, improve_ratio_type="1"))
    # print(abnormal_max_recursion(target={"肺健康异常":"1"}, evidence={"肺功能评估":"1","肺部感染风险":"1","压力值":"2","深睡比例":"1","血氧":"1"}, user_problem_type={"肺健康异常":"1"}, bn=model, improve_ratio_type="1"))
    # print(abnormal_max_recursion(target={"肺健康异常":"1"}, evidence={"肺功能评估":"1","肺部感染风险":"1","压力值":"2","深睡比例":"1","血氧":"0"}, user_problem_type={"肺健康异常":"1"}, bn=model, improve_ratio_type="1"))
    # print(abnormal_max_recursion(target={"肺健康异常":"1"}, evidence={"肺功能评估":"1","肺部感染风险":"1","压力值":"2","深睡比例":"0","血氧":"0"}, user_problem_type={"肺健康异常":"1"}, bn=model, improve_ratio_type="1"))
    # print(abnormal_max_recursion(target={"肺健康异常":"1"}, evidence={"肺功能评估":"1","肺部感染风险":"1"}, user_problem_type={"肺健康异常":"1"}, bn=model, improve_ratio_type="1"))
    # print(improve_sort(target={"心电图":"2"}, evidence={"压力值":"3", "体重":"1", "睡眠时长":"0", "血氧":"1"}, user_problem_type={"心脏健康异常":"1"}, bn=model))
    print(abnormal_sort(target={"肺健康异常":"1"}, evidence={"肺部感染风险":"1","肺功能评估":"1","压力值":"2","深睡比例":"1","血氧":"1"}, user_problem_type={"肺健康异常":"1"}, bn=model, max_nums=4, improve_ratio_type="1"))
    print(abnormal_sort(target={"心电图":"2"}, evidence={"体重":"0", "睡眠时长":"1", "深睡比例":"1", "血氧":"1", "血管弹性":"1", "体脂率":"1"}, user_problem_type={"心脏健康异常":"1"}, bn=model, max_nums=5))
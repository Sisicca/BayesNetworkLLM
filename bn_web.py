import streamlit as st
from pgmpy.models import BayesianModel
import networkx as nx
from utils import (
    problem_classifier, bn_posterior, bn_infer,
    answer_with_infer, draw_subgraph, softmax,
    check_normal, classify_sleep, classify_pressure, classify_age
)

st.write("# Bayes Network LLM")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    base_url = st.text_input("请输入Base URL：")
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-keys)")

st.write("## 贝叶斯网络模型")

tab1, tab2 = st.tabs(["导入模型", "模型图例"])

with tab1:
    import_bn = st.button("导入模型")
    if import_bn:
        with st.spinner("模型正在导入，请稍后..."):
            if import_bn and "bn_model" not in st.session_state:
                st.session_state.bn_model = BayesianModel.load(filename='BayesNetwork_hw.bif', filetype='bif')
                st.session_state.G = nx.DiGraph()
                st.session_state.G.add_edges_from(st.session_state.bn_model.edges)   
        st.write("模型已导入成功！")
        
with tab2:
    if "bn_model" in st.session_state:
        st.image("graph_hw.png")
        
st.divider()

st.write("## 用户输入")

sample = st.radio("**请选择样例**:",
                  ["手动输入", "样例一", "样例二"],
                  horizontal=True)

indicators_dict = {"手动输入" : None,
                   "样例一" : ["睡眠时长", "深睡比例", "睡眠血氧", "心电图", "静息心率", "肺功能评估", "压力值", "运动心率", "活动热量", "睡眠得分"],
                   "样例二" : ["睡眠时长", "深睡比例", "清醒次数", "睡眠呼吸率", "心电图", "心血管风险", "压力值", "肺功能评估", "慢阻肺风险", "静息心率", "活动热量"]
                   }

data_dict = {
    "手动输入" : {
        "年龄" : 18,
        "体重" : 60,
        "体脂率" : 15,
        
        "睡眠时长" : 8,
        "深睡比例" : 15,
        "清醒次数" : 0,
        "睡眠心率" : 60,
        "REM" : 1,
        "睡眠血氧" : 95,
        "睡眠呼吸率" : 15,
        "睡眠得分" : 85,
        
        "静息心率" : 70,
        "血氧" : 96,
        "DBP" : 80,
        "SBP" : 120,
        "压力值" : 43,
        
        "步数" : 5000,
        "活动热量" : 600,
        "运动心率" : 150,
        
        "心电图" : "正常", # 正常，窦性心率，失常
        "心血管风险" : "低", # 低，中，高
        "脉搏波传导速度" : "慢", # 慢，中，快
        "血管弹性" : "正常", # 正常，稍差
        "肺功能评估" : "良好", # 良好，中等，较差
        "肺部感染风险" : "低", # 低，中，高
        "慢阻肺风险" : "低", # 低，中，高  
    },
    "样例一" : {
        "年龄" : 28,
        "体重" : 85,
        "体脂率" : 28,
        
        "睡眠时长" : 6,
        "深睡比例" : 15,
        "清醒次数" : 0,
        "睡眠心率" : 60,
        "REM" : 1,
        "睡眠血氧" : 94,
        "睡眠呼吸率" : 15,
        "睡眠得分" : 65,
        
        "静息心率" : 105,
        "血氧" : 96,
        "DBP" : 80,
        "SBP" : 120,
        "压力值" : 73,
        
        "步数" : 5000,
        "活动热量" : 600,
        "运动心率" : 205,
        
        "心电图" : "正常", # 正常，窦性心率，失常
        "心血管风险" : "低", # 低，中，高
        "脉搏波传导速度" : "慢", # 慢，中，快
        "血管弹性" : "正常", # 正常，稍差
        "肺功能评估" : "较差", # 良好，中等，较差
        "肺部感染风险" : "低", # 低，中，高
        "慢阻肺风险" : "低", # 低，中，高 
    },
    "样例二" : {
        "年龄" : 33,
        "体重" : 68,
        "体脂率" : 22,
        
        "睡眠时长" : 9,
        "深睡比例" : 9,
        "清醒次数" : 3,
        "睡眠心率" : 60,
        "REM" : 1,
        "睡眠血氧" : 95,
        "睡眠呼吸率" : 8,
        "睡眠得分" : 60,
        
        "静息心率" : 65,
        "血氧" : 96,
        "DBP" : 80,
        "SBP" : 120,
        "压力值" : 25,
        
        "步数" : 5000,
        "活动热量" : 300,
        "运动心率" : 205,
        
        "心电图" : "正常", # 正常，窦性心率，失常
        "心血管风险" : "低", # 低，中，高
        "脉搏波传导速度" : "慢", # 慢，中，快
        "血管弹性" : "正常", # 正常，稍差
        "肺功能评估" : "中等", # 良好，中等，较差
        "肺部感染风险" : "低", # 低，中，高
        "慢阻肺风险" : "中", # 低，中，高
    }
}

problems_dict = {
    "手动输入" : "",
    "样例一" : "我最近跑步的表现一直不好，有点喘不上气，我该如何调整?",
    "样例二" : "我明明每天都睡很久，但还是感觉没有精神，头昏脑胀，这是怎么回事？"
}

evidence = {}
data_records = {}

st.write("### 选择指标")
indicators = st.multiselect("请选择需要输入的健康指标：",
                            ["睡眠时长", "深睡比例", "清醒次数", "睡眠心率", "REM", "睡眠血氧", "睡眠呼吸率", "睡眠得分",
                             "心电图", "心血管风险", "静息心率", "血氧", "DBP", "SBP", "脉搏波传导速度", "血管弹性", "肺功能评估", "肺部感染风险", "慢阻肺风险", "压力值",
                             "步数", "活动热量", "运动心率",
                             ],
                            default=indicators_dict[sample])

column1, column2, column3 = st.columns([1, 2, 1])

with column1:
    st.write("### 基本信息")
    age = st.number_input("年龄（岁）：",
                          value=data_dict[sample].get("年龄"),
                          min_value=0,
                          max_value=100,
                          step=1)
    evidence["年龄"] = classify_age(age)
    data_records["年龄"] = f"{age}岁"
    weight = st.number_input("体重（kg）：",
                          value=data_dict[sample].get("体重"),
                          min_value=30,
                          max_value=300,
                          step=5)
    evidence["体重"] = check_normal(weight, 50, 80)
    data_records["体重"] = f"{weight}kg"
    fat_rate = st.number_input("体脂率（%）：",
                          value=data_dict[sample].get("体脂率"),
                          min_value=5,
                          max_value=50,
                          step=1)
    evidence["体脂率"] = check_normal(fat_rate, 15, 25)
    data_records["体脂率"] = f"{fat_rate}%"

with column2:
    st.write("### 数值指标")
    
    sub_column1, sub_column2 = st.columns([1, 1])
    
    with sub_column1:
        
        value_indicators = ["睡眠时长", "深睡比例", "清醒次数", "睡眠心率","REM", "睡眠血氧", "睡眠呼吸率", "睡眠得分",
                            "静息心率", "血氧", "DBP", "SBP", "压力值",
                            "步数", "活动热量", "运动心率"]
        value_indicators_unit = ["hr", "%", "次", "次/分","hr", "饱和度%", "%", "分",
                                "次每分", "饱和度%", "mmhg", "mmhg", "分",
                                "步", "kcal", "次/分"]
        value_indicators_min = [0, 0, 0, 40, 0, 80, 5, 0,
                                40, 80, 40, 70, 0,
                                0, 0, 80]
        value_indicators_max = [24, 50, 10, 120, 8, 100, 40, 100,
                                120, 100, 120, 200, 100,
                                100000, 10000, 220]
        value_indicators_step = [1, 1, 1, 5, 1, 1, 1, 5,
                                5, 1, 5, 5, 5,
                                100, 100, 5]
        value_indicators_check = [(7, 9), (15, 25), (0, 2), (60, 100), (1, 2), (95, 100), (12, 20), (),
                                (60, 79), (96, 100), (60, 80), (90, 120), (),
                                (6000, float('inf')), (300, 700), (100, 160)]
        
        it = iter(value_indicators)
        it_unit = iter(value_indicators_unit)
        it_min = iter(value_indicators_min)
        it_max = iter(value_indicators_max)
        it_step = iter(value_indicators_step)
        it_check = iter(value_indicators_check)
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"

    with sub_column2:
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"
        
        name = next(it)
        unit = next(it_unit)
        value_min = next(it_min)
        value_max = next(it_max)
        step = next(it_step)
        check = next(it_check)
        if name in indicators:
            temp = st.number_input(f"{name}（{unit}）：",
                                value=data_dict[sample].get(name),
                                min_value=value_min,
                                max_value=value_max,
                                step=step)
            if name == "睡眠得分":
                evidence[name] = classify_sleep(temp)
            elif name == "压力值":
                evidence[name] = classify_pressure(temp)
            else:
                evidence[name] = check_normal(temp, *check)
            data_records[name] = f"{temp}{unit}"

with column3:
    st.write("### 分类指标")
    
    type_indicators = ["心电图", "心血管风险", "脉搏波传导速度", "血管弹性", "肺功能评估", "肺部感染风险", "慢阻肺风险"]
    type_indicators_map = [{"正常" : '0', "窦性心率" : '1', "失常" : '2'},
                           {"低" : '0', "中" : '1', "高" : '2'},
                           {"慢" : '0', "中" : '1', "快" : '2'},
                           {"正常" : '0', "稍差" : '1'},
                           {"良好" : '0', "中等" : '1', "较差" : '2'},
                           {"低" : '0', "中" : '1', "高" : '2'},
                           {"低" : '0', "中" : '1', "高" : '2'}]
    
    it_type = iter(type_indicators)
    it_map = iter(type_indicators_map)
    
    name = next(it_type)
    map_dict = next(it_map)
    if name in indicators:
        temp = st.radio(f"{name}：",
                        map_dict.keys(),
                        horizontal=True,
                        index=int(map_dict.get(data_dict[sample].get(name))))
        evidence[name] = map_dict.get(temp)
        data_records[name] = f"{temp}"
    
    name = next(it_type)
    map_dict = next(it_map)
    if name in indicators:
        temp = st.radio(f"{name}：",
                        map_dict.keys(),
                        horizontal=True,
                        index=int(map_dict.get(data_dict[sample].get(name))))
        evidence[name] = map_dict.get(temp)
        data_records[name] = f"{temp}"
    
    name = next(it_type)
    map_dict = next(it_map)
    if name in indicators:
        temp = st.radio(f"{name}：",
                        map_dict.keys(),
                        horizontal=True,
                        index=int(map_dict.get(data_dict[sample].get(name))))
        evidence[name] = map_dict.get(temp)
        data_records[name] = f"{temp}"
    
    name = next(it_type)
    map_dict = next(it_map)
    if name in indicators:
        temp = st.radio(f"{name}：",
                        map_dict.keys(),
                        horizontal=True,
                        index=int(map_dict.get(data_dict[sample].get(name))))
        evidence[name] = map_dict.get(temp)
        data_records[name] = f"{temp}"
    
    name = next(it_type)
    map_dict = next(it_map)
    if name in indicators:
        temp = st.radio(f"{name}：",
                        map_dict.keys(),
                        horizontal=True,
                        index=int(map_dict.get(data_dict[sample].get(name))))
        evidence[name] = map_dict.get(temp)
        data_records[name] = f"{temp}"
    
    name = next(it_type)
    map_dict = next(it_map)
    if name in indicators:
        temp = st.radio(f"{name}：",
                        map_dict.keys(),
                        horizontal=True,
                        index=int(map_dict.get(data_dict[sample].get(name))))
        evidence[name] = map_dict.get(temp)
        data_records[name] = f"{temp}"
    
    name = next(it_type)
    map_dict = next(it_map)
    if name in indicators:
        temp = st.radio(f"{name}：",
                        map_dict.keys(),
                        horizontal=True,
                        index=int(map_dict.get(data_dict[sample].get(name))))
        evidence[name] = map_dict.get(temp)
        data_records[name] = f"{temp}"

evidence_list = list(evidence.keys())

st.write("### 用户问题")
user_problem = st.text_area("请描述您的运动健康问题：",
                            value=problems_dict[sample])
user_problem_type = ""
if user_problem:
    with st.spinner("正在分类问题，请稍后..."):
        if sample == "样例一":
            user_problem_type = "运动表现异常、肺健康异常"
        elif sample == "样例二":
            user_problem_type = "睡眠异常、其他问题异常"
        else:
            if openai_api_key and base_url:
                user_problem_type = problem_classifier(user_problem, openai_api_key, base_url)
            else:
                st.warning("请输入OpenAI API密钥和Base URL", icon="⚠️")
        st.write(f"您的问题属于：{user_problem_type}。")
        user_problem_type = user_problem_type.split("、")
        for problem in user_problem_type:
            evidence[problem] = "1"

st.divider()

st.write("## 贝叶斯网络推理")

tab2_1, tab2_2 = st.tabs(["推理", "展示子图"])

with tab2_1:
    @st.cache_data
    def bn_posterior_cache(evidence, user_problem_type):
        return bn_posterior(st.session_state.bn_model, evidence, user_problem_type)
    
    @st.cache_data
    def bn_infer_cache(evidence):
        return bn_infer(st.session_state.bn_model, evidence)
    
    @st.cache_data
    def draw_subgraph_cache(evidence_list, user_problem_type, infer_result):
        return draw_subgraph(st.session_state.G, evidence_list, user_problem_type, infer_result)
    
    infer_result = ""
    infer = st.radio("是否进行推理",
                     ["是", "否"],
                     horizontal=True,
                     index=1)
    if infer == "是":
        if "bn_model" not in st.session_state:
            st.warning("请先导入贝叶斯网络模型。", icon="⚠️")
        else:
            if user_problem_type:
                with st.spinner("正在推理，请稍后..."):
                    
                    infer_result = bn_posterior_cache(evidence, user_problem_type)
                    # st.write(evidence)
                    # st.write(user_problem_type)
                    # st.write(infer_result)
                    if not infer_result:
                        st.write("您目前提供的指标中未检测到异常，请做进一步的检查。")
                    else:
                        infer_score = [x[1] for x in infer_result]
                        infer_result = [x[0] for x in infer_result]
                        infer_score = softmax(infer_score)
                        infer_merge = [f"{x}：{y}" for x, y in zip(infer_result, infer_score)]
                        infer_result_str = "，".join(infer_merge)
                        st.write(f"对用户问题造成影响的指标中，根据影响程度，从大到小依次是：{infer_result_str}")
                        st.write(f"“{infer_result[0]}”是造成用户{'、'.join(user_problem_type)}的最主要原因。")
                    
                    infer_potential = bn_infer_cache(evidence)
                    if infer_potential:
                        st.write(f"根据您提供的信息，可能存在异常的指标有：{'、'.join(infer_potential)}，可针对这些指标做进一步的检查。")
            else:
                st.warning("请先描述您的问题。", icon="⚠️")

with tab2_2:
    if infer == "是" and ("bn_model" in st.session_state):
        if user_problem_type:
            st.pyplot(draw_subgraph_cache(evidence_list, user_problem_type, infer_result))
        else:
            st.warning("请先描述您的问题。", icon="⚠️")

st.divider()

st.write("## LLM回答问题")

infer_result_for_llm = [f"{x}:{data_records[x]}" for x in infer_result]
infer_result_for_llm = "、".join(infer_result_for_llm[:2])
extra_data = [f"{x}:{y}" for x, y in data_records.items() if x not in infer_result_for_llm[:2]]
extra_data = "、".join(extra_data)

st.write(f"将参考“{infer_result_for_llm}”给出回答。")
is_answer = st.button("生成回答")
if is_answer:
    if openai_api_key and base_url:
        if user_problem:
            with st.spinner("LLM正在生成回答，请稍后..."):
                answer = answer_with_infer(user_problem, infer_result_for_llm, extra_data, openai_api_key, base_url)
                st.write(answer)
        else:
            st.warning("请先描述您的问题。", icon="⚠️")
    else:
        st.warning("请输入OpenAI API密钥和Base URL", icon="⚠️")

import streamlit as st
import networkx as nx
from pgmpy.models import BayesianNetwork
from user_utils import (
    SAMPLE_INDICATORS, SAMPLE_DATA, SAMPLE_PROBLEMS, VALUE_INDICATORS, TYPE_INDICATORS,
    check_normal, classify_age, classify_pressure, classify_sleep
)
from llm_utils import (
    problem_classify
)
from bn_utils import (
    risk_sort, unknown_predict, improve_sort, is_abnormal, is_improve
)

st.write("# Bayes Network LLM")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    base_url = st.text_input("请输入Base URL：")
    st.markdown("[获取OpenAI API密钥](https://platform.openai.com/account/api-keys)")

st.write("## 贝叶斯网络模型")

tab1_1, tab1_2 = st.tabs(["导入模型", "模型图例"])

with tab1_1:
    import_bn = st.button("导入模型")
    if import_bn:
        with st.spinner("模型正在导入，请稍后..."):
            if import_bn and "bn_model" not in st.session_state:
                st.session_state.bn_model = BayesianNetwork.load(filename='BayesNetwork_hw.bif', filetype='bif')
                st.session_state.G = nx.DiGraph()
                st.session_state.G.add_edges_from(st.session_state.bn_model.edges)   
        st.write("模型已导入成功！")
        
with tab1_2:
    if "bn_model" in st.session_state:
        st.image("graph_hw.png")
        
st.divider()

# evidence 记录用户指标状态 不包括用户问题 用于BN推理
evidence = {}
# data_records 记录用户指标数值或类别名称 用于LLM参考
data_records = {}
# user_problem_type 记录用户问题中涉及的异常 用于BN推理
user_problem_type = {}

st.write("## 用户输入")

sample = st.radio("**请选择样例**:", list(SAMPLE_INDICATORS.keys()), horizontal=True)

st.write("### 选择指标")

indicators = st.multiselect("请选择需要输入的健康指标：", VALUE_INDICATORS[0]+TYPE_INDICATORS[0], default=SAMPLE_INDICATORS[sample])

column1, column2, column3 = st.columns([1, 2, 1])

with column1:
    st.write("### 基本信息")
    age = st.number_input("年龄（岁）：", value=SAMPLE_DATA[sample].get("年龄"), min_value=0, max_value=100, step=1)
    evidence["年龄"] = classify_age(age)
    data_records["年龄"] = f"{age}岁"
    weight = st.number_input("体重（kg）：", value=SAMPLE_DATA[sample].get("体重"), min_value=30, max_value=300, step=5)
    evidence["体重"] = check_normal(weight, 50, 80)
    data_records["体重"] = f"{weight}kg"
    fat_rate = st.number_input("体脂率（%）：", value=SAMPLE_DATA[sample].get("体脂率"), min_value=5, max_value=50, step=1)
    evidence["体脂率"] = check_normal(fat_rate, 15, 25)
    data_records["体脂率"] = f"{fat_rate}%"

with column2:
    st.write("### 数值指标")
    
    sub_column1, sub_column2 = st.columns([1, 1])
    
    sub_value_indicatos_1 = [sub_list[:len(sub_list)//2] for sub_list in VALUE_INDICATORS]
    sub_value_indicatos_2 = [sub_list[len(sub_list)//2:] for sub_list in VALUE_INDICATORS]
    
    with sub_column1:
        for name, unit, low, up, step, check in zip(*sub_value_indicatos_1):
            if name in indicators:
                temp = st.number_input(f"{name}（{unit}）：",
                                    value=SAMPLE_DATA[sample].get(name),
                                    min_value=low,
                                    max_value=up,
                                    step=step)
                if name == "睡眠得分":
                    evidence[name] = classify_sleep(temp)
                elif name == "压力值":
                    evidence[name] = classify_pressure(temp)
                else:
                    evidence[name] = check_normal(temp, *check)
                data_records[name] = f"{temp}{unit}"
    
    with sub_column2:
        for name, unit, low, up, step, check in zip(*sub_value_indicatos_2):
            if name in indicators:
                temp = st.number_input(f"{name}（{unit}）：",
                                    value=SAMPLE_DATA[sample].get(name),
                                    min_value=low,
                                    max_value=up,
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
    
    for name, map_dict in zip(*TYPE_INDICATORS):
        if name in indicators:
            temp = st.radio(f"{name}：",
                            map_dict.keys(),
                            horizontal=True,
                            index=int(map_dict.get(SAMPLE_DATA[sample].get(name))))
            evidence[name] = map_dict.get(temp)
            data_records[name] = f"{temp}"

st.write("### 用户问题")

user_problem = st.text_area("请描述您的运动健康问题：", value=SAMPLE_PROBLEMS[sample].get("user_problem"))

if user_problem:
    with st.spinner("正在分类问题，请稍后..."):
        if sample != "手动输入":
            user_problem_type = SAMPLE_PROBLEMS[sample].get("user_problem_type")
        else:
            if openai_api_key and base_url:
                user_problem_type = problem_classify(user_problem, openai_api_key, base_url)
            else:
                st.warning("请输入OpenAI API密钥和Base URL", icon="⚠️")
        user_problem_type_str = "、".join([k for k, v in user_problem_type.items() if v=="1"])
        st.write(f"您的问题属于：{user_problem_type_str}。")

st.divider()

st.write("## 贝叶斯网络推理")

tab2_1, tab2_2, tab2_3, tab2_4 = st.tabs(["归因", "预测", "改善", "子图"])

with tab2_1:
    do_ascribe = st.radio("是否进行归因", ["是", "否"], horizontal=True, index=1)
    
    if do_ascribe == "是":
        if "bn_model" not in st.session_state:
            st.warning("请先导入贝叶斯网络模型。", icon="⚠️")
        else:
            if user_problem_type:
                for indicator_name, indicator_type in user_problem_type.items():
                    if is_abnormal(indicator_name, indicator_type):
                        ascribe_result = risk_sort(target={indicator_name:indicator_type}, evidence=evidence, user_problem_type=user_problem_type, bn=st.session_state.bn_model)
                        st.write(f"对“{indicator_name}”造成影响，从大到小依次是：")
                        st.write("、".join([f"{indicator}-{data_records.get(indicator)}-{score}" for indicator, score in ascribe_result]))
            else:
                st.warning("请先描述您的问题。", icon="⚠️")

with tab2_2:
    if len(indicators) == len(VALUE_INDICATORS[0]+TYPE_INDICATORS[0]):
        st.write("您的指标已输入完全。")
    else:
        do_predict = st.radio("您的指标输入不完全，是否对缺失指标进行预测", ["是", "否"], horizontal=True, index=1)
        if do_predict == "是":
            st.write(unknown_predict(evidence=evidence, user_problem_type=user_problem_type, bn=st.session_state.bn_model, only_abnormal=False))
    
with tab2_3:
    do_improve = st.radio("是否对异常问题进行改善推理", ["是", "否"], horizontal=True, index=1)
    if do_improve == "是":
        targets = [{k : v} for k, v in user_problem_type.items() if is_abnormal(k, v)]
        improve_results = []
        for target in targets:
            st.write(f"为改善{target}，可先依次改善以下指标：")
            improve_result = improve_sort(target=target, evidence=evidence, user_problem_type=user_problem_type, bn=st.session_state.bn_model)
            print(improve_result)
            st.write("、".join([f"{indicator}-{score}" for indicator, score in improve_result]))
            improve_results.append((target, improve_result))
    
    abnormal_indicator_name_list = [indicator_name 
                                    for indicator_name, indicator_type in evidence.items()
                                    if (is_abnormal(indicator_name, indicator_type) and not is_improve(indicator_name, indicator_type))]
    do_custom_improve = st.radio("请选择需要改善的节点", ["无"]+abnormal_indicator_name_list, horizontal=True, index=0)
    if do_custom_improve != "无":
        target = {do_custom_improve : evidence.get(do_custom_improve)}
        st.write(f"为改善{target}，可先依次改善以下指标：")
        improve_result = improve_sort(target=target, evidence=evidence, user_problem_type=user_problem_type, bn=st.session_state.bn_model)
        print(improve_result)
        st.write("、".join([f"{indicator}-{score}" for indicator, score in improve_result if is_improve(indicator_name=indicator, indicator_type=evidence.get(indicator))]))
with tab2_4:
    ...
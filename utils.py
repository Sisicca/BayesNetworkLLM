import streamlit as st

# LLM 回答

from openai import OpenAI

@st.cache_data
def problem_classifier(user_problem, api_key, base_url):
    
    client = OpenAI(api_key= api_key,
                base_url=base_url)
    
    problem_list = ["运动表现异常", "睡眠异常", "其他问题异常", "心脏健康异常", "肺健康异常"]
    problem_str = '、'.join(problem_list)
    
    prompt = """
    请你根据用户的问题‘{user_problem}’，从异常类别‘{problem_str}’中，找出问题对应的异常类别。
    问题对应的异常类别可以有多个，不同异常类别之间用、符号分隔，除此之外不要输出任何额外文本。
    """
    
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": prompt.format(user_problem="我卧推的重量下降了，这是怎么回事？",
                                     problem_str=problem_str)
        },
        {
        "role": "assistant",
        "content": "运动表现异常"
        },
        {
        "role": "user",
        "content": prompt.format(user_problem="我昨天失眠了，今天打球没有状态，该如何调整？",
                                 problem_str=problem_str)
        },
        {
        "role": "assistant",
        "content": "运动表现异常、睡眠异常"
        },
        {
            "role": "user",
            "content": prompt.format(user_problem=user_problem,
                                     problem_str=problem_str)
        }
    ]
    )
    content = response.choices[0].message.content
    
    return content

@st.cache_data
def answer_with_infer(user_problem, infer_result, extra_data, api_key, base_url):
    client = OpenAI(api_key= api_key,
                base_url=base_url)
    
    prompt = """
    请你重点针对‘{infer_result}’中所提到的指标及其数值进行分析，从而回答我的问题：‘{user_problem}’。
    此外，如果有必要，可以使用的额外数据有{extra_data}。
    ###名词解释说明
        **压力值数值范围说明：
            - 放松：1-29
            - 正常：30-59
            - 中等：60-79
            - 偏高：80-99
        **睡眠得分说明：
            - 优秀睡眠：80-100
            - 一般睡眠：60-79
            - 问题睡眠：40-59
            - 睡眠障碍：20-39
            - 几乎无有效睡眠：0-19
    """
    
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一位专业的运动健康顾问，拥有丰富的知识和经验，能够从专业的角度解答我所遇到的运动健康问题，并提供精准且专业建议。"},
        {"role": "user", "content": prompt.format(infer_result=infer_result, user_problem=user_problem, extra_data=extra_data)}
    ]
    )
    content = response.choices[0].message.content
    
    return content

import numpy as np

def softmax(x):
    x = np.array(x)
    x_max = np.max(x)
    x = x - x_max
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    softmax_x = np.around(softmax_x, 3)
    return list(softmax_x)

# Bayes Network 推理

from pgmpy.inference import VariableElimination

def bn_posterior(model, evidence: dict, user_problem: list):
    query_list = [x for x, y in evidence.items() if y!='0']
    infer = VariableElimination(model)
    posterior_list = []
    for mask in query_list:
        if mask in user_problem:
            continue
        temp_evidence = {x : y for x, y in evidence.items() if x!=mask}
        posterior = infer.query(variables=[mask], evidence=temp_evidence)
        posterior_list.append((mask, posterior.values[int(evidence[mask])] if not np.isnan(posterior.values[int(evidence[mask])]) else 0))
    posterior_list = sorted(posterior_list, key=lambda x: x[1], reverse=True)
    #posterior_str = '，'.join([x[0] for x in posterior_list])
    #user_problem_str = '，'.join(user_problem)
    return posterior_list

def bn_infer(model, evidence: dict):
    nodes_list = list(model.nodes)
    problem_list = ["运动表现异常", "睡眠异常", "其他问题异常", "心脏健康异常", "肺健康异常"]
    query_list = [x for x in nodes_list if (x not in evidence and x not in problem_list)]
    infer = VariableElimination(model)
    posterior = infer.query(variables=query_list, evidence=evidence)
    max_prob_list = posterior.assignment([posterior.values.argmax()])[0]
    abnormal_list = [x[0] for x in max_prob_list if x[1]!='0']
    # for key, value in evidence.items():
    #     if value!='0':
    #         abnormal_list.append(key)
    # abnormal_str = "，".join(abnormal_list)
    # return f"可能异常的健康指标有：{abnormal_str}。"
    return abnormal_list

# 生成子图

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def get_paths(graph, start, end):
    paths = find_all_paths(graph, start, end)
    if not paths:
        return []
    return paths

def draw_filtered_paths(graph, paths, red_spots:list=[]):
    
    # 设置中文字体
    font_path = 'STHeiti Light.ttc'  # 替换为你系统中支持中文显示的字体路径
    font_prop = fm.FontProperties(fname=font_path)
    
    # 创建一个新的子图，保证每次都能独立地显示图像
    fig = plt.figure(figsize=(12, 8))
    
    # 找到所有涉及的节点和边
    sub_nodes = set()
    sub_edges = set()
    for path in paths:
        sub_nodes.update(path)
        edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        sub_edges.update(edges)
    
    # 创建子图
    subgraph = graph.edge_subgraph(sub_edges).copy()
    
    # 布局
    pos = nx.kamada_kawai_layout(subgraph)
    
    # 设置节点颜色：睡眠、健康、运动、用户异常颜色各不同。
    node_colors = []
    for node in subgraph.nodes():
        if node in red_spots:
            node_colors.append("#eb2c2d")
        elif node in {"心电图", "心血管风险", "静息心率", "血氧", "DBP", "SBP", "脉搏波传导速度", "血管弹性", "肺功能评估", "肺部感染风险", "慢阻肺风险", "压力值"}:
            node_colors.append('#57c7e3')
        elif node in {"睡眠时长", "深睡比例", "清醒次数", "睡眠心率", "REM", "睡眠血氧", "睡眠呼吸率", "睡眠得分"}:
            node_colors.append('#f79767')
        elif node in {"步数", "活动热量", "运动心率"}:
            node_colors.append('#f16667')
        elif node in {"年龄", "体重", "体脂率"}:
            node_colors.append('#d9c8ae')
        else:
            node_colors.append('#ffc454')
    
    # 绘制子图
    nx.draw(subgraph, pos, with_labels=True,
            node_size=5000, node_color=node_colors,
            font_size=0, font_weight='bold',
            arrowsize=10, edge_color='gray')
    
    # 高亮路径的边
    """for path in paths:
        edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(subgraph, pos, edgelist=edges, edge_color='r', width=2.5)"""
    
    # 使用 matplotlib 绘制中文标签
    for node, (x, y) in pos.items():
        plt.text(x, y, node, fontsize=20, fontproperties=font_prop, ha='center', va='center')
 
    # 图形美化
    #plt.title("Filtered Paths from Start to End Node", fontsize=15)
    plt.axis('off')
    return fig

def draw_subgraph(graph, sources:list, targets:list, red_spots:list=[]):
    result = []
    for target in targets:
        for source in sources:
            result = result + get_paths(graph, source, target)
    return draw_filtered_paths(graph, result, red_spots)

# 转换数据
    
def check_normal(value, low, high):
    return '0' if low <= value <= high else '1'
"""
def classify_sleep(sleep):
    if sleep >= 80:
        return '0'
    elif sleep >= 60:
        return '1'
    elif sleep >= 40:
        return '2'
    elif sleep >= 20:
        return '3'
    return '4'
"""

def classify_sleep(sleep):
    if sleep >= 80:
        return '0'
    elif sleep >= 60:
        return '1'
    return '2'

def classify_pressure(pressure):
    if pressure >= 80:
        return '3'
    elif pressure >= 60:
        return '2'
    elif pressure >= 30:
        return '1'
    return '0'

def classify_age(age):
    if age < 40:
        return '0'
    elif 40 <= age < 60:
        return '1'
    else:
        return '2'
from typing import Literal, List
from langgraph.graph import StateGraph, END, START

from src.Multi_Agent.states import DebateAgentState
from src.Multi_Agent.nodes import (
    optimistic_initial_node, pessimistic_initial_node, 
    optimistic_debate_node, pessimistic_debate_node, 
    summary_node, save_debate_node)
from src.Multi_Agent.functions import should_continue

def agent_debate_graph():
    """
    멀티에이전트 토론 그래프 생성
    """
    workflow = StateGraph(DebateAgentState)

    # 노드 추가
    workflow.add_node("optimist_initial", optimistic_initial_node)
    workflow.add_node("pessimist_initial", pessimistic_initial_node)
    workflow.add_node("optimist_debate", optimistic_debate_node)
    workflow.add_node("pessimist_debate", pessimistic_debate_node)
    workflow.add_node("summary", summary_node)
    workflow.add_node("save_debate", save_debate_node)

    # 엣지 추가
    
    # 1. 낙관론자 초기 의견
    workflow.add_edge(START, "optimist_initial")
    
    # 2. 비관론자 초기 의견 
    workflow.add_edge("optimist_initial", "pessimist_initial")

    # 3. 토론 시작 - 낙관론자
    workflow.add_edge("pessimist_initial", "optimist_debate")
    
    # 4. 비관론자 반박 
    workflow.add_edge("optimist_debate", "pessimist_debate")

    # 5. 조건부 루프 
    workflow.add_conditional_edges(
        "pessimist_debate",
        should_continue,
        {
            "optimist" : "optimist_debate",
            "summary" : "summary"
        }
    )

    workflow.add_edge("summary", "save_debate")
    workflow.add_edge("save_debate", END)
    return workflow.compile()

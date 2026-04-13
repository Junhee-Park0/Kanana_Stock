from typing import Literal, List
from langgraph.graph import StateGraph, END, START

from src.Agent.states import DebateAgentState
from src.Agent.nodes import (
    optimistic_initial_node, pessimistic_initial_node, 
    optimistic_debate_node, pessimistic_debate_node, 
    summary_node, save_debate_node)
from src.Agent.functions import should_continue

def agent_debate_graph():
    """
    멀티에이전트 토론 그래프 생성
    """
    workflow = StateGraph(DebateAgentState)

    # 노드 추가
    workflow.add_node("[Initial] Optimist", optimistic_initial_node)
    workflow.add_node("[Initial] Pessimist", pessimistic_initial_node)
    workflow.add_node("[Debate] Optimist", optimistic_debate_node)
    workflow.add_node("[Debate] Pessimist", pessimistic_debate_node)
    workflow.add_node("Consensus Generator", summary_node)
    workflow.add_node("Save Session", save_debate_node)

    # 엣지 추가
    
    # 1. 낙관론자 초기 의견
    workflow.add_edge(START, "[Initial] Optimist")
    
    # 2. 비관론자 초기 의견 
    workflow.add_edge("[Initial] Optimist", "[Initial] Pessimist")

    # 3. 토론 시작 - 낙관론자
    workflow.add_edge("[Initial] Pessimist", "[Debate] Optimist")
    
    # 4. 비관론자 반박 
    workflow.add_edge("[Debate] Optimist", "[Debate] Pessimist")

    # 5. 조건부 루프 
    workflow.add_conditional_edges(
        "[Debate] Pessimist",
        should_continue,
        {
            "optimist" : "[Debate] Optimist",
            "summary" : "Consensus Generator"
        }
    )

    workflow.add_edge("Consensus Generator", "Save Session")
    workflow.add_edge("Save Session", END)
    return workflow.compile()

from typing import List, Dict, Any, TypedDict, Optional, Literal
from typing import Annotated
import operator

class DebateAgentState(TypedDict):
    """각 토론 에이전트의 상태"""
    ticker : str
    context : str 

    optimist_initial : str
    pessimist_initial : str

    tool_calls : Annotated[List[Dict[str, Any]], operator.add]
    sources : Annotated[List[Dict[str, Any]], operator.add]

    debate_history : Annotated[List[str], operator.add]
    optimist_used_arguments: Annotated[list[str], operator.add]
    pessimist_used_arguments: Annotated[list[str], operator.add]

    turn_count : int 
    max_turns : int 
    current_agent : str
    should_continue : Literal["continue", "stop"]

    final_consensus : Optional[str]

from typing import List, Dict, Any, TypedDict, Optional
from typing import Annotated
import operator

class DebateAgentState(TypedDict):
    """각 토론 에이전트의 상태"""
    ticker : str
    keywords : List[str]
    context : str 

    optimist_initial : str
    pessimist_initial : str

    debate_history : Annotated[List[str], operator.add]

    turn_count : int 
    max_turns : int 
    current_agent : str
    
    final_consensus : Optional[str]

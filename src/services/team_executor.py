"""
Team Executor Service using LangGraph for agent team orchestration.
"""
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path

from src.graphs.tenant_team_graph import TenantTeamGraph

logger = logging.getLogger(__name__)

class TeamExecutorService:
    """Service to manage multi-agent conversation execution using LangGraph."""
    
    def __init__(self, agents: Dict[str, Any], flow: Optional[List[str]] = None, max_turns: int = 5):
        """
        Initialize with a dictionary of AutoGen agents.
        
        Args:
            agents: Dictionary mapping agent roles to AutoGen agent instances
            flow: Optional ordered list of agent roles to determine speaking order
            max_turns: Maximum number of conversation turns before stopping
        """
        self.agents = agents
        
        # Determine speaking order if flow is provided
        self.flow = flow if flow else list(agents.keys())
        
        # Convert string roles to actual roles if needed
        self.flow = [role for role in self.flow if role in self.agents]
        
        # Set up parameters
        self.max_turns = max_turns
        self.conversation_log = []
        
        # Create the graph
        checkpoint_dir = Path(".langgraph")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.graph = TenantTeamGraph(
            agents=self.agents, 
            flow=self.flow, 
            max_turns=self.max_turns,
            checkpoint_dir=str(checkpoint_dir)
        )
        
    def run_conversation(self, user_query: str) -> str:
        """
        Execute the multi-agent conversation using LangGraph.
        
        Args:
            user_query: The initial user query to start the conversation
            
        Returns:
            The final answer (or combined result) from the team
        """
        # Initial state with user query
        initial_state = {
            "query": user_query
        }
        
        # Run the graph and collect results
        result = self.graph.invoke(initial_state)
        
        # Update conversation log for API response
        self.conversation_log = result["conversation_log"]
        
        # Return the final answer
        return result["final_answer"]

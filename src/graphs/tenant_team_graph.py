"""LangGraph implementation for agent team orchestration."""
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

class TenantTeamGraph:
    """
    Creates and manages a LangGraph StateGraph for agent team orchestration.
    
    This class creates a graph whose nodes are AutoGen agents and whose edges
    encode the turn-taking order specified in AgentTeamConfig.
    """
    
    def __init__(self, agents: Dict[str, Any], flow: Optional[List[str]] = None, 
                 max_turns: int = 5, checkpoint_dir: Optional[str] = None):
        """
        Initialize the graph with AutoGen agents and flow configuration.
        
        Args:
            agents: Dictionary mapping agent roles to AutoGen agent instances
            flow: List of agent roles defining the conversation order, if None uses all agents in order
            max_turns: Maximum number of conversation turns before ending
            checkpoint_dir: Directory to store graph checkpoints, defaults to .langgraph/
        """
        self.agents = agents
        self.roles = list(agents.keys())
        self.flow = flow if flow else self.roles
        self.max_turns = max_turns
        
        # Set up checkpoint directory
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else ".langgraph"
        Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        
        # Create and configure the StateGraph
        self.graph = self._build_graph()
        
        # Set up checkpoint manager for persistence
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
    
    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph StateGraph with nodes and edges.
        """
        # Initialize the graph with agent message and conversation state
        graph = StateGraph(
            channels={
                "message": lambda: {"content": "", "sender": ""},
                "state": lambda: {
                    "turns": 0,
                    "final_answer": "",
                    "conversation_log": []
                }
            }
        )
        
        # Add agent nodes
        for role, agent in self.agents.items():
            graph.add_node(role, self._create_agent_node(role, agent))
        
        # Define conditional edge routing
        def should_end(state):
            # Check if we reached the maximum turns
            if state["state"]["turns"] >= self.max_turns:
                return True
            
            # Check if the message contains FINAL_ANSWER signal
            message = state["message"]["content"].lower()
            if "final answer:" in message or "final_answer:" in message:
                # Save the final answer
                state["state"]["final_answer"] = state["message"]["content"]
                return True
            
            return False
            
        # Set up edges based on agent_team_flow
        for i, role in enumerate(self.flow):
            # Determine next agent in the flow (circular)
            next_role = self.flow[(i + 1) % len(self.flow)]
            
            # Add edge with conditional branching
            graph.add_conditional_edges(
                role,
                lambda state, current_role=role: self._next_agent(state, current_role),
                {
                    END: should_end,
                    next_role: lambda _: True  # Default case
                }
            )
            
        # Define the entry point (assumes user query is passed in initial state)
        graph.set_entry_point("state")
        
        return graph
    
    def _create_agent_node(self, role: str, agent: Any):
        """
        Creates a graph node function for an agent.
        
        Args:
            role: The agent's role identifier
            agent: The AutoGen agent instance
            
        Returns:
            A node function for the StateGraph
        """
        def agent_node(state):
            # Get the current message
            message = state["message"]
            
            # Update the conversation log
            if message["content"]:
                state["state"]["conversation_log"].append((message["sender"], message["content"]))
            
            # Get the conversation history
            history = state["state"]["conversation_log"]
            
            # If this is the first agent and there's no message yet, use the initial query
            if not message["content"] and len(history) == 0:
                # No action needed, the agent will use the next code path
                pass
            
            # Generate a response from the agent
            # Note: This assumes AutoGen agents have a generate_reply method
            # that returns a dict with at least a "content" field
            response = agent.generate_reply(message["content"], state["state"])
            
            # Update state with the agent's response
            message["content"] = response["content"] 
            message["sender"] = role
            
            # Increment turn counter
            state["state"]["turns"] += 1
            
            return {"message": message, "state": state["state"]}
        
        return agent_node
    
    def _next_agent(self, state: Dict[str, Any], current_role: str) -> str:
        """
        Determines the next agent in the conversation flow.
        
        Args:
            state: The current graph state
            current_role: The role of the agent that just finished
            
        Returns:
            The role of the next agent or END
        """
        # Default to the next agent in the flow
        current_index = self.flow.index(current_role)
        next_role = self.flow[(current_index + 1) % len(self.flow)]
        
        return next_role
    
    def invoke(self, initial_state: Dict[str, Any], checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the graph with an initial state and optional checkpoint.
        
        Args:
            initial_state: The initial state containing at least the user query
            checkpoint_id: Optional checkpoint ID to resume from
            
        Returns:
            The final state with conversation log and final answer
        """
        # Initialize state with the user query
        state = {
            "message": {"content": initial_state.get("query", ""), "sender": "User"},
            "state": {
                "turns": 0,
                "final_answer": "",
                "conversation_log": [("User", initial_state.get("query", ""))]
            }
        }
        
        # Load from checkpoint if provided
        if checkpoint_id:
            try:
                state = self.checkpoint_manager.get(checkpoint_id)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_id}: {e}")
        
        # Run the graph
        config = {"checkpointing": True}
        result = self.graph.invoke(state, config=config)
        
        # Return the final state
        return {
            "final_answer": result["state"]["final_answer"] or result["message"]["content"],
            "conversation_log": result["state"]["conversation_log"]
        }

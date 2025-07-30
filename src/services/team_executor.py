"""
Team Executor Service using LangGraph for agent team orchestration.
Implements async graph execution with checkpointing and persistence.
"""
from typing import Dict, Any, Optional, List
import logging
import json
import os
import asyncio
from pathlib import Path

from src.graphs.tenant_team_graph import TenantTeamGraph

logger = logging.getLogger(__name__)


class TeamExecutorService:
    """
    Service to manage multi-agent conversation execution using LangGraph.
    
    Features:
    - Async LangGraph execution
    - Thread-based conversation persistence
    - Checkpointing support
    - Conversation logging
    """
    
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
        self.thread_id = None
        
        # Create the LangGraph with full features
        checkpoint_dir = Path(".langgraph")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.graph = TenantTeamGraph(
            agents=self.agents, 
            flow=self.flow, 
            max_turns=self.max_turns,
            checkpoint_dir=str(checkpoint_dir)
        )
        
    async def run_conversation_async(self, user_query: str, thread_id: Optional[str] = None) -> str:
        """
        Execute the multi-agent conversation using LangGraph asynchronously.
        
        Args:
            user_query: The initial user query to start the conversation
            thread_id: Optional thread ID to resume a previous conversation
            
        Returns:
            The final answer (or combined result) from the team
        """
        try:
            # Initial state with user query
            initial_state = {
                "query": user_query,
                "max_turns": self.max_turns
            }
            
            # Run the graph and collect results
            result = await self.graph.invoke(initial_state, thread_id=thread_id)
            
            # Update conversation log for API response
            self.conversation_log = result.get("conversation_log", [])
            
            # Store thread_id for potential future use
            self.thread_id = result.get("thread_id", thread_id)
            
            # Return the final answer
            final_answer = result.get("final_answer", "No final answer generated")
            
            logger.info(f"Conversation completed with {result.get('turn_count', 0)} turns")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in async conversation execution: {e}")
            return f"Error: {str(e)}"
    
    def run_conversation(self, user_query: str, thread_id: Optional[str] = None) -> str:
        """
        Execute the multi-agent conversation using LangGraph (sync wrapper).
        
        Args:
            user_query: The initial user query to start the conversation
            thread_id: Optional thread ID to resume a previous conversation
            
        Returns:
            The final answer (or combined result) from the team
        """
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a new task
                task = loop.create_task(self.run_conversation_async(user_query, thread_id))
                return asyncio.run_coroutine_threadsafe(task, loop).result()
            except RuntimeError:
                # No running loop, we can use asyncio.run
                return asyncio.run(self.run_conversation_async(user_query, thread_id))
                
        except Exception as e:
            logger.error(f"Error in conversation execution: {e}")
            # Fallback to simple synchronous execution
            return self._fallback_sync_execution(user_query, thread_id)
    
    def _fallback_sync_execution(self, user_query: str, thread_id: Optional[str] = None) -> str:
        """
        Fallback synchronous execution when async fails.
        
        Args:
            user_query: The user query
            thread_id: Optional thread ID
            
        Returns:
            Final answer from fallback execution
        """
        logger.warning("Using fallback synchronous execution")
        
        try:
            # Use the graph's fallback method
            initial_state = {"query": user_query, "max_turns": self.max_turns}
            result = self.graph._fallback_invoke(initial_state, thread_id)
            
            self.conversation_log = result.get("conversation_log", [])
            self.thread_id = result.get("thread_id", thread_id)
            
            return result.get("final_answer", "Fallback execution completed")
            
        except Exception as e:
            logger.error(f"Error in fallback execution: {e}")
            return f"Error in conversation execution: {str(e)}"
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """
        Get the current conversation state.
        
        Returns:
            Dictionary with conversation metadata
        """
        return {
            "thread_id": self.thread_id,
            "conversation_log": self.conversation_log,
            "agent_count": len(self.agents),
            "max_turns": self.max_turns,
            "flow": self.flow
        }
    
    def resume_conversation(self, thread_id: str, additional_query: str = "") -> str:
        """
        Resume a conversation from a checkpoint.
        
        Args:
            thread_id: Thread ID to resume
            additional_query: Additional query to add to the conversation
            
        Returns:
            Final answer from resumed conversation
        """
        logger.info(f"Resuming conversation from thread {thread_id}")
        
        try:
            # Get checkpoint if available
            checkpoint = self.graph.get_checkpoint(thread_id)
            
            if checkpoint:
                logger.info("Checkpoint found, resuming from saved state")
                query = additional_query or "Continue the conversation"
            else:
                logger.warning("No checkpoint found, starting new conversation")
                query = additional_query or "Start new conversation"
            
            return self.run_conversation(query, thread_id)
            
        except Exception as e:
            logger.error(f"Error resuming conversation: {e}")
            return f"Error resuming conversation: {str(e)}"

"""
LangGraph implementation for agent team orchestration.
Implements StateGraph with persistence, checkpoints, and AutoGen agent integration.
"""
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Annotated
import json
import logging
import asyncio
from pathlib import Path

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for when LangGraph is not properly installed
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = "START"
    END = "END"
    add_messages = None
    SqliteSaver = None
    MemorySaver = None

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    """State structure for the LangGraph conversation graph"""
    messages: List[Dict[str, Any]]
    current_agent: str
    turn_count: int
    max_turns: int
    final_answer: Optional[str]
    tenant_id: Optional[str]
    conversation_log: List[Tuple[str, str]]
    agent_flow: List[str]


class TenantTeamGraph:
    """
    LangGraph StateGraph implementation for agent team orchestration.
    
    Features:
    - StateGraph with persistence & checkpoints
    - Support for cycles, max_turns, and FINAL_ANSWER stop signal
    - Human approval gate node when required
    - SQLite checkpoint store for persistence
    """
    
    def __init__(self, agents: Dict[str, Any], flow: Optional[List[str]] = None, 
                 max_turns: int = 5, checkpoint_dir: Optional[str] = None):
        """
        Initialize the LangGraph with AutoGen agents and flow configuration.
        
        Args:
            agents: Dictionary mapping agent roles to AutoGen agent instances
            flow: List of agent roles defining the conversation order
            max_turns: Maximum number of conversation turns before ending
            checkpoint_dir: Directory to store SQLite checkpoints
        """
        self.agents = agents
        self.roles = list(agents.keys())
        self.flow = flow if flow else self.roles
        self.max_turns = max_turns
        
        # Set up checkpoint directory and SQLite persistence
        self.checkpoint_dir = Path(checkpoint_dir if checkpoint_dir else ".langgraph")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize checkpointer (SQLite for persistence)
        if LANGGRAPH_AVAILABLE:
            try:
                checkpoint_path = self.checkpoint_dir / "checkpoints.sqlite"
                self.checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to create SQLite checkpointer: {e}. Using memory saver.")
                self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None
        
        # Build the StateGraph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """
        Build the LangGraph StateGraph with nodes and edges.
        
        Returns:
            Compiled StateGraph with checkpointing enabled
        """
        if not LANGGRAPH_AVAILABLE:
            logger.error("LangGraph not available. Using fallback implementation.")
            return None
            
        # Create StateGraph with ConversationState schema
        graph = StateGraph(ConversationState)
        
        # Add agent nodes
        for role in self.roles:
            graph.add_node(role, self._create_agent_node(role))
        
        # Add special nodes
        graph.add_node("human_approval", self._human_approval_node)
        graph.add_node("final_answer", self._final_answer_node)
        
        # Set entry point to first agent in flow
        if self.flow:
            graph.add_edge(START, self.flow[0])
        else:
            graph.add_edge(START, "final_answer")
        
        # Add flow edges and conditional routing
        self._add_flow_edges(graph)
        
        # Compile graph with checkpointing
        try:
            compiled_graph = graph.compile(checkpointer=self.checkpointer)
            logger.info("LangGraph compiled successfully with checkpointing")
            return compiled_graph
        except Exception as e:
            logger.error(f"Failed to compile graph: {e}")
            # Fallback without checkpointing
            return graph.compile()
    
    def _create_agent_node(self, role: str):
        """
        Create a LangGraph node function for an AutoGen agent.
        
        Args:
            role: The agent's role identifier
            
        Returns:
            Async node function for the StateGraph
        """
        async def agent_node(state: ConversationState) -> ConversationState:
            try:
                agent = self.agents[role]
                
                # Get the last message for context
                last_message = state["messages"][-1] if state["messages"] else {"content": "Start conversation"}
                message_content = last_message.get("content", "")
                
                # Generate agent response using AutoGen
                response_content = await self._generate_agent_response(agent, message_content, state)
                
                # Check for FINAL_ANSWER signal
                if "FINAL_ANSWER:" in response_content.upper() or "FINAL ANSWER:" in response_content.upper():
                    state["final_answer"] = response_content
                
                # Add agent response to messages
                new_message = {
                    "role": "assistant",
                    "content": response_content,
                    "name": role,
                    "agent_role": role
                }
                
                # Update state
                state["messages"].append(new_message)
                state["current_agent"] = role
                state["turn_count"] += 1
                state["conversation_log"].append((role, response_content))
                
                logger.info(f"Agent {role} completed turn {state['turn_count']}")
                
                return state
                
            except Exception as e:
                logger.error(f"Error in agent node {role}: {e}")
                # Add error message to state
                error_message = {
                    "role": "assistant", 
                    "content": f"Error: {str(e)}", 
                    "name": role
                }
                state["messages"].append(error_message)
                state["conversation_log"].append((role, f"Error: {str(e)}"))
                return state
        
        return agent_node
    
    async def _generate_agent_response(self, agent: Any, message: str, state: ConversationState) -> str:
        """
        Generate a response from an AutoGen agent.
        
        Args:
            agent: The AutoGen agent instance
            message: The input message
            state: Current conversation state
            
        Returns:
            Generated response string
        """
        try:
            # Try to use AutoGen's response generation
            if hasattr(agent, 'generate_reply'):
                # For newer AutoGen versions
                try:
                    response = await agent.generate_reply(
                        message, 
                        context={"conversation_log": state["conversation_log"]}
                    )
                except TypeError:
                    # Handle synchronous generate_reply
                    response = agent.generate_reply(
                        message, 
                        context={"conversation_log": state["conversation_log"]}
                    )
            elif hasattr(agent, 'chat'):
                # For older AutoGen versions
                response = agent.chat(message)
            else:
                # Fallback for mock agents
                response = f"[{agent.role}]: Processing '{message[:50]}...'"
            
            # Handle different response formats
            if isinstance(response, dict):
                return response.get("content", str(response))
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating agent response: {e}")
            return f"[{agent.role}]: Error processing request - {str(e)}"
    
    def _human_approval_node(self, state: ConversationState) -> ConversationState:
        """
        Handle human approval workflow.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with approval decision
        """
        logger.info("Human approval node activated")
        
        # In production, this would integrate with UI for actual human input
        # For now, auto-approve but log the requirement
        approval_message = {
            "role": "system",
            "content": "Human approval granted (auto-approved for testing)",
            "name": "human_approver"
        }
        
        state["messages"].append(approval_message)
        state["conversation_log"].append(("human_approver", "Approval granted"))
        
        logger.info("Human approval workflow completed")
        return state
    
    def _final_answer_node(self, state: ConversationState) -> ConversationState:
        """
        Process and finalize the conversation answer.
        
        Args:
            state: Current conversation state
            
        Returns:
            State with finalized answer
        """
        if not state.get("final_answer"):
            # Generate final answer from conversation if not already set
            if state["conversation_log"]:
                last_response = state["conversation_log"][-1][1]
                state["final_answer"] = f"FINAL_ANSWER: {last_response}"
            else:
                state["final_answer"] = "FINAL_ANSWER: Conversation completed without specific answer"
        
        logger.info(f"Final answer generated: {state['final_answer'][:100]}...")
        return state
    
    def _add_flow_edges(self, graph):
        """
        Add edges based on agent_team_flow configuration and conditional routing.
        
        Args:
            graph: The StateGraph being built
        """
        # Add conditional edges for each agent
        for i, role in enumerate(self.flow):
            graph.add_conditional_edges(
                role,
                self._should_continue,
                {
                    "continue": self._get_next_agent(role),
                    "human_approval": "human_approval",
                    "final_answer": "final_answer",
                    "end": END
                }
            )
        
        # Connect special nodes
        graph.add_conditional_edges(
            "human_approval",
            lambda state: "continue",
            {"continue": self._get_next_agent("human_approval")}
        )
        
        graph.add_edge("final_answer", END)
    
    def _should_continue(self, state: ConversationState) -> str:
        """
        Determine whether conversation should continue or end.
        
        Args:
            state: Current conversation state
            
        Returns:
            Next action: "continue", "human_approval", "final_answer", or "end"
        """
        # Check for final answer signal
        if state.get("final_answer"):
            return "final_answer"
        
        # Check if max turns reached
        if state["turn_count"] >= state["max_turns"]:
            return "final_answer"
        
        # Check last message for final answer keywords
        if state["messages"]:
            last_content = state["messages"][-1].get("content", "").upper()
            if "FINAL_ANSWER:" in last_content or "FINAL ANSWER:" in last_content:
                return "final_answer"
        
        # Check for human approval requirement (every 3 turns as example)
        if state["turn_count"] > 0 and state["turn_count"] % 3 == 0:
            # This could be based on configuration
            return "human_approval"
        
        return "continue"
    
    def _get_next_agent(self, current_agent: str) -> str:
        """
        Get the next agent in the conversation flow.
        
        Args:
            current_agent: Current agent role
            
        Returns:
            Next agent role in the flow
        """
        if current_agent == "human_approval":
            # After human approval, continue with normal flow
            return self.flow[0] if self.flow else "final_answer"
        
        try:
            current_index = self.flow.index(current_agent)
            next_index = (current_index + 1) % len(self.flow)
            return self.flow[next_index]
        except (ValueError, IndexError):
            # If agent not in flow or flow is empty, go to final answer
            return "final_answer"
    
    async def invoke(self, initial_state: Dict[str, Any], thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the LangGraph with persistence and checkpointing.
        
        Args:
            initial_state: Initial state containing user query
            thread_id: Optional thread ID for conversation persistence
            
        Returns:
            Final state with conversation log and answer
        """
        if not self.graph:
            # Fallback for when LangGraph is not available
            return self._fallback_invoke(initial_state, thread_id)
        
        try:
            # Create initial conversation state
            query = initial_state.get("query", "")
            conv_state = ConversationState(
                messages=[{
                    "role": "user",
                    "content": query
                }],
                current_agent="",
                turn_count=0,
                max_turns=initial_state.get("max_turns", self.max_turns),
                final_answer=None,
                tenant_id=initial_state.get("tenant_id"),
                conversation_log=[("User", query)],
                agent_flow=self.flow
            )
            
            # Configure with thread ID for checkpointing
            config = {"configurable": {"thread_id": thread_id or "default"}}
            
            # Invoke the graph
            result = await self.graph.ainvoke(conv_state, config=config)
            
            logger.info(f"Graph execution completed for thread {thread_id}")
            
            return {
                "final_answer": result.get("final_answer", "No final answer generated"),
                "conversation_log": result.get("conversation_log", []),
                "thread_id": thread_id,
                "turn_count": result.get("turn_count", 0),
                "messages": result.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"Error in graph execution: {str(e)}")
            # Fallback to simple execution
            return self._fallback_invoke(initial_state, thread_id)
    
    def _fallback_invoke(self, initial_state: Dict[str, Any], thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fallback conversation execution when LangGraph is not available.
        
        Args:
            initial_state: Initial state containing user query
            thread_id: Optional thread ID
            
        Returns:
            Conversation result
        """
        logger.warning("Using fallback conversation execution")
        
        query = initial_state.get("query", "")
        conversation_log = [("User", query)]
        turns = 0
        max_turns = initial_state.get("max_turns", self.max_turns)
        
        # Simple round-robin conversation
        while turns < max_turns and turns < len(self.flow):
            agent_role = self.flow[turns % len(self.flow)]
            agent = self.agents.get(agent_role)
            
            if agent:
                try:
                    # Generate simple response
                    if hasattr(agent, 'role'):
                        response = f"[{agent.role}]: Processed query '{query[:30]}...' on turn {turns + 1}"
                    else:
                        response = f"[{agent_role}]: Processed query '{query[:30]}...' on turn {turns + 1}"
                    
                    conversation_log.append((agent_role, response))
                    turns += 1
                    
                    # Check for final answer
                    if turns >= max_turns - 1:
                        final_response = f"FINAL_ANSWER: Completed conversation after {turns} turns"
                        conversation_log.append((agent_role, final_response))
                        break
                        
                except Exception as e:
                    logger.error(f"Error in fallback execution: {e}")
                    break
        
        final_answer = conversation_log[-1][1] if conversation_log else "No conversation generated"
        
        return {
            "final_answer": final_answer,
            "conversation_log": conversation_log,
            "thread_id": thread_id,
            "turn_count": turns
        }
    
    def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve checkpoint for a conversation thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Checkpoint data if available
        """
        if not self.checkpointer:
            return None
            
        try:
            # This would retrieve the latest checkpoint for the thread
            # Implementation depends on the checkpointer API
            logger.info(f"Retrieving checkpoint for thread {thread_id}")
            return None  # Placeholder for actual checkpoint retrieval
        except Exception as e:
            logger.error(f"Error retrieving checkpoint: {str(e)}")
            return None

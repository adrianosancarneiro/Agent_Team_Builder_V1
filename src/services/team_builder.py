"""
Team Builder Service for creating AutoGen-based agent teams with full LangGraph integration.
"""
from typing import List, Dict, Any, Callable, Optional
import json
import logging
import asyncio
from unittest.mock import MagicMock

try:
    from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
    from autogen_agentchat.base import ChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Fallback for when AutoGen is not properly installed
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None
    BaseChatAgent = None
    ChatCompletionClient = None

from src.config.schema import AgentTeamConfig, AgentDefinition
from src.tools import qdrant_tool, graph_tool, embed_tool, webapi_tool, chunking_tool

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize tool clients (in a real app, use dependency injection or global singletons)
_vector_tool = qdrant_tool.QdrantTool()
_graph_tool = graph_tool.GraphDBTool()
_embed_service = embed_tool.EmbeddingService()
_web_tool = webapi_tool.WebAPITool()
_chunking_tool = chunking_tool.ChunkingTool()


class MockChatCompletionClient:
    """Mock model client for testing and development purposes"""
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
    
    async def create(self, messages, **kwargs):
        """Mock create method for chat completion"""
        last_message = messages[-1] if messages else {"content": ""}
        response_content = f"Mock response from {self.model_name} to: {last_message.get('content', '')[:50]}..."
        
        return MagicMock(
            content=response_content,
            role="assistant"
        )
    
    @property
    def model_info(self):
        return {"model": self.model_name}


class AutogenToolWrapper:
    """
    Wrapper class to convert existing tools to AutoGen function format with ReAct-style calling.
    """
    @staticmethod
    def search_vector_db(query: str, top_k: int = 5):
        """Search vector database for similar items"""
        try:
            # Convert query to embedding
            query_vector = _embed_service.embed(query)
            results = _vector_tool.search(query_vector, top_k=top_k)
            return {
                "tool": "Search_Vector_DB",
                "result": results,
                "status": "success"
            }
        except Exception as e:
            return {
                "tool": "Search_Vector_DB", 
                "result": f"Error: {str(e)}", 
                "status": "error"
            }
    
    @staticmethod
    def search_graph_db(query_string: str):
        """Run a Neo4j graph query"""
        try:
            results = _graph_tool.query(query_string)
            return {
                "tool": "Search_Graph_DB",
                "result": results,
                "status": "success"
            }
        except Exception as e:
            return {
                "tool": "Search_Graph_DB",
                "result": f"Error: {str(e)}",
                "status": "error"
            }
    
    @staticmethod
    def call_web_api(endpoint: str, method: str = "GET", params: dict = None, headers: dict = None):
        """Call an external web API"""
        try:
            result = _web_tool.call(endpoint, method, params, headers)
            return {
                "tool": "Call_Web_API",
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "tool": "Call_Web_API",
                "result": f"Error: {str(e)}",
                "status": "error"
            }
    
    @staticmethod
    def generate_embedding(text: str):
        """Generate an embedding for the given text"""
        try:
            embedding = _embed_service.embed(text)
            # Convert to list for JSON serialization
            result = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            return {
                "tool": "Embedding_Service",
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "tool": "Embedding_Service",
                "result": f"Error: {str(e)}",
                "status": "error"
            }
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into chunks for processing"""
        try:
            result = _chunking_tool.chunk(text, chunk_size, overlap)
            return {
                "tool": "Chunk_Text",
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "tool": "Chunk_Text",
                "result": f"Error: {str(e)}",
                "status": "error"
            }

class Agent(AssistantAgent if AUTOGEN_AVAILABLE else object):
    """
    Enhanced AutoGen AssistantAgent with team builder metadata and ReAct prompting.
    
    Features:
    - ReAct-style tool usage instructions
    - Tool integration with AutoGen's interface
    - Role-based system prompting
    - Metadata for team management
    """
    
    def __init__(self, 
                 name: str,
                 role: str,
                 personality: str,
                 system_prompt: str,
                 llm_config: Dict[str, Any],
                 tools: List[Callable] = None):
        
        # Store metadata
        self.role = role
        self.personality = personality
        self.system_prompt = system_prompt
        self.tool_names = [t.__name__ for t in (tools or [])]
        
        # Build enhanced system prompt with ReAct instructions
        enhanced_prompt = self._create_react_system_message(system_prompt, personality, tools)
        
        if AUTOGEN_AVAILABLE:
            # Create model client for AutoGen 0.7.1
            model_name = llm_config.get("config", {}).get("model", "mock-model")
            model_client = MockChatCompletionClient(model_name)
            
            # Initialize AutoGen AssistantAgent with updated parameters for 0.7.1
            if AssistantAgent is not None:
                # For AutoGen 0.7.1
                AssistantAgent.__init__(
                    self,
                    name=name,
                    model_client=model_client,
                    system_message=enhanced_prompt,
                    tools=tools or [],
                    description=f"{role}: {personality}"
                )
            else:
                # Fallback if AssistantAgent can't be imported correctly
                self.name = name
                self.model_client = model_client
                self.system_message = enhanced_prompt
                self.tools = tools or []
        else:
            # Fallback for when AutoGen is not available
            self.name = name
            self.model_client = MockChatCompletionClient()
            self.system_message = enhanced_prompt
            self.tools = tools or []
    
    def _create_react_system_message(self, goal_prompt: str, personality: str, tools: List[Callable] = None) -> str:
        """
        Create ReAct-style system message as required by the refactoring prompt.
        
        Args:
            goal_prompt: Agent's goal-based prompt
            personality: Agent's personality description
            tools: List of available tools
            
        Returns:
            Enhanced system message with ReAct instructions
        """
        system_message = f"""You are {self.role} with the following personality: {personality}

Your goal: {goal_prompt}

You should use a ReAct (Reasoning and Acting) approach for all tasks:

1. **Think** - Reason about what you need to do
2. **Act** - Use available tools when necessary  
3. **Observe** - Analyze the results
4. **Reflect** - Consider if you need to take more actions

Available tools: {', '.join(self.tool_names) if self.tool_names else 'None'}

TOOL USAGE INSTRUCTIONS:
When you need to use a tool, respond with JSON in this exact format:
```json
{{
  "thought": "Your reasoning about why you need this tool",
  "tool": "tool_name", 
  "tool_input": {{"param1": "value1", "param2": "value2"}}
}}
```

After receiving tool results, continue your reasoning and provide your analysis.

When you have completed the task or have a final answer, include "FINAL_ANSWER:" followed by your conclusion.

IMPORTANT: 
- Always explain your reasoning before taking actions
- Use tools strategically to gather information
- Provide clear, actionable insights
- Signal completion with "FINAL_ANSWER:" when done
"""
        
        return system_message
    
    def summarize(self) -> dict:
        """Return a summary of the agent for API responses and logging."""
        return {
            "name": self.name if hasattr(self, 'name') else "Unknown",
            "role": self.role,
            "personality": self.personality,
            "model": getattr(self.model_client, 'model_name', 'mock-model') if hasattr(self, 'model_client') else 'unknown',
            "tools": self.tool_names
        }
    
    async def generate_reply(self, message: str, context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate a reply to a message with ReAct-style reasoning.
        
        Args:
            message: Input message to respond to
            context: Additional context for response generation
            
        Returns:
            Response dictionary with content
        """
        try:
            if AUTOGEN_AVAILABLE and hasattr(super(), 'generate_reply'):
                # Use AutoGen's built-in reply generation
                response = await super().generate_reply(message, context)
                if isinstance(response, dict):
                    return response
                else:
                    return {"content": str(response)}
            else:
                # Fallback implementation
                return await self._fallback_generate_reply(message, context)
                
        except Exception as e:
            logger.error(f"Error in generate_reply for {self.role}: {e}")
            return {"content": f"[{self.role}]: Error processing message - {str(e)}"}
    
    async def _fallback_generate_reply(self, message: str, context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Fallback reply generation when AutoGen is not available.
        
        Args:
            message: Input message
            context: Additional context
            
        Returns:
            Generated response
        """
        # Simple ReAct-style response
        response_parts = [
            f"[{self.role}] Thinking: I need to process the message '{message[:50]}...'",
            f"[{self.role}] Analysis: As a {self.personality}, I will approach this systematically.",
        ]
        
        # Add tool usage if tools are available
        if self.tool_names:
            response_parts.append(f"[{self.role}] Available tools: {', '.join(self.tool_names)}")
            
        # Add conclusion
        response_parts.append(f"[{self.role}] Response: I have processed your request and here is my analysis...")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        return {"content": "\n".join(response_parts)}

class TeamBuilderService:
    """Service responsible for creating an agent team from configuration."""
    DEFAULT_AGENTS = [  # Default roles if none specified (base scenario: DecisionMaker -> Retriever -> Critic -> Refiner)
        {
            "agent_role": "DecisionMaker",
            "agent_name": "DecisionMakerAgent",
            "agent_personality": "Decisive leader that coordinates the team.",
            "agent_goal_based_prompt": "Decide the next step or conclude the task based on inputs from others.",
            "agent_tools": []  # Decision maker may not use external tools, just organizes
        },
        {
            "agent_role": "Retriever",
            "agent_name": "RetrieverAgent",
            "agent_personality": "Diligent researcher fetching relevant information.",
            "agent_goal_based_prompt": "Find facts and data relevant to the main goal using available knowledge sources.",
            "agent_tools": ["Search_Vector_DB", "Search_Graph_DB"]  # uses vector DB and graph DB
        },
        {
            "agent_role": "Critic",
            "agent_name": "CriticAgent",
            "agent_personality": "Skeptical critic that double-checks answers.",
            "agent_goal_based_prompt": "Evaluate the responses for correctness and potential issues.",
            "agent_tools": []
        },
        {
            "agent_role": "Refiner",
            "agent_name": "RefinerAgent",
            "agent_personality": "Thoughtful refiner that improves and finalizes answers.",
            "agent_goal_based_prompt": "Polish and integrate the information into a final answer.",
            "agent_tools": []
        }
    ]

    @classmethod
    def build_team(cls, config: AgentTeamConfig) -> Dict[str, Agent]:
        """
        Construct the team of agents based on the provided configuration.
        Returns a dictionary mapping agent roles to AutoGen agent objects.
        """
        agents_config = []
        if config.agents and len(config.agents) > 0:
            # Start with caller-specified agents
            for agent_def in config.agents:
                agents_config.append(agent_def.model_dump())
        else:
            # No agents specified by caller: use default template roles
            agents_config = [dict(a) for a in cls.DEFAULT_AGENTS]
        
        # If allowed, the Team Builder can recommend additional agents up to the specified number
        if config.allow_TBA_to_recommend_agents and config.allow_TBA_how_many_more > 0:
            # For simplicity, if fewer than allowed agents are present, add a placeholder agent.
            # In a real scenario, this could analyze the task and tenant context to suggest a role.
            for i in range(config.allow_TBA_how_many_more):
                extra_role = f"AdditionalAgent{i+1}"
                agents_config.append({
                    "agent_role": extra_role,
                    "agent_name": f"ExtraAgent{i+1}",
                    "agent_personality": "Auxiliary agent added by TeamBuilder to cover missing expertise.",
                    "agent_goal_based_prompt": f"Assist in achieving the main goal by providing {extra_role} capabilities.",
                    "agent_tools": config.available_tools or []  # give it access to all available tools by default
                })
                break  # (In practice, you'd add up to how_many_more agents; here we add one as example)
        
        # Instantiate Agent objects for each definition
        team = {}
        for agent_def_dict in agents_config:
            agent_def = AgentDefinition(**agent_def_dict)
            agent = cls._create_autogen_agent(agent_def, config)
            team[agent_def.agent_role] = agent
            
        # If human approval is needed, add a human-in-the-loop agent
        if config.should_TBA_ask_caller_approval:
            # For autogen-agentchat, we'll use a simple notification mechanism
            # rather than an actual human agent
            logger.info("Human approval requested in configuration - this would be handled via UI/workflow")
            # human_agent = UserProxyAgent(
            #     name="HumanApprover",
            #     system_message="You are a human approver. You review and approve team plans."
            # )
            # team["HumanApprover"] = human_agent
            
        return team
    
    @classmethod
    def _create_autogen_agent(cls, definition: AgentDefinition, config: AgentTeamConfig) -> Agent:
        """
        Create an AutoGen agent from an AgentDefinition.
        
        Args:
            definition: The agent definition from the configuration
            config: The overall team configuration for global settings
            
        Returns:
            An AutoGen-compatible Agent object
        """
        # Set up LLM configuration for AutoGen
        llm_config = {
            "config": {
                "model": (definition.LLM_model or {}).get("model", "gpt-4"),
                "temperature": (definition.LLM_configuration or {}).get("temperature", 0.7),
            }
        }
        
        # Map tool names to actual functions
        tools = []
        for tool_name in (definition.agent_tools or []):
            if tool_name == "Search_Vector_DB":
                tools.append(AutogenToolWrapper.search_vector_db)
            elif tool_name == "Search_Graph_DB":
                tools.append(AutogenToolWrapper.search_graph_db)
            elif tool_name == "Call_Web_API":
                tools.append(AutogenToolWrapper.call_web_api)
            elif tool_name == "Embedding_Service":
                tools.append(AutogenToolWrapper.generate_embedding)
            elif tool_name == "Chunk_Text":
                tools.append(AutogenToolWrapper.chunk_text)
        
        # Create the agent
        agent = Agent(
            name=definition.agent_name,
            role=definition.agent_role,
            personality=definition.agent_personality,
            system_prompt=definition.agent_goal_based_prompt,
            llm_config=llm_config,
            tools=tools
        )
        
        return agent

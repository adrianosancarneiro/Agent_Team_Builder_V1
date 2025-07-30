"""
Team Builder Service for creating AutoGen-based agent teams.
"""
from typing import List, Dict, Any, Callable, Optional
import json
import logging

from autogen import ConversableAgent

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

class AutogenToolWrapper:
    """
    Wrapper class to convert existing tools to AutoGen function format.
    """
    @staticmethod
    def search_vector_db(query: str, top_k: int = 5):
        """Search vector database for similar items"""
        # Convert query to embedding
        query_vector = _embed_service.embed(query)
        results = _vector_tool.search(query_vector, top_k=top_k)
        return results
    
    @staticmethod
    def search_graph_db(query_string: str):
        """Run a Neo4j graph query"""
        results = _graph_tool.query(query_string)
        return results
    
    @staticmethod
    def call_web_api(endpoint: str, method: str = "GET", params: dict = None, headers: dict = None):
        """Call an external web API"""
        result = _web_tool.call(endpoint, method, params, headers)
        return result
    
    @staticmethod
    def generate_embedding(text: str):
        """Generate an embedding for the given text"""
        embedding = _embed_service.embed(text)
        # Convert to list for JSON serialization
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into chunks for processing"""
        return _chunking_tool.chunk(text, chunk_size, overlap)

class Agent(ConversableAgent):
    """Enhanced ConversableAgent with team builder metadata"""
    def __init__(self, 
                 name: str,
                 role: str,
                 personality: str,
                 system_prompt: str,
                 llm_config: Dict[str, Any],
                 tools: List[Callable] = None):
        
        # Build the enhanced system prompt with personality and ReAct instructions
        enhanced_prompt = f"{system_prompt}\n\nYou are a {personality}.\n\n"
        
        # Add ReAct instruction if tools are provided
        if tools:
            enhanced_prompt += (
                "INSTRUCTIONS FOR USING TOOLS:\n"
                "1. To use a tool, respond with JSON in this exact format:\n"
                "```json\n"
                "{\n"
                '  "thought": "your reasoning here",\n'
                '  "tool": "tool_name",\n'
                '  "tool_input": {"param1": "value1", ...}\n'
                "}\n"
                "```\n"
                "2. Wait for the tool result before continuing.\n"
                "3. When you have a final answer or recommendation, include 'FINAL_ANSWER:' in your response.\n"
            )
            
        # Create the AutoGen agent
        super().__init__(
            name=name,
            system_message=enhanced_prompt,
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Register tools if provided
        if tools:
            for tool in tools:
                self.register_function(tool)
        
        # Store metadata for the team builder
        self.role = role
        self.personality = personality
        self.tool_names = [t.__name__ for t in (tools or [])]

    def summarize(self) -> dict:
        """Return a summary of the agent (for API responses or logging)."""
        return {
            "name": self.name,
            "role": self.role,
            "personality": self.personality,
            "model": self.llm_config.get("config", {}).get("model", "default-LLM") if hasattr(self, "llm_config") else "default-LLM",
            "tools": self.tool_names
        }

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
            human_agent = ConversableAgent(
                name="HumanApprover",
                system_message="You are a human approver. You review and approve team plans.",
                human_input_mode="ALWAYS"
            )
            team["HumanApprover"] = human_agent
            
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

from src.config.schema import AgentTeamConfig, AgentDefinition
from src.tools import qdrant_tool, graph_tool, embed_tool

# Initialize tool clients (in a real app, use dependency injection or global singletons)
_vector_tool = qdrant_tool.QdrantTool()
_graph_tool = graph_tool.GraphDBTool()
_embed_service = embed_tool.EmbeddingService()

class Agent:
    """Representation of an AI agent (LLM-powered) with certain tools and persona."""
    def __init__(self, definition: AgentDefinition):
        self.role = definition.agent_role
        self.name = definition.agent_name
        self.personality = definition.agent_personality
        self.prompt = definition.agent_goal_based_prompt
        self.model_info = definition.LLM_model or {"model": "default-LLM"}
        self.allow_model_override = definition.allow_team_builder_to_override_model
        self.model_config = definition.LLM_configuration or {}
        # Tools that this agent can use, instantiated from tool names
        self.tools = []
        for tool_name in (definition.agent_tools or []):
            if tool_name == "Search_Vector_DB":
                self.tools.append(_vector_tool)
            elif tool_name == "Search_Graph_DB":
                self.tools.append(_graph_tool)
            elif tool_name == "Call_Web_API":
                from src.tools import webapi_tool
                self.tools.append(webapi_tool.WebAPITool())
            elif tool_name == "Embedding_Service":
                self.tools.append(_embed_service)
            # Add other tool name checks (e.g., "Chunk_Text") as needed
        # If EmbeddingService is used as a tool, maybe it's for generating embeddings on the fly.

    def summarize(self) -> dict:
        """Return a summary of the agent (for API responses or logging)."""
        return {
            "name": self.name,
            "role": self.role,
            "personality": self.personality,
            "model": self.model_info,
            "tools": [t.__class__.__name__ for t in self.tools]
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
    def build_team(cls, config: AgentTeamConfig) -> list:
        """
        Construct the team of agents based on the provided configuration.
        Returns a list of Agent objects.
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
        team = [Agent(AgentDefinition(**agent_def)) for agent_def in agents_config]
        # If a static flow is defined (agent_team_flow), we could reorder team or mark the order somewhere
        # For simplicity, assume order in list will be the conversation order if flow is given.
        # (In a full implementation, parse agent_team_flow string to enforce speaking order.)
        return team

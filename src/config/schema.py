from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class AgentToolConfig(BaseModel):
    """Configuration for a tool that an agent can use."""
    name: str  # e.g., "Search_Vector_DB", "Search_Graph_DB", "Call_Web_API"
    params: Optional[Dict[str, Any]] = None  # parameters or settings for the tool (if any)

class AgentDefinition(BaseModel):
    """Definition of a single agent in the team."""
    agent_role: str = Field(..., description="The agent's role or specialization (e.g. Retriever, Critic)")
    agent_name: str = Field(..., description="Human-readable identifier for the agent")
    agent_personality: str = Field(..., description="Brief description of the agent's personality or perspective")
    agent_goal_based_prompt: str = Field(..., description="Role-specific instructions or prompt for this agent")
    LLM_model: Optional[Dict[str, str]] = Field(None, description="Model name or config (e.g., {'model': 'gpt-4'})")
    allow_team_builder_to_override_model: bool = Field(True, description="If true, Team Builder can change the model")
    LLM_configuration: Optional[Dict[str, float]] = Field(None, description="Model params like temperature, etc.")
    agent_tools: Optional[List[str]] = Field(None, description="List of tool names this agent can use")

class AgentTeamConfig(BaseModel):
    """Pydantic model for the agent team configuration schema."""
    agent_team_main_goal: str = Field(..., description="Primary goal or problem statement for the agent team")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for context loading and multi-tenancy isolation")
    # Flags controlling dynamic team building:
    allow_TBA_to_recommend_agents: bool = Field(False, description="Allow Team Builder Agent to add extra agents beyond those specified")
    allow_TBA_how_many_more: int = Field(0, description="Max number of additional agents TBA can add if allowed")
    should_TBA_ask_caller_approval: bool = Field(False, description="If true, pause and require human approval for additions")
    # Optional conversation flow and limits:
    agent_team_flow: Optional[str] = Field(None, description="Preset conversation turn order, e.g. 'Retriever->Critic->Refiner'")
    max_turns: int = Field(5, description="Max number of conversation rounds to execute")
    # Tools available globally (for assignment to agents):
    available_tools: Optional[List[str]] = Field(None, description="Whitelist of tools/APIs available for agents")
    # Pre-defined agents (partial or full specification):
    agents: Optional[List[AgentDefinition]] = Field(None, description="List of agent definitions to include in the team")

"""
Test the team builder's ability to create AutoGen agents.
"""
import pytest
from autogen import ConversableAgent
from src.config.schema import AgentTeamConfig
from src.services.team_builder import TeamBuilderService

def test_build_team_with_autogen():
    """Test that team builder returns AutoGen ConversableAgent instances"""
    config = AgentTeamConfig(
        agent_team_main_goal="Test the team builder",
        tenant_id="test_tenant",
        agents=[
            {
                "agent_role": "TestAgent",
                "agent_name": "Tester",
                "agent_personality": "Thorough tester",
                "agent_goal_based_prompt": "Test everything carefully",
                "agent_tools": ["Search_Vector_DB"]
            }
        ]
    )
    
    team = TeamBuilderService.build_team(config)
    
    # Check that we got a dictionary with at least one agent
    assert team, "Team should not be empty"
    assert isinstance(team, dict), "Team should be a dictionary"
    assert "TestAgent" in team, "Team should contain the requested agent"
    
    # Check that the agent is an AutoGen ConversableAgent
    agent = team["TestAgent"]
    assert isinstance(agent, ConversableAgent), "Agent should be a ConversableAgent"
    
    # Check that the agent has the expected attributes
    assert agent.name == "Tester", "Agent should have the right name"
    assert hasattr(agent, "role"), "Agent should have a role attribute"
    assert agent.role == "TestAgent", "Agent should have the right role"
    
    # Check the tool configuration
    assert hasattr(agent, "tool_names"), "Agent should have tool_names attribute"
    assert "search_vector_db" in agent.tool_names, "Agent should have search_vector_db tool"

def test_human_approval_agent():
    """Test that the human approval agent is created when requested"""
    config = AgentTeamConfig(
        agent_team_main_goal="Test human approval",
        tenant_id="test_tenant",
        should_TBA_ask_caller_approval=True,
        agents=[
            {
                "agent_role": "TestAgent",
                "agent_name": "Tester",
                "agent_personality": "Thorough tester",
                "agent_goal_based_prompt": "Test everything carefully"
            }
        ]
    )
    
    team = TeamBuilderService.build_team(config)
    
    # Check that the human approver is included
    assert "HumanApprover" in team, "Team should include a HumanApprover"
    human_agent = team["HumanApprover"]
    assert human_agent.human_input_mode == "ALWAYS", "Human agent should be set to ALWAYS get input"

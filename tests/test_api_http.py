"""
Test the FastAPI endpoints with the LangGraph implementation.
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.main import app
from src.services.team_builder import TeamBuilderService, Agent

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("src.services.team_builder.TeamBuilderService.build_team")
@patch("src.services.team_executor.TenantTeamGraph")
def test_build_team_minimal(mock_graph, mock_build_team):
    """Test building a team with minimal configuration"""
    # Mock the AutoGen agents
    mock_agent = MagicMock(spec=Agent)
    mock_agent.summarize.return_value = {
        "name": "TestAgent",
        "role": "Tester",
        "personality": "Thorough",
        "model": "test-model",
        "tools": []
    }
    mock_build_team.return_value = {"Tester": mock_agent}
    
    config = {
        "agent_team_main_goal": "Test goal for team building",
        "tenant_id": "test_tenant"
    }
    response = client.post("/build_team", json=config)
    assert response.status_code == 200
    data = response.json()
    assert "agent_team" in data
    assert len(data["agent_team"]) > 0

@patch("src.services.team_builder.TeamBuilderService.build_team")
@patch("src.services.team_executor.TenantTeamGraph")
def test_build_and_execute(mock_graph, mock_build_team):
    """Test the full build and execute workflow with mocks"""
    # Mock the AutoGen agents
    mock_agent = MagicMock(spec=Agent)
    mock_agent.summarize.return_value = {
        "name": "AnalystAgent",
        "role": "Analyst",
        "personality": "Analytical",
        "model": "test-model",
        "tools": []
    }
    mock_build_team.return_value = {"Analyst": mock_agent}
    
    # Mock the graph execution
    import asyncio
    async def mock_invoke(*args, **kwargs):
        return {
            "final_answer": "This is the final answer",
            "conversation_log": [
                ("User", "Analyze customer feedback"),
                ("Analyst", "This is the final answer")
            ],
            "thread_id": "test-thread-id"
        }
        
    mock_graph_instance = MagicMock()
    mock_graph_instance.invoke = mock_invoke
    mock_graph.return_value = mock_graph_instance
    
    config = {
        "agent_team_main_goal": "Analyze customer feedback",
        "tenant_id": "test_tenant",
        "max_turns": 2,
        "agents": [
            {
                "agent_role": "Analyst",
                "agent_name": "DataAnalyst",
                "agent_personality": "Analytical thinker",
                "agent_goal_based_prompt": "Analyze the given data",
                "agent_tools": []  # No external tools
            }
        ]
    }
    response = client.post("/build_and_execute", json=config)
    assert response.status_code == 200
    data = response.json()
    assert "agent_team" in data
    assert "conversation_log" in data
    assert "final_answer" in data
    assert data["final_answer"] == "This is the final answer"

@patch("src.services.team_builder.TeamBuilderService.build_team")
@patch("src.services.team_executor.TenantTeamGraph")
def test_example_config(mock_graph, mock_build_team):
    """Test with the example configuration file"""
    # Mock the AutoGen agents
    mock_agents = {}
    for role in ["Retriever", "Critic", "Refiner"]:
        mock_agent = MagicMock(spec=Agent)
        mock_agent.summarize.return_value = {
            "name": f"{role}Agent",
            "role": role,
            "personality": "Professional",
            "model": "test-model",
            "tools": []
        }
        mock_agents[role] = mock_agent
    mock_build_team.return_value = mock_agents
    
    with open("configs/example_config.json", "r") as f:
        config = json.load(f)
    
    response = client.post("/build_team", json=config)
    assert response.status_code == 200
    data = response.json()
    assert "agent_team" in data
    # Should have at least the 3 predefined agents
    assert len(data["agent_team"]) >= 3

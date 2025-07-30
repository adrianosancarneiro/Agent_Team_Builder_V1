import pytest
import json
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_build_team_minimal():
    """Test building a team with minimal configuration"""
    config = {
        "agent_team_main_goal": "Test goal for team building",
        "tenant_id": "test_tenant"
    }
    response = client.post("/build_team", json=config)
    assert response.status_code == 200
    data = response.json()
    assert "agent_team" in data
    assert len(data["agent_team"]) > 0

def test_build_and_execute():
    """Test the full build and execute workflow"""
    config = {
        "agent_team_main_goal": "Analyze customer feedback",
        "tenant_id": "test_tenant",
        "max_turns": 2,
        # Use agents without tools to avoid external service dependencies
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

def test_example_config():
    """Test with the example configuration file"""
    with open("configs/example_config.json", "r") as f:
        config = json.load(f)
    
    response = client.post("/build_team", json=config)
    assert response.status_code == 200
    data = response.json()
    assert "agent_team" in data
    # Should have at least the 3 predefined agents
    assert len(data["agent_team"]) >= 3

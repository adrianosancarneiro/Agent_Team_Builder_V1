"""
Test the tool-calling capabilities of the agents.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.config.schema import AgentTeamConfig, AgentDefinition
from src.services.team_builder import TeamBuilderService, AutogenToolWrapper
from src.tools import qdrant_tool, graph_tool, embed_tool

@patch("src.services.team_builder._embed_service")
@patch("src.services.team_builder._vector_tool")
def test_vector_search_tool_call(mock_vector_tool, mock_embed_service):
    """Test that a Retriever agent can call the vector search tool"""
    # Mock the embedding service
    mock_embed_service.embed.return_value = np.array([0.1, 0.2, 0.3])
    
    # Mock the Qdrant tool
    mock_vector_tool.search.return_value = [
        {"id": "doc1", "score": 0.95, "payload": {"text": "Sample document"}}
    ]
    
    # Create a wrapper and test direct invocation
    result = AutogenToolWrapper.search_vector_db("test query")
    
    # Check that the tool called the right dependencies
    mock_embed_service.embed.assert_called_once_with("test query")
    mock_vector_tool.search.assert_called_once()
    
    # Check the result format
    assert isinstance(result, dict)
    assert "tool" in result
    assert "result" in result
    assert result["tool"] == "Search_Vector_DB"
    assert isinstance(result["result"], list)

@patch("src.services.team_builder._graph_tool")
def test_graph_query_tool_call(mock_graph_tool):
    """Test that an agent can call the graph database query tool"""
    # Mock the Neo4j tool
    mock_graph_tool.query.return_value = [
        {"node": {"name": "TestNode", "type": "Entity"}}
    ]
    
    # Create a wrapper and test direct invocation
    result = AutogenToolWrapper.search_graph_db("MATCH (n) RETURN n LIMIT 1")
    
    # Check that the tool called the right dependencies
    mock_graph_tool.query.assert_called_once_with("MATCH (n) RETURN n LIMIT 1")
    
    # Check the result format
    assert isinstance(result, dict)
    assert "tool" in result
    assert "result" in result
    assert result["tool"] == "Search_Graph_DB"
    assert isinstance(result["result"], list)

@patch("src.services.team_builder._vector_tool")
@patch("src.services.team_builder._embed_service")
def test_agent_with_tools(mock_embed_service, mock_vector_tool):
    """Test that an agent is created with the right tools and can use them"""
    # Create a test configuration
    config = AgentTeamConfig(
        agent_team_main_goal="Test tools",
        tenant_id="test_tenant",
        agents=[
            {
                "agent_role": "Retriever",
                "agent_name": "DataRetriever",
                "agent_personality": "Thorough researcher",
                "agent_goal_based_prompt": "Search for information",
                "agent_tools": ["Search_Vector_DB", "Search_Graph_DB"]
            }
        ]
    )

    # Update for AutoGen 0.7.1
    with patch("src.services.team_builder.AUTOGEN_AVAILABLE", return_value=True):
        # Build the team
        team = TeamBuilderService.build_team(config)

        # Check that the agent was created with tools
        assert "Retriever" in team
        agent = team["Retriever"]
        
        # Instead of checking register_function calls, check the tools attribute
        assert hasattr(agent, "tool_names"), "Agent should have tool_names attribute"
        assert "search_vector_db" in agent.tool_names
        assert "search_graph_db" in agent.tool_names

"""
Test the tool-calling capabilities of the agents.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.config.schema import AgentTeamConfig, AgentDefinition
from src.services.team_builder import TeamBuilderService, AutogenToolWrapper
from src.tools import qdrant_tool, graph_tool, embed_tool

@patch("src.tools.qdrant_tool.QdrantTool")
@patch("src.tools.embed_tool.EmbeddingService")
def test_vector_search_tool_call(mock_embed_service, mock_qdrant):
    """Test that a Retriever agent can call the vector search tool"""
    # Mock the embedding service
    mock_embed_instance = MagicMock()
    mock_embed_instance.embed.return_value = np.array([0.1, 0.2, 0.3])
    mock_embed_service.return_value = mock_embed_instance
    
    # Mock the Qdrant tool
    mock_qdrant_instance = MagicMock()
    mock_qdrant_instance.search.return_value = [
        {"id": "doc1", "score": 0.95, "payload": {"text": "Sample document"}}
    ]
    mock_qdrant.return_value = mock_qdrant_instance
    
    # Create a wrapper and test direct invocation
    result = AutogenToolWrapper.search_vector_db("test query")
    
    # Check that the tool called the right dependencies
    mock_embed_instance.embed.assert_called_once_with("test query")
    mock_qdrant_instance.search.assert_called_once()
    
    # Check the result format
    assert isinstance(result, list)
    assert len(result) > 0
    assert "id" in result[0]
    assert "score" in result[0]

@patch("src.tools.graph_tool.GraphDBTool")
def test_graph_query_tool_call(mock_graph_db):
    """Test that an agent can call the graph database query tool"""
    # Mock the Neo4j tool
    mock_graph_instance = MagicMock()
    mock_graph_instance.query.return_value = [
        {"node": {"name": "TestNode", "type": "Entity"}}
    ]
    mock_graph_db.return_value = mock_graph_instance
    
    # Create a wrapper and test direct invocation
    result = AutogenToolWrapper.search_graph_db("MATCH (n) RETURN n LIMIT 1")
    
    # Check that the tool called the right dependencies
    mock_graph_instance.query.assert_called_once_with("MATCH (n) RETURN n LIMIT 1")
    
    # Check the result format
    assert isinstance(result, list)
    assert len(result) > 0
    assert "node" in result[0]

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
    
    # Create a mock for the agent's register_function method
    with patch("autogen.ConversableAgent.register_function") as mock_register:
        # Build the team
        team = TeamBuilderService.build_team(config)
        
        # Check that the agent was created with tools
        assert "Retriever" in team
        agent = team["Retriever"]
        
        # Check that the register_function was called for each tool
        assert mock_register.call_count >= 2, "Should register at least 2 tools"
        
        # Check that the tool_names attribute contains the expected tools
        assert "search_vector_db" in agent.tool_names
        assert "search_graph_db" in agent.tool_names

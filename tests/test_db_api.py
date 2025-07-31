"""
Tests for the new database-backed team management functionality.

This tests the database models, services, and API endpoints for
team creation, updating, versioning, and deletion.
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.config.models import Base, AgentTeam, TenantAppConfigFile
from src.config.schema import AgentTeamConfig, AgentDefinition
from src.services.team_builder import TeamBuilderService
from src.services.team_updater import TeamUpdaterService
from src.services.team_deleter import TeamDeleterService
from src.main import app, get_db


# Setup file-based SQLite database for tests
# Using a file-based database avoids concurrency issues in FastAPI testing
import os
import tempfile

# Create a temporary file for the SQLite database
DB_FILE = tempfile.NamedTemporaryFile(delete=False).name
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Cleanup function to delete the DB file when done
def cleanup_db_file():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


# Import the text construct for SQLAlchemy 2.0
from sqlalchemy import text

# Create tables for tests
@pytest.fixture(scope="session")
def setup_db():
    """Create test database tables at session level to ensure they stay available"""
    # Create all tables defined in Base metadata
    Base.metadata.create_all(bind=engine)
    
    # Verify the tables were created
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]
        print(f"Available tables: {tables}")
    
    yield
    
    # Clean up after all tests are done
    Base.metadata.drop_all(bind=engine)
    cleanup_db_file()


# Function to clear the database between tests
def clear_database():
    """Clear all data from tables but keep the tables themselves"""
    with engine.begin() as conn:
        # Delete all data from tables in correct order to avoid constraint violations
        conn.execute(text("DELETE FROM tenant_app_config_file"))
        conn.execute(text("DELETE FROM agent_team"))

# Create a shared dependency override for the entire session
@pytest.fixture(scope="session")
def override_dependency():
    # Override the get_db dependency to use our test database
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()

# Create a test client that uses the test database
@pytest.fixture(scope="function")
def client(setup_db, override_dependency):
    # Clear data before each test
    clear_database()
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up data after test
    clear_database()


# Sample test data
@pytest.fixture
def sample_agent_config():
    return AgentTeamConfig(
        agent_team_main_goal="Solve the test problem",
        max_turns=5,
        agents=[
            AgentDefinition(
                agent_role="TestAgent",
                agent_name="TestAgent1",
                agent_personality="Helpful test agent",
                agent_goal_based_prompt="Help with testing",
                agent_tools=[]
            )
        ]
    )


@pytest.fixture
def sample_team_id():
    return uuid.uuid4()


@pytest.fixture
def sample_tenant_id():
    return uuid.uuid4()


@pytest.fixture
def sample_app_id():
    return uuid.uuid4()


# Mock for TeamBuilderService.build_team to avoid actual model calls
@pytest.fixture(autouse=True)
def mock_team_builder():
    with patch.object(TeamBuilderService, 'build_team') as mock:
        # Return a dict with at least one agent that has a summarize method
        agent_mock = MagicMock()
        agent_mock.summarize.return_value = {
            "name": "TestAgent1",
            "role": "TestAgent",
            "personality": "Helpful test agent",
            "tools": []
        }
        mock.return_value = {"TestAgent": agent_mock}
        yield mock


# Test creating a new team
def test_create_team(client, sample_agent_config, sample_tenant_id, sample_app_id):
    # Make request to create team
    response = client.post(
        "/build",
        json={
            "tenant_id": str(sample_tenant_id),
            "app_id": str(sample_app_id),
            "config": sample_agent_config.model_dump()
        }
    )
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["tenant_id"] == str(sample_tenant_id)
    assert data["config"]["agent_team_main_goal"] == "Solve the test problem"
    
    # Verify team exists in database
    with TestingSessionLocal() as db:
        team = db.query(AgentTeam).filter(AgentTeam.id == uuid.UUID(data["id"])).first()
        assert team is not None
        assert team.tenant_id == sample_tenant_id
        
        # Verify version was created
        version = db.query(TenantAppConfigFile).filter(
            TenantAppConfigFile.agent_team_id == team.id
        ).first()
        assert version is not None
        assert version.version == 1


# Test retrieving a team
def test_get_team(client, sample_agent_config, sample_tenant_id, sample_app_id):
    # Create a team first
    response = client.post(
        "/build",
        json={
            "tenant_id": str(sample_tenant_id),
            "app_id": str(sample_app_id),
            "config": sample_agent_config.model_dump()
        }
    )
    team_id = response.json()["id"]
    
    # Get the team
    response = client.get(f"/teams/{team_id}")
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == team_id
    assert data["config"]["agent_team_main_goal"] == "Solve the test problem"


# Test updating a team
def test_update_team(client, sample_agent_config, sample_tenant_id, sample_app_id):
    # Create a team first
    response = client.post(
        "/build",
        json={
            "tenant_id": str(sample_tenant_id),
            "app_id": str(sample_app_id),
            "config": sample_agent_config.model_dump()
        }
    )
    team_id = response.json()["id"]
    
    # Update the team with a new config
    updated_config = sample_agent_config.model_copy(deep=True)
    updated_config.agent_team_main_goal = "Updated Goal"
    
    response = client.post(
        f"/update_team/{team_id}",
        json={"config": updated_config.model_dump()}
    )
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["team_id"] == team_id
    assert data["new_version"] == 2  # Should be version 2 after update
    
    # Verify team was updated in database
    with TestingSessionLocal() as db:
        team = db.query(AgentTeam).filter(AgentTeam.id == uuid.UUID(team_id)).first()
        assert team.config_jsonb["agent_team_main_goal"] == "Updated Goal"
        
        # Verify version history
        versions = db.query(TenantAppConfigFile).filter(
            TenantAppConfigFile.agent_team_id == team.id
        ).order_by(TenantAppConfigFile.version).all()
        
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2
        assert versions[1].config_json["agent_team_main_goal"] == "Updated Goal"


# Test version history and restoration
def test_version_history_and_restore(client, sample_agent_config, sample_tenant_id, sample_app_id):
    # Create a team first
    response = client.post(
        "/build",
        json={
            "tenant_id": str(sample_tenant_id),
            "app_id": str(sample_app_id),
            "config": sample_agent_config.model_dump()
        }
    )
    team_id = response.json()["id"]
    
    # Update the team twice
    for i in range(2):
        updated_config = sample_agent_config.model_copy(deep=True)
        updated_config.agent_team_main_goal = f"Updated Goal {i+1}"
        client.post(
            f"/update_team/{team_id}",
            json={"config": updated_config.model_dump()}
        )
    
    # Get version history
    response = client.get(f"/version_history/{team_id}")
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["team_id"] == team_id
    assert len(data["versions"]) == 3  # Should have 3 versions
    assert data["versions"][0]["version"] == 3  # Latest version first
    assert data["versions"][2]["version"] == 1  # Original version last
    
    # Restore to version 1
    response = client.post(f"/restore_version/{team_id}/1")
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["team_id"] == team_id
    assert data["restored_version"] == 1
    assert data["new_version"] == 4  # Should be version 4 after restoration
    
    # Verify team was restored in database
    with TestingSessionLocal() as db:
        team = db.query(AgentTeam).filter(AgentTeam.id == uuid.UUID(team_id)).first()
        assert team.config_jsonb["agent_team_main_goal"] == "Solve the test problem"  # Original goal
# Test deleting a team
def test_delete_team(client, sample_agent_config, sample_tenant_id, sample_app_id):
    # Create a team first
    response = client.post(
        "/build",
        json={
            "tenant_id": str(sample_tenant_id),
            "app_id": str(sample_app_id),
            "config": sample_agent_config.model_dump()
        }
    )
    team_id = response.json()["id"]
    
    # Delete the team
    response = client.delete(f"/delete_team/{team_id}")
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["team_id"] == team_id
    assert data["status"] == "deleted"
    
    # Verify team is marked as deleted in database
    with TestingSessionLocal() as db:
        team = db.query(AgentTeam).filter(AgentTeam.id == uuid.UUID(team_id)).first()
        assert team.is_deleted is True
        assert team.deleted_at is not None
    
    # Attempt to get the deleted team should fail
    response = client.get(f"/teams/{team_id}")
    assert response.status_code == 404


# Test restoring a deleted team
def test_restore_deleted_team(client, sample_agent_config, sample_tenant_id, sample_app_id):
    # Create a team first
    response = client.post(
        "/build",
        json={
            "tenant_id": str(sample_tenant_id),
            "app_id": str(sample_app_id),
            "config": sample_agent_config.model_dump()
        }
    )
    team_id = response.json()["id"]
    
    # Delete the team
    client.delete(f"/delete_team/{team_id}")
    
    # Restore the team
    response = client.post(f"/restore_team/{team_id}")
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["team_id"] == team_id
    assert data["status"] == "restored"
    
    # Verify team is restored in database
    with TestingSessionLocal() as db:
        team = db.query(AgentTeam).filter(AgentTeam.id == uuid.UUID(team_id)).first()
        assert team.is_deleted is False
        assert team.deleted_at is None
    
    # Get the restored team
    response = client.get(f"/teams/{team_id}")
    assert response.status_code == 200


@patch("src.services.team_executor.TeamExecutorService.run_conversation")
def test_execute_team(mock_run_conversation, client, sample_agent_config, sample_tenant_id, sample_app_id):
    """Test executing a team from database"""
    # Mock the conversation result
    mock_run_conversation.return_value = "Test execution response"
    
    # Create a team first
    response = client.post(
        "/build",
        json={
            "tenant_id": str(sample_tenant_id),
            "app_id": str(sample_app_id),
            "config": sample_agent_config.model_dump()
        }
    )
    team_id = response.json()["id"]
    
    # Execute the team
    execution_request = {
        "user_query": "Test query for execution",
        "max_turns": 3
    }
    response = client.post(f"/execute/{team_id}", json=execution_request)
    
    # Assert response
    assert response.status_code == 200
    data = response.json()
    assert data["team_id"] == team_id
    assert data["final_answer"] == "Test execution response"
    assert "conversation_log" in data
    assert "agent_team" in data
    
    # Verify the run_conversation was called with the user query
    mock_run_conversation.assert_called_once_with(user_query=execution_request["user_query"])

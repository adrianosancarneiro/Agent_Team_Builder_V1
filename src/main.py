from fastapi import FastAPI, HTTPException, Depends, Body
from uuid import UUID
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, UUID4

from sqlalchemy.orm import Session

from src.config.schema import AgentTeamConfig
from src.services.team_builder import TeamBuilderService
from src.services.team_executor import TeamExecutorService
from src.services.team_updater import TeamUpdaterService
from src.services.team_deleter import TeamDeleterService
from src.config.db_models import AgentTeam, AgentTeamConfigVersion
from src.config.database import get_db

app = FastAPI(title="AI Team Builder Agent Service", version="2.0")


class TeamCreationRequest(BaseModel):
    """Request model for team creation."""
    tenant_id: UUID4
    config: AgentTeamConfig


class TeamResponse(BaseModel):
    """Response model for team data."""
    id: str
    tenant_id: str
    config: Dict[str, Any]
    created_at: str
    updated_at: str
    is_deleted: bool = False
    deleted_at: Optional[str] = None


class TeamUpdateRequest(BaseModel):
    """Request model for team update."""
    config: AgentTeamConfig


class TeamExecutionRequest(BaseModel):
    """Request model for team execution."""
    user_query: str
    max_turns: Optional[int] = 10


class VersionHistoryResponse(BaseModel):
    """Response model for version history."""
    team_id: str
    versions: List[Dict[str, Any]]

# Build the team and optionally run the conversation in one go (for simplicity).
@app.post("/build_and_execute")
def build_and_execute(config: AgentTeamConfig):
    """
    Build an AI agent team according to the provided configuration and run the multi-agent conversation.
    Returns the team composition and the final answer.
    """
    # Build the team of agents
    try:
        team = TeamBuilderService.build_team(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")
    # Log or store team info
    team_summary = [agent.summarize() for agent in team.values()]

    # Execute the conversation workflow
    flow = None
    if config.agent_team_flow:
        flow = [s.strip() for s in config.agent_team_flow.split("->")]
    
    executor = TeamExecutorService(
        agents=team, 
        flow=flow,
        max_turns=config.max_turns)
    final_answer = executor.run_conversation(user_query=config.agent_team_main_goal)

    # Return both the team details and the final answer
    return {
        "agent_team": team_summary,
        "conversation_log": executor.conversation_log,
        "final_answer": final_answer,
        "thread_id": executor.thread_id  # Include thread_id for conversation resumption
    }

    # (Optional) Separate endpoint to just build team without execution
@app.post("/build_team")
def build_team_endpoint(config: AgentTeamConfig):
    """
    Endpoint to build the agent team from config, without running the conversation.
    Returns the team composition.
    """
    try:
        team = TeamBuilderService.build_team(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")
    return {"agent_team": [agent.summarize() for agent in team.values()]}


# New database-backed team endpoints
@app.post("/build", response_model=TeamResponse)
def build_team_db(request: TeamCreationRequest, db: Session = Depends(get_db)):
    """
    Create a new agent team configuration in the database.
    
    - Validates and builds the team (ensuring config is correct)
    - Persists the team config and initial version to the database
    - Returns the new team details
    """
    # Validate and build team (does not persist, just to ensure no errors)
    try:
        team_agents = TeamBuilderService.build_team(request.config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    
    # Persist the new team config
    new_team = AgentTeam(
        tenant_id=request.tenant_id,
        config_json=request.config.model_dump()
    )
    db.add(new_team)
    db.flush()  # Get new_team.id assigned
    
    # Create initial version entry (version 1)
    version_entry = AgentTeamConfigVersion(
        team_id=new_team.id,
        version=1,
        config_json=request.config.model_dump()
    )
    db.add(version_entry)
    db.commit()
    db.refresh(new_team)
    
    # Return response with team details
    return TeamResponse(
        id=str(new_team.id),
        tenant_id=str(new_team.tenant_id),
        config=new_team.config_json,
        created_at=new_team.created_at.isoformat(),
        updated_at=new_team.updated_at.isoformat()
    )


@app.get("/teams/{team_id}", response_model=TeamResponse)
def get_team(team_id: UUID4, db: Session = Depends(get_db)):
    """
    Get a team configuration by ID.
    """
    # Retrieve team from database
    team = db.query(AgentTeam).filter(
        AgentTeam.id == team_id,
        AgentTeam.is_deleted == False
    ).first()
    
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found or is deleted")
    
    # Return team details
    return TeamResponse(
        id=str(team.id),
        tenant_id=str(team.tenant_id),
        config=team.config_json,
        created_at=team.created_at.isoformat(),
        updated_at=team.updated_at.isoformat()
    )


@app.get("/teams", response_model=List[TeamResponse])
def get_teams_by_tenant(tenant_id: UUID4, db: Session = Depends(get_db)):
    """
    Get all teams for a tenant.
    """
    # Retrieve teams from database
    teams = db.query(AgentTeam).filter(
        AgentTeam.tenant_id == tenant_id,
        AgentTeam.is_deleted == False
    ).all()
    
    # Return team details
    return [
        TeamResponse(
            id=str(team.id),
            tenant_id=str(team.tenant_id),
            config=team.config_json,
            created_at=team.created_at.isoformat(),
            updated_at=team.updated_at.isoformat()
        )
        for team in teams
    ]


@app.post("/execute/{team_id}")
def execute_team(team_id: UUID4, request: TeamExecutionRequest, db: Session = Depends(get_db)):
    """
    Execute a saved agent team by team_id.
    
    Loads the latest configuration from the database and runs the multi-agent conversation.
    Returns the team composition, conversation log, and final answer.
    """
    # Retrieve team config
    team_record = db.query(AgentTeam).filter(
        AgentTeam.id == team_id,
        AgentTeam.is_deleted == False
    ).first()
    
    if not team_record:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found or is deleted")
    
    # Parse stored config JSON back into AgentTeamConfig model
    try:
        config = AgentTeamConfig.model_validate(team_record.config_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stored configuration is invalid: {str(e)}")
    
    # Re-build agent instances from config and execute conversation
    agents = TeamBuilderService.build_team(config)
    
    # Extract flow if present
    flow = None
    if config.agent_team_flow:
        flow = [s.strip() for s in config.agent_team_flow.split("->")]
    
    executor = TeamExecutorService(
        agents=agents,
        flow=flow,
        max_turns=request.max_turns or config.max_turns
    )
    
    # Use provided user_query instead of the stored goal
    final_answer = executor.run_conversation(user_query=request.user_query)
    
    # Return execution results
    return {
        "team_id": str(team_id),
        "agent_team": [agent.summarize() for agent in agents.values()],
        "conversation_log": executor.conversation_log,
        "final_answer": final_answer,
        "thread_id": executor.thread_id
    }


@app.post("/update_team/{team_id}")
def update_team_endpoint(team_id: UUID4, request: TeamUpdateRequest, db: Session = Depends(get_db)):
    """
    Update an existing team's configuration with versioning.
    
    - Validates the new configuration
    - Creates a new version entry
    - Updates the main team record
    """
    try:
        result = TeamUpdaterService.update_team(db, team_id, request.config)
        return {
            "team_id": result["team_id"],
            "new_version": result["new_version"],
            "updated_at": result["updated_at"]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating team: {str(e)}")


@app.post("/restore_version/{team_id}/{version}")
def restore_version_endpoint(team_id: UUID4, version: int, db: Session = Depends(get_db)):
    """
    Restore a team to a previous configuration version.
    
    - Creates a new version based on the target version
    - Updates the main team record
    """
    try:
        result = TeamUpdaterService.restore_version(db, team_id, version)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restoring version: {str(e)}")


@app.get("/version_history/{team_id}", response_model=VersionHistoryResponse)
def get_version_history(team_id: UUID4, db: Session = Depends(get_db)):
    """
    Get the version history for a team.
    """
    try:
        versions = TeamUpdaterService.get_version_history(db, team_id)
        return {
            "team_id": str(team_id),
            "versions": versions
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving version history: {str(e)}")


@app.delete("/delete_team/{team_id}")
def delete_team_endpoint(team_id: UUID4, db: Session = Depends(get_db)):
    """
    Soft-delete a team by ID.
    
    - Sets is_deleted flag to True
    - Records deleted_at timestamp
    """
    try:
        result = TeamDeleterService.delete_team(db, team_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting team: {str(e)}")


@app.post("/restore_team/{team_id}")
def restore_team_endpoint(team_id: UUID4, db: Session = Depends(get_db)):
    """
    Restore a previously soft-deleted team.
    
    - Sets is_deleted flag to False
    - Clears deleted_at timestamp
    """
    try:
        result = TeamDeleterService.restore_deleted_team(db, team_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restoring team: {str(e)}")


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

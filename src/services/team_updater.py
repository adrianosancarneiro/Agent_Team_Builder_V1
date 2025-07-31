"""
Team Updater Service for updating and versioning agent team configurations.

This module provides functionality to update existing team configurations and track version history.
"""
from datetime import datetime
import json
import uuid
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy.future import select
from pydantic import UUID4

from src.config.schema import AgentTeamConfig
from src.services.team_builder import TeamBuilderService
from src.config.models import AgentTeam, TenantAppConfigFile


class TeamUpdaterService:
    """Service responsible for updating an existing agent team configuration with versioning."""
    
    @staticmethod
    def update_team(session: Session, team_id: UUID4, new_config: AgentTeamConfig) -> Dict[str, Any]:
        """
        Update the stored team configuration for the given team_id.
        - Validates new_config via TeamBuilderService.
        - Increments version and stores new config in TenantAppConfigFile.
        - Updates the AgentTeam record's config to the new version.
        
        Args:
            session: SQLAlchemy DB session
            team_id: UUID of the team to update
            new_config: Updated team configuration
            
        Returns:
            Dict with team_id, new_version, and updated_at timestamp
            
        Raises:
            ValueError: If team not found or is deleted
        """
        # 1. Retrieve existing team record
        team = session.query(AgentTeam).filter(
            AgentTeam.id == team_id,
            AgentTeam.is_deleted == False
        ).first()
        
        if not team:
            raise ValueError(f"Team ID {team_id} not found or is deleted")
        
        # 2. Validate new configuration by attempting to build the team
        TeamBuilderService.build_team(new_config)  # raises exception if invalid
        
        # 3. Determine next version number
        last_version = session.query(TenantAppConfigFile) \
                             .filter(TenantAppConfigFile.agent_team_id == team_id) \
                             .order_by(TenantAppConfigFile.version.desc()) \
                             .first()
        
        new_version_number = 1 if last_version is None else last_version.version + 1
        
        # 4. Store the new config in the versions history
        version_entry = TenantAppConfigFile(
            agent_team_id=team_id,
            version=new_version_number,
            config_json=new_config.model_dump()
        )
        session.add(version_entry)
        
        # 5. Update the main AgentTeam record with new config
        team.config_jsonb = new_config.model_dump()
        team.updated_at = datetime.now()
        
        # 6. Commit changes
        session.commit()
        
        # 7. Return update summary
        return {
            "team_id": str(team_id),
            "new_version": new_version_number,
            "updated_at": team.updated_at.isoformat()
        }
    
    @staticmethod
    def restore_version(session: Session, team_id: UUID4, target_version: int) -> Dict[str, Any]:
        """
        Rollback the team configuration to a previous version.
        Creates a new version entry identical to the target version and sets it as current.
        
        Args:
            session: SQLAlchemy DB session
            team_id: UUID of the team
            target_version: Version number to restore
            
        Returns:
            Dict with restoration details including team_id, restored_version, new_version
            
        Raises:
            ValueError: If team not found, deleted, or target version doesn't exist
        """
        # 1. Find the team
        team = session.query(AgentTeam).filter(
            AgentTeam.id == team_id,
            AgentTeam.is_deleted == False
        ).first()
        
        if not team:
            raise ValueError(f"Team ID {team_id} not found or is deleted")
        
        # 2. Find the target version entry
        version_entry = session.query(TenantAppConfigFile) \
                              .filter(
                                  TenantAppConfigFile.agent_team_id == team_id,
                                  TenantAppConfigFile.version == target_version
                              ).first()
        
        if not version_entry:
            raise ValueError(f"Version {target_version} not found for Team ID {team_id}")
        
        # 3. Determine next version number
        last_version = session.query(TenantAppConfigFile) \
                             .filter(TenantAppConfigFile.agent_team_id == team_id) \
                             .order_by(TenantAppConfigFile.version.desc()) \
                             .first()
        
        new_version_number = last_version.version + 1
        
        # 4. Create a new version as a copy of target_version
        new_version_entry = TenantAppConfigFile(
            agent_team_id=team_id,
            version=new_version_number,
            config_json=version_entry.config_json
        )
        session.add(new_version_entry)
        
        # 5. Update AgentTeam record to use the restored config
        team.config_jsonb = version_entry.config_json
        team.updated_at = datetime.now()
        
        # 6. Commit changes
        session.commit()
        
        # 7. Return restoration details
        return {
            "team_id": str(team_id),
            "restored_version": target_version,
            "new_version": new_version_number,
            "restored_at": team.updated_at.isoformat()
        }
    
    @staticmethod
    def get_version_history(session: Session, team_id: UUID4) -> list:
        """
        Get the version history for a team.
        
        Args:
            session: SQLAlchemy DB session
            team_id: UUID of the team
            
        Returns:
            List of versions with metadata (no config JSON)
            
        Raises:
            ValueError: If team not found or is deleted
        """
        # Check if team exists and is not deleted
        team = session.query(AgentTeam).filter(
            AgentTeam.id == team_id,
            AgentTeam.is_deleted == False
        ).first()
        
        if not team:
            raise ValueError(f"Team ID {team_id} not found or is deleted")
        
        # Get all versions for this team
        versions = session.query(TenantAppConfigFile) \
                         .filter(TenantAppConfigFile.agent_team_id == team_id) \
                         .order_by(TenantAppConfigFile.version.desc()) \
                         .all()
        
        # Return version summary information (without full config)
        return [
            {
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "id": str(v.id)
            }
            for v in versions
        ]

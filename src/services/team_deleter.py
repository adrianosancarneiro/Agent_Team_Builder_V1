"""
Team Deleter Service for soft-deleting agent team configurations.

This module provides functionality to soft-delete teams, preserving history while hiding them from active use.
"""
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import UUID4

from sqlalchemy.orm import Session

from src.config.db_models import AgentTeam


class TeamDeleterService:
    """Service responsible for soft-deleting (archiving) an agent team configuration."""
    
    @staticmethod
    def delete_team(session: Session, team_id: UUID4) -> Dict[str, Any]:
        """
        Soft delete the team by setting is_deleted True and recording deleted_at timestamp.
        
        Args:
            session: SQLAlchemy DB session
            team_id: UUID of the team to delete
            
        Returns:
            Dict with deletion details including team_id and deleted_at timestamp
            
        Raises:
            ValueError: If team not found or already deleted
        """
        # Find the team
        team = session.query(AgentTeam).filter(
            AgentTeam.id == team_id,
            AgentTeam.is_deleted == False
        ).first()
        
        if not team:
            raise ValueError(f"Team ID {team_id} not found or already deleted")
        
        # Set deletion flags
        team.is_deleted = True
        team.deleted_at = datetime.now()
        
        # Commit changes
        session.commit()
        
        return {
            "team_id": str(team_id),
            "deleted_at": team.deleted_at.isoformat(),
            "status": "deleted"
        }
    
    @staticmethod
    def restore_deleted_team(session: Session, team_id: UUID4) -> Dict[str, Any]:
        """
        Restore a previously soft-deleted team.
        
        Args:
            session: SQLAlchemy DB session
            team_id: UUID of the team to restore
            
        Returns:
            Dict with restoration details including team_id and restored_at timestamp
            
        Raises:
            ValueError: If team not found or not deleted
        """
        # Find the team
        team = session.query(AgentTeam).filter(
            AgentTeam.id == team_id,
            AgentTeam.is_deleted == True
        ).first()
        
        if not team:
            raise ValueError(f"Team ID {team_id} not found or not deleted")
        
        # Restore the team
        team.is_deleted = False
        team.deleted_at = None
        team.updated_at = datetime.now()
        
        # Commit changes
        session.commit()
        
        return {
            "team_id": str(team_id),
            "restored_at": team.updated_at.isoformat(),
            "status": "restored"
        }

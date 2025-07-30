"""Database helper module for the Agent Team Builder service.

This module provides functions to interact with the PostgreSQL database.
It handles connections, queries, and data manipulation for agent teams.
"""

import os
import uuid
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection parameters from environment variables
PG_DSN = os.getenv("PG_DSN", "postgresql://postgres_user:postgres_pass@localhost:5432/agentteambuilder")


async def get_pool() -> asyncpg.Pool:
    """Create and return a connection pool to the PostgreSQL database.
    
    Returns:
        asyncpg.Pool: A connection pool object
    """
    return await asyncpg.create_pool(dsn=PG_DSN)


async def create_team(tenant_id: Union[str, uuid.UUID], config_json: Dict[str, Any]) -> str:
    """Create a new agent team in the database.
    
    Args:
        tenant_id: The UUID of the tenant
        config_json: The team configuration JSON
    
    Returns:
        str: The UUID of the created team as a string
    """
    # Generate a new UUID for the team
    team_id = uuid.uuid4()
    
    # Get a connection from the pool
    pool = await get_pool()
    async with pool.acquire() as connection:
        # Insert the team
        await connection.execute(
            """
            INSERT INTO agent_teams (id, tenant_id, config_json, updated_at, created_at)
            VALUES ($1, $2, $3, now(), now())
            """,
            team_id,
            tenant_id if isinstance(tenant_id, uuid.UUID) else uuid.UUID(tenant_id),
            json.dumps(config_json)
        )
    
    return str(team_id)


async def get_team(team_id: Union[str, uuid.UUID]) -> Optional[Dict[str, Any]]:
    """Get a team by its ID.
    
    Args:
        team_id: The UUID of the team
    
    Returns:
        Optional[Dict[str, Any]]: The team data or None if not found
    """
    # Get a connection from the pool
    pool = await get_pool()
    async with pool.acquire() as connection:
        # Query the team
        row = await connection.fetchrow(
            """
            SELECT id, tenant_id, config_json, updated_at, created_at
            FROM agent_teams
            WHERE id = $1
            """,
            team_id if isinstance(team_id, uuid.UUID) else uuid.UUID(team_id)
        )
    
    if not row:
        return None
    
    # Convert the row to a dictionary
    return {
        "id": str(row["id"]),
        "tenant_id": str(row["tenant_id"]),
        "config": json.loads(row["config_json"]),
        "updated_at": row["updated_at"].isoformat(),
        "created_at": row["created_at"].isoformat()
    }


async def get_teams_by_tenant(tenant_id: Union[str, uuid.UUID]) -> List[Dict[str, Any]]:
    """Get all teams for a tenant.
    
    Args:
        tenant_id: The UUID of the tenant
    
    Returns:
        List[Dict[str, Any]]: A list of team data dictionaries
    """
    # Get a connection from the pool
    pool = await get_pool()
    async with pool.acquire() as connection:
        # Query the teams
        rows = await connection.fetch(
            """
            SELECT id, tenant_id, config_json, updated_at, created_at
            FROM agent_teams
            WHERE tenant_id = $1
            ORDER BY created_at DESC
            """,
            tenant_id if isinstance(tenant_id, uuid.UUID) else uuid.UUID(tenant_id)
        )
    
    # Convert the rows to dictionaries
    return [
        {
            "id": str(row["id"]),
            "tenant_id": str(row["tenant_id"]),
            "config": json.loads(row["config_json"]),
            "updated_at": row["updated_at"].isoformat(),
            "created_at": row["created_at"].isoformat()
        }
        for row in rows
    ]


async def update_team(team_id: Union[str, uuid.UUID], config_json: Dict[str, Any]) -> bool:
    """Update an existing team configuration.
    
    Args:
        team_id: The UUID of the team
        config_json: The updated team configuration
    
    Returns:
        bool: True if the team was updated, False if not found
    """
    # Get a connection from the pool
    pool = await get_pool()
    async with pool.acquire() as connection:
        # Update the team
        result = await connection.execute(
            """
            UPDATE agent_teams
            SET config_json = $1, updated_at = now()
            WHERE id = $2
            """,
            json.dumps(config_json),
            team_id if isinstance(team_id, uuid.UUID) else uuid.UUID(team_id)
        )
    
    # Check if any rows were updated
    return result != "UPDATE 0"


async def delete_team(team_id: Union[str, uuid.UUID]) -> bool:
    """Delete a team by its ID.
    
    Args:
        team_id: The UUID of the team
    
    Returns:
        bool: True if the team was deleted, False if not found
    """
    # Get a connection from the pool
    pool = await get_pool()
    async with pool.acquire() as connection:
        # Delete the team
        result = await connection.execute(
            """
            DELETE FROM agent_teams
            WHERE id = $1
            """,
            team_id if isinstance(team_id, uuid.UUID) else uuid.UUID(team_id)
        )
    
    # Check if any rows were deleted
    return result != "DELETE 0"

#!/bin/bash
#
# This script sets up Airbyte to sync data from PostgreSQL to Neo4j
# It uses the Airbyte API to create source and destination connections,
# and then creates a connection between them.
#
# Dependencies: curl, jq

set -e

# Load environment variables
source .env 2>/dev/null || echo "No .env file found, using default values."

# Airbyte settings
AIRBYTE_API_HOST=${AIRBYTE_API_HOST:-"localhost:8000"}
AIRBYTE_API_URL="http://${AIRBYTE_API_HOST}/api/v1"

# PostgreSQL source settings
PG_HOST=${PG_HOST:-"localhost"}
PG_PORT=${PG_PORT:-"5432"}
PG_DB=${PG_DB:-"agentteambuilder"}
PG_USER=${PG_USER:-"postgres_user"}
PG_PASSWORD=${PG_PASSWORD:-"postgres_pass"}

# Neo4j destination settings
NEO4J_HOST=${NEO4J_HOST:-"localhost"}
NEO4J_PORT=${NEO4J_PORT:-"7687"}
NEO4J_USER=${NEO4J_USER:-"neo4j"}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-"password"}
NEO4J_DATABASE=${NEO4J_DATABASE:-"neo4j"}

# Function to check if Airbyte is available
check_airbyte_availability() {
  echo "Checking if Airbyte API is available..."
  if ! curl -s "${AIRBYTE_API_URL}/health" > /dev/null; then
    echo "Error: Airbyte API is not available at ${AIRBYTE_API_URL}"
    echo "Please make sure Airbyte is running and the API host is correct."
    exit 1
  fi
  echo "Airbyte API is available."
}

# Function to create PostgreSQL source in Airbyte
create_postgres_source() {
  echo "Creating PostgreSQL source in Airbyte..."
  
  # Source configuration
  SOURCE_CONFIG='{
    "name": "Agent Team Builder PostgreSQL",
    "sourceDefinitionId": "decd338e-5647-4c0b-adf4-da0e75f5a750",
    "workspaceId": "5ae6b09b-fdec-41af-aaf7-7d94cfc33ef6",
    "connectionConfiguration": {
      "host": "'"$PG_HOST"'",
      "port": '"$PG_PORT"',
      "database": "'"$PG_DB"'",
      "username": "'"$PG_USER"'",
      "password": "'"$PG_PASSWORD"'",
      "schema": "public",
      "ssl": false
    }
  }'
  
  # Create source
  SOURCE_RESPONSE=$(curl -s -X POST "${AIRBYTE_API_URL}/sources/create" \
    -H "Content-Type: application/json" \
    -d "$SOURCE_CONFIG")
  
  # Extract source ID
  SOURCE_ID=$(echo "$SOURCE_RESPONSE" | jq -r '.sourceId')
  
  if [ "$SOURCE_ID" == "null" ] || [ -z "$SOURCE_ID" ]; then
    echo "Error creating PostgreSQL source:"
    echo "$SOURCE_RESPONSE" | jq .
    exit 1
  fi
  
  echo "PostgreSQL source created with ID: $SOURCE_ID"
  
  # Return source ID
  echo "$SOURCE_ID"
}

# Function to create Neo4j destination in Airbyte
create_neo4j_destination() {
  echo "Creating Neo4j destination in Airbyte..."
  
  # Destination configuration
  DESTINATION_CONFIG='{
    "name": "Agent Team Builder Neo4j",
    "destinationDefinitionId": "f2be84c5-3e8f-48aa-9bc7-1a08bc17bece",
    "workspaceId": "5ae6b09b-fdec-41af-aaf7-7d94cfc33ef6",
    "connectionConfiguration": {
      "host": "'"$NEO4J_HOST"'",
      "port": '"$NEO4J_PORT"',
      "database": "'"$NEO4J_DATABASE"'",
      "username": "'"$NEO4J_USER"'",
      "password": "'"$NEO4J_PASSWORD"'",
      "scheme": "bolt"
    }
  }'
  
  # Create destination
  DESTINATION_RESPONSE=$(curl -s -X POST "${AIRBYTE_API_URL}/destinations/create" \
    -H "Content-Type: application/json" \
    -d "$DESTINATION_CONFIG")
  
  # Extract destination ID
  DESTINATION_ID=$(echo "$DESTINATION_RESPONSE" | jq -r '.destinationId')
  
  if [ "$DESTINATION_ID" == "null" ] || [ -z "$DESTINATION_ID" ]; then
    echo "Error creating Neo4j destination:"
    echo "$DESTINATION_RESPONSE" | jq .
    exit 1
  fi
  
  echo "Neo4j destination created with ID: $DESTINATION_ID"
  
  # Return destination ID
  echo "$DESTINATION_ID"
}

# Function to create connection between source and destination
create_connection() {
  local source_id="$1"
  local destination_id="$2"
  
  echo "Creating connection between PostgreSQL source and Neo4j destination..."
  
  # Connection configuration
  CONNECTION_CONFIG='{
    "name": "Agent Team Builder - PostgreSQL to Neo4j",
    "sourceId": "'"$source_id"'",
    "destinationId": "'"$destination_id"'",
    "namespaceDefinition": "source",
    "namespaceFormat": "${SOURCE_NAMESPACE}",
    "prefix": "",
    "operationIds": [],
    "syncCatalog": {
      "streams": [
        {
          "stream": {
            "name": "agent_teams",
            "jsonSchema": {
              "type": "object",
              "properties": {}
            },
            "supportedSyncModes": ["full_refresh", "incremental"],
            "sourceDefinedCursor": true,
            "defaultCursorField": ["updated_at"],
            "sourceDefinedPrimaryKey": [["id"]]
          },
          "config": {
            "syncMode": "incremental",
            "cursorField": ["updated_at"],
            "destinationSyncMode": "append",
            "primaryKey": [["id"]],
            "aliasName": "agent_teams",
            "selected": true
          }
        }
      ]
    },
    "schedule": {
      "scheduleType": "cron",
      "cron": "0 * * * *"
    },
    "status": "active"
  }'
  
  # Create connection
  CONNECTION_RESPONSE=$(curl -s -X POST "${AIRBYTE_API_URL}/connections/create" \
    -H "Content-Type: application/json" \
    -d "$CONNECTION_CONFIG")
  
  # Extract connection ID
  CONNECTION_ID=$(echo "$CONNECTION_RESPONSE" | jq -r '.connectionId')
  
  if [ "$CONNECTION_ID" == "null" ] || [ -z "$CONNECTION_ID" ]; then
    echo "Error creating connection:"
    echo "$CONNECTION_RESPONSE" | jq .
    exit 1
  fi
  
  echo "Connection created with ID: $CONNECTION_ID"
  
  # Return connection ID
  echo "$CONNECTION_ID"
}

# Main script

echo "Setting up Airbyte sync from PostgreSQL to Neo4j..."

# Check if Airbyte API is available
check_airbyte_availability

# Create source, destination, and connection
SOURCE_ID=$(create_postgres_source)
DESTINATION_ID=$(create_neo4j_destination)
CONNECTION_ID=$(create_connection "$SOURCE_ID" "$DESTINATION_ID")

echo
echo "Setup complete!"
echo "PostgreSQL Source ID: $SOURCE_ID"
echo "Neo4j Destination ID: $DESTINATION_ID"
echo "Connection ID: $CONNECTION_ID"
echo
echo "Data will sync hourly according to the schedule configuration."
echo "You can trigger a manual sync from the Airbyte UI or API if needed."

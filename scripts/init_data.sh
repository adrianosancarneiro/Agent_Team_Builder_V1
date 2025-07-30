#!/usr/bin/env bash
# Optional script to initialize external services or seed data (e.g., ensure Qdrant and Neo4j are ready).

# 1. Create Qdrant collection (if not exists) for vector storage
QDRANT_HOST="${QDRANT_HOST:-http://192.168.0.83:6333}"
COLLECTION="${QDRANT_COLLECTION:-agent_vectors}"

echo "Checking Qdrant collection '$COLLECTION'..."
if command -v curl &> /dev/null; then
    # Use Qdrant Collections API to create collection if needed
    COLLECTIONS_URL="$QDRANT_HOST/collections/$COLLECTION"
    resp=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$COLLECTIONS_URL")
    if [ "$resp" != "200" ]; then
        echo "Creating Qdrant collection: $COLLECTION"
        curl -s -X PUT "$COLLECTIONS_URL" -H "Content-Type: application/json" \
             -d '{"vector_size": 768, "distance": "Cosine"}'
        echo ""  # newline
    else
        echo "Collection $COLLECTION already exists."
    fi
else
    echo "curl not available, please ensure Qdrant collection exists manually."
fi

# 2. (Optional) Load initial data or schema into Neo4j
NEO4J_URI="${NEO4J_URI:-bolt://192.168.0.83:7687}"  # bolt port for neo4j
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-pJnssz3khcLtn6T}"
# This part requires Neo4j CLI or driver to run commands; for demo, we'll skip actual commands.
echo "Ensure Neo4j is running at $NEO4J_URI (user: $NEO4J_USER). Load schema or data as needed."
# e.g., using cypher-shell:
# cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CREATE CONSTRAINT IF NOT EXISTS ON (n:AgentMemory) ASSERT n.id IS UNIQUE;"

echo "Initialization of external services complete (if applicable)."

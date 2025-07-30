import os
from neo4j import GraphDatabase, basic_auth

class GraphDBTool:
    """Minimal client for Neo4j graph database queries."""
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "neo4j://192.168.0.83:7474")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "pJnssz3khcLtn6T")  # Note: use env var in practice for security
        # Initialize Neo4j driver (encrypted=False for local dev)
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, pwd), encrypted=False)

    def query(self, cypher: str, params: dict = None) -> list:
        """Run a Cypher query and return results (as list of records)."""
        records = []
        with self.driver.session() as session:
            results = session.run(cypher, params or {})
            for record in results:
                records.append(record.data())
        return records

    def close(self):
        """Close the database connection (call on app shutdown)."""
        self.driver.close()

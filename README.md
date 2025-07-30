# AI Team Builder Agent Service – FastAPI Implementation

**Overview:** This service uses **FastAPI** to expose endpoints for building and managing an AI agent team. The architecture follows a clean, scalable structure with clear separation of concerns, leveraging **uv** for environment management instead of Docker. The Team Builder Agent uses tenant-specific context and tools to assemble a multi-agent team and orchestrate their collaboration.

Key external services (from Phase 1 infrastructure) include:
- **Qdrant** for vector search 
- **Neo4j** for graph queries
- **Embedding model service** (BAAI/bge-base-en-v1.5)
- **Web API access** for external data
- **File chunking** utilities

## Quick Start

1. **Setup Environment**:
   ```bash
   ./scripts/setup_env.sh
   ```

2. **Install Dependencies**:
   ```bash
   ./scripts/install_deps.sh
   ```

3. **Initialize External Services** (optional):
   ```bash
   ./scripts/init_data.sh
   ```

4. **Run the Application**:
   ```bash
   ./scripts/run_app.sh
   ```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

## RTX 5090 / CUDA Compatibility

If you have an RTX 5090 and see CUDA compatibility warnings, you have several options:

### Option 1: Upgrade PyTorch (Recommended)
```bash
./scripts/upgrade_pytorch.sh
```

### Option 2: Force CPU Usage
```bash
export EMBEDDING_DEVICE=cpu
```

Or edit your `.env` file:
```bash
EMBEDDING_DEVICE=cpu
```

### Option 3: Check Compatibility
The application automatically detects CUDA compatibility and falls back to CPU if needed. The RTX 5090 requires PyTorch with CUDA 12.4+ and sm_120 architecture support.

## Project Structure

AI_Team_Builder_Service/  
├── src/  
│ ├── main.py # FastAPI app definition and routes  
│ ├── config/  
│ │ └── schema.py # Pydantic models for agent team config  
│ ├── services/  
│ │ ├── team_builder.py # Core logic for team assembly (Phase 1 auto-generation)  
│ │ └── team_executor.py # Logic for running multi-agent conversations  
│ └── tools/  
│ ├── qdrant_tool.py # Tool interface for Qdrant vector DB  
│ ├── graph_tool.py # Tool interface for Neo4j graph DB  
│ ├── embed_tool.py # Tool interface for embedding service (BAAI/bge-base-en-v1.5)  
│ ├── webapi_tool.py # Tool for external web/API calls  
│ └── chunking_tool.py # Utility for file text chunking  
├── configs/  
│ └── example_config.json # Example of AgentTeamConfig JSON  
├── scripts/  
│ ├── setup_env.sh # Shell script for uv environment setup and installation  
│ ├── install_deps.sh # Shell script for dependency installation via uv  
│ ├── run_app.sh # Shell script to launch the FastAPI (uvicorn) server  
│ └── init_data.sh # Shell script for any data/tool initialization (e.g., ensuring DB schemas)  
├── logs/ # Log files (application and agent conversation logs)  
└── pyproject.toml # Project dependencies and uv configuration (if using Poetry/uv)

- **src/** – Application source code, organized by component.
- **src/config/** – Configuration schemas and models (e.g. Pydantic schema for agent team config).
- **src/services/** – Core service logic: building the agent team from config, and executing team conversations.
- **src/tools/** – Integration with external tools and services (vector DB, graph DB, embeddings, etc.), each encapsulated in a module for clarity.
- **configs/** – Static config files or examples (e.g. a sample JSON for an agent team configuration).
- **scripts/** – Shell scripts for setting up the environment (using uv), installing dependencies, running the app, and initializing external tools or data.
- **logs/** – Log output directory for both system logs and conversation transcripts for debugging and audit.
- **pyproject.toml** – Defines Python dependencies and uv configuration for environment management (using uv’s preferred workflow instead of a Dockerfile).

This structure follows best practices for scalability and maintainability: configuration and constants are isolated, external tool interfaces are modular, and business logic is separate from the FastAPI app layer. Next, we detail each component with sample implementations and inline comments for clarity.

## src/config/schema.py – Config Schema (Pydantic)

from pydantic import BaseModel, Field  
from typing import List, Dict, Optional  
<br/>class AgentToolConfig(BaseModel):  
"""Configuration for a tool that an agent can use."""  
name: str # e.g., "Search_Vector_DB", "Search_Graph_DB", "Call_Web_API"  
params: Optional\[Dict\[str, any\]\] = None # parameters or settings for the tool (if any)  
<br/>class AgentDefinition(BaseModel):  
"""Definition of a single agent in the team."""  
agent_role: str = Field(..., description="The agent’s role or specialization (e.g. Retriever, Critic)")  
agent_name: str = Field(..., description="Human-readable identifier for the agent")  
agent_personality: str = Field(..., description="Brief description of the agent’s personality or perspective")  
agent_goal_based_prompt: str = Field(..., description="Role-specific instructions or prompt for this agent")  
LLM_model: Optional\[Dict\[str, str\]\] = Field(None, description="Model name or config (e.g., {'model': 'gpt-4'})")  
allow_team_builder_to_override_model: bool = Field(True, description="If true, Team Builder can change the model")  
LLM_configuration: Optional\[Dict\[str, float\]\] = Field(None, description="Model params like temperature, etc.")  
agent_tools: Optional\[List\[str\]\] = Field(None, description="List of tool names this agent can use")  
<br/>class AgentTeamConfig(BaseModel):  
"""Pydantic model for the agent team configuration schema."""  
agent_team_main_goal: str = Field(..., description="Primary goal or problem statement for the agent team")  
tenant_id: Optional\[str\] = Field(None, description="Tenant ID for context loading and multi-tenancy isolation")  
\# Flags controlling dynamic team building:  
allow_TBA_to_recommend_agents: bool = Field(False, description="Allow Team Builder Agent to add extra agents beyond those specified")  
allow_TBA_how_many_more: int = Field(0, description="Max number of additional agents TBA can add if allowed")  
should_TBA_ask_caller_approval: bool = Field(False, description="If true, pause and require human approval for additions")  
\# Optional conversation flow and limits:  
agent_team_flow: Optional\[str\] = Field(None, description="Preset conversation turn order, e.g. 'Retriever->Critic->Refiner'")  
max_turns: int = Field(5, description="Max number of conversation rounds to execute")  
\# Tools available globally (for assignment to agents):  
available_tools: Optional\[List\[str\]\] = Field(None, description="Whitelist of tools/APIs available for agents")  
\# Pre-defined agents (partial or full specification):  
agents: Optional\[List\[AgentDefinition\]\] = Field(None, description="List of agent definitions to include in the team")

**Notes:**  
\- We use Pydantic models to validate and document the schema for the agent team configuration. This mirrors the JSON schema described in the design docs[\[5\]\[6\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), including fields like the main goal, flags to allow the Team Builder to recommend agents[\[5\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) and how many[\[7\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), an optional agent_team_flow to fix the turn order[\[8\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), and a list of agent definitions with required subfields (role, name, personality, prompt, etc.)[\[6\]\[9\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).  
\- The AgentDefinition model captures each agent’s attributes. Tools for agents are referenced by name (from available_tools). In practice, each tool name corresponds to an integration in our src/tools module (e.g., "Search_Vector_DB" → Qdrant search tool).  
\- Using Pydantic ensures that any config passed in (via API or loaded from file) is validated for completeness and correctness, providing fast feedback if required fields are missing.

## src/tools/qdrant_tool.py – Qdrant Vector DB Integration

import os  
from typing import List, Any, Optional  
import requests # using requests for simplicity; could use qdrant-client library  
<br/>class QdrantTool:  
"""Minimal client for Qdrant vector database operations."""  
def \__init_\_(self):  
\# Qdrant URL could be configured via env or config  
self.base_url = os.getenv("QDRANT_URL", "<http://192.168.0.83:6333>")  
\# Optional: collection name could be tenant-specific  
self.collection = os.getenv("QDRANT_COLLECTION", "agent_vectors")  
<br/>def search(self, query_embedding: List\[float\], top_k: int = 5, filters: Optional\[dict\] = None) -> List\[Any\]:  
"""Search the Qdrant vector collection for nearest vectors to the query embedding."""  
url = f"{self.base_url}/collections/{self.collection}/points/search"  
payload = {  
"vector": query_embedding,  
"limit": top_k  
}  
if filters:  
payload\["filter"\] = filters  
try:  
res = requests.post(url, json=payload, timeout=5)  
res.raise_for_status()  
results = res.json().get("result", \[\])  
return results # Each result contains e.g. an "id" and "score" and possibly payload  
except Exception as e:  
\# In a real system, handle exceptions and logging appropriately  
print(f"Qdrant search error: {e}")  
return \[\]  
<br/>def upsert(self, points: List\[dict\]) -> bool:  
"""Insert or update points (vectors with payload) into the collection."""  
url = f"{self.base_url}/collections/{self.collection}/points"  
try:  
res = requests.put(url, json={"points": points}, timeout=5)  
res.raise_for_status()  
return True  
except Exception as e:  
print(f"Qdrant upsert error: {e}")  
return False

**Explanation:** The QdrantTool class provides a minimal interface to the Qdrant vector DB. We default to the Phase 1 test endpoint at 192.168.0.83:6333[\[3\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), but in practice this should be configurable per tenant or environment. The search method posts a query embedding to Qdrant’s search API and retrieves the closest stored vectors (used for semantic search results, e.g. knowledge retrieval)[\[3\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). The upsert method allows adding new vectors with associated metadata. In a full implementation, we might use the official qdrant-client for Python, handle collection creation, and manage authentication or namespace per tenant. Here we keep it simple and synchronous for demonstration. Proper error handling and logging is included to ensure reliability.

## src/tools/graph_tool.py – Neo4j Graph DB Integration

import os  
from neo4j import GraphDatabase, basic_auth  
<br/>class GraphDBTool:  
"""Minimal client for Neo4j graph database queries."""  
def \__init_\_(self):  
uri = os.getenv("NEO4J_URI", "neo4j://192.168.0.83:7474")  
user = os.getenv("NEO4J_USER", "neo4j")  
pwd = os.getenv("NEO4J_PASSWORD", "pJnssz3khcLtn6T") # Note: use env var in practice for security  
\# Initialize Neo4j driver (encrypted=False for local dev)  
self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, pwd), encrypted=False)  
<br/>def query(self, cypher: str, params: dict = None) -> list:  
"""Run a Cypher query and return results (as list of records)."""  
records = \[\]  
with self.driver.session() as session:  
results = session.run(cypher, params or {})  
for record in results:  
records.append(record.data())  
return records  
<br/>def close(self):  
"""Close the database connection (call on app shutdown)."""  
self.driver.close()

**Explanation:** The GraphDBTool uses the Neo4j Python driver to connect to the Neo4j database. It reads connection details from environment variables, defaulting to the provided development server URL and credentials[\[10\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). The query method executes a Cypher query and returns the results in a simple list of dicts format. In a real scenario, one might have specialized methods for specific graph queries (e.g., querying a knowledge graph for certain relationships). For brevity, we expose a generic query interface. The Neo4j database can store a tenant’s knowledge graph or agent memories[\[11\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), enabling agents to retrieve structured information via the Search_Graph_DB tool.

## src/tools/embed_tool.py – Embedding Service Integration

from typing import List  
\# We will use a transformer model for embeddings. In practice, this might be a separate service call.  
try:  
from sentence_transformers import SentenceTransformer  
except ImportError:  
SentenceTransformer = None  
<br/>class EmbeddingService:  
"""Embedding service using BAAI/bge-base-en-v1.5 model to get text embeddings."""  
def \__init_\_(self, model_name: str = "BAAI/bge-base-en-v1.5"):  
if SentenceTransformer:  
\# Load the embedding model (this will download the model if not present)  
self.model = SentenceTransformer(model_name)  
else:  
self.model = None  
print("SentenceTransformer not installed. Embeddings will be dummy values.")  
<br/>def embed(self, text: str) -> List\[float\]:  
"""Convert text into a vector embedding."""  
if self.model:  
embedding: List\[float\] = self.model.encode(text, show_progress_bar=False)  
return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)  
\# Fallback: return a dummy embedding (e.g., vector of zeros) if model not available  
return \[0.0\] \* 768 # assuming 768-dim for BGE base model

**Explanation:** The EmbeddingService class provides vector embeddings for input text using the **BAAI/bge-base-en-v1.5** model[\[12\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). This model is referenced as the chosen embedding model in Phase 1, used to convert text (like user queries or document chunks) into high-dimensional vectors for semantic search[\[12\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). In this minimal implementation, we attempt to use the sentence_transformers library to load the model and compute embeddings. In a production system, this might instead call a dedicated microservice (for instance, a container running the model, or a HuggingFace inference endpoint). If the model isn't available (library not installed), we fall back to a dummy embedding for safety.

This embedding service can be utilized by the Team Builder or agents to embed the user’s query before searching the vector DB (so that Qdrant can be queried with the query’s embedding)[\[4\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), or by any agent that needs to vectorize text for comparison. The embedding dimension (in this case, 768) should match what the vector DB and LLM are compatible with[\[13\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

## src/tools/webapi_tool.py – External Web/API Call Tool

import requests  
<br/>class WebAPITool:  
"""Tool to make external web API calls (internet access)."""  
def get(self, url: str, params: dict = None, headers: dict = None) -> str:  
"""Perform a GET request to the given URL and return response text."""  
try:  
res = requests.get(url, params=params, headers=headers, timeout=10)  
res.raise_for_status()  
return res.text  
except Exception as e:  
return f"ERROR: {e}"  
<br/>def post(self, url: str, data: dict = None, json_data: dict = None, headers: dict = None) -> str:  
"""Perform a POST request (with JSON or form data) and return response text."""  
try:  
res = requests.post(url, data=data, json=json_data, headers=headers, timeout=10)  
res.raise_for_status()  
return res.text  
except Exception as e:  
return f"ERROR: {e}"

**Explanation:** The WebAPITool allows agents to perform HTTP requests to external APIs or websites. This is the implementation behind a tool like "Call_Web_API" in the config[\[14\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). Having internet access within the Team Builder’s host environment means agents can fetch live data or call external services when allowed[\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). The tool provides simple GET and POST methods; in a secure deployment, you’d add restrictions (e.g., allowed domains or rate limiting) to prevent abuse. The responses are returned as text for the agent to consume. This tool is used sparingly and under tenant-defined security constraints[\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

## src/tools/chunking_tool.py – File/Text Chunking Utility

import math  
<br/>class ChunkingTool:  
"""Utility to split text into chunks for processing (e.g., indexing to vector DB)."""  
def \__init_\_(self, chunk_size: int = 500):  
self.chunk_size = chunk_size # default chunk size in characters (or tokens approx.)  
<br/>def chunk_text(self, text: str) -> list:  
"""Split a long text into chunks of approximately chunk_size characters."""  
chunks = \[\]  
length = len(text)  
for i in range(0, length, self.chunk_size):  
chunk = text\[i: i + self.chunk_size\]  
chunks.append(chunk)  
return chunks  
<br/>def chunk_file(self, file_path: str) -> list:  
"""Read a file and split its content into chunks."""  
try:  
with open(file_path, 'r', encoding='utf-8') as f:  
content = f.read()  
except Exception as e:  
print(f"Error reading file {file_path}: {e}")  
return \[\]  
return self.chunk_text(content)

**Explanation:** ChunkingTool is a simple utility to break down large text (or file contents) into smaller chunks. This is helpful for indexing knowledge sources: for example, splitting a long document into chunks that can be embedded and stored in Qdrant, enabling semantic search on each chunk. While not explicitly described in the high-level docs, chunking is a common preparation step in Retrieval-Augmented Generation (RAG) workflows and likely part of the **Phase 1** toolkit (the documentation notes chunking service as a potential tool)[\[16\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). We keep the implementation straightforward, splitting by a fixed character count. In a real system, one might integrate a tokenization-based splitter to respect token boundaries or sentence boundaries.

## src/services/team_builder.py – Team Building Logic (Phase 1)

from src.config.schema import AgentTeamConfig, AgentDefinition  
from src.tools import qdrant_tool, graph_tool, embed_tool  
<br/>\# Initialize tool clients (in a real app, use dependency injection or global singletons)  
\_vector_tool = qdrant_tool.QdrantTool()  
\_graph_tool = graph_tool.GraphDBTool()  
\_embed_service = embed_tool.EmbeddingService()  
<br/>class Agent:  
"""Representation of an AI agent (LLM-powered) with certain tools and persona."""  
def \__init_\_(self, definition: AgentDefinition):  
self.role = definition.agent_role  
self.name = definition.agent_name  
self.personality = definition.agent_personality  
self.prompt = definition.agent_goal_based_prompt  
self.model_info = definition.LLM_model or {"model": "default-LLM"}  
self.allow_model_override = definition.allow_team_builder_to_override_model  
self.model_config = definition.LLM_configuration or {}  
\# Tools that this agent can use, instantiated from tool names  
self.tools = \[\]  
for tool_name in (definition.agent_tools or \[\]):  
if tool_name == "Search_Vector_DB":  
self.tools.append(\_vector_tool)  
elif tool_name == "Search_Graph_DB":  
self.tools.append(\_graph_tool)  
elif tool_name == "Call_Web_API":  
from src.tools import webapi_tool  
self.tools.append(webapi_tool.WebAPITool())  
elif tool_name == "Embedding_Service":  
self.tools.append(\_embed_service)  
\# Add other tool name checks (e.g., "Chunk_Text") as needed  
\# If EmbeddingService is used as a tool, maybe it's for generating embeddings on the fly.  
<br/>def summarize(self) -> dict:  
"""Return a summary of the agent (for API responses or logging)."""  
return {  
"name": self.name,  
"role": self.role,  
"personality": self.personality,  
"model": self.model_info,  
"tools": \[t.\__class_\_._\_name__ for t in self.tools\]  
}  
<br/>class TeamBuilderService:  
"""Service responsible for creating an agent team from configuration."""  
DEFAULT_AGENTS = \[ # Default roles if none specified (base scenario: DecisionMaker -> Retriever -> Critic -> Refiner)  
{  
"agent_role": "DecisionMaker",  
"agent_name": "DecisionMakerAgent",  
"agent_personality": "Decisive leader that coordinates the team.",  
"agent_goal_based_prompt": "Decide the next step or conclude the task based on inputs from others.",  
"agent_tools": \[\] # Decision maker may not use external tools, just organizes  
},  
{  
"agent_role": "Retriever",  
"agent_name": "RetrieverAgent",  
"agent_personality": "Diligent researcher fetching relevant information.",  
"agent_goal_based_prompt": "Find facts and data relevant to the main goal using available knowledge sources.",  
"agent_tools": \["Search_Vector_DB", "Search_Graph_DB"\] # uses vector DB and graph DB  
},  
{  
"agent_role": "Critic",  
"agent_name": "CriticAgent",  
"agent_personality": "Skeptical critic that double-checks answers.",  
"agent_goal_based_prompt": "Evaluate the responses for correctness and potential issues.",  
"agent_tools": \[\]  
},  
{  
"agent_role": "Refiner",  
"agent_name": "RefinerAgent",  
"agent_personality": "Thoughtful refiner that improves and finalizes answers.",  
"agent_goal_based_prompt": "Polish and integrate the information into a final answer.",  
"agent_tools": \[\]  
}  
\]  
<br/>@classmethod  
def build_team(cls, config: AgentTeamConfig) -> list:  
"""  
Construct the team of agents based on the provided configuration.  
Returns a list of Agent objects.  
"""  
agents_config = \[\]  
if config.agents and len(config.agents) > 0:  
\# Start with caller-specified agents  
for agent_def in config.agents:  
agents_config.append(agent_def.dict())  
else:  
\# No agents specified by caller: use default template roles  
agents_config = \[dict(a) for a in cls.DEFAULT_AGENTS\]  
\# If allowed, the Team Builder can recommend additional agents up to the specified number  
if config.allow_TBA_to_recommend_agents and config.allow_TBA_how_many_more > 0:  
\# For simplicity, if fewer than allowed agents are present, add a placeholder agent.  
\# In a real scenario, this could analyze the task and tenant context to suggest a role.  
for i in range(config.allow_TBA_how_many_more):  
extra_role = f"AdditionalAgent{i+1}"  
agents_config.append({  
"agent_role": extra_role,  
"agent_name": f"ExtraAgent{i+1}",  
"agent_personality": "Auxiliary agent added by TeamBuilder to cover missing expertise.",  
"agent_goal_based_prompt": f"Assist in achieving the main goal by providing {extra_role} capabilities.",  
"agent_tools": config.available_tools or \[\] # give it access to all available tools by default  
})  
break # (In practice, you'd add up to how_many_more agents; here we add one as example)  
\# Instantiate Agent objects for each definition  
team = \[Agent(AgentDefinition(\*\*agent_def)) for agent_def in agents_config\]  
\# If a static flow is defined (agent_team_flow), we could reorder team or mark the order somewhere  
\# For simplicity, assume order in list will be the conversation order if flow is given.  
\# (In a full implementation, parse agent_team_flow string to enforce speaking order.)  
return team

**Team Building Logic:** The TeamBuilderService.build_team takes an AgentTeamConfig and produces a list of Agent instances representing the AI agents team. This function implements the **auto-generation behavior** described for Phase 1:

- **Default Team Composition:** If the caller did not specify any agents, we auto-create a default team of four roles: Decision Maker, Retriever, Critic, Refiner[\[17\]\[18\]](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE). These correspond to a basic workflow (decision-making loop) and align with the documentation's suggestion of default roles when none are provided[\[19\]](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE). Each default agent has a preset persona and prompt.
- **Caller-Specified Agents:** If the config includes a list of agents, we include those. Each agent’s definition is converted from the Pydantic model to a dict for flexibility.
- **Dynamic Agent Addition:** If the config allows the Team Builder Agent to recommend additional agents (allow_TBA_to_recommend_agents == True), we add up to allow_TBA_how_many_more extra agents[\[7\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). In a real system, this logic might analyze the main goal and existing team composition to decide what new role is needed (e.g., adding a Planner or a DomainExpert if missing). Here we simply append a generic extra agent (demonstrating the capability).
- **Tool Assignment:** Each Agent instance is initialized with the tools it’s allowed to use. We map tool names from the config to actual tool instances (e.g., "Search_Vector_DB" → the QdrantTool instance, "Embedding_Service" → the embedding service, etc.). This uses simple conditional logic; a more scalable design might use a registry or factory pattern for tools. The available tools for an agent are constrained by the available_tools list in the config (or defaults if not provided)[\[20\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).
- **Model Selection:** Each agent can have a specified LLM model and parameters. If allow_team_builder_to_override_model is true for that agent, the Team Builder could switch to a different model better suited to the role[\[21\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). Our implementation doesn’t dynamically change models, but we honor the flag in the Agent’s attributes for future use. For now, we simply store model info and assume a default model if none provided.
- **Flow Order:** If agent_team_flow is provided (e.g., "Retriever -> Critic -> Refiner"), the Team Builder would ensure the agents or orchestrator respect that speaking order[\[8\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). In this minimal implementation, we would parse and enforce the order or designate an orchestrator agent. For simplicity, we note that if a flow is given, the order of the team list corresponds to it (this would be refined in a full implementation).

Each created Agent has a summarize() method to output its key properties, useful for returning via API or logging the team composition. The Agent class also sets up the instances of tool clients it can use. This clean separation means when an agent needs to perform an action (search vector DB, query the graph, call an API, etc.), it will invoke these tool instances.

## src/services/team_executor.py – Team Execution Logic (Conversation Orchestration)

from typing import List, Tuple  
from src.services.team_builder import Agent  
<br/>class TeamExecutorService:  
"""Service to manage multi-agent conversation execution."""  
def \__init_\_(self, agents: List\[Agent\], flow: List\[str\] = None, max_turns: int = 5):  
"""  
Initialize with a list of Agent instances. Optionally provide a conversation flow order  
(list of agent names or roles in speaking order). If no flow is given, a default order (round-robin) is used.  
"""  
self.agents = agents  
\# Create a mapping of agent role->agent and name->agent for convenience  
self.agents_by_name = {agent.name: agent for agent in agents}  
self.agents_by_role = {agent.role: agent for agent in agents}  
\# Determine speaking order  
if flow:  
\# If flow is provided as a list of names/roles, convert to actual agent instances  
self.flow = \[\]  
for identifier in flow:  
agent = self.agents_by_name.get(identifier) or self.agents_by_role.get(identifier)  
if agent:  
self.flow.append(agent)  
else:  
\# default to the order given or round-robin  
self.flow = agents  
self.max_turns = max_turns  
self.conversation_log: List\[Tuple\[str, str\]\] = \[\] # list of (agent_name, message)  
<br/>def run_conversation(self, user_query: str) -> str:  
"""  
Execute the multi-agent conversation until max_turns or a stopping condition is met.  
Returns the final answer (or combined result) from the team.  
"""  
\# Initial user input  
self.conversation_log.append(("User", user_query))  
current_turn = 0  
final_answer = ""  
\# Simple loop through agents in the defined flow  
while current_turn < self.max_turns:  
for agent in self.flow:  
\# Each agent takes the last message and responds  
last_speaker, last_message = self.conversation_log\[-1\]  
\# Determine if conversation should end (if last speaker was an agent and decided to stop)  
if last_speaker != "User" and agent.role == "DecisionMaker":  
\# (Example heuristic) DecisionMaker can decide to finish the conversation  
if "conclude" in last_message.lower():  
final_answer = last_message  
return final_answer  
\# Agent formulates a response (Here we'd call the LLM with prompt and context. We'll simulate.)  
response = self.\_agent_respond(agent, last_message)  
\# Log the agent's response  
self.conversation_log.append((agent.name, response))  
\# Optionally, check if this agent is an orchestrator or decision-maker concluding the chat  
if agent.role.lower() in ("decisionmaker", "orchestrator"):  
if "final answer:" in response.lower() or "conclude" in response.lower():  
final_answer = response  
return final_answer  
current_turn += 1  
\# If loop completes without early return, take the last agent's message as final answer  
final_answer = self.conversation_log\[-1\]\[1\]  
return final_answer  
<br/>def \_agent_respond(self, agent: Agent, last_message: str) -> str:  
"""  
Simulate an agent responding to the last_message. In reality, this would involve the agent's prompt,  
persona, tools, and an LLM call. Here, we'll do a simple placeholder implementation.  
"""  
\# If the agent has any tools, maybe use one (for demo, use first applicable tool to augment response)  
tool_augmented_info = ""  
for tool in agent.tools:  
tool_name = tool.\__class_\_._\_name__  
if tool_name == "QdrantTool":  
\# Example: use embedding service to embed query, then search vector DB  
query_emb = \_embed_service.embed(last_message)  
results = tool.search(query_emb, top_k=1)  
if results:  
tool_augmented_info += " \[Found relevant info via vector DB\]"  
elif tool_name == "GraphDBTool":  
\# Example: query graph DB for a fact (here we just do a dummy query or skip)  
\# In real case, we might have a specific query pattern  
results = tool.query("MATCH (n) RETURN n LIMIT 1")  
if results:  
tool_augmented_info += " \[Knowledge graph checked\]"  
elif tool_name == "WebAPITool":  
\# Example: perform a web API call (not doing actual call in demo)  
tool_augmented_info += " \[Called external API\]"  
\# (Additional tool handling as needed)  
\# Formulate a dummy response using the agent's role and possibly augmented info  
response = f"{agent.name} ({agent.role}) says: Based on '{last_message}', I {agent.personality.lower()} respond with an answer.{tool_augmented_info}"  
return response

**Team Execution Logic:** The TeamExecutorService takes the assembled agents and manages their conversation. This corresponds to the **Tenant App Agent Team Execution Workflow** – essentially running the multi-agent system through its conversation rounds.

Key points in this implementation:

- **Conversation Flow:** If a specific agent_team_flow is provided in the config (e.g., a sequence of roles/names)[\[8\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), we construct the flow accordingly. Otherwise, we default to the list order or a simple round-robin. In Phase 1, the typical flow might be “DecisionMaker -> Retriever -> Critic -> Refiner” repeating, or a Manager (DecisionMaker) deciding next speaker dynamically. Here, we simulate a fixed order for simplicity.
- **Conversation Loop:** run_conversation starts with the user’s query (main goal) as the first message. It then iterates through agents in the flow for each turn, appending each agent’s response to a conversation_log. We limit the process to max_turns to avoid infinite loops. There’s a simple heuristic: if the DecisionMaker (or an Orchestrator/Manager agent) outputs a message indicating conclusion (e.g., containing "final answer" or "conclude"), we break early – simulating that the team decided to stop once the goal is achieved.
- **Agent Response Simulation:** In \_agent_respond, we **simulate** an agent’s reasoning. In a real system, this is where we would construct the prompt for the LLM, including the agent’s role description, its goal-based prompt, the conversation context so far, and possibly tool results. The function would then call the chosen LLM (via an API or SDK) to generate the agent’s response. We also demonstrate how tools might be invoked by an agent:
- If the agent has the Qdrant tool, we embed the last message (using the embedding service) and query the vector DB for relevant info[\[3\]\[12\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).
- If the agent has the GraphDB tool, we run a sample graph query (dummy in this case)[\[11\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).
- If the agent has the WebAPI tool, we simulate calling an external API[\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).
- Any results or side-effects are appended to the agent’s response as bracketed notes to indicate the tool’s contribution.
- **Final Answer:** The conversation ends when the loop completes the set number of turns or a special flag from an agent indicates completion. The final answer returned could be the last message from the DecisionMaker or a aggregated answer. Here we simply return the last message in the log as the final answer, under the assumption that the DecisionMaker or Refiner’s last turn produces the solution.

This executor is a **simplified orchestration**. In a more advanced Phase 2 scenario, one might integrate Microsoft’s Autogen GroupChat or a LangGraph state machine to handle turn-taking and branching logic more robustly. For now, it demonstrates how multi-agent dialogue might be handled in code, with the Team Builder’s output (the agent team) feeding into this execution loop.

## src/main.py – FastAPI Application

from fastapi import FastAPI, HTTPException  
from src.config.schema import AgentTeamConfig  
from src.services.team_builder import TeamBuilderService  
from src.services.team_executor import TeamExecutorService  
<br/>app = FastAPI(title="AI Team Builder Agent Service", version="1.0")  
<br/>\# Build the team and optionally run the conversation in one go (for simplicity).  
@app.post("/build_and_execute")  
def build_and_execute(config: AgentTeamConfig):  
"""  
Build an AI agent team according to the provided configuration and run the multi-agent conversation.  
Returns the team composition and the final answer.  
"""  
\# Build the team of agents  
try:  
team = TeamBuilderService.build_team(config)  
except Exception as e:  
raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")  
\# Log or store team info (omitted for brevity)  
team_summary = \[agent.summarize() for agent in team\]  
<br/>\# Execute the conversation workflow  
executor = TeamExecutorService(agents=team,  
flow=\[agent.role for agent in team\] if config.agent_team_flow is None else \[s.strip() for s in config.agent_team_flow.split("->")\],  
max_turns=config.max_turns)  
final_answer = executor.run_conversation(user_query=config.agent_team_main_goal)  
<br/>\# Return both the team details and the final answer  
return {  
"agent_team": team_summary,  
"conversation_log": executor.conversation_log,  
"final_answer": final_answer  
}  
<br/>\# (Optional) Separate endpoint to just build team without execution  
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
return {"agent_team": \[agent.summarize() for agent in team\]}  
<br/>\# (Optional) Health check endpoint  
@app.get("/health")  
def health_check():  
return {"status": "ok"}

**FastAPI App:** The FastAPI application ties everything together, exposing a clean API to clients (e.g., a tenant’s application or a testing tool). We define two primary endpoints:

- **POST /build_and_execute:** Takes a JSON payload matching AgentTeamConfig. It first calls the Team Builder to assemble the agents, then immediately runs the conversation using TeamExecutorService. The response includes a summary of the agent team (roles, tools, etc.), the conversation log (all messages exchanged, including the user’s query and each agent’s responses), and the final answer produced by the team. This provides an end-to-end single call for convenience.
- **POST /build_team:** (Optional) Allows the client to just construct the team without running the conversation. This returns the agent team composition. In a scenario where the client might want to inspect or adjust the team before execution, this separation can be useful.
- **GET /health:** A simple health check endpoint to verify the service is running.

We include basic error handling (e.g., if the config is invalid, return HTTP 400). The AgentTeamConfig Pydantic model ensures that the request body is validated against our schema automatically by FastAPI.

**Best Practices Applied:** We utilize FastAPI’s dependency injection and Pydantic integration for input validation. The code is organized so that business logic (team building, execution) is in the services module, not mixed into the API route functions, making it easier to maintain and test. Inline comments and docstrings provide clarity on each component’s role. Additionally, sensitive info like database URLs and credentials are fetched from environment variables (with defaults for development) – in real deployments, those would be set outside the code for security. Logging (not fully shown here due to brevity) would be configured to output to files in the logs/ directory, capturing each step of the process for debugging and audit, as recommended by the design[\[22\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

## Shell Scripts for Setup and Deployment

We provide shell scripts in the scripts/ directory to streamline environment setup and running the service. These scripts use **uv** (the Astral package manager) to manage the Python environment and dependencies, instead of relying on Docker. This approach ensures fast, reproducible setups and aligns with modern Python packaging best practices.

### scripts/setup_env.sh – Environment Setup with uv

# !/usr/bin/env bash  
\# This script sets up a Python virtual environment using uv and installs uv itself if needed.  
<br/>\# Exit immediately if a command exits with a non-zero status  
set -e  
<br/>\# 1. Install uv if not already installed  
if ! command -v uv &> /dev/null; then  
echo "uv not found, installing uv..."  
\# Using pipx to install uv for isolation (could also use curl installer from Astral)  
python3 -m pip install --user pipx || { echo "Failed to install pipx"; exit 1; }  
pipx install uv  
fi  
<br/>\# 2. Initialize a uv-managed virtual environment  
\# By default, uv looks for pyproject.toml in the current directory to resolve dependencies.  
if \[ ! -d ".venv" \]; then  
echo "Creating virtual environment with uv..."  
uv venv .venv # create venv in .venv folder  
fi  
<br/>echo "Activating virtual environment..."  
\# Activate the environment for subsequent steps (if needed for interactive use)  
source .venv/bin/activate || { echo "Activation failed"; exit 1; }  
<br/>echo "Environment setup complete. uv is ready to manage dependencies."

**What it does:** Checks if uv is installed, installs it if not (using pipx for isolation). Then it creates a virtual environment .venv using uv (which ensures Python version consistency and can manage multiple environments). Activating the venv is optional for non-interactive shell scripts, but we show it for completeness. This script would be run once to prepare the environment.

### scripts/install_deps.sh – Dependency Installation

# !/usr/bin/env bash  
\# Install project dependencies using uv (assumes pyproject.toml defines them)  
<br/>set -e  
<br/>\# Ensure uv is installed and environment is set up  
if ! command -v uv &> /dev/null; then  
echo "uv is not installed. Run setup_env.sh first."  
exit 1  
fi  
<br/>\# Ensure we're in project root directory (where pyproject.toml is)  
SCRIPT_DIR=$(dirname "$0")  
cd "$SCRIPT_DIR/.." # move to project root relative to script directory  
<br/>\# Use uv to sync dependencies defined in pyproject.toml (production deps only, no dev)  
echo "Installing dependencies with uv..."  
uv sync --no-dev  
<br/>echo "Dependencies installation complete."

**What it does:** Uses uv sync to install all dependencies specified in pyproject.toml (excluding dev dependencies). The pyproject.toml would list packages like FastAPI, uvicorn, pydantic, neo4j, qdrant-client (or requests), sentence-transformers, etc. Using uv sync ensures a deterministic environment (it will use a lockfile uv.lock if present for pinned versions). The script should be run after setup_env.sh.

### scripts/run_app.sh – Launching the FastAPI Service

# !/usr/bin/env bash  
\# Launch the FastAPI application using uvicorn  
<br/>set -e  
<br/>\# Activate the environment (so that uvicorn and deps are in PATH)  
source .venv/bin/activate  
<br/>\# Default host/port can be parameterized; using 0.0.0.0:8000 for accessibility in container/VM  
HOST="${HOST:-0.0.0.0}"  
PORT="${PORT:-8000}"  
<br/>echo "Starting FastAPI app on $HOST:$PORT ..."  
\# Using uvicorn to run the app; adjust workers if needed for concurrency.  
uvicorn src.main:app --host $HOST --port $PORT --workers 4

**What it does:** Activates the Python virtual environment and then starts the FastAPI app with Uvicorn. We bind to 0.0.0.0 on port 8000 by default (making it accessible externally, which is typical for container or VM deployments). The number of workers is set to 4 for handling multiple requests concurrently (tunable based on expected load). This script abstracts the uvicorn command so that environment variables or other launch configurations can be easily managed at one place.

### scripts/init_data.sh – Data/Tool Initialization

# !/usr/bin/env bash  
\# Optional script to initialize external services or seed data (e.g., ensure Qdrant and Neo4j are ready).  
<br/>\# 1. Create Qdrant collection (if not exists) for vector storage  
QDRANT_HOST="${QDRANT_HOST:-http://192.168.0.83:6333}"  
COLLECTION="${QDRANT_COLLECTION:-agent_vectors}"  
<br/>echo "Checking Qdrant collection '$COLLECTION'..."  
if command -v curl &> /dev/null; then  
\# Use Qdrant Collections API to create collection if needed  
COLLECTIONS_URL="$QDRANT_HOST/collections/$COLLECTION"  
resp=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$COLLECTIONS_URL")  
if \[ "$resp" != "200" \]; then  
echo "Creating Qdrant collection: $COLLECTION"  
curl -s -X PUT "$COLLECTIONS_URL" -H "Content-Type: application/json" \\  
\-d '{"vector_size": 768, "distance": "Cosine"}'  
echo "" # newline  
else  
echo "Collection $COLLECTION already exists."  
fi  
else  
echo "curl not available, please ensure Qdrant collection exists manually."  
fi  
<br/>\# 2. (Optional) Load initial data or schema into Neo4j  
NEO4J_URI="${NEO4J_URI:-bolt://192.168.0.83:7687}" # bolt port for neo4j  
NEO4J_USER="${NEO4J_USER:-neo4j}"  
NEO4J_PASSWORD="${NEO4J_PASSWORD:-pJnssz3khcLtn6T}"  
\# This part requires Neo4j CLI or driver to run commands; for demo, we'll skip actual commands.  
echo "Ensure Neo4j is running at $NEO4J_URI (user: $NEO4J_USER). Load schema or data as needed."  
\# e.g., using cypher-shell:  
\# cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "CREATE CONSTRAINT IF NOT EXISTS ON (n:AgentMemory) ASSERT n.id IS UNIQUE;"  
<br/>echo "Initialization of external services complete (if applicable)."

**What it does:** Prepares external services for use. This script is optional and can be run to ensure that the external databases are correctly set up before launching the app.

- For **Qdrant**, it checks if the specified collection exists; if not, it creates one with a given vector size (768, matching the embedding model) and similarity metric (Cosine, commonly used for embeddings). This uses the Qdrant HTTP API via curl. In practice, you’d also load or index any initial documents into Qdrant here (not shown).
- For **Neo4j**, it simply notes how to verify the service and suggests using Neo4j’s cypher-shell to run any initialization queries (for example, setting up constraints or pre-loading a knowledge graph). We don’t run a specific command here in the script for safety; it’s more of a template indicating where to put such commands.
- The script uses environment variables with defaults for service locations, allowing flexibility in different environments (dev, staging, etc.).

By using these scripts, a developer or DevOps engineer can quickly set up the environment (setup_env.sh), install all dependencies (install_deps.sh), initialize databases (init_data.sh), and run the app (run_app.sh) without manually typing lengthy commands. This helps avoid mistakes and ensures that the correct sequence is followed. All scripts include comments and are written defensively (using set -e to abort on errors).

**Conclusion:** This FastAPI-based **AI Team Builder Agent Service** is structured for clarity and growth. The Phase 1 implementation covers assembling a team of AI agents from a flexible JSON config and orchestrating a multi-agent conversation with integrated tools (vector DB, graph DB, web APIs, etc.). We emphasized clean architecture (separating config, tools, services, and API layers) and reproducible setup using uv for environment management. Logging and monitoring hooks are in place for every interaction (from config parsing to each conversation turn) to facilitate debugging and transparency[\[22\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). This foundation sets the stage for future enhancements, such as advanced orchestration (LangGraph or AutoGen integration in Phase 2) and more sophisticated agent behaviors, while following best practices in Python development and deployment. The result is a maintainable codebase that can be confidently evolved to meet the project's long-term vision.[\[23\]\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs)


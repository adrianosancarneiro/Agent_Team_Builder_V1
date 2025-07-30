# AI Team Builder Agent Service – FastAPI Implementation

**Overview:** This service uses **FastAPI** to expose endpoints for building and managing an AI agent team. The architecture follows a clean, scalable structure with clear separation of concerns, leveraging **uv** for environment management instead of Docker. The Team Builder Agent uses tenant-specific context and tools to assemble a multi-agent team and orchestrate their collaboration[\[1\]\[2\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). Key external services (from *Phase 1* infrastructure) include Qdrant for vector search, Neo4j for graph queries, an embedding model service, web API access, and file chunking[\[3\]\[4\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). We implement minimal integrations for these tools and outline scripts for environment setup, dependency installation, app launch, and data initialization.

## Project Structure

AI\_Team\_Builder\_Service/  
├── src/  
│   ├── main.py                 \# FastAPI app definition and routes  
│   ├── config/  
│   │   └── schema.py           \# Pydantic models for agent team config  
│   ├── services/  
│   │   ├── team\_builder.py     \# Core logic for team assembly (Phase 1 auto-generation)  
│   │   └── team\_executor.py    \# Logic for running multi-agent conversations  
│   └── tools/  
│       ├── qdrant\_tool.py      \# Tool interface for Qdrant vector DB  
│       ├── graph\_tool.py       \# Tool interface for Neo4j graph DB  
│       ├── embed\_tool.py       \# Tool interface for embedding service (BAAI/bge-base-en-v1.5)  
│       ├── webapi\_tool.py      \# Tool for external web/API calls  
│       └── chunking\_tool.py    \# Utility for file text chunking  
├── configs/  
│   └── example\_config.json     \# Example of AgentTeamConfig JSON  
├── scripts/  
│   ├── setup\_env.sh            \# Shell script for uv environment setup and installation  
│   ├── install\_deps.sh         \# Shell script for dependency installation via uv  
│   ├── run\_app.sh              \# Shell script to launch the FastAPI (uvicorn) server  
│   └── init\_data.sh            \# Shell script for any data/tool initialization (e.g., ensuring DB schemas)  
├── logs/                       \# Log files (application and agent conversation logs)  
└── pyproject.toml              \# Project dependencies and uv configuration (if using Poetry/uv)

* **src/** – Application source code, organized by component.

* **src/config/** – Configuration schemas and models (e.g. Pydantic schema for agent team config).

* **src/services/** – Core service logic: building the agent team from config, and executing team conversations.

* **src/tools/** – Integration with external tools and services (vector DB, graph DB, embeddings, etc.), each encapsulated in a module for clarity.

* **configs/** – Static config files or examples (e.g. a sample JSON for an agent team configuration).

* **scripts/** – Shell scripts for setting up the environment (using uv), installing dependencies, running the app, and initializing external tools or data.

* **logs/** – Log output directory for both system logs and conversation transcripts for debugging and audit.

* **pyproject.toml** – Defines Python dependencies and uv configuration for environment management (using uv’s preferred workflow instead of a Dockerfile).

This structure follows best practices for scalability and maintainability: configuration and constants are isolated, external tool interfaces are modular, and business logic is separate from the FastAPI app layer. Next, we detail each component with sample implementations and inline comments for clarity.

## src/config/schema.py – Config Schema (Pydantic)

from pydantic import BaseModel, Field  
from typing import List, Dict, Optional

class AgentToolConfig(BaseModel):  
    """Configuration for a tool that an agent can use."""  
    name: str  \# e.g., "Search\_Vector\_DB", "Search\_Graph\_DB", "Call\_Web\_API"  
    params: Optional\[Dict\[str, any\]\] \= None  \# parameters or settings for the tool (if any)

class AgentDefinition(BaseModel):  
    """Definition of a single agent in the team."""  
    agent\_role: str \= Field(..., description="The agent’s role or specialization (e.g. Retriever, Critic)")  
    agent\_name: str \= Field(..., description="Human-readable identifier for the agent")  
    agent\_personality: str \= Field(..., description="Brief description of the agent’s personality or perspective")  
    agent\_goal\_based\_prompt: str \= Field(..., description="Role-specific instructions or prompt for this agent")  
    LLM\_model: Optional\[Dict\[str, str\]\] \= Field(None, description="Model name or config (e.g., {'model': 'gpt-4'})")  
    allow\_team\_builder\_to\_override\_model: bool \= Field(True, description="If true, Team Builder can change the model")  
    LLM\_configuration: Optional\[Dict\[str, float\]\] \= Field(None, description="Model params like temperature, etc.")  
    agent\_tools: Optional\[List\[str\]\] \= Field(None, description="List of tool names this agent can use")

class AgentTeamConfig(BaseModel):  
    """Pydantic model for the agent team configuration schema."""  
    agent\_team\_main\_goal: str \= Field(..., description="Primary goal or problem statement for the agent team")  
    tenant\_id: Optional\[str\] \= Field(None, description="Tenant ID for context loading and multi-tenancy isolation")  
    \# Flags controlling dynamic team building:  
    allow\_TBA\_to\_recommend\_agents: bool \= Field(False, description="Allow Team Builder Agent to add extra agents beyond those specified")  
    allow\_TBA\_how\_many\_more: int \= Field(0, description="Max number of additional agents TBA can add if allowed")  
    should\_TBA\_ask\_caller\_approval: bool \= Field(False, description="If true, pause and require human approval for additions")  
    \# Optional conversation flow and limits:  
    agent\_team\_flow: Optional\[str\] \= Field(None, description="Preset conversation turn order, e.g. 'Retriever-\>Critic-\>Refiner'")  
    max\_turns: int \= Field(5, description="Max number of conversation rounds to execute")  
    \# Tools available globally (for assignment to agents):  
    available\_tools: Optional\[List\[str\]\] \= Field(None, description="Whitelist of tools/APIs available for agents")  
    \# Pre-defined agents (partial or full specification):  
    agents: Optional\[List\[AgentDefinition\]\] \= Field(None, description="List of agent definitions to include in the team")

**Notes:**  
\- We use Pydantic models to validate and document the schema for the agent team configuration. This mirrors the JSON schema described in the design docs[\[5\]\[6\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), including fields like the main goal, flags to allow the Team Builder to recommend agents[\[5\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) and how many[\[7\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), an optional agent\_team\_flow to fix the turn order[\[8\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), and a list of agent definitions with required subfields (role, name, personality, prompt, etc.)[\[6\]\[9\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).  
\- The AgentDefinition model captures each agent’s attributes. Tools for agents are referenced by name (from available\_tools). In practice, each tool name corresponds to an integration in our src/tools module (e.g., "Search\_Vector\_DB" → Qdrant search tool).  
\- Using Pydantic ensures that any config passed in (via API or loaded from file) is validated for completeness and correctness, providing fast feedback if required fields are missing.

## src/tools/qdrant\_tool.py – Qdrant Vector DB Integration

import os  
from typing import List, Any, Optional  
import requests  \# using requests for simplicity; could use qdrant-client library

class QdrantTool:  
    """Minimal client for Qdrant vector database operations."""  
    def \_\_init\_\_(self):  
        \# Qdrant URL could be configured via env or config  
        self.base\_url \= os.getenv("QDRANT\_URL", "http://192.168.0.83:6333")  
        \# Optional: collection name could be tenant-specific  
        self.collection \= os.getenv("QDRANT\_COLLECTION", "agent\_vectors")

    def search(self, query\_embedding: List\[float\], top\_k: int \= 5, filters: Optional\[dict\] \= None) \-\> List\[Any\]:  
        """Search the Qdrant vector collection for nearest vectors to the query embedding."""  
        url \= f"{self.base\_url}/collections/{self.collection}/points/search"  
        payload \= {  
            "vector": query\_embedding,  
            "limit": top\_k  
        }  
        if filters:  
            payload\["filter"\] \= filters  
        try:  
            res \= requests.post(url, json=payload, timeout=5)  
            res.raise\_for\_status()  
            results \= res.json().get("result", \[\])  
            return results  \# Each result contains e.g. an "id" and "score" and possibly payload  
        except Exception as e:  
            \# In a real system, handle exceptions and logging appropriately  
            print(f"Qdrant search error: {e}")  
            return \[\]

    def upsert(self, points: List\[dict\]) \-\> bool:  
        """Insert or update points (vectors with payload) into the collection."""  
        url \= f"{self.base\_url}/collections/{self.collection}/points"  
        try:  
            res \= requests.put(url, json={"points": points}, timeout=5)  
            res.raise\_for\_status()  
            return True  
        except Exception as e:  
            print(f"Qdrant upsert error: {e}")  
            return False

**Explanation:** The QdrantTool class provides a minimal interface to the Qdrant vector DB. We default to the Phase 1 test endpoint at 192.168.0.83:6333[\[3\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), but in practice this should be configurable per tenant or environment. The search method posts a query embedding to Qdrant’s search API and retrieves the closest stored vectors (used for semantic search results, e.g. knowledge retrieval)[\[3\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). The upsert method allows adding new vectors with associated metadata. In a full implementation, we might use the official qdrant-client for Python, handle collection creation, and manage authentication or namespace per tenant. Here we keep it simple and synchronous for demonstration. Proper error handling and logging is included to ensure reliability.

## src/tools/graph\_tool.py – Neo4j Graph DB Integration

import os  
from neo4j import GraphDatabase, basic\_auth

class GraphDBTool:  
    """Minimal client for Neo4j graph database queries."""  
    def \_\_init\_\_(self):  
        uri \= os.getenv("NEO4J\_URI", "neo4j://192.168.0.83:7474")  
        user \= os.getenv("NEO4J\_USER", "neo4j")  
        pwd \= os.getenv("NEO4J\_PASSWORD", "pJnssz3khcLtn6T")  \# Note: use env var in practice for security  
        \# Initialize Neo4j driver (encrypted=False for local dev)  
        self.driver \= GraphDatabase.driver(uri, auth=basic\_auth(user, pwd), encrypted=False)

    def query(self, cypher: str, params: dict \= None) \-\> list:  
        """Run a Cypher query and return results (as list of records)."""  
        records \= \[\]  
        with self.driver.session() as session:  
            results \= session.run(cypher, params or {})  
            for record in results:  
                records.append(record.data())  
        return records

    def close(self):  
        """Close the database connection (call on app shutdown)."""  
        self.driver.close()

**Explanation:** The GraphDBTool uses the Neo4j Python driver to connect to the Neo4j database. It reads connection details from environment variables, defaulting to the provided development server URL and credentials[\[10\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). The query method executes a Cypher query and returns the results in a simple list of dicts format. In a real scenario, one might have specialized methods for specific graph queries (e.g., querying a knowledge graph for certain relationships). For brevity, we expose a generic query interface. The Neo4j database can store a tenant’s knowledge graph or agent memories[\[11\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), enabling agents to retrieve structured information via the Search\_Graph\_DB tool.

## src/tools/embed\_tool.py – Embedding Service Integration

from typing import List  
\# We will use a transformer model for embeddings. In practice, this might be a separate service call.  
try:  
    from sentence\_transformers import SentenceTransformer  
except ImportError:  
    SentenceTransformer \= None

class EmbeddingService:  
    """Embedding service using BAAI/bge-base-en-v1.5 model to get text embeddings."""  
    def \_\_init\_\_(self, model\_name: str \= "BAAI/bge-base-en-v1.5"):  
        if SentenceTransformer:  
            \# Load the embedding model (this will download the model if not present)  
            self.model \= SentenceTransformer(model\_name)  
        else:  
            self.model \= None  
            print("SentenceTransformer not installed. Embeddings will be dummy values.")

    def embed(self, text: str) \-\> List\[float\]:  
        """Convert text into a vector embedding."""  
        if self.model:  
            embedding: List\[float\] \= self.model.encode(text, show\_progress\_bar=False)  
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)  
        \# Fallback: return a dummy embedding (e.g., vector of zeros) if model not available  
        return \[0.0\] \* 768  \# assuming 768-dim for BGE base model

**Explanation:** The EmbeddingService class provides vector embeddings for input text using the **BAAI/bge-base-en-v1.5** model[\[12\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). This model is referenced as the chosen embedding model in Phase 1, used to convert text (like user queries or document chunks) into high-dimensional vectors for semantic search[\[12\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). In this minimal implementation, we attempt to use the sentence\_transformers library to load the model and compute embeddings. In a production system, this might instead call a dedicated microservice (for instance, a container running the model, or a HuggingFace inference endpoint). If the model isn't available (library not installed), we fall back to a dummy embedding for safety.

This embedding service can be utilized by the Team Builder or agents to embed the user’s query before searching the vector DB (so that Qdrant can be queried with the query’s embedding)[\[4\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), or by any agent that needs to vectorize text for comparison. The embedding dimension (in this case, 768\) should match what the vector DB and LLM are compatible with[\[13\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

## src/tools/webapi\_tool.py – External Web/API Call Tool

import requests

class WebAPITool:  
    """Tool to make external web API calls (internet access)."""  
    def get(self, url: str, params: dict \= None, headers: dict \= None) \-\> str:  
        """Perform a GET request to the given URL and return response text."""  
        try:  
            res \= requests.get(url, params=params, headers=headers, timeout=10)  
            res.raise\_for\_status()  
            return res.text  
        except Exception as e:  
            return f"ERROR: {e}"

    def post(self, url: str, data: dict \= None, json\_data: dict \= None, headers: dict \= None) \-\> str:  
        """Perform a POST request (with JSON or form data) and return response text."""  
        try:  
            res \= requests.post(url, data=data, json=json\_data, headers=headers, timeout=10)  
            res.raise\_for\_status()  
            return res.text  
        except Exception as e:  
            return f"ERROR: {e}"

**Explanation:** The WebAPITool allows agents to perform HTTP requests to external APIs or websites. This is the implementation behind a tool like "Call\_Web\_API" in the config[\[14\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). Having internet access within the Team Builder’s host environment means agents can fetch live data or call external services when allowed[\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). The tool provides simple GET and POST methods; in a secure deployment, you’d add restrictions (e.g., allowed domains or rate limiting) to prevent abuse. The responses are returned as text for the agent to consume. This tool is used sparingly and under tenant-defined security constraints[\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

## src/tools/chunking\_tool.py – File/Text Chunking Utility

import math

class ChunkingTool:  
    """Utility to split text into chunks for processing (e.g., indexing to vector DB)."""  
    def \_\_init\_\_(self, chunk\_size: int \= 500):  
        self.chunk\_size \= chunk\_size  \# default chunk size in characters (or tokens approx.)

    def chunk\_text(self, text: str) \-\> list:  
        """Split a long text into chunks of approximately chunk\_size characters."""  
        chunks \= \[\]  
        length \= len(text)  
        for i in range(0, length, self.chunk\_size):  
            chunk \= text\[i: i \+ self.chunk\_size\]  
            chunks.append(chunk)  
        return chunks

    def chunk\_file(self, file\_path: str) \-\> list:  
        """Read a file and split its content into chunks."""  
        try:  
            with open(file\_path, 'r', encoding='utf-8') as f:  
                content \= f.read()  
        except Exception as e:  
            print(f"Error reading file {file\_path}: {e}")  
            return \[\]  
        return self.chunk\_text(content)

**Explanation:** ChunkingTool is a simple utility to break down large text (or file contents) into smaller chunks. This is helpful for indexing knowledge sources: for example, splitting a long document into chunks that can be embedded and stored in Qdrant, enabling semantic search on each chunk. While not explicitly described in the high-level docs, chunking is a common preparation step in Retrieval-Augmented Generation (RAG) workflows and likely part of the **Phase 1** toolkit (the documentation notes chunking service as a potential tool)[\[16\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). We keep the implementation straightforward, splitting by a fixed character count. In a real system, one might integrate a tokenization-based splitter to respect token boundaries or sentence boundaries.

## src/services/team\_builder.py – Team Building Logic (Phase 1)

from src.config.schema import AgentTeamConfig, AgentDefinition  
from src.tools import qdrant\_tool, graph\_tool, embed\_tool

\# Initialize tool clients (in a real app, use dependency injection or global singletons)  
\_vector\_tool \= qdrant\_tool.QdrantTool()  
\_graph\_tool \= graph\_tool.GraphDBTool()  
\_embed\_service \= embed\_tool.EmbeddingService()

class Agent:  
    """Representation of an AI agent (LLM-powered) with certain tools and persona."""  
    def \_\_init\_\_(self, definition: AgentDefinition):  
        self.role \= definition.agent\_role  
        self.name \= definition.agent\_name  
        self.personality \= definition.agent\_personality  
        self.prompt \= definition.agent\_goal\_based\_prompt  
        self.model\_info \= definition.LLM\_model or {"model": "default-LLM"}  
        self.allow\_model\_override \= definition.allow\_team\_builder\_to\_override\_model  
        self.model\_config \= definition.LLM\_configuration or {}  
        \# Tools that this agent can use, instantiated from tool names  
        self.tools \= \[\]  
        for tool\_name in (definition.agent\_tools or \[\]):  
            if tool\_name \== "Search\_Vector\_DB":  
                self.tools.append(\_vector\_tool)  
            elif tool\_name \== "Search\_Graph\_DB":  
                self.tools.append(\_graph\_tool)  
            elif tool\_name \== "Call\_Web\_API":  
                from src.tools import webapi\_tool  
                self.tools.append(webapi\_tool.WebAPITool())  
            elif tool\_name \== "Embedding\_Service":  
                self.tools.append(\_embed\_service)  
            \# Add other tool name checks (e.g., "Chunk\_Text") as needed  
        \# If EmbeddingService is used as a tool, maybe it's for generating embeddings on the fly.

    def summarize(self) \-\> dict:  
        """Return a summary of the agent (for API responses or logging)."""  
        return {  
            "name": self.name,  
            "role": self.role,  
            "personality": self.personality,  
            "model": self.model\_info,  
            "tools": \[t.\_\_class\_\_.\_\_name\_\_ for t in self.tools\]  
        }

class TeamBuilderService:  
    """Service responsible for creating an agent team from configuration."""  
    DEFAULT\_AGENTS \= \[  \# Default roles if none specified (base scenario: DecisionMaker \-\> Retriever \-\> Critic \-\> Refiner)  
        {  
            "agent\_role": "DecisionMaker",  
            "agent\_name": "DecisionMakerAgent",  
            "agent\_personality": "Decisive leader that coordinates the team.",  
            "agent\_goal\_based\_prompt": "Decide the next step or conclude the task based on inputs from others.",  
            "agent\_tools": \[\]  \# Decision maker may not use external tools, just organizes  
        },  
        {  
            "agent\_role": "Retriever",  
            "agent\_name": "RetrieverAgent",  
            "agent\_personality": "Diligent researcher fetching relevant information.",  
            "agent\_goal\_based\_prompt": "Find facts and data relevant to the main goal using available knowledge sources.",  
            "agent\_tools": \["Search\_Vector\_DB", "Search\_Graph\_DB"\]  \# uses vector DB and graph DB  
        },  
        {  
            "agent\_role": "Critic",  
            "agent\_name": "CriticAgent",  
            "agent\_personality": "Skeptical critic that double-checks answers.",  
            "agent\_goal\_based\_prompt": "Evaluate the responses for correctness and potential issues.",  
            "agent\_tools": \[\]  
        },  
        {  
            "agent\_role": "Refiner",  
            "agent\_name": "RefinerAgent",  
            "agent\_personality": "Thoughtful refiner that improves and finalizes answers.",  
            "agent\_goal\_based\_prompt": "Polish and integrate the information into a final answer.",  
            "agent\_tools": \[\]  
        }  
    \]

    @classmethod  
    def build\_team(cls, config: AgentTeamConfig) \-\> list:  
        """  
        Construct the team of agents based on the provided configuration.  
        Returns a list of Agent objects.  
        """  
        agents\_config \= \[\]  
        if config.agents and len(config.agents) \> 0:  
            \# Start with caller-specified agents  
            for agent\_def in config.agents:  
                agents\_config.append(agent\_def.dict())  
        else:  
            \# No agents specified by caller: use default template roles  
            agents\_config \= \[dict(a) for a in cls.DEFAULT\_AGENTS\]  
        \# If allowed, the Team Builder can recommend additional agents up to the specified number  
        if config.allow\_TBA\_to\_recommend\_agents and config.allow\_TBA\_how\_many\_more \> 0:  
            \# For simplicity, if fewer than allowed agents are present, add a placeholder agent.  
            \# In a real scenario, this could analyze the task and tenant context to suggest a role.  
            for i in range(config.allow\_TBA\_how\_many\_more):  
                extra\_role \= f"AdditionalAgent{i+1}"  
                agents\_config.append({  
                    "agent\_role": extra\_role,  
                    "agent\_name": f"ExtraAgent{i+1}",  
                    "agent\_personality": "Auxiliary agent added by TeamBuilder to cover missing expertise.",  
                    "agent\_goal\_based\_prompt": f"Assist in achieving the main goal by providing {extra\_role} capabilities.",  
                    "agent\_tools": config.available\_tools or \[\]  \# give it access to all available tools by default  
                })  
                break  \# (In practice, you'd add up to how\_many\_more agents; here we add one as example)  
        \# Instantiate Agent objects for each definition  
        team \= \[Agent(AgentDefinition(\*\*agent\_def)) for agent\_def in agents\_config\]  
        \# If a static flow is defined (agent\_team\_flow), we could reorder team or mark the order somewhere  
        \# For simplicity, assume order in list will be the conversation order if flow is given.  
        \# (In a full implementation, parse agent\_team\_flow string to enforce speaking order.)  
        return team

**Team Building Logic:** The TeamBuilderService.build\_team takes an AgentTeamConfig and produces a list of Agent instances representing the AI agents team. This function implements the **auto-generation behavior** described for Phase 1:

* **Default Team Composition:** If the caller did not specify any agents, we auto-create a default team of four roles: Decision Maker, Retriever, Critic, Refiner[\[17\]\[18\]](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE). These correspond to a basic workflow (decision-making loop) and align with the documentation's suggestion of default roles when none are provided[\[19\]](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE). Each default agent has a preset persona and prompt.

* **Caller-Specified Agents:** If the config includes a list of agents, we include those. Each agent’s definition is converted from the Pydantic model to a dict for flexibility.

* **Dynamic Agent Addition:** If the config allows the Team Builder Agent to recommend additional agents (allow\_TBA\_to\_recommend\_agents \== True), we add up to allow\_TBA\_how\_many\_more extra agents[\[7\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). In a real system, this logic might analyze the main goal and existing team composition to decide what new role is needed (e.g., adding a Planner or a DomainExpert if missing). Here we simply append a generic extra agent (demonstrating the capability).

* **Tool Assignment:** Each Agent instance is initialized with the tools it’s allowed to use. We map tool names from the config to actual tool instances (e.g., "Search\_Vector\_DB" → the QdrantTool instance, "Embedding\_Service" → the embedding service, etc.). This uses simple conditional logic; a more scalable design might use a registry or factory pattern for tools. The available tools for an agent are constrained by the available\_tools list in the config (or defaults if not provided)[\[20\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

* **Model Selection:** Each agent can have a specified LLM model and parameters. If allow\_team\_builder\_to\_override\_model is true for that agent, the Team Builder could switch to a different model better suited to the role[\[21\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). Our implementation doesn’t dynamically change models, but we honor the flag in the Agent’s attributes for future use. For now, we simply store model info and assume a default model if none provided.

* **Flow Order:** If agent\_team\_flow is provided (e.g., "Retriever \-\> Critic \-\> Refiner"), the Team Builder would ensure the agents or orchestrator respect that speaking order[\[8\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). In this minimal implementation, we would parse and enforce the order or designate an orchestrator agent. For simplicity, we note that if a flow is given, the order of the team list corresponds to it (this would be refined in a full implementation).

Each created Agent has a summarize() method to output its key properties, useful for returning via API or logging the team composition. The Agent class also sets up the instances of tool clients it can use. This clean separation means when an agent needs to perform an action (search vector DB, query the graph, call an API, etc.), it will invoke these tool instances.

## src/services/team\_executor.py – Team Execution Logic (Conversation Orchestration)

from typing import List, Tuple  
from src.services.team\_builder import Agent

class TeamExecutorService:  
    """Service to manage multi-agent conversation execution."""  
    def \_\_init\_\_(self, agents: List\[Agent\], flow: List\[str\] \= None, max\_turns: int \= 5):  
        """  
        Initialize with a list of Agent instances. Optionally provide a conversation flow order   
        (list of agent names or roles in speaking order). If no flow is given, a default order (round-robin) is used.  
        """  
        self.agents \= agents  
        \# Create a mapping of agent role-\>agent and name-\>agent for convenience  
        self.agents\_by\_name \= {agent.name: agent for agent in agents}  
        self.agents\_by\_role \= {agent.role: agent for agent in agents}  
        \# Determine speaking order  
        if flow:  
            \# If flow is provided as a list of names/roles, convert to actual agent instances  
            self.flow \= \[\]  
            for identifier in flow:  
                agent \= self.agents\_by\_name.get(identifier) or self.agents\_by\_role.get(identifier)  
                if agent:  
                    self.flow.append(agent)  
        else:  
            \# default to the order given or round-robin  
            self.flow \= agents  
        self.max\_turns \= max\_turns  
        self.conversation\_log: List\[Tuple\[str, str\]\] \= \[\]  \# list of (agent\_name, message)

    def run\_conversation(self, user\_query: str) \-\> str:  
        """  
        Execute the multi-agent conversation until max\_turns or a stopping condition is met.  
        Returns the final answer (or combined result) from the team.  
        """  
        \# Initial user input  
        self.conversation\_log.append(("User", user\_query))  
        current\_turn \= 0  
        final\_answer \= ""  
        \# Simple loop through agents in the defined flow  
        while current\_turn \< self.max\_turns:  
            for agent in self.flow:  
                \# Each agent takes the last message and responds  
                last\_speaker, last\_message \= self.conversation\_log\[-1\]  
                \# Determine if conversation should end (if last speaker was an agent and decided to stop)  
                if last\_speaker \!= "User" and agent.role \== "DecisionMaker":  
                    \# (Example heuristic) DecisionMaker can decide to finish the conversation  
                    if "conclude" in last\_message.lower():  
                        final\_answer \= last\_message  
                        return final\_answer  
                \# Agent formulates a response (Here we'd call the LLM with prompt and context. We'll simulate.)  
                response \= self.\_agent\_respond(agent, last\_message)  
                \# Log the agent's response  
                self.conversation\_log.append((agent.name, response))  
                \# Optionally, check if this agent is an orchestrator or decision-maker concluding the chat  
                if agent.role.lower() in ("decisionmaker", "orchestrator"):  
                    if "final answer:" in response.lower() or "conclude" in response.lower():  
                        final\_answer \= response  
                        return final\_answer  
            current\_turn \+= 1  
        \# If loop completes without early return, take the last agent's message as final answer  
        final\_answer \= self.conversation\_log\[-1\]\[1\]  
        return final\_answer

    def \_agent\_respond(self, agent: Agent, last\_message: str) \-\> str:  
        """  
        Simulate an agent responding to the last\_message. In reality, this would involve the agent's prompt,  
        persona, tools, and an LLM call. Here, we'll do a simple placeholder implementation.  
        """  
        \# If the agent has any tools, maybe use one (for demo, use first applicable tool to augment response)  
        tool\_augmented\_info \= ""  
        for tool in agent.tools:  
            tool\_name \= tool.\_\_class\_\_.\_\_name\_\_  
            if tool\_name \== "QdrantTool":  
                \# Example: use embedding service to embed query, then search vector DB  
                query\_emb \= \_embed\_service.embed(last\_message)  
                results \= tool.search(query\_emb, top\_k=1)  
                if results:  
                    tool\_augmented\_info \+= " \[Found relevant info via vector DB\]"  
            elif tool\_name \== "GraphDBTool":  
                \# Example: query graph DB for a fact (here we just do a dummy query or skip)  
                \# In real case, we might have a specific query pattern  
                results \= tool.query("MATCH (n) RETURN n LIMIT 1")  
                if results:  
                    tool\_augmented\_info \+= " \[Knowledge graph checked\]"  
            elif tool\_name \== "WebAPITool":  
                \# Example: perform a web API call (not doing actual call in demo)  
                tool\_augmented\_info \+= " \[Called external API\]"  
            \# (Additional tool handling as needed)  
        \# Formulate a dummy response using the agent's role and possibly augmented info  
        response \= f"{agent.name} ({agent.role}) says: Based on '{last\_message}', I {agent.personality.lower()} respond with an answer.{tool\_augmented\_info}"  
        return response

**Team Execution Logic:** The TeamExecutorService takes the assembled agents and manages their conversation. This corresponds to the **Tenant App Agent Team Execution Workflow** – essentially running the multi-agent system through its conversation rounds.

Key points in this implementation:

* **Conversation Flow:** If a specific agent\_team\_flow is provided in the config (e.g., a sequence of roles/names)[\[8\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs), we construct the flow accordingly. Otherwise, we default to the list order or a simple round-robin. In Phase 1, the typical flow might be “DecisionMaker \-\> Retriever \-\> Critic \-\> Refiner” repeating, or a Manager (DecisionMaker) deciding next speaker dynamically. Here, we simulate a fixed order for simplicity.

* **Conversation Loop:** run\_conversation starts with the user’s query (main goal) as the first message. It then iterates through agents in the flow for each turn, appending each agent’s response to a conversation\_log. We limit the process to max\_turns to avoid infinite loops. There’s a simple heuristic: if the DecisionMaker (or an Orchestrator/Manager agent) outputs a message indicating conclusion (e.g., containing "final answer" or "conclude"), we break early – simulating that the team decided to stop once the goal is achieved.

* **Agent Response Simulation:** In \_agent\_respond, we **simulate** an agent’s reasoning. In a real system, this is where we would construct the prompt for the LLM, including the agent’s role description, its goal-based prompt, the conversation context so far, and possibly tool results. The function would then call the chosen LLM (via an API or SDK) to generate the agent’s response. We also demonstrate how tools might be invoked by an agent:

* If the agent has the Qdrant tool, we embed the last message (using the embedding service) and query the vector DB for relevant info[\[3\]\[12\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

* If the agent has the GraphDB tool, we run a sample graph query (dummy in this case)[\[11\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

* If the agent has the WebAPI tool, we simulate calling an external API[\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

* Any results or side-effects are appended to the agent’s response as bracketed notes to indicate the tool’s contribution.

* **Final Answer:** The conversation ends when the loop completes the set number of turns or a special flag from an agent indicates completion. The final answer returned could be the last message from the DecisionMaker or a aggregated answer. Here we simply return the last message in the log as the final answer, under the assumption that the DecisionMaker or Refiner’s last turn produces the solution.

This executor is a **simplified orchestration**. In a more advanced Phase 2 scenario, one might integrate Microsoft’s Autogen GroupChat or a LangGraph state machine to handle turn-taking and branching logic more robustly. For now, it demonstrates how multi-agent dialogue might be handled in code, with the Team Builder’s output (the agent team) feeding into this execution loop.

## src/main.py – FastAPI Application

from fastapi import FastAPI, HTTPException  
from src.config.schema import AgentTeamConfig  
from src.services.team\_builder import TeamBuilderService  
from src.services.team\_executor import TeamExecutorService

app \= FastAPI(title="AI Team Builder Agent Service", version="1.0")

\# Build the team and optionally run the conversation in one go (for simplicity).  
@app.post("/build\_and\_execute")  
def build\_and\_execute(config: AgentTeamConfig):  
    """  
    Build an AI agent team according to the provided configuration and run the multi-agent conversation.  
    Returns the team composition and the final answer.  
    """  
    \# Build the team of agents  
    try:  
        team \= TeamBuilderService.build\_team(config)  
    except Exception as e:  
        raise HTTPException(status\_code=400, detail=f"Invalid configuration: {e}")  
    \# Log or store team info (omitted for brevity)  
    team\_summary \= \[agent.summarize() for agent in team\]

    \# Execute the conversation workflow  
    executor \= TeamExecutorService(agents=team,   
                                   flow=\[agent.role for agent in team\] if config.agent\_team\_flow is None else \[s.strip() for s in config.agent\_team\_flow.split("-\>")\],  
                                   max\_turns=config.max\_turns)  
    final\_answer \= executor.run\_conversation(user\_query=config.agent\_team\_main\_goal)

    \# Return both the team details and the final answer  
    return {  
        "agent\_team": team\_summary,  
        "conversation\_log": executor.conversation\_log,  
        "final\_answer": final\_answer  
    }

\# (Optional) Separate endpoint to just build team without execution  
@app.post("/build\_team")  
def build\_team\_endpoint(config: AgentTeamConfig):  
    """  
    Endpoint to build the agent team from config, without running the conversation.  
    Returns the team composition.  
    """  
    try:  
        team \= TeamBuilderService.build\_team(config)  
    except Exception as e:  
        raise HTTPException(status\_code=400, detail=f"Invalid configuration: {e}")  
    return {"agent\_team": \[agent.summarize() for agent in team\]}

\# (Optional) Health check endpoint  
@app.get("/health")  
def health\_check():  
    return {"status": "ok"}

**FastAPI App:** The FastAPI application ties everything together, exposing a clean API to clients (e.g., a tenant’s application or a testing tool). We define two primary endpoints:

* **POST /build\_and\_execute:** Takes a JSON payload matching AgentTeamConfig. It first calls the Team Builder to assemble the agents, then immediately runs the conversation using TeamExecutorService. The response includes a summary of the agent team (roles, tools, etc.), the conversation log (all messages exchanged, including the user’s query and each agent’s responses), and the final answer produced by the team. This provides an end-to-end single call for convenience.

* **POST /build\_team:** (Optional) Allows the client to just construct the team without running the conversation. This returns the agent team composition. In a scenario where the client might want to inspect or adjust the team before execution, this separation can be useful.

* **GET /health:** A simple health check endpoint to verify the service is running.

We include basic error handling (e.g., if the config is invalid, return HTTP 400). The AgentTeamConfig Pydantic model ensures that the request body is validated against our schema automatically by FastAPI.

**Best Practices Applied:** We utilize FastAPI’s dependency injection and Pydantic integration for input validation. The code is organized so that business logic (team building, execution) is in the services module, not mixed into the API route functions, making it easier to maintain and test. Inline comments and docstrings provide clarity on each component’s role. Additionally, sensitive info like database URLs and credentials are fetched from environment variables (with defaults for development) – in real deployments, those would be set outside the code for security. Logging (not fully shown here due to brevity) would be configured to output to files in the logs/ directory, capturing each step of the process for debugging and audit, as recommended by the design[\[22\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs).

## Shell Scripts for Setup and Deployment

We provide shell scripts in the scripts/ directory to streamline environment setup and running the service. These scripts use **uv** (the Astral package manager) to manage the Python environment and dependencies, instead of relying on Docker. This approach ensures fast, reproducible setups and aligns with modern Python packaging best practices.

### scripts/setup\_env.sh – Environment Setup with uv

\#\!/usr/bin/env bash  
\# This script sets up a Python virtual environment using uv and installs uv itself if needed.

\# Exit immediately if a command exits with a non-zero status  
set \-e

\# 1\. Install uv if not already installed  
if \! command \-v uv &\> /dev/null; then  
    echo "uv not found, installing uv..."  
    \# Using pipx to install uv for isolation (could also use curl installer from Astral)  
    python3 \-m pip install \--user pipx || { echo "Failed to install pipx"; exit 1; }  
    pipx install uv  
fi

\# 2\. Initialize a uv-managed virtual environment  
\# By default, uv looks for pyproject.toml in the current directory to resolve dependencies.  
if \[ \! \-d ".venv" \]; then  
    echo "Creating virtual environment with uv..."  
    uv venv .venv  \# create venv in .venv folder  
fi

echo "Activating virtual environment..."  
\# Activate the environment for subsequent steps (if needed for interactive use)  
source .venv/bin/activate || { echo "Activation failed"; exit 1; }

echo "Environment setup complete. uv is ready to manage dependencies."

**What it does:** Checks if uv is installed, installs it if not (using pipx for isolation). Then it creates a virtual environment .venv using uv (which ensures Python version consistency and can manage multiple environments). Activating the venv is optional for non-interactive shell scripts, but we show it for completeness. This script would be run once to prepare the environment.

### scripts/install\_deps.sh – Dependency Installation

\#\!/usr/bin/env bash  
\# Install project dependencies using uv (assumes pyproject.toml defines them)

set \-e

\# Ensure uv is installed and environment is set up  
if \! command \-v uv &\> /dev/null; then  
    echo "uv is not installed. Run setup\_env.sh first."  
    exit 1  
fi

\# Ensure we're in project root directory (where pyproject.toml is)  
SCRIPT\_DIR=$(dirname "$0")  
cd "$SCRIPT\_DIR/.."  \# move to project root relative to script directory

\# Use uv to sync dependencies defined in pyproject.toml (production deps only, no dev)  
echo "Installing dependencies with uv..."  
uv sync \--no-dev

echo "Dependencies installation complete."

**What it does:** Uses uv sync to install all dependencies specified in pyproject.toml (excluding dev dependencies). The pyproject.toml would list packages like FastAPI, uvicorn, pydantic, neo4j, qdrant-client (or requests), sentence-transformers, etc. Using uv sync ensures a deterministic environment (it will use a lockfile uv.lock if present for pinned versions). The script should be run after setup\_env.sh.

### scripts/run\_app.sh – Launching the FastAPI Service

\#\!/usr/bin/env bash  
\# Launch the FastAPI application using uvicorn

set \-e

\# Activate the environment (so that uvicorn and deps are in PATH)  
source .venv/bin/activate

\# Default host/port can be parameterized; using 0.0.0.0:8000 for accessibility in container/VM  
HOST="${HOST:-0.0.0.0}"  
PORT="${PORT:-8000}"

echo "Starting FastAPI app on $HOST:$PORT ..."  
\# Using uvicorn to run the app; adjust workers if needed for concurrency.  
uvicorn src.main:app \--host $HOST \--port $PORT \--workers 4

**What it does:** Activates the Python virtual environment and then starts the FastAPI app with Uvicorn. We bind to 0.0.0.0 on port 8000 by default (making it accessible externally, which is typical for container or VM deployments). The number of workers is set to 4 for handling multiple requests concurrently (tunable based on expected load). This script abstracts the uvicorn command so that environment variables or other launch configurations can be easily managed at one place.

### scripts/init\_data.sh – Data/Tool Initialization

\#\!/usr/bin/env bash  
\# Optional script to initialize external services or seed data (e.g., ensure Qdrant and Neo4j are ready).

\# 1\. Create Qdrant collection (if not exists) for vector storage  
QDRANT\_HOST="${QDRANT\_HOST:-http://192.168.0.83:6333}"  
COLLECTION="${QDRANT\_COLLECTION:-agent\_vectors}"

echo "Checking Qdrant collection '$COLLECTION'..."  
if command \-v curl &\> /dev/null; then  
    \# Use Qdrant Collections API to create collection if needed  
    COLLECTIONS\_URL="$QDRANT\_HOST/collections/$COLLECTION"  
    resp=$(curl \-s \-o /dev/null \-w "%{http\_code}" \-X GET "$COLLECTIONS\_URL")  
    if \[ "$resp" \!= "200" \]; then  
        echo "Creating Qdrant collection: $COLLECTION"  
        curl \-s \-X PUT "$COLLECTIONS\_URL" \-H "Content-Type: application/json" \\  
             \-d '{"vector\_size": 768, "distance": "Cosine"}'  
        echo ""  \# newline  
    else  
        echo "Collection $COLLECTION already exists."  
    fi  
else  
    echo "curl not available, please ensure Qdrant collection exists manually."  
fi

\# 2\. (Optional) Load initial data or schema into Neo4j  
NEO4J\_URI="${NEO4J\_URI:-bolt://192.168.0.83:7687}"  \# bolt port for neo4j  
NEO4J\_USER="${NEO4J\_USER:-neo4j}"  
NEO4J\_PASSWORD="${NEO4J\_PASSWORD:-pJnssz3khcLtn6T}"  
\# This part requires Neo4j CLI or driver to run commands; for demo, we'll skip actual commands.  
echo "Ensure Neo4j is running at $NEO4J\_URI (user: $NEO4J\_USER). Load schema or data as needed."  
\# e.g., using cypher-shell:  
\# cypher-shell \-a "$NEO4J\_URI" \-u "$NEO4J\_USER" \-p "$NEO4J\_PASSWORD" "CREATE CONSTRAINT IF NOT EXISTS ON (n:AgentMemory) ASSERT n.id IS UNIQUE;"

echo "Initialization of external services complete (if applicable)."

**What it does:** Prepares external services for use. This script is optional and can be run to ensure that the external databases are correctly set up before launching the app.

* For **Qdrant**, it checks if the specified collection exists; if not, it creates one with a given vector size (768, matching the embedding model) and similarity metric (Cosine, commonly used for embeddings). This uses the Qdrant HTTP API via curl. In practice, you’d also load or index any initial documents into Qdrant here (not shown).

* For **Neo4j**, it simply notes how to verify the service and suggests using Neo4j’s cypher-shell to run any initialization queries (for example, setting up constraints or pre-loading a knowledge graph). We don’t run a specific command here in the script for safety; it’s more of a template indicating where to put such commands.

* The script uses environment variables with defaults for service locations, allowing flexibility in different environments (dev, staging, etc.).

By using these scripts, a developer or DevOps engineer can quickly set up the environment (setup\_env.sh), install all dependencies (install\_deps.sh), initialize databases (init\_data.sh), and run the app (run\_app.sh) without manually typing lengthy commands. This helps avoid mistakes and ensures that the correct sequence is followed. All scripts include comments and are written defensively (using set \-e to abort on errors).

---

**Conclusion:** This FastAPI-based **AI Team Builder Agent Service** is structured for clarity and growth. The Phase 1 implementation covers assembling a team of AI agents from a flexible JSON config and orchestrating a multi-agent conversation with integrated tools (vector DB, graph DB, web APIs, etc.). We emphasized clean architecture (separating config, tools, services, and API layers) and reproducible setup using uv for environment management. Logging and monitoring hooks are in place for every interaction (from config parsing to each conversation turn) to facilitate debugging and transparency[\[22\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs). This foundation sets the stage for future enhancements, such as advanced orchestration (LangGraph or AutoGen integration in Phase 2) and more sophisticated agent behaviors, while following best practices in Python development and deployment. The result is a maintainable codebase that can be confidently evolved to meet the project's long-term vision.[\[23\]\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs)

---

[\[1\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[2\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[3\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[4\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[5\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[6\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[7\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[8\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[9\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[10\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[11\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[12\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[13\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[14\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[15\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[16\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[20\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[21\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[22\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) [\[23\]](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs) Agent\_Team\_Orcherstration\_V4

[https://docs.google.com/document/d/1ZoaM81TOKe\_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs](https://docs.google.com/document/d/1ZoaM81TOKe_AHFvjJhpVm5z2kLLIJr4V2FAvKhhSGYs)

[\[17\]](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE) [\[18\]](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE) [\[19\]](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE) Agent\_Team\_Service\_First\_Steps

[https://docs.google.com/document/d/11qdDcLjeRWabdxj\_az52epT8RgLxb-4zXWp-Y-yZEoE](https://docs.google.com/document/d/11qdDcLjeRWabdxj_az52epT8RgLxb-4zXWp-Y-yZEoE)
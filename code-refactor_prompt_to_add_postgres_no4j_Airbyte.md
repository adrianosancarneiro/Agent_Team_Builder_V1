## **code-refactor\_prompt\_to\_add\_postgres\_no4j\_Airbyte**

You are an expert Python architect.  
Your task is to upgrade **Agent Team Builder V1** so each tenant’s agent‑team configuration is 

* **persisted** once in **PostgreSQL (JSONB)**,

* **automatically mirrored** into **Neo4j** by an **Airbyte Postgres‑►Neo4j connection**, and

* reused on subsequent `/execute` calls unless the caller supplies an updated config document.

**High‑level flow**  
 1\. Client → `POST /teams` – create / update a team; FastAPI writes JSONB row ➜ Airbyte CDC sends the delta to Neo4j.  
 2\. Client → `POST /execute` with `team_id` only – FastAPI pulls JSONB, rebuilds agents in‑memory (no validation cost) and runs the conversation.

Below is the **your local PostgreSQL instance**

`Host      : http://localhost`  
`Port      : 5432`  
`Database  : agentteambuilder`  
`Username  : postgres_user`  
`Password  : postgres_pass`

Below is the **your local neo4j instance**

`Host      : http://localhost`  
`Port      : 7474`  
`Database  : neo4j`  
`Username  : neo4j`  
`Password  : pJnssz3khcLtn6T`

All other logic (Airbyte → Neo4j mirror; reuse of configs by `team_id`) is unchanged.

## **✅ 1\. PostgreSQL Setup**

### **1.1 Add Dependencies in `pyproject.toml`**

diff  
CopyEdit  
`[tool.poetry.dependencies]`  
 `fastapi = "^0.110"`  
 `uvicorn = {extras=["standard"], version = "^0.29"}`  
`-# …`  
`+psycopg[binary] = "^3.1"      # async PG driver`  
`+asyncpg = "^0.29"`  
`+alembic = "^1.13"`

---

### **1.2 Create Alembic Migration (`alembic/versions/<timestamp>_init_agent_teams.py`)**

sql  
CopyEdit  
`-- create table to store agent team config per tenant`  
`CREATE TABLE agent_teams (`  
  `team_id     UUID PRIMARY KEY,`  
  `tenant_id   TEXT NOT NULL,`  
  `config_json JSONB NOT NULL,`  
  `updated_at  TIMESTAMP DEFAULT now()`  
`);`

---

## **🛠️ 2\. Database Helper: `src/database.py`**

python  
CopyEdit  
`import os, json`  
`from uuid import UUID`  
`import psycopg_pool`  
`from src.config.schema import AgentTeamConfig`

`# Initialize an async connection pool to local Postgres`  
`_pool = psycopg_pool.AsyncConnectionPool(os.getenv(`  
    `"PG_DSN", "postgresql://postgres_user:postgres_pass@localhost:5432/agentteambuilder"`  
`))`

`async def fetch_team(team_id: UUID) -> AgentTeamConfig | None:`  
    `"""Retrieve previously saved team config JSONB and construct AgentTeamConfig."""`  
    `async with _pool.connection() as conn:`  
        `row = await conn.fetchrow(`  
            `"SELECT config_json FROM agent_teams WHERE team_id = $1",`   
            `team_id`  
        `)`  
    `return AgentTeamConfig(**row["config_json"]) if row else None`

`async def upsert_team(team_id: UUID, cfg: AgentTeamConfig):`  
    `"""Insert or update team config JSONB. Tenant ID and full config stored."""`  
    `async with _pool.connection() as conn:`  
        `await conn.execute(`  
            `"""`  
            `INSERT INTO agent_teams(team_id, tenant_id, config_json)`  
            `VALUES ($1, $2, $3)`  
            `ON CONFLICT (team_id) DO UPDATE`  
              `SET config_json = $3, updated_at = now();`  
            `""",`  
            `team_id, cfg.tenant_id, json.dumps(cfg.dict())`  
        `)`

---

## **🧭 3\. FastAPI Changes: `src/main.py`**

diff  
CopyEdit  
`from fastapi import FastAPI, HTTPException`  
`+from uuid import UUID, uuid4`  
 `from src.config.schema import AgentTeamConfig`  
 `from src.services.team_builder import TeamBuilderService`  
 `from src.services.team_executor import TeamExecutorService`  
`+from src import database  # new import for data persistence`

 `app = FastAPI(title="AI Team Builder Agent Service (with persistence)", version="1.0")`

`-@app.post("/build_and_execute")`  
`-def build_and_execute(cfg: AgentTeamConfig):`  
`+@app.post("/teams", status_code=201)`  
`+async def create_or_update_team(cfg: AgentTeamConfig,`  
`+                                team_id: UUID | None = None):`  
`+    """`  
`+    Save or update agent-team config in PostgreSQL.`  
``+    Returns `team_id` for future reference.``  
`+    """`  
`+    team_id = team_id or uuid4()`  
`+    await database.upsert_team(team_id, cfg)`  
`+    return {"team_id": str(team_id), "message": "Team config saved"}`

`+@app.post("/execute")`  
`+async def execute_team(team_id: UUID,`  
`+                       cfg: AgentTeamConfig | None = None):`  
`+    """`  
`+    Execute a team: either reuse stored config or update first.`  
`` +    - If `cfg` is None: load stored config by `team_id` ``  
``+    - If `cfg` is provided: overwrite stored config before execution.``  
`+    """`  
`+    if cfg is None:`  
`+        cfg = await database.fetch_team(team_id)`  
`+        if cfg is None:`  
`+            raise HTTPException(status_code=404, detail="team_id not found")`  
`+    else:`  
`+        await database.upsert_team(team_id, cfg)`

`+    # Build agents in memory`  
`+    team = TeamBuilderService.build_team(cfg)`

`+    # Determine conversation flow`  
`+    flow = ([agent.role for agent in team]`   
`+            if cfg.agent_team_flow is None`   
`+            else [s.strip() for s in cfg.agent_team_flow.split("->")])`

`+    execsvc = TeamExecutorService(team, flow=flow, max_turns=cfg.max_turns)`  
`+    final_answer = execsvc.run_conversation(cfg.agent_team_main_goal)`

`+    return {`  
`+        "team_id": str(team_id),`  
`+        "final_answer": final_answer,`  
`+        "conversation_log": execsvc.conversation_log`  
`+    }`

---

## **🚀 4\. Airbyte Setup for Local Credentials**

Airbyte will watch your local PostgreSQL and mirror changes to your local Neo4j. Using the credentials you provided:

* **Postgres Source**

  * host: `localhost`

  * port: `5432`

  * database: `agentteambuilder`

  * username: `postgres_user`

  * password: `postgres_pass`

  * replication method: `CDC` or `xmin`

* **Neo4j Destination**

  * host: `localhost`

  * port: `7474` (or bolt 7687 depending on connector)

  * database: `neo4j` (default)

  * username: `neo4j`

  * password: `pJnssz3khcLtn6T`

### **Airbyte bootstrapping script: `scripts/airbyte_setup.sh`**

bash  
CopyEdit  
`#!/usr/bin/env bash`  
`# Starts Airbyte and configures sync pipelines using local credentials`

`AB_URL=${AB_URL:-http://localhost:8000/api/v1}`

`# Create Postgres source`  
`SRC_ID=$(curl -s $AB_URL/sources/create -d @- <<EOF | jq -r '.sourceId'`  
`{"sourceDefinitionId":"<postgres-source-id>",`  
 `"name":"pg_local_source",`  
 `"connectionConfiguration":{`  
   `"host":"localhost","port":5432,`  
   `"database":"agentteambuilder",`  
   `"username":"postgres_user","password":"postgres_pass",`  
   `"schema":"public","replication_method":"CDC"}}`  
`EOF)`

`# Create Neo4j destination`  
`DST_ID=$(curl -s $AB_URL/destinations/create -d @- <<EOF | jq -r '.destinationId'`  
`{"destinationDefinitionId":"<neo4j-destination-id>",`  
 `"name":"neo4j_local",`  
 `"connectionConfiguration":{`  
   `"host":"localhost","port":7474,`  
   `"username":"neo4j","password":"pJnssz3khcLtn6T"}}`  
`EOF)`

`# Create connection & kick off sync`  
`CONN_ID=$(curl -s $AB_URL/connections/create -d \`  
  `"{\"sourceId\":\"$SRC_ID\",\"destinationId\":\"$DST_ID\",`  
    `\"syncCatalog\":{\"streams\":[]},\"scheduleType\":\"manual\"}" \`  
  `| jq -r '.connectionId')`

`curl -X POST $AB_URL/connections/sync -d "{\"connectionId\":\"$CONN_ID\"}"`

---

## **🔍 5\. Build-Team Caching (Optional)**

diff  
CopyEdit  
`from functools import lru_cache`

 `class TeamBuilderService:`  
`-    def build_team(cls, cfg: AgentTeamConfig):`  
`+    @classmethod`  
`+    @lru_cache(maxsize=128)`  
`+    def build_team(cls, cfg_json_str: str):`  
         `# cfg_json_str is JSON dump of config; builds same team deterministically`

* When calling: serialize config with `json.dumps(cfg.dict(), sort_keys=True)` before passing to `build_team`. This speeds up repeated in-memory builds.

---

## **🧾 Summary**

* **Postgres** runs locally with your credentials; JSONB stores the full agent‑team config per `team_id`.

* **Neo4j** mirrors that data automatically via Airbyte CDC with your local credentials.

* **FastAPI** now supports two endpoints:

  * `POST /teams` — to create/update config.

  * `POST /execute` — to run the orchestration using saved config or update if provided.

* Each code file is fully documented with inline comments, making it easier for junior developers to follow.

* **Airbyte handles data synchronization** between Postgres and Neo4j—no custom dual‑write logic required, ensuring consistency and real‑time mirroring.


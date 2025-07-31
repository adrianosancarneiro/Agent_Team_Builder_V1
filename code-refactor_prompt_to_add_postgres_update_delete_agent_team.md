## **code-refactor\_prompt\_to\_add\_postgres\_update\_delete\_agent\_team**

**Refactoring Agent Team Builder for Persistence, Versioning, and New Endpoints**

You are an expert Python architect.  
Your task is to upgrade **Agent Team Builder V1** it will **preserve all existing logic** in `team_builder.py` and `team_executor.py` while introducing a database layer and new service modules. We add endpoints for building, executing, updating (with versioning), and soft-deleting agent team configurations. Each step below outlines the changes with **old\_code \=\> recommended\_code** comparisons, followed by full updated code listings (with docstrings and inline comments) for the affected files.

Below is the **your local PostgreSQL instance**

`Host      : http://localhost`  
`Port      : 5432`  
`Database  : agentteambuilder`  
`Username  : postgres_user`  
`Password  : postgres_pass`

**Step 1: Preserve Core Team Builder/Executor Behavior (No Logic Changes)**

We ensure that the core behavior in `TeamBuilderService` and `TeamExecutorService` remains unchanged. These services will be **reused as-is** by the new endpoints to maintain the original team assembly and execution logic.

**Old Code (src/services/team\_builder.py `TeamBuilderService.build_team` excerpt):**

python  
CopyEdit  
`# old_code (TeamBuilderService.build_team)`  
`class TeamBuilderService:`  
    `@classmethod`  
    `def build_team(cls, config: AgentTeamConfig) -> list:`  
        `# ... (existing logic assembling Agent instances from config)`  
        `team = [Agent(AgentDefinition(**agent_def)) for agent_def in agents_config]`  
        `return team`

**Recommended Code:** *No changes to logic.* We will continue calling `TeamBuilderService.build_team(config)` to construct the agent team. The same applies for `TeamExecutorService.run_conversation` from `team_executor.py`. Any new functionality (like saving to DB) will wrap around these calls rather than altering them.

## **Step 2: Introduce Database Models and Migration Scripts**

To support persistence and versioning, we add a **SQLAlchemy ORM model** for the agent team configuration and a separate model for config versions. We also prepare an **Alembic migration** to add the new fields and table:

* **`AgentTeam` model:** Represents the team configuration record (with `id`, `config` JSON, timestamps, and new `is_deleted` & `deleted_at` fields for soft deletion).

* **`AgentTeamConfigVersion` model:** Tracks each version of a team's config (with `team_id` foreign key, `version` number, config JSON, and timestamp).

**Alembic Migration – Add `is_deleted`, `deleted_at` and Versioning Table:**

python  
CopyEdit  
`# recommended_code (Alembic revision script: e.g., 20250730_add_versioning.py)`  
`from alembic import op`  
`import sqlalchemy as sa`

`# (Assuming a previous migration created agent_teams table with at least 'id' and 'config')`  
`def upgrade():`  
    `# Add soft-delete columns to agent_teams`  
    `op.add_column('agent_teams', sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='0'))`  
    `op.add_column('agent_teams', sa.Column('deleted_at', sa.DateTime(), nullable=True))`  
    `# Create new table for storing config versions`  
    `op.create_table('agent_team_config_versions',`  
        `sa.Column('id', sa.Integer(), primary_key=True),`  
        `sa.Column('team_id', sa.Integer(), sa.ForeignKey('agent_teams.id'), nullable=False),`  
        `sa.Column('version', sa.Integer(), nullable=False),`  
        `sa.Column('config', sa.Text(), nullable=False),`  
        `sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False)`  
    `)`  
    `op.create_index('ix_agent_team_config_versions_team_id', 'agent_team_config_versions', ['team_id'])`

`def downgrade():`  
    `op.drop_index('ix_agent_team_config_versions_team_id', table_name='agent_team_config_versions')`  
    `op.drop_table('agent_team_config_versions')`  
    `op.drop_column('agent_teams', 'deleted_at')`  
    `op.drop_column('agent_teams', 'is_deleted')`

This migration modifies the `agent_teams` table to include `is_deleted` (boolean) and `deleted_at` (timestamp), and creates a new `agent_team_config_versions` table for version history.

**SQLAlchemy ORM Models (e.g., in `src/config/db_models.py`):**

python  
CopyEdit  
`# recommended_code (Database models definition)`  
`from sqlalchemy import Column, Integer, Boolean, DateTime, Text, ForeignKey`  
`from sqlalchemy.ext.declarative import declarative_base`  
`from datetime import datetime`

`Base = declarative_base()`

`class AgentTeam(Base):`  
    `__tablename__ = "agent_teams"`  
    `id = Column(Integer, primary_key=True, index=True)`  
    `config = Column(Text, nullable=False)           # Store full JSON of AgentTeamConfig`  
    `is_deleted = Column(Boolean, default=False)     # Soft-delete flag`  
    `deleted_at = Column(DateTime, nullable=True)    # Timestamp of deletion`  
    `created_at = Column(DateTime, default=datetime.utcnow)`  
    `updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)`

`class AgentTeamConfigVersion(Base):`  
    `__tablename__ = "agent_team_config_versions"`  
    `id = Column(Integer, primary_key=True)`  
    `team_id = Column(Integer, ForeignKey("agent_teams.id"), nullable=False)`  
    `version = Column(Integer, nullable=False)`  
    `config = Column(Text, nullable=False)           # JSON of the config for this version`  
    `created_at = Column(DateTime, default=datetime.utcnow)`

**Note:** In practice, you would also configure a database `engine` and session (e.g., using `create_engine` and `sessionmaker`). For brevity, assume we have a SQLAlchemy session factory `SessionLocal` available to use in our service code. The configuration string for the engine (database URL) would come from environment or config (e.g., `DATABASE_URL` env variable), and Alembic would be set up to use these models for migrations.

## **Step 3: Add New Service Modules for Update and Delete Operations**

We create two new service classes, `TeamUpdaterService` and `TeamDeleterService`, in the `src/services` directory. These will encapsulate the logic for updating an existing team configuration (with versioning) and soft-deleting a team, respectively. We **carefully reuse** parts of `TeamBuilderService` logic (like validation) without altering their behavior.

### **3.1 TeamUpdaterService (`src/services/team_updater.py`)**

**Old Code:** *No dedicated updater existed.* Previously, updating a team would have required rebuilding or was not supported.

**Recommended Code (new file `team_updater.py`):**

python  
CopyEdit  
`# recommended_code (File: /home/mentorius/AI_Services/Agent_Team_Builder_V1/src/services/team_updater.py)`  
`from datetime import datetime`  
`from sqlalchemy.orm import Session`  
`from src.config.schema import AgentTeamConfig`  
`from src.services.team_builder import TeamBuilderService`  
`from src.config.db_models import AgentTeam, AgentTeamConfigVersion`

`class TeamUpdaterService:`  
    `"""Service responsible for updating an existing agent team configuration with versioning."""`  
      
    `@staticmethod`  
    `def update_team(session: Session, team_id: int, new_config: AgentTeamConfig) -> dict:`  
        `"""`  
        `Update the stored team configuration for the given team_id.`  
        `- Validates new_config via TeamBuilderService.`  
        `- Increments version and stores new config in AgentTeamConfigVersion.`  
        `- Updates the AgentTeam record's config to the new version.`  
        `Returns a summary of the update (including new version number).`  
        `"""`  
        `# 1. Retrieve existing team record`  
        `team = session.query(AgentTeam).filter(AgentTeam.id == team_id, AgentTeam.is_deleted == False).first()`  
        `if not team:`  
            `raise ValueError(f"Team ID {team_id} not found or is deleted")`  
          
        `# 2. Validate new configuration by attempting to build the team (does not alter DB)`  
        `TeamBuilderService.build_team(new_config)  # if invalid, this will raise an exception`  
          
        `# 3. Determine next version number`  
        `last_version = session.query(AgentTeamConfigVersion) \`  
                              `.filter(AgentTeamConfigVersion.team_id == team_id) \`  
                              `.order_by(AgentTeamConfigVersion.version.desc()) \`  
                              `.first()`  
        `new_version_number = 1 if last_version is None else last_version.version + 1`  
          
        `# 4. Store the new config in the versions history`  
        `version_entry = AgentTeamConfigVersion(team_id=team_id, version=new_version_number,`   
                                               `config=new_config.json())`  
        `session.add(version_entry)`  
          
        `# 5. Update the main AgentTeam record to the new config (and updated_at timestamp auto-updates)`  
        `team.config = new_config.json()`  
        `session.commit()`  
          
        `return {`  
            `"team_id": team_id,`  
            `"new_version": new_version_number,`  
            `"updated_at": datetime.utcnow().isoformat()`  
        `}`  
      
    `@staticmethod`  
    `def restore_version(session: Session, team_id: int, target_version: int) -> dict:`  
        `"""`  
        `Rollback the team configuration to a previous version.`  
        `Creates a new version entry identical to the target version and sets it as current.`  
        `Returns details of the restoration.`  
        `"""`  
        `team = session.query(AgentTeam).filter(AgentTeam.id == team_id, AgentTeam.is_deleted == False).first()`  
        `if not team:`  
            `raise ValueError(f"Team ID {team_id} not found or is deleted")`  
        `# Find the target version entry`  
        `version_entry = session.query(AgentTeamConfigVersion) \`  
                               `.filter(AgentTeamConfigVersion.team_id == team_id, AgentTeamConfigVersion.version == target_version) \`  
                               `.first()`  
        `if not version_entry:`  
            `raise ValueError(f"Version {target_version} not found for Team ID {team_id}")`  
          
        `# Create a new version as a copy of target_version`  
        `last_version = session.query(AgentTeamConfigVersion) \`  
                              `.filter(AgentTeamConfigVersion.team_id == team_id) \`  
                              `.order_by(AgentTeamConfigVersion.version.desc()) \`  
                              `.first()`  
        `new_version_number = 1 if last_version is None else last_version.version + 1`  
        `new_config_json = version_entry.config  # JSON string of the old config`  
          
        `new_version_entry = AgentTeamConfigVersion(team_id=team_id, version=new_version_number, config=new_config_json)`  
        `session.add(new_version_entry)`  
          
        `# Update AgentTeam record to use the restored config`  
        `team.config = new_config_json`  
        `team.updated_at = datetime.utcnow()`  
        `session.commit()`  
          
        `return {`  
            `"team_id": team_id,`  
            `"restored_version": target_version,`  
            `"new_version": new_version_number,`  
            `"restored_at": datetime.utcnow().isoformat()`  
        `}`

Key points in `TeamUpdaterService`:

* It uses a **database session** (assumed to be provided per request) to query and update records.

* Before applying an update, it calls `TeamBuilderService.build_team(new_config)` to reuse the validation and assembly logic. This ensures the new configuration can actually produce a valid team of agents (maintaining original behavior).

* The `update_team` method then inserts a new version row and updates the main record’s `config`.

* The `restore_version` method implements **rollback**: given a `team_id` and a `target_version`, it finds that version’s config and creates a new version entry (with an incremented version number) making it the current config in `AgentTeam`. This approach preserves the history (the rollback itself becomes a new version entry).

### **3.2 TeamDeleterService (`src/services/team_deleter.py`)**

**Old Code:** *No dedicated deleter; deletion not previously supported.*

**Recommended Code (new file `team_deleter.py`):**

python  
CopyEdit  
`# recommended_code (File: /home/mentorius/AI_Services/Agent_Team_Builder_V1/src/services/team_deleter.py)`  
`from datetime import datetime`  
`from sqlalchemy.orm import Session`  
`from src.config.db_models import AgentTeam`

`class TeamDeleterService:`  
    `"""Service responsible for soft-deleting (archiving) an agent team configuration."""`  
      
    `@staticmethod`  
    `def delete_team(session: Session, team_id: int) -> None:`  
        `"""`  
        `Soft delete the team by setting is_deleted True and recording deleted_at timestamp.`  
        `Raises ValueError if team not found.`  
        `"""`  
        `team = session.query(AgentTeam).filter(AgentTeam.id == team_id, AgentTeam.is_deleted == False).first()`  
        `if not team:`  
            `raise ValueError(f"Team ID {team_id} not found or already deleted")`  
        `team.is_deleted = True`  
        `team.deleted_at = datetime.utcnow()`  
        `session.commit()`

`TeamDeleterService.delete_team` simply flags a team as deleted. We do not remove any records, thus **preserving history** and enabling potential future restoration if needed. (Re-enabling a deleted team is not explicitly required, but the data remains in the DB.)

## **Step 4: Implement New FastAPI Endpoints in the Router (src/main.py)**

Next, we add four API endpoints to `src/main.py` (or the appropriate router). These endpoints utilize the above services:

* **POST `/build`** – Create a new team config: validate & build the team, then save it to the database.

* **POST `/execute`** – Execute a stored team by ID: load config from DB, build agent instances, run the conversation.

* **POST `/update_team`** – Update an existing team’s config: accept a new config, update DB (with versioning).

* **DELETE `/delete_team`** – Soft-delete a team by ID.

Each endpoint uses a database session (e.g., via FastAPI dependency injection or a simple SessionLocal instance) and returns JSON responses. We include error handling for missing teams or invalid configs.

**Old Code (excerpt from old `main.py` for build\_and\_execute and build\_team):**

python  
CopyEdit  
`# old_code (src/main.py excerpt for building team)`  
`@app.post("/build_and_execute")`  
`def build_and_execute(config: AgentTeamConfig):`  
    `# ... build team and run conversation, returns agent_team and final_answer`

`@app.post("/build_team")`  
`def build_team_endpoint(config: AgentTeamConfig):`  
    `# ... build team and return composition`

In the new design, these combined behaviors are split and expanded. We'll introduce separate endpoints and integrate the database:

**Recommended Code (modified `src/main.py` with new endpoints):**

python  
CopyEdit  
`# recommended_code (File: /home/mentorius/AI_Services/Agent_Team_Builder_V1/src/main.py)`  
`from fastapi import FastAPI, HTTPException, Depends`  
`from sqlalchemy.orm import Session`  
`from src.config.schema import AgentTeamConfig`  
`from src.services.team_builder import TeamBuilderService`  
`from src.services.team_executor import TeamExecutorService`  
`from src.services.team_updater import TeamUpdaterService`  
`from src.services.team_deleter import TeamDeleterService`  
`from src.config.db_models import AgentTeam, AgentTeamConfigVersion`  
`from src.config.database import SessionLocal  # assume SessionLocal is our session factory`

`app = FastAPI(title="AI Team Builder Agent Service", version="2.0")`

`# Dependency to get DB session (FastAPI will call this for each request)`  
`def get_db():`  
    `db = SessionLocal()`  
    `try:`  
        `yield db`  
    `finally:`  
        `db.close()`

`@app.post("/build")`  
`def build_team(config: AgentTeamConfig, db: Session = Depends(get_db)):`  
    `"""`  
    `Create a new agent team configuration.`  
    `- Validates and builds the team (ensuring config is correct).`  
    `- Persists the team config and initial version to the database.`  
    `Returns the new team ID and a summary of the team.`  
    `"""`  
    `# Validate and build team (does not persist, just to ensure no errors)`  
    `try:`  
        `team_agents = TeamBuilderService.build_team(config)`  
    `except Exception as e:`  
        `raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")`  
    `# Persist the new team config`  
    `new_team = AgentTeam(config=config.json())`  
    `db.add(new_team)`  
    `db.flush()  # get new_team.id assigned`  
    `# Create initial version entry (version 1)`  
    `version_entry = AgentTeamConfigVersion(team_id=new_team.id, version=1, config=config.json())`  
    `db.add(version_entry)`  
    `db.commit()`  
    `# Prepare response: team summary (from built agents) and assigned team_id`  
    `team_summary = [agent.summarize() for agent in team_agents]`  
    `return {"team_id": new_team.id, "agent_team": team_summary}`

`@app.post("/execute")`  
`def execute_team(team_id: int, db: Session = Depends(get_db)):`  
    `"""`  
    `Execute a saved agent team by team_id.`  
    `Loads the latest configuration from the database and runs the multi-agent conversation.`  
    `Returns the team composition, conversation log, and final answer.`  
    `"""`  
    `# Retrieve team config`  
    `team_record = db.query(AgentTeam).filter(AgentTeam.id == team_id, AgentTeam.is_deleted == False).first()`  
    `if not team_record:`  
        `raise HTTPException(status_code=404, detail=f"Team ID {team_id} not found or is deleted")`  
    `# Parse stored config JSON back into AgentTeamConfig model`  
    `try:`  
        `config = AgentTeamConfig.parse_raw(team_record.config)`  
    `except Exception as e:`  
        `raise HTTPException(status_code=500, detail=f"Stored configuration is invalid: {e}")`  
    `# Re-build agent instances from config and execute conversation`  
    `agents = TeamBuilderService.build_team(config)`  
    `executor = TeamExecutorService(agents=agents,`   
                                   `flow=[agent.role for agent in agents] if not config.agent_team_flow else`   
                                        `[role.strip() for role in config.agent_team_flow.split("->")],`  
                                   `max_turns=config.max_turns)`  
    `final_answer = executor.run_conversation(user_query=config.agent_team_main_goal)`  
    `return {`  
        `"agent_team": [agent.summarize() for agent in agents],`  
        `"conversation_log": executor.conversation_log,`  
        `"final_answer": final_answer`  
    `}`

`@app.post("/update_team")`  
`def update_team_endpoint(team_id: int, config: AgentTeamConfig, db: Session = Depends(get_db)):`  
    `"""`  
    `Update an existing team's configuration (creates a new version).`  
    `Returns the new version number and timestamps.`  
    `"""`  
    `try:`  
        `result = TeamUpdaterService.update_team(db, team_id, config)`  
    `except ValueError as e:`  
        `raise HTTPException(status_code=404, detail=str(e))`  
    `except Exception as e:`  
        `# Catch validation errors from TeamBuilderService or others`  
        `raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")`  
    `return {"team_id": team_id, "new_version": result["new_version"], "updated_at": result["updated_at"]}`

`@app.delete("/delete_team")`  
`def delete_team_endpoint(team_id: int, db: Session = Depends(get_db)):`  
    `"""`  
    `Soft-delete the specified team configuration.`  
    `Marks the team as deleted without removing data.`  
    `"""`  
    `try:`  
        `TeamDeleterService.delete_team(db, team_id)`  
    `except ValueError as e:`  
        `raise HTTPException(status_code=404, detail=str(e))`  
    `return {"team_id": team_id, "deleted": True}`

**Explanation of the new endpoints:**

* **`/build`:** We create a new `AgentTeam` record. Before committing, we use `TeamBuilderService.build_team` to validate that the provided `AgentTeamConfig` is valid and to generate the list of Agent objects (for returning a summary). We then save the config JSON in the DB and create an initial version entry (`version=1`). The response includes the new `team_id` and the agent team summary (roles, names, etc.) for confirmation.

* **`/execute`:** This endpoint fetches the stored config by `team_id`. We parse the JSON into an `AgentTeamConfig` Pydantic object (leveraging `parse_raw` for convenience). Then we rebuild the agents using `TeamBuilderService.build_team(config)` and run the conversation via `TeamExecutorService`. We return the team summary, the full conversation log, and the final answer. This mirrors the old `/build_and_execute` behavior, but now operates on a persisted config. We ensure to check `is_deleted` – if the team was soft-deleted or not found, we return 404\.

* **`/update_team`:** Accepts a new config for an existing team. We call `TeamUpdaterService.update_team`, which validates the config and handles the database updates (inserting a new version and updating the main record). We handle exceptions: if the team ID is not found, or if validation fails, appropriate HTTP errors are raised. On success, we return the new version number and timestamp.

* **`/delete_team`:** Calls `TeamDeleterService.delete_team` to soft-delete the record. If the team is not found (or already deleted), a 404 is returned. Otherwise, it returns a confirmation that the team was marked deleted.

All new endpoints use **dependency injection** (`db: Session = Depends(get_db)`) to obtain a SQLAlchemy session for the request. After each operation, changes are committed to the database.

## **Step 5: Implement Configuration Rollback Capability**

To fulfill the rollback feature, we use the `TeamUpdaterService.restore_version` method introduced in Step 3.1. Although we did not expose a separate API endpoint in the instructions, this method can be invoked internally or via an extension of the update endpoint to restore a previous version.

For example, we could add an optional query parameter or a separate route to trigger rollback. One approach is shown below (not strictly required by the prompt but demonstrates usage):

python  
CopyEdit  
`# recommended_code (Additional endpoint for rollback, if needed)`  
`@app.post("/rollback_team")`  
`def rollback_team_endpoint(team_id: int, target_version: int, db: Session = Depends(get_db)):`  
    `"""`  
    `Roll back the team configuration to a specified previous version.`  
    `Creates a new version identical to the target and makes it current.`  
    `"""`  
    `try:`  
        `result = TeamUpdaterService.restore_version(db, team_id, target_version)`  
    `except ValueError as e:`  
        `raise HTTPException(status_code=404, detail=str(e))`  
    `return {`  
        `"team_id": team_id,`   
        `"restored_version": target_version,`   
        `"new_version": result["new_version"],`   
        `"restored_at": result["restored_at"]`  
    `}`

In this snippet, `restore_version` looks up the old config and applies it as a new version. After calling this, the team’s current config is the same as an earlier state. This ensures even rollbacks are tracked as a forward-moving version history (no data loss or overwriting historical records).

*(If a dedicated rollback endpoint is not desired, similar logic could be integrated into `/update_team` by checking for a special field in the request to indicate a rollback to a given version.)*

## **Step 6: Summary and Verification of Unchanged Behavior**

* **Original Behavior Maintained:** The core agent team assembly (`build_team`) and execution (`run_conversation`) logic is unchanged. We have only **refactored how they are used**, not how they work internally. By reusing these services, we ensure that all existing functionality (such as default agent roles, tool assignment, conversation flow, etc.) remains consistent with V1[github.com](https://github.com/adrianosancarneiro/Agent_Team_Builder_V1#:~:text=class%20Agent%3A%20,LLM)[github.com](https://github.com/adrianosancarneiro/Agent_Team_Builder_V1#:~:text=If%20the%20agent%20has%20any,search%28query_emb%2C%20top_k%3D1).

* **Isolation of New Logic:** The new `TeamUpdaterService` and `TeamDeleterService` handle their respective concerns without modifying the builder/executor. They interact with the database and use the builder for validation, thus **avoiding code duplication**. Any logic extracted for reuse (in our case, validation via `TeamBuilderService`) was leveraged through method calls rather than rewriting it.

* **Database Integration:** We introduced SQLAlchemy models and Alembic migrations to persist configurations. The code examples show how the team config (as JSON) is saved and versioned. By storing full config JSON, we keep the schema flexible – the Pydantic model ensures validity on input/output.

* **Testing the Flow:** After migration, a typical flow would be:

  1. **Build a team**: `POST /build` with a JSON body (AgentTeamConfig) → returns `team_id` and team summary.

  2. **Execute the team**: `POST /execute` with that `team_id` → returns conversation result.

  3. **Update the team**: `POST /update_team` with `team_id` and a new JSON config → returns new version number.

  4. **(Optional) Rollback**: `POST /rollback_team` with `team_id` and a previous version number → returns confirmation of restoration.

  5. **Delete the team**: `DELETE /delete_team` with `team_id` → marks the team as deleted.

Throughout these steps, the system logs and behaviors from V1 are preserved, but we now have a robust persistence layer that supports version tracking and soft deletion for better maintainability. The code is organized into clear service modules, aligning with the original design principle of separation of concerns, and each function is documented for clarity.


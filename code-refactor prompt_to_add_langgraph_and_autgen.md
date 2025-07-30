## **code-refactor prompt\_to\_add\_langgraph\_and\_autgen**

You are an expert Python architect.  
 Your task is to migrate the repo **`Agent_Team_Builder_V1`** (folder layout shown below) from its current bespoke orchestration loop to a **LangGraph ▶ AutoGen** architecture, without breaking the public FastAPI contract.

arduino  
CopyEdit  
`Agent_Team_Builder_V1/`  
`├─ src/`  
`│  ├─ main.py`  
`│  ├─ config/`  
`│  ├─ services/`  
`│  │  ├─ team_builder.py`  
`│  │  └─ team_executor.py      # ← custom while-loop TODAY`  
`│  └─ tools/`  
`├─ scripts/`  
`└─ pyproject.toml`

### **1\. Design targets**

| Layer | Old (Phase 1\) | New (Phase 2—what you must implement) |
| ----- | ----- | ----- |
| **Graph orchestration** | `TeamExecutorService.run_conversation()` while-loop | **LangGraph `StateGraph`** with persistence & checkpoints |
| **Agent implementation** | Plain `Agent` classes with `chat()` | **AutoGen `ConversableAgent`** subclasses |
| **Prompting approach** | Hand-written role prompts | **ReAct / tool-augmented prompts** \+ system messages |
| **Tool routing** | Custom python methods | AutoGen tool-calling interface |
| **Testing** | Minimal unit tests | *Full* pytest suite incl. API, graph, tool calls |

### **2\. Required code changes (high level)**

**Dependencies** – Add

 toml  
CopyEdit  
`langgraph>=0.0.32`  
`pyautogen>=0.2.25`

1.  to **`pyproject.toml`**, then run `uv sync --no-dev`.

2. **New module** → `src/graphs/tenant_team_graph.py`

   * Build a `StateGraph` whose **nodes** are the AutoGen agents returned by the Team Builder and whose **edges** encode the turn-taking order specified by `AgentTeamConfig.agent_team_flow`.

   * Support cycles, `max_turns`, and a “FINAL\_ANSWER” stop signal.

3. **Refactor `team_builder.py`**

   * Instead of returning plain `Agent` instances, return concrete **AutoGen `ConversableAgent`** objects.

   * Each `AgentDefinition` in the JSON config becomes one AutoGen subclass with:

     * a role prompt (system)

     * ReAct-style instructions for tool usage

     * a list of allowed tools (vector search, Neo4j query, web search, etc.)

   * Add an optional **human-approval gate** node if `should_TBA_ask_caller_approval` is `true`.

4. **Refactor `team_executor.py`**

   * Delete the while-loop.

   * Accept a LangGraph graph object and call `graph.invoke(initial_state)` to run the conversation.

   * Gather `conversation_log` from LangGraph callbacks.

5. **API adjustments (`main.py`)**

   * Keep endpoint signatures **unchanged**.

   * Internally call the new builder → graph runner workflow.

6. **Scripts** – If needed, update `scripts/run_app.sh` to run database migrations before starting the server.

### **3\. Best-practice constraints**

* **Context-window management** – Use AutoGen’s streaming mode and LangGraph’s checkpointing so agents can resume without re-sending the full transcript.

**Tool calls** – Implement ReAct-style messages:

 json  
CopyEdit  
`{`  
  `"thought": "...",`  
  `"tool": "Search_Vector_DB",`  
  `"tool_input": {"query": "..."}`  
`}`

*  and inject the tool result into the next agent turn.

* **Persistence** – Enable LangGraph’s SQLite checkpoint store (`.langgraph/`) so partial runs survive restarts.

* **Logging** – Continue writing JSONL logs to `logs/`, but now source them from LangGraph’s streaming callbacks.

### **4\. Testing requirements**

Create / update **pytest** tests so CI passes:

| Test file | Purpose |
| ----- | ----- |
| `tests/test_team_builder.py` | Builds agents from a sample `AgentTeamConfig` and asserts each is an AutoGen `ConversableAgent` with right tools. |
| `tests/test_graph_execution.py` | Instantiates the graph with a fake user query and verifies it produces a `"final_answer"` key within `max_turns`. |
| `tests/test_api_http.py` | Uses `httpx.AsyncClient` to call `/build_and_execute`; checks **200 OK**, expected JSON schema, and non-empty `final_answer`. |
| `tests/test_tool_calls.py` | Mocks Qdrant & Neo4j; ensures a Retriever agent calls the right tool when prompted. |

Run everything with:

bash  
CopyEdit  
`pytest -q`

### **5\. Deliverables & format**

1. **Git-style diffs** for each modified / added file.

2. A **brief commit message** summarizing the migration.

3. All tests green (`pytest -q` returns exit code 0).

*Do not* include unrelated files or formatting-only changes.

### **6\. Handoff checklist for you, the LLM**

* Modify `pyproject.toml` dependencies.

* Add `src/graphs/tenant_team_graph.py`.

* Refactor `team_builder.py` to create AutoGen agents.

* Refactor `team_executor.py` to use LangGraph.

* Keep FastAPI endpoints unchanged externally.

* Update / write pytest tests.

* Ensure `scripts/*` still work.

* Output diffs \+ commit message.

When finished, **test everything** with `pytest -q` to ensure all tests pass add additional tests if needed.




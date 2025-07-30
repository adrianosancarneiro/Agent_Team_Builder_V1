from fastapi import FastAPI, HTTPException
from src.config.schema import AgentTeamConfig
from src.services.team_builder import TeamBuilderService
from src.services.team_executor import TeamExecutorService

app = FastAPI(title="AI Team Builder Agent Service", version="1.0")

# Build the team and optionally run the conversation in one go (for simplicity).
@app.post("/build_and_execute")
def build_and_execute(config: AgentTeamConfig):
    """
    Build an AI agent team according to the provided configuration and run the multi-agent conversation.
    Returns the team composition and the final answer.
    """
    # Build the team of agents
    try:
        team = TeamBuilderService.build_team(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")
    # Log or store team info
    team_summary = [agent.summarize() for agent in team.values()]

    # Execute the conversation workflow
    flow = None
    if config.agent_team_flow:
        flow = [s.strip() for s in config.agent_team_flow.split("->")]
    
    executor = TeamExecutorService(
        agents=team, 
        flow=flow,
        max_turns=config.max_turns)
    final_answer = executor.run_conversation(user_query=config.agent_team_main_goal)

    # Return both the team details and the final answer
    return {
        "agent_team": team_summary,
        "conversation_log": executor.conversation_log,
        "final_answer": final_answer,
        "thread_id": executor.thread_id  # Include thread_id for conversation resumption
    }

    # (Optional) Separate endpoint to just build team without execution
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
    return {"agent_team": [agent.summarize() for agent in team.values()]}# (Optional) Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

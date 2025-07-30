from typing import List, Tuple
from src.services.team_builder import Agent

class TeamExecutorService:
    """Service to manage multi-agent conversation execution."""
    def __init__(self, agents: List[Agent], flow: List[str] = None, max_turns: int = 5):
        """
        Initialize with a list of Agent instances. Optionally provide a conversation flow order 
        (list of agent names or roles in speaking order). If no flow is given, a default order (round-robin) is used.
        """
        self.agents = agents
        # Create a mapping of agent role->agent and name->agent for convenience
        self.agents_by_name = {agent.name: agent for agent in agents}
        self.agents_by_role = {agent.role: agent for agent in agents}
        # Determine speaking order
        if flow:
            # If flow is provided as a list of names/roles, convert to actual agent instances
            self.flow = []
            for identifier in flow:
                agent = self.agents_by_name.get(identifier) or self.agents_by_role.get(identifier)
                if agent:
                    self.flow.append(agent)
        else:
            # default to the order given or round-robin
            self.flow = agents
        self.max_turns = max_turns
        self.conversation_log: List[Tuple[str, str]] = []  # list of (agent_name, message)

    def run_conversation(self, user_query: str) -> str:
        """
        Execute the multi-agent conversation until max_turns or a stopping condition is met.
        Returns the final answer (or combined result) from the team.
        """
        # Initial user input
        self.conversation_log.append(("User", user_query))
        current_turn = 0
        final_answer = ""
        # Simple loop through agents in the defined flow
        while current_turn < self.max_turns:
            for agent in self.flow:
                # Each agent takes the last message and responds
                last_speaker, last_message = self.conversation_log[-1]
                # Determine if conversation should end (if last speaker was an agent and decided to stop)
                if last_speaker != "User" and agent.role == "DecisionMaker":
                    # (Example heuristic) DecisionMaker can decide to finish the conversation
                    if "conclude" in last_message.lower():
                        final_answer = last_message
                        return final_answer
                # Agent formulates a response (Here we'd call the LLM with prompt and context. We'll simulate.)
                response = self._agent_respond(agent, last_message)
                # Log the agent's response
                self.conversation_log.append((agent.name, response))
                # Optionally, check if this agent is an orchestrator or decision-maker concluding the chat
                if agent.role.lower() in ("decisionmaker", "orchestrator"):
                    if "final answer:" in response.lower() or "conclude" in response.lower():
                        final_answer = response
                        return final_answer
            current_turn += 1
        # If loop completes without early return, take the last agent's message as final answer
        final_answer = self.conversation_log[-1][1]
        return final_answer

    def _agent_respond(self, agent: Agent, last_message: str) -> str:
        """
        Simulate an agent responding to the last_message. In reality, this would involve the agent's prompt,
        persona, tools, and an LLM call. Here, we'll do a simple placeholder implementation.
        """
        # If the agent has any tools, maybe use one (for demo, use first applicable tool to augment response)
        tool_augmented_info = ""
        for tool in agent.tools:
            tool_name = tool.__class__.__name__
            if tool_name == "QdrantTool":
                # Example: use embedding service to embed query, then search vector DB
                from src.tools.embed_tool import EmbeddingService
                embed_service = EmbeddingService()
                query_emb = embed_service.embed(last_message)
                results = tool.search(query_emb, top_k=1)
                if results:
                    tool_augmented_info += " [Found relevant info via vector DB]"
            elif tool_name == "GraphDBTool":
                # Example: query graph DB for a fact (here we just do a dummy query or skip)
                # In real case, we might have a specific query pattern
                results = tool.query("MATCH (n) RETURN n LIMIT 1")
                if results:
                    tool_augmented_info += " [Knowledge graph checked]"
            elif tool_name == "WebAPITool":
                # Example: perform a web API call (not doing actual call in demo)
                tool_augmented_info += " [Called external API]"
            # (Additional tool handling as needed)
        # Formulate a dummy response using the agent's role and possibly augmented info
        response = f"{agent.name} ({agent.role}) says: Based on '{last_message}', I {agent.personality.lower()} respond with an answer.{tool_augmented_info}"
        return response

-- PostgreSQL trigger function and triggers to auto-refresh AgentTeam.config_jsonb
-- whenever related Agent or AgentTool entities change.
CREATE OR REPLACE FUNCTION refresh_agent_team_config()
RETURNS TRIGGER AS $$
DECLARE
    team_uuid UUID;
BEGIN
    -- Determine the affected team UUID based on triggering table and operation
    IF TG_TABLE_NAME = 'agent_tool' THEN
        IF (TG_OP = 'DELETE') THEN
            SELECT agent_team_id INTO team_uuid FROM agent WHERE id = OLD.agent_id;
        ELSE
            SELECT agent_team_id INTO team_uuid FROM agent WHERE id = NEW.agent_id;
        END IF;
    ELSIF TG_TABLE_NAME = 'agent' THEN
        IF (TG_OP = 'DELETE') THEN
            team_uuid := OLD.agent_team_id;
        ELSE
            team_uuid := NEW.agent_team_id;
        END IF;
    END IF;

    IF team_uuid IS NULL THEN
        RETURN NULL;  -- No team to refresh
    END IF;

    -- Update the config_jsonb of the corresponding AgentTeam by rebuilding the "agents" list
    UPDATE agent_team AS at
    SET config_jsonb = jsonb_set(
        at.config_jsonb,
        '{agents}',
        COALESCE(
            (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'agent_role', a.role,
                        'agent_name', a.name,
                        'agent_personality', a.personality,
                        'agent_goal_based_prompt', a.goal_based_prompt,
                        'LLM_model', a.llm_model,
                        'allow_team_builder_to_override_model', a.allow_override,
                        'LLM_configuration', a.llm_config,
                        'agent_tools', COALESCE(
                            (SELECT jsonb_agg(t.tool_name)
                             FROM agent_tool t
                             WHERE t.agent_id = a.id),
                            '[]'::jsonb
                        )
                    )
                )
                FROM agent a
                WHERE a.agent_team_id = team_uuid
            ),
            '[]'::jsonb
        ),
        true
    )
    WHERE at.id = team_uuid;

    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger on Agent table (after insert/update/delete)
CREATE TRIGGER trg_refresh_team_config_on_agent
AFTER INSERT OR UPDATE OR DELETE ON agent
FOR EACH ROW
EXECUTE FUNCTION refresh_agent_team_config();

-- Trigger on AgentTool table (after insert/update/delete)
CREATE TRIGGER trg_refresh_team_config_on_agent_tool
AFTER INSERT OR UPDATE OR DELETE ON agent_tool
FOR EACH ROW
EXECUTE FUNCTION refresh_agent_team_config();


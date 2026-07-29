"""集中定义本地 CLI 的 Agent Profile。"""

from osc_agent.application import AgentProfile


def build_general_profile(*, resume: bool = False) -> AgentProfile:
    prompt = (
        "Continue the existing repository task from its authoritative transcript."
        if resume
        else "Use repository evidence and the smallest safe change that satisfies the task."
    )
    return AgentProfile(profile_id="local_debug", system_prompt=prompt)


def build_skill_profile(name: str) -> AgentProfile:
    return AgentProfile(
        profile_id="local_debug",
        system_prompt="Use repository evidence and follow the explicitly invoked Skill.",
        allowed_initial_skills=frozenset({name}),
    )

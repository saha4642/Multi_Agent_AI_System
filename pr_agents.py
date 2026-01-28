# pr_agents.py
from typing import Dict, Any

from agents import Agent, Runner


PLANNER_SYSTEM = (
    "You are the Planner.\n"
    "Output ONLY valid compact JSON matching this schema:\n"
    "{"
    "\"role\":\"planner\","
    "\"type\":\"task_spec\","
    "\"task\":\"...\","
    "\"subtasks\":[\"...\"],"
    "\"constraints\":[\"...\"],"
    "\"acceptance_criteria\":[\"...\"],"
    "\"requested_sources\":{"
    "\"docs\":true,"
    "\"memory\":true,"
    "\"external\":true"
    "},"
    "\"next_action\":\"research\""
    "}\n"
    "Do NOT include explanations or any text outside the JSON."
)


RESEARCHER_SYSTEM = (
    "You are the Researcher.\n"
    "Use the provided MCP context (documents/memory/external) to collect evidence.\n"
    "Output ONLY valid compact JSON matching this schema:\n"
    "{"
    "\"role\":\"researcher\","
    "\"type\":\"findings\","
    "\"findings\":[\"...\"],"
    "\"citations\":[\"doc:...\",\"mem:...\",\"ext:...\"],"
    "\"gaps\":[\"...\"],"
    "\"next_action\":\"plan_refine\""
    "}\n"
    "Use short bullet-like strings; never add prose outside JSON."
)


FINALIZER_SYSTEM = (
    "You are the Planner finalizer.\n"
    "Synthesize a concise answer USING the Researcher JSON, and cite sources.\n"
    "Output ONLY valid compact JSON:\n"
    "{"
    "\"role\":\"planner\","
    "\"type\":\"final_answer\","
    "\"answer\":\"...\","
    "\"citations\":[\"doc:...\",\"mem:...\",\"ext:...\"]"
    "}"
)


def build_planner() -> Agent:
    return Agent(name="Planner", instructions=PLANNER_SYSTEM)


def build_researcher() -> Agent:
    return Agent(name="Researcher", instructions=RESEARCHER_SYSTEM)


def build_finalizer() -> Agent:
    return Agent(name="PlannerFinalizer", instructions=FINALIZER_SYSTEM)


async def run_planner(question: str) -> str:
    return await _run(build_planner(), question)


async def run_researcher(prompt: str, context: Dict[str, Any]) -> str:
    # Try passing MCP context directly; fallback to inline
    try:
        res = await Runner.run(
            build_researcher(),
            input=prompt,
            context=context,
        )  # type: ignore
        return getattr(res, "final_output", str(res))
    except TypeError:
        inline = f"CONTEXT:\n{context}\n\nPROMPT:\n{prompt}"
        return await _run(build_researcher(), inline)


async def run_finalizer(prompt: str, context: Dict[str, Any]) -> str:
    try:
        res = await Runner.run(
            build_finalizer(),
            input=prompt,
            context=context,
        )  # type: ignore
        return getattr(res, "final_output", str(res))
    except TypeError:
        inline = f"CONTEXT:\n{context}\n\nPROMPT:\n{prompt}"
        return await _run(build_finalizer(), inline)


async def _run(agent: Agent, inp: str) -> str:
    res = await Runner.run(agent, input=inp)
    return getattr(res, "final_output", str(res))

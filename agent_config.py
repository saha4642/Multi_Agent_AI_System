# agent_config.py

from agents import Agent, function_tool


@function_tool
def get_order_status(order_id: str) -> str:
    """Look up an order's shipping status by order_id."""
    return f"Order {order_id} is packed and will ship today."


def build_agent() -> Agent:
    """Create and configure the support agent."""
    return Agent(
        name="SupportAgent",
        instructions=(
            "You are a helpful support agent. "
            "Remember user details shared in this session "
            "and use tools when relevant."
        ),
        tools=[get_order_status],
    )

"""Private MCP server exposing check_order_status (Task 14).  Run: python -m mcp_layer.server"""
import os

from fastmcp import FastMCP

from agent.tools import check_order_status as _check_order_status

mcp = FastMCP("nykaa-orders")


@mcp.tool()
def check_order_status(record_id: str) -> dict:
    """Look up a Nykaa order by ID and return its status, value and escalation score.

    Args:
        record_id: Order ID such as "NYK-1003".

    Returns:
        dict with status, order_value_inr, escalation_score (0-1), escalation_threshold and
        recommend_escalation; or an "error" key if the order does not exist.
    """
    return _check_order_status(record_id)


if __name__ == "__main__":
    # HTTP transport mounts at /mcp -> http://127.0.0.1:<port>/mcp
    mcp.run(transport="http", host="127.0.0.1", port=int(os.getenv("MCP_PORT", "8000")))

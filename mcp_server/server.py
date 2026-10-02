"""
SalesSaathi MCP Server
-----------------------
A real Model Context Protocol (MCP) server exposing three tools that stand in
for a dealer's CRM (Salesforce Sales Cloud), DMS (Dealer Management System),
and showroom calendar.

In a production rollout, these three functions would call the real Salesforce
/ DMS / calendar APIs. The point of putting them behind MCP rather than
custom integrations is that any future AI feature -- this agent, a different
agent, a different team's tool -- can reuse the same tool contracts instead
of each one re-implementing its own CRM/DMS client.

Run standalone (for manual testing):
    python mcp_server/server.py
"""

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

# Keep the demo terminal clean -- the MCP SDK logs each request at INFO by default.
logging.getLogger().setLevel(logging.WARNING)

DATA_DIR = os.path.join(os.path.dirname(__file__), "mock_data")

mcp = FastMCP("tata-motors-sales-tools")


def _load(name: str):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


@mcp.tool()
def get_lead_from_crm(lead_id: str) -> dict:
    """Fetch a lead record from the CRM (mock Salesforce Sales Cloud) by lead ID."""
    leads = _load("leads.json")
    for lead in leads:
        if lead["id"] == lead_id:
            return lead
    return {"error": f"lead {lead_id} not found"}


@mcp.tool()
def check_inventory(model: str, region: str) -> dict:
    """Check dealer inventory (mock DMS) for a given vehicle model and region."""
    inventory = _load("inventory.json")
    key = f"{model}|{region}"
    return inventory.get(key, {"inStock": False, "units": 0})


@mcp.tool()
def schedule_test_drive(lead_id: str, preferred_slot: str = "next available") -> dict:
    """Hold a test-drive slot on the showroom calendar for a given lead."""
    # A real implementation would check actual calendar availability.
    return {
        "leadId": lead_id,
        "slot": "Tomorrow, 4:30 PM",
        "status": "held (pending customer confirmation)",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

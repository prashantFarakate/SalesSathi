#!/usr/bin/env python3
"""
SalesSaathi CLI
----------------
A demo-ready command-line interface tying together:
  - a real MCP server/client (mock CRM, DMS, calendar tools)
  - an embedding-based RAG assistant (ChromaDB + local MiniLM) that generates
    grounded, cited answers and refuses to guess when nothing matches
  - an agent pipeline with LLM-based (or heuristic fallback) scoring & drafting

Usage:
    python cli.py leads              List mock leads
    python cli.py run L-2291         Run the full agent pipeline for a lead
    python cli.py ask "<question>"   Query the knowledge assistant directly
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.agent import run_pipeline  # noqa: E402

console = Console()


def cmd_leads():
    with open(os.path.join("mcp_server", "mock_data", "leads.json")) as f:
        leads = json.load(f)
    table = Table(title="Mock CRM leads")
    for col in ["ID", "Name", "Source", "Model", "Region", "Budget"]:
        table.add_column(col)
    for l in leads:
        table.add_row(l["id"], l["name"], l["source"], l["model"], l["region"], l["budget"])
    console.print(table)
    console.print("\n[dim]Run one with:[/dim] python cli.py run <ID>")


def cmd_ask(query: str):
    from rag.answerer import answer  # local import keeps `leads`/`run` fast

    console.print(Panel(f"[bold]{query}[/bold]", title="Query", border_style="cyan"))
    result = answer(query)

    border = {"no_context": "yellow", "degraded": "yellow"}.get(result.mode, "green")
    console.print(Panel(result.text.strip(), title=f"Answer  [dim](mode: {result.mode})[/dim]", border_style=border))

    if result.sources:
        console.print(f"[dim]Sources: {', '.join(result.sources)}[/dim]")
    for c in result.chunks:
        console.print(Panel(
            c.text.strip()[:500] + ("..." if len(c.text) > 500 else ""),
            title=f"grounding: {c.source}  (score={c.score:.3f})",
            border_style="blue",
        ))


def cmd_run(lead_id: str):
    console.rule(f"[bold]Agent run — {lead_id}[/bold]")

    result_holder = {}

    def on_step(step, detail):
        if step == "error":
            console.print(f"[red]✗ {detail}[/red]")
            return
        console.print(f"[bold cyan]→ {step}[/bold cyan]")
        console.print(json.dumps(detail, indent=2, ensure_ascii=False))
        console.print()

    result = asyncio.run(run_pipeline(lead_id, on_step=on_step))
    if not result:
        return

    score = result["score"]
    tier_color = {"hot": "green", "warm": "yellow", "cold": "blue"}.get(score["tier"], "white")
    console.print(Panel(
        f"[bold {tier_color}]{score['score']}/100 — {score['tier'].upper()}[/bold {tier_color}]\n"
        + "\n".join(f"· {r}" for r in score["reasons"])
        + f"\n\n[dim]scored via: {score['mode']}[/dim]",
        title="Lead score",
        border_style=tier_color,
    ))

    draft = result["draft"]
    console.print(Panel(
        draft["message"] + f"\n\n[dim]drafted via: {draft['mode']}[/dim]",
        title="Draft follow-up message",
        border_style="magenta",
    ))

    if Confirm.ask("Approve and queue this message to send?"):
        console.print("[bold green]✓ Approved — queued to send via WhatsApp.[/bold green]")
    else:
        console.print("[yellow]Not sent. Edit the draft and re-run, or handle manually.[/yellow]")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "leads":
        cmd_leads()
    elif cmd == "run" and len(sys.argv) > 2:
        cmd_run(sys.argv[2])
    elif cmd == "ask" and len(sys.argv) > 2:
        cmd_ask(" ".join(sys.argv[2:]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

"""
server.py — FastAPI health + inference server for HF Spaces deployment.

Endpoints:
    GET  /         → environment info
    GET  /health   → 200 OK (required by HF automated ping)
    GET  /tasks    → list all task IDs and metadata
    POST /reset    → reset environment for a task
    POST /step     → advance environment one step
    GET  /state    → current task state snapshot
    POST /run      → run full inference baseline for one or all tasks
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import SalesOpsEnv
from models import Action, ActionType
from tasks import get_task_ids, get_task_meta

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SalesOps OpenEnv",
    description="Enterprise CRM workflow RL environment — OpenEnv Hackathon v1",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global environment instance (single-session; use /reset to start a new episode)
_env = SalesOpsEnv()
_active_task: Optional[str] = None

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str
    value: Optional[str] = None
    reasoning: Optional[str] = None


class RunRequest(BaseModel):
    task_id: Optional[str] = None   # None = run all tasks


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "env": SalesOpsEnv.ENV_NAME,
        "version": SalesOpsEnv.ENV_VERSION,
        "tasks": get_task_ids(),
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> JSONResponse:
    """HF Spaces automated ping — must return 200."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "env": SalesOpsEnv.ENV_NAME,
            "timestamp": int(time.time()),
        },
    )


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    tasks_info = []
    for tid in get_task_ids():
        meta = get_task_meta(tid)
        tasks_info.append({
            "task_id":    meta.task_id,
            "name":       meta.name,
            "difficulty": meta.difficulty,
            "max_steps":  meta.max_steps,
            "tags":       meta.tags,
            "description": meta.description,
        })
    return {"tasks": tasks_info, "count": len(tasks_info)}


@app.post("/reset")
async def reset_env(req: Optional[ResetRequest] = None) -> Dict[str, Any]:
    global _active_task
    
    # If no body or no task_id provided, default to the first task
    task_id = req.task_id if (req and req.task_id) else get_task_ids()[0]
    
    if task_id not in get_task_ids():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {get_task_ids()}",
        )
    obs = _env.reset(task_id)
    _active_task = task_id
    return {
        "task_id": task_id,
        "observation": obs.model_dump(mode="json"),
    }


@app.post("/step")
async def step_env(req: StepRequest) -> Dict[str, Any]:
    if _active_task is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        action = Action(
            action_type=ActionType(req.action_type),
            value=req.value,
            reasoning=req.reasoning,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    result = _env.step(action)
    return {
        "observation": result.observation.model_dump(mode="json"),
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
        "error":       result.error,
    }


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    if _active_task is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    s = _env.state()
    return s.model_dump(mode="json")


@app.post("/run")
async def run_inference(req: RunRequest) -> Dict[str, Any]:
    """
    Runs the baseline inference.py agent programmatically (non-blocking wrapper).
    Returns the structured results for all requested tasks.
    """
    # Import here to avoid circular import at module load
    from inference import run_task

    task_ids = [req.task_id] if req.task_id else get_task_ids()

    if req.task_id and req.task_id not in get_task_ids():
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'.")

    results = []
    for tid in task_ids:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_task, tid
            )
            results.append(result)
        except Exception as exc:
            results.append({
                "task_id": tid,
                "error": str(exc),
                "success": False,
                "score": 0.0,
            })

    return {"results": results, "tasks_run": len(results)}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False, workers=1)

if __name__ == "__main__":
    main()

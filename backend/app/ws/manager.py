import json
from collections import defaultdict
from fastapi import WebSocket
from typing import Any


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, project_id: str, ws: WebSocket):
        await ws.accept()
        self._connections[project_id].append(ws)

    def disconnect(self, project_id: str, ws: WebSocket):
        conns = self._connections.get(project_id, [])
        if ws in conns:
            conns.remove(ws)
        if not conns and project_id in self._connections:
            del self._connections[project_id]

    async def broadcast(self, project_id: str, data: dict[str, Any]):
        msg = json.dumps(data)
        dead = []
        for ws in self._connections.get(project_id, []):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(project_id, ws)

    async def send_log(self, project_id: str, line: str):
        await self.broadcast(project_id, {"type": "log", "line": line})

    async def send_progress(self, project_id: str, step: str, substep: str, percent: float):
        await self.broadcast(project_id, {
            "type": "progress", "step": step, "substep": substep, "percent": percent
        })

    async def send_status(self, project_id: str, step: str, state: str, error: str | None = None):
        await self.broadcast(project_id, {
            "type": "status", "step": step, "state": state, "error": error
        })

    async def send_metric(self, project_id: str, **kwargs):
        await self.broadcast(project_id, {"type": "metric", **kwargs})


manager = ConnectionManager()

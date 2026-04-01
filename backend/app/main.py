import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# On Windows, ensure ProactorEventLoop is used so asyncio.create_subprocess_exec
# works even when uvicorn runs with --reload (which can swap the loop policy).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from .config import settings
from .database import init_db
from .logging_config import setup_logging
from .routers import projects, pipeline, system
from .ws.manager import manager

# Configure logging before anything else
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GaussianSplat Studio starting up")
    await init_db()
    logger.info("Database initialized")
    logger.info("Log file: %s", settings.log_file)
    yield
    logger.info("GaussianSplat Studio shutting down")


app = FastAPI(title=settings.project_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Allow all origins so Quest 3 and other LAN devices can connect directly.
    # WebSocket connects directly to port 8000 (bypasses Next.js proxy), so
    # the origin will be the LAN IP (e.g. http://192.168.1.196:3000).
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(projects.router)
app.include_router(pipeline.router)
app.include_router(system.router)


@app.websocket("/ws/projects/{project_id}/logs")
async def ws_logs(websocket: WebSocket, project_id: str):
    await manager.connect(project_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(project_id, websocket)


@app.get("/api/health")
async def health():
    return {"status": "ok"}

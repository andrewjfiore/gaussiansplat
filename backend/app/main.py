import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

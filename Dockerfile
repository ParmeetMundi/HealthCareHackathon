# ── CrewAI Healthcare Orchestrator — Container ──────────────────────────────────
#
# Builds and runs the single orchestrator service that hosts all specialist
# agents in-process via CrewAI's hierarchical crew.
#
# Local build + test:
#   docker build -t crewai-healthcare .
#   docker run --rm -p 8003:8003 \
#     -e GOOGLE_API_KEY=your-key-here \
#     crewai-healthcare
#
# Cloud Run deployment:
#   gcloud run deploy orchestrator \
#     --source . \
#     --set-env-vars "PORT=8080" \
#     --set-secrets "GOOGLE_API_KEY=google-api-key:latest" ...

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full source tree.
COPY . .

# Default port for the orchestrator.
ENV PORT=8003

ENV AGENT_MODULE=orchestrator.app:a2a_app

# exec replaces the shell so uvicorn is PID 1 and receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn ${AGENT_MODULE} --host 0.0.0.0 --port ${PORT}"]

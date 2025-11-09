## Container for the Case Closed agent (Flask HTTP server)
## Notes:
## - Defaults to running agent.py on PORT 5008 (configurable via env var PORT)
## - Uses Python 3.12 slim image, with layer caching for pip installs
## - Runs as a non-root user

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5008

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source
COPY . .

# Create and use a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose the application port (matches agent.py default PORT)
EXPOSE 5008

# For production you may prefer gunicorn:
#   RUN pip install --no-cache-dir gunicorn
#   CMD ["gunicorn", "-b", "0.0.0.0:${PORT}", "agent:app"]

# Default: run the Flask app directly
CMD ["python", "agent.py"]

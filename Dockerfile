FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive mode to prevent tzdata issues
ENV DEBIAN_FRONTEND=noninteractive

# Set up the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git curl wget nano bash \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Install `uv` package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure `uv` is in the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set up virtual environment using full path to `uv`
RUN /root/.local/bin/uv venv --python=python3.10

# Install dependencies using `uv`
RUN /root/.local/bin/uv sync

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Expose ports if needed (e.g., Flask API, Jupyter Notebook)
EXPOSE 8080 8888

# Default command (modify for your use case)
CMD ["python", "cli.py", "--help"]


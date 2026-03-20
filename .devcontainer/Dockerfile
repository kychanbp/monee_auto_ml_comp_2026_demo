FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  curl \
  sudo \
  zsh \
  fzf \
  jq \
  nano \
  vim \
  procps \
  build-essential \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
  apt-get install -y nodejs && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code@latest

# Create non-root user
ARG USERNAME=dev
RUN useradd -m -s /bin/zsh ${USERNAME} && \
  echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# Persist bash history
RUN mkdir /commandhistory && \
  touch /commandhistory/.bash_history && \
  chown -R ${USERNAME} /commandhistory

# Create workspace and config directories
RUN mkdir -p /workspace /home/${USERNAME}/.claude && \
  chown -R ${USERNAME}:${USERNAME} /workspace /home/${USERNAME}/.claude

USER ${USERNAME}
WORKDIR /workspace

# Install base Python ML packages
RUN pip install --no-cache-dir numpy pandas scikit-learn lightgbm xgboost catboost scipy pyarrow

ENV SHELL=/bin/zsh

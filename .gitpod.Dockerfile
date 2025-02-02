# Base image: use any official Gitpod workspace image or another base
FROM gitpod/workspace-full:latest

# Switch to root
USER root

# Install the packages you need
RUN apt-get update \
  && apt-get install -y libgl1-mesa-glx curl vim \
  && rm -rf /var/lib/apt/lists/*

# (Optional) switch back to a non-root user 'gitpod'
USER gitpod
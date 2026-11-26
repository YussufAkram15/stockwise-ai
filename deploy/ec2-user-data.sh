#!/bin/bash
# EC2 "User data" script (Amazon Linux 2023). It runs once, as root, when the server first starts:
# install Docker + Git, clone the repo (main), build the dashboard image and start it on port 8501.
# Log: /var/log/cloud-init-output.log
set -euxo pipefail

dnf install -y docker git
systemctl enable --now docker

git clone --depth 1 https://github.com/YussufAkram15/stockwise-ai.git /opt/stockwise-ai
cd /opt/stockwise-ai

docker build -f deploy/Dockerfile -t stockwise-dashboard .
docker run -d --name stockwise --restart unless-stopped -p 8501:8501 stockwise-dashboard

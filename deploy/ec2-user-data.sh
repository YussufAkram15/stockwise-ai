#!/bin/bash
# EC2 "User data" script (Amazon Linux 2023). It runs once, as root, when the server first starts:
# install Docker + Git, clone the repo (main), build the dashboard image and start it on port 8501.
# Log: /var/log/cloud-init-output.log
set -euxo pipefail

# 1 GB of swap (spare memory on disk): t3.micro has only 1 GB RAM, and dnf + pip come close to it.
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

dnf install -y docker git
systemctl enable --now docker

git clone --depth 1 https://github.com/YussufAkram15/stockwise-ai.git /opt/stockwise-ai
cd /opt/stockwise-ai

docker build -f deploy/Dockerfile -t stockwise-dashboard .
docker run -d --name stockwise --restart unless-stopped -p 8501:8501 stockwise-dashboard

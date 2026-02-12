#!/bin/sh
cd "$(dirname "$0")"
docker compose -f docker-compose-nvidia.yml pull
docker compose -f docker-compose-nvidia.yml up --force-recreate -d

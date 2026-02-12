#!/bin/sh
cd "$(dirname "$0")"
docker compose -f docker-compose-amd.yml pull
docker compose -f docker-compose-amd.yml up --force-recreate -d

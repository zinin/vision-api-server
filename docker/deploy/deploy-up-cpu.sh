#!/bin/sh
cd "$(dirname "$0")"
docker compose -f docker-compose-cpu.yml pull
docker compose -f docker-compose-cpu.yml up --force-recreate

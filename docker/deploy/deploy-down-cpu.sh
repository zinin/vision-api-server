#!/bin/sh
cd "$(dirname "$0")"
docker compose -f docker-compose-cpu.yml down

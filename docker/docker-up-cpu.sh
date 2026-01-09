#!/bin/sh

docker compose -f docker-compose-cpu.yml pull
docker compose -f docker-compose-cpu.yml up --build --force-recreate

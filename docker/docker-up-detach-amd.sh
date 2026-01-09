#!/bin/sh

docker compose -f docker-compose-amd.yml pull
docker compose -f docker-compose-amd.yml up --build --force-recreate -d

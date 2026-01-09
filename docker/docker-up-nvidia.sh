#!/bin/sh

docker compose -f docker-compose-nvidia.yml pull
docker compose -f docker-compose-nvidia.yml up --build --force-recreate

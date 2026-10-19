set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

default:
    @just --list

up:
    docker compose up -d

down:
    docker compose down

logs:
    docker compose logs -f

ps:
    docker compose ps

salt:
    docker compose exec bbb bbb-conf --salt

html5:
    docker compose exec -u bigbluebutton bbb bash -lc 'cd ~/src/bigbluebutton-html5 && ./run-dev.sh'

sync:
    uv --directory edusync sync

test:
    uv --directory edusync run pytest --cov

lint:
    uv --directory edusync run ruff check .
    uv --directory edusync run ruff format --check .

fmt:
    uv --directory edusync run ruff format .
    uv --directory edusync run ruff check --fix .

check: lint test

config:
    docker compose config -q

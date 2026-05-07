#!/bin/bash
# =============================================================================
# Postgres init script — creates additional databases listed in
# POSTGRES_MULTIPLE_DATABASES (comma-separated). Runs automatically on first
# container boot via the postgres image's /docker-entrypoint-initdb.d hook.
# =============================================================================
set -e
set -u

if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
  echo "[init-multi-db] Creating additional databases: $POSTGRES_MULTIPLE_DATABASES"
  for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
    echo "[init-multi-db]   -> $db"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
      SELECT 'CREATE DATABASE $db'
      WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
      GRANT ALL PRIVILEGES ON DATABASE $db TO $POSTGRES_USER;
EOSQL
  done

  # Enable PostGIS on the primary application database.
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS postgis_topology;
EOSQL
  echo "[init-multi-db] Done."
fi

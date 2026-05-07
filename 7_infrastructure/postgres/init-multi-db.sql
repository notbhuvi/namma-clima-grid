-- =============================================================================
-- NammaClimaGrid — Postgres initialization
-- Creates the airflow and superset side-databases and enables PostGIS on all.
-- This file is executed automatically by the postgres image on first boot
-- via the /docker-entrypoint-initdb.d/ hook (no execute permission required).
-- =============================================================================

-- Airflow metadata database
SELECT 'CREATE DATABASE airflow OWNER ncg_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

-- Superset metadata database
SELECT 'CREATE DATABASE superset OWNER ncg_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'superset')\gexec

-- Enable PostGIS on main application database (already selected by Docker env)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Switch to airflow and enable PostGIS there too
\c airflow
CREATE EXTENSION IF NOT EXISTS postgis;

-- Switch to superset
\c superset
CREATE EXTENSION IF NOT EXISTS postgis;

-- Return to main db
\c namma_clima_grid

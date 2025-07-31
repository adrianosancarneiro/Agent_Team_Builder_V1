#!/bin/bash
set -e

PG_VERSION=$(psql -V | awk '{print $3}' | cut -d. -f1)
PG_CONF="/etc/postgresql/$PG_VERSION/main/postgresql.conf"
PG_HBA="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"

# Update configs
echo -e "\n🔧 Updating PostgreSQL configuration..."
sudo sed -i \
  -e "s/^#listen_addresses.*/listen_addresses = '*'/" \
  "$PG_CONF"

echo -e "\n🔒 Configuring pg_hba.conf..."
echo "host all all 0.0.0.0/0 md5" | sudo tee -a "$PG_HBA"
echo "host all all ::/0 md5" | sudo tee -a "$PG_HBA"

# Restart PostgreSQL
echo "\n🔄 Restarting PostgreSQL..."
sudo systemctl restart postgresql

# Create user and DB
echo -e "\n👤 Creating user and database..."
sudo -u postgres psql <<EOF
CREATE ROLE postgres_user WITH LOGIN PASSWORD 'postgres_pass' CREATEDB SUPERUSER;
CREATE DATABASE "agentteambuilder" WITH OWNER postgres_user;
EOF

# Wait a moment for the database to be ready
sleep 2

echo -e "\n👤 Setting permissions..."
sudo -u postgres psql -d "agentteambuilder" <<EOF
GRANT USAGE ON SCHEMA public TO postgres_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO postgres_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO postgres_user;
EOF

# Enable pg_stat_statements
echo -e "\n📦 Enabling extension and granting privileges..."
if ! grep -q "pg_stat_statements" "$PG_CONF"; then
  sudo sed -i "s/^#*shared_preload_libraries.*/shared_preload_libraries = 'pg_stat_statements'/" "$PG_CONF"
  sudo systemctl restart postgresql
fi

sudo -u postgres psql <<EOF
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
GRANT SELECT ON pg_stat_statements TO postgres_user;
EOF

echo -e "\n✅ PostgreSQL is now fully configured for OpenMetadata."
echo "🔗 Use host.docker.internal:5432 in OpenMetadata UI"
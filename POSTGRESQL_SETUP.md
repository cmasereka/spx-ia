# PostgreSQL Setup for SPX AI Trading Platform

## Overview

This document explains how to set up PostgreSQL for the SPX AI Trading Platform. The system uses PostgreSQL as the primary database for storing trading data, backtest results, and system logs.

## Prerequisites

1. PostgreSQL 12+ installed and running
2. Python dependencies installed: `pip install -r requirements.txt`

## Environment Variables

Set the following environment variables or create a `.env` file:

```bash
# PostgreSQL Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/spx_ai
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=spx_ai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

## Database Setup

### 1. Install PostgreSQL

**macOS (using Homebrew):**
```bash
brew install postgresql
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Create Database User

```bash
sudo -u postgres psql
CREATE USER spx_user WITH PASSWORD 'your_password';
CREATE DATABASE spx_ai OWNER spx_user;
GRANT ALL PRIVILEGES ON DATABASE spx_ai TO spx_user;
\q
```

### 3. Initialize Database

Run the initialization script:
```bash
python init_database.py
```

This will:
- Create the database if it doesn't exist
- Run Alembic migrations
- Create all necessary tables

## Database Management

Use the database manager script for common operations:

```bash
# Test connection
python db_manager.py test

# Create tables
python db_manager.py create

# Drop tables
python db_manager.py drop

# Reset database (drop and recreate)
python db_manager.py reset

# Show database info
python db_manager.py info
```

## Database Schema

### Main Tables

1. **backtest_runs** - Stores backtest configuration and results
2. **trades** - Individual trade records from backtests
3. **market_data** - SPX index price data
4. **option_data** - Options pricing and Greeks data
5. **system_logs** - Application logs and audit trail

### Key Features

- UUID primary keys for better scalability
- JSONB columns for flexible metadata storage
- Proper indexing for query performance
- Foreign key relationships for data integrity
- Automatic timestamps for audit trails

## Migrations

The system uses Alembic for database migrations:

```bash
# Generate a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Downgrade to previous version
alembic downgrade -1

# Show migration history
alembic history
```

## Performance Considerations

1. **Connection Pooling**: Configured with SQLAlchemy's QueuePool
2. **Indexes**: Added on frequently queried columns
3. **JSONB**: Used for flexible schema while maintaining query performance
4. **Partitioning**: Consider partitioning by date for large datasets

## Backup and Recovery

### Create Backup
```bash
pg_dump -h localhost -U spx_user -d spx_ai > backup.sql
```

### Restore Backup
```bash
psql -h localhost -U spx_user -d spx_ai < backup.sql
```

## Troubleshooting

### Connection Issues

1. Check PostgreSQL is running:
   ```bash
   sudo systemctl status postgresql  # Linux
   brew services list | grep postgres  # macOS
   ```

2. Verify connection parameters in `.env` file
3. Test connection: `python db_manager.py test`

### Permission Issues

```bash
# Grant necessary permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE spx_ai TO spx_user;"
```

### Migration Issues

```bash
# Reset migrations (caution: loses data)
alembic downgrade base
alembic upgrade head
```

## Development Notes

- Always run migrations in a transaction
- Test migrations on a copy of production data
- Use JSONB for flexible schema, but don't overuse it
- Monitor query performance with `EXPLAIN ANALYZE`
- Consider read replicas for heavy analytical queries
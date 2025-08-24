-- Flight Schedule Optimization Database Initialization Script
-- This script runs when the PostgreSQL Docker container starts

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create the database if it doesn't exist (this will be handled by docker-compose environment)
-- The database 'flight_schedule_optimization' will be created automatically

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE flight_schedule_optimization TO flight_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO flight_user;

-- Set search path
SET search_path TO public;

-- Note: Tables will be created by SQLAlchemy models when the application starts
-- This script ensures the database and user permissions are set up correctly

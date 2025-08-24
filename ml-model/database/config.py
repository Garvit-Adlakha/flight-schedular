"""
PostgreSQL Database Configuration for Flight Schedule Optimization
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'flight_schedule_optimization'),
    'username': os.getenv('POSTGRES_USER', 'flight_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'flight_password')
}

def get_database_url() -> str:
    """Get PostgreSQL database URL"""
    return (
        f"postgresql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
    )

# SQLAlchemy setup
engine = create_engine(
    get_database_url(),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_database_if_not_exists():
    """Create database if it doesn't exist"""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['username'],
            password=DATABASE_CONFIG['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", 
                      (DATABASE_CONFIG['database'],))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {DATABASE_CONFIG['database']}")
            print(f"✅ Created database: {DATABASE_CONFIG['database']}")
        else:
            print(f"✅ Database already exists: {DATABASE_CONFIG['database']}")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Error creating database: {e}")
        print("💡 Make sure PostgreSQL is running and credentials are correct")
        raise

def test_connection() -> bool:
    """Test database connection"""
    try:
        engine.connect()
        print("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing PostgreSQL configuration...")
    print(f"Database URL: {get_database_url()}")
    
    try:
        create_database_if_not_exists()
        test_connection()
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

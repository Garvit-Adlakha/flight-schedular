"""
PostgreSQL Database Setup Script
Automated setup for flight schedule optimization database
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.config import create_database_if_not_exists, test_connection, DATABASE_CONFIG
from database.models import create_all_tables

def check_postgresql_installation():
    """Check if PostgreSQL is installed and running"""
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ PostgreSQL found: {result.stdout.strip()}")
            return True
        else:
            print("❌ PostgreSQL not found")
            return False
    except FileNotFoundError:
        print("❌ PostgreSQL not installed")
        return False

def check_postgresql_service():
    """Check if PostgreSQL service is running"""
    try:
        # Try to connect to default PostgreSQL port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((DATABASE_CONFIG['host'], int(DATABASE_CONFIG['port'])))
        sock.close()
        
        if result == 0:
            print("✅ PostgreSQL service is running")
            return True
        else:
            print("❌ PostgreSQL service is not running")
            return False
    except Exception as e:
        print(f"❌ Error checking PostgreSQL service: {e}")
        return False

def create_user_and_database():
    """Create database user and database"""
    try:
        print("🔧 Setting up database and user...")
        
        # Commands to create user and database
        commands = [
            f"CREATE USER {DATABASE_CONFIG['username']} WITH PASSWORD '{DATABASE_CONFIG['password']}';",
            f"CREATE DATABASE {DATABASE_CONFIG['database']} OWNER {DATABASE_CONFIG['username']};",
            f"GRANT ALL PRIVILEGES ON DATABASE {DATABASE_CONFIG['database']} TO {DATABASE_CONFIG['username']};"
        ]
        
        for cmd in commands:
            try:
                subprocess.run([
                    'psql', 
                    '-h', DATABASE_CONFIG['host'],
                    '-p', DATABASE_CONFIG['port'],
                    '-U', 'postgres',  # Assume postgres is the superuser
                    '-c', cmd
                ], check=True, capture_output=True, text=True)
                print(f"✅ Executed: {cmd}")
            except subprocess.CalledProcessError as e:
                if "already exists" in e.stderr:
                    print(f"⚠️  Already exists: {cmd}")
                else:
                    print(f"❌ Error executing: {cmd}")
                    print(f"   Error: {e.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def install_python_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    try:
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ], check=True)
        
        print("✅ Python dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_database_schema():
    """Create database tables and indexes"""
    print("🏗️  Creating database schema...")
    
    try:
        create_database_if_not_exists()
        create_all_tables()
        print("✅ Database schema created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating schema: {e}")
        return False

def run_database_tests():
    """Run basic database connectivity tests"""
    print("🧪 Running database tests...")
    
    try:
        if test_connection():
            print("✅ Database connection test passed")
            return True
        else:
            print("❌ Database connection test failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def print_connection_info():
    """Print database connection information"""
    print("\n📋 Database Configuration:")
    print(f"   Host: {DATABASE_CONFIG['host']}")
    print(f"   Port: {DATABASE_CONFIG['port']}")
    print(f"   Database: {DATABASE_CONFIG['database']}")
    print(f"   Username: {DATABASE_CONFIG['username']}")
    print(f"   Password: {'*' * len(DATABASE_CONFIG['password'])}")

def print_docker_instructions():
    """Print Docker setup instructions"""
    print("\n🐳 Docker PostgreSQL Setup (Alternative):")
    print("If you prefer to use Docker, run these commands:")
    print()
    print("1. Pull PostgreSQL image:")
    print("   docker pull postgres:15")
    print()
    print("2. Run PostgreSQL container:")
    print(f"""   docker run --name flight-postgres \\
     -e POSTGRES_USER={DATABASE_CONFIG['username']} \\
     -e POSTGRES_PASSWORD={DATABASE_CONFIG['password']} \\
     -e POSTGRES_DB={DATABASE_CONFIG['database']} \\
     -p {DATABASE_CONFIG['port']}:5432 \\
     -d postgres:15""")
    print()
    print("3. Then rerun this setup script")

def main():
    """Main setup function"""
    print("🚀 Flight Schedule Optimization - PostgreSQL Setup")
    print("=" * 60)
    
    print_connection_info()
    
    # Check PostgreSQL installation
    if not check_postgresql_installation():
        print("\n💡 PostgreSQL Installation Instructions:")
        print("   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib")
        print("   CentOS/RHEL: sudo yum install postgresql-server postgresql-contrib")
        print("   macOS: brew install postgresql")
        print("   Windows: Download from https://www.postgresql.org/download/")
        print_docker_instructions()
        return False
    
    # Check if service is running
    if not check_postgresql_service():
        print("\n💡 Start PostgreSQL service:")
        print("   Ubuntu/Debian: sudo systemctl start postgresql")
        print("   CentOS/RHEL: sudo systemctl start postgresql")
        print("   macOS: brew services start postgresql")
        print("   Windows: Start via Services or pg_ctl")
        print_docker_instructions()
        return False
    
    # Install Python dependencies
    if not install_python_dependencies():
        return False
    
    # Create user and database
    print(f"\n🔑 You may be prompted for the PostgreSQL superuser (postgres) password")
    if not create_user_and_database():
        print("⚠️  Manual setup may be required. Try running these SQL commands as postgres user:")
        print(f"   CREATE USER {DATABASE_CONFIG['username']} WITH PASSWORD '{DATABASE_CONFIG['password']}';")
        print(f"   CREATE DATABASE {DATABASE_CONFIG['database']} OWNER {DATABASE_CONFIG['username']};")
        print(f"   GRANT ALL PRIVILEGES ON DATABASE {DATABASE_CONFIG['database']} TO {DATABASE_CONFIG['username']};")
    
    # Set up schema
    if not setup_database_schema():
        return False
    
    # Run tests
    if not run_database_tests():
        return False
    
    print("\n🎉 Database setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the API server: cd api && python start_api.py")
    print("2. Upload flight data via API: POST /api/upload-data")
    print("3. Query analytics: GET /api/peak-hours")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

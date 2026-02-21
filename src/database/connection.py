"""
Database configuration and connection management for PostgreSQL
"""

import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from config.settings import DATABASE_URL, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseManager:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or DATABASE_URL
        self._engine = None
        self._session_factory = None
        
    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False
            )
        return self._engine
    
    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_database_info(self) -> dict:
        """Get database information"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                
                result = conn.execute(text("SELECT current_database()"))
                database = result.scalar()
                
                return {
                    "database": database,
                    "version": version,
                    "url": self.database_url.split('@')[1] if '@' in self.database_url else self.database_url
                }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}

# Global database manager instance
db_manager = DatabaseManager()

def get_database_url(host: str = None, port: int = None, database: str = None, 
                    user: str = None, password: str = None) -> str:
    """Construct database URL from components"""
    host = host or POSTGRES_HOST
    port = port or POSTGRES_PORT
    database = database or POSTGRES_DB
    user = user or POSTGRES_USER
    password = password or POSTGRES_PASSWORD
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

def init_database():
    """Initialize database connection and create tables"""
    try:
        if db_manager.test_connection():
            db_manager.create_tables()
            logger.info("Database initialized successfully")
            return True
        else:
            logger.error("Database initialization failed - connection test failed")
            return False
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False
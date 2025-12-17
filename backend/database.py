from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from redis import Redis
import asyncpg
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

from config import settings
from models.tick_data import Base

# Synchronous engine for SQLAlchemy ORM
sync_engine = create_engine(
    settings.database_url.replace("postgresql", "postgresql+psycopg2"),
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=30
)

# Async engine for FastAPI
async_engine = create_async_engine(
    settings.database_url.replace("postgresql", "postgresql+asyncpg"),
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

SyncSessionLocal = sessionmaker(
    sync_engine,
    expire_on_commit=False
)

# Redis connection
redis_client = Redis.from_url(settings.redis_url, decode_responses=True)

@asynccontextmanager
async def get_async_db():
    """Async database session context manager"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

def get_sync_db():
    """Synchronous database session"""
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

async def init_database():
    """Initialize database tables"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database tables created successfully")

async def test_connections():
    """Test database and Redis connections"""
    try:
        # Test PostgreSQL
        async with async_engine.begin() as conn:
            await conn.execute("SELECT 1")
        print("PostgreSQL connection successful")
        
        # Test Redis
        redis_client.ping()
        print("Redis connection successful")
        
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
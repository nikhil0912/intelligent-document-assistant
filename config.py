import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    DEBUG = False
    TESTING = False
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '2000'))
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '384'))
    
    # Retrieval Configuration
    TOP_K_DOCS = int(os.getenv('TOP_K_DOCS', '3'))
    SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', '0.5'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '5000'))
    API_DEBUG = os.getenv('API_DEBUG', 'false').lower() == 'true'
    
    # File Upload Configuration
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '50000000'))  # 50MB
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'md'}
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', 'data/vector_store')
    USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    
    # Memory Configuration
    MEMORY_TYPE = os.getenv('MEMORY_TYPE', 'conversation')  # conversation, summary, entity
    MEMORY_MAX_TOKENS = int(os.getenv('MEMORY_MAX_TOKENS', '4000'))
    
    # Feature Flags
    ENABLE_STREAMING = os.getenv('ENABLE_STREAMING', 'true').lower() == 'true'
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
    ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    LLM_MODEL = 'gpt-3.5-turbo'
    
# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig)

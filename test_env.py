# test_env.py
import os
from dotenv import load_dotenv
import pathlib

# Print current working directory
print("Current directory:", os.getcwd())

# Print location of .env file
env_path = pathlib.Path('.') / '.env'
print("\nLooking for .env at:", env_path.absolute())
print("File exists:", env_path.exists())

# Try to load .env
print("\nTrying to load .env file...")
load_dotenv(verbose=True)  # verbose=True will print debug info

# Print environment variables
print("\nEnvironment variables after loading:")
print("APP_ENV:", os.getenv('APP_ENV'))
print("DATABASE_URL:", os.getenv('DATABASE_URL'))
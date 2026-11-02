import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database.models import configure_database

configure_database()
print("Database tables initialized.")

from sqlmodel import create_engine, Session
from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)


def get_session() -> Session:
    return Session(engine)

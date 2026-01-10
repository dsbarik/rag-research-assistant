from sqlmodel import SQLModel

from src.infra.db.session import get_session, engine
from src.orchestrator.service import OrchestratorService


class ResearchAssistant:
    def __init__(self) -> None:
        SQLModel.metadata.create_all(engine)
        

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            with get_session() as session:
                service = OrchestratorService(session)

                method = getattr(service, name)

                return method(*args, **kwargs)

        return wrapper

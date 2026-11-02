from fastapi.testclient import TestClient
from app.database.models import configure_database, get_session_factory, Experiment
from app.api.main import app


def test_database_insert(tmp_path):
    configure_database(f"sqlite:///{tmp_path / 'test.db'}")
    Session = get_session_factory()
    with Session() as session:
        item = Experiment(provider="mock", model="test", embedding_model="hashing",
                          chunk_size=500, top_k=3, total_questions=0)
        session.add(item)
        session.commit()
        assert item.id is not None


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

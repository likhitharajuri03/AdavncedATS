import pytest
from app import app
from models.database import Base, engine, SessionLocal
from utils.security import get_password_hash
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

class TestAuth:
    def test_register(self, client):
        response = client.post('/api/auth/register', json={
            'email': 'test@example.com',
            'password': 'testpass123',
            'name': 'Test User',
            'user_type': 'job_seeker'
        })
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'access_token' in data
        assert data['user']['email'] == 'test@example.com'

    def test_login(self, client, db):
        # Create test user
        hashed_password = get_password_hash('testpass123')
        db.execute(
            "INSERT INTO users (email, hashed_password, name, user_type) "
            "VALUES (:email, :password, :name, :user_type)",
            {
                'email': 'test@example.com',
                'password': hashed_password,
                'name': 'Test User',
                'user_type': 'job_seeker'
            }
        )
        db.commit()

        response = client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'testpass123'
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'access_token' in data

class TestResumeProcessing:
    def test_upload_resume(self, client):
        with open('tests/data/test_resume.pdf', 'rb') as f:
            response = client.post(
                '/api/resume/upload',
                data={'file': (f, 'test_resume.pdf')},
                headers={'Authorization': f'Bearer {self._get_token(client)}'}
            )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'skills' in data

    def _get_token(self, client):
        response = client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'testpass123'
        })
        return json.loads(response.data)['access_token']

class TestJobMatching:
    def test_match_jobs(self, client, db):
        # Add test job to database
        db.execute(
            "INSERT INTO jobs (title, description, it_skills, soft_skills) "
            "VALUES (:title, :desc, :it_skills, :soft_skills)",
            {
                'title': 'Test Job',
                'desc': 'Test job description',
                'it_skills': json.dumps(['Python', 'SQL']),
                'soft_skills': json.dumps(['Communication'])
            }
        )
        db.commit()

        response = client.post(
            '/api/jobs/match',
            json={
                'resume_text': 'Python developer with SQL experience',
                'resume_skills': ['Python', 'SQL', 'Communication']
            },
            headers={'Authorization': f'Bearer {self._get_token(client)}'}
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['matches']) > 0

    def _get_token(self, client):
        response = client.post('/api/auth/login', json={
            'email': 'test@example.com',
            'password': 'testpass123'
        })
        return json.loads(response.data)['access_token']
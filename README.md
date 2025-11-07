# AI-Powered ATS Documentation

## Overview
AI-Powered ATS is an Applicant Tracking System that uses artificial intelligence to analyze resumes, match candidates with jobs, and provide intelligent insights for both job seekers and recruiters.

## Features
- Resume parsing and analysis
- Skill extraction and matching
- AI-powered job recommendations
- Personality insights
- Virtual career assistant
- Interview question generation

## Architecture
The system consists of several components:
- Frontend (React)
- Backend (Python/Flask)
- Database (PostgreSQL)
- Cache (Redis)
- ML Models (scikit-learn)
- Monitoring (Prometheus/Grafana)

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Node.js 14+
- Python 3.8+

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   # Backend
   cd backend
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt

   # Frontend
   cd frontend
   npm install
   ```

3. Set up environment variables:
   Copy `.env.example` to `.env` and fill in the required values.

4. Start the services:
   ```powershell
   # Start Docker Desktop (GUI) and ensure it reports "Docker Desktop is running".
   # From project root (where docker-compose.yml is):
   docker-compose up --build -d
   ```

5. Run the included smoke test (PowerShell):
   ```powershell
   # Wait until services are healthy, then run the smoke test which performs a health check, register/login and resume analyze flow
   .\scripts\smoke_test.ps1 -HostUrl http://localhost:8080
   ```

### Development
1. Run backend tests:
   ```bash
   cd backend
   pytest
   ```

2. Run frontend tests:
   ```bash
   cd frontend
   npm test
   ```

## API Documentation

### Authentication
- POST /api/auth/register
- POST /api/auth/login

### Resume Processing
- POST /api/resume/upload
- POST /api/resume/analyze
- POST /api/resume/personality

### Job Matching
- POST /api/jobs/match
- POST /api/jobs/skill-gap/:id
- POST /api/jobs/validate-skills

### AI Assistant
- POST /api/assistant/chat
- POST /api/assistant/interview-questions
- POST /api/assistant/improve-section

## Security
The system implements several security measures:
- JWT authentication
- Password hashing
- Rate limiting
- CSRF protection
- Input sanitization

## Monitoring
Monitor the application using:
- Prometheus (metrics): http://localhost:9090
- Grafana (dashboards): http://localhost:3000

## Production Deployment
1. Configure SSL certificates
2. Set up proper domain names
3. Configure proper environment variables
4. Set up backup strategy
5. Configure monitoring alerts

## Performance Optimization
- Redis caching for frequently accessed data
- Database indexing for common queries
- Rate limiting for API endpoints
- Efficient ML model serving

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License
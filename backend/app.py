from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import os
from datetime import timedelta
import logging

# Import custom modules
from services.document_processor import DocumentProcessor
from services.gemini_service import GeminiService
from models.resume_classifier import ResumeClassifier
from models.skill_extractor import SkillExtractor
from models.job_matcher import JobMatcher
from utils.preprocessing import preprocess_text
from utils.similarity import calculate_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

CORS(app)
jwt = JWTManager(app)

# Initialize services
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
JOBS_DATASET_PATH = 'data/jobs_dataset.csv'
MODEL_PATH = 'models/resume_classifier.pkl'

document_processor = DocumentProcessor()
gemini_service = GeminiService(api_key=GEMINI_API_KEY)
skill_extractor = SkillExtractor(skills_database_path=JOBS_DATASET_PATH)
job_matcher = JobMatcher(jobs_dataset_path=JOBS_DATASET_PATH)

# Load or train classifier
try:
    resume_classifier = ResumeClassifier()
    resume_classifier.load_model(MODEL_PATH)
    logger.info("Loaded pre-trained classifier")
except:
    logger.info("Classifier not found, will train on first use")
    resume_classifier = None

# In-memory user storage (use database in production)
users_db = {}

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user (Job Seeker or Recruiter)"""
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')
    user_type = data.get('user_type')  # 'job_seeker' or 'recruiter'
    name = data.get('name')
    
    if not all([email, password, user_type, name]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if email in users_db:
        return jsonify({'error': 'User already exists'}), 409
    
    # Store user (hash password in production!)
    users_db[email] = {
        'email': email,
        'password': password,  # HASH THIS IN PRODUCTION!
        'user_type': user_type,
        'name': name
    }
    
    # Create access token
    access_token = create_access_token(identity=email)
    
    return jsonify({
        'message': 'User registered successfully',
        'access_token': access_token,
        'user': {
            'email': email,
            'name': name,
            'user_type': user_type
        }
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    email = data.get('email')
    password = data.get('password')
    
    if not all([email, password]):
        return jsonify({'error': 'Missing credentials'}), 400
    
    user = users_db.get(email)
    if not user or user['password'] != password:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    access_token = create_access_token(identity=email)
    
    return jsonify({
        'access_token': access_token,
        'user': {
            'email': user['email'],
            'name': user['name'],
            'user_type': user['user_type']
        }
    }), 200

# ==================== RESUME PROCESSING ROUTES ====================

@app.route('/api/resume/upload', methods=['POST'])
@jwt_required()
def upload_resume():
    """Upload and process resume (supports PDF, DOCX, PPT, JPEG, PNG)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read file
        file_bytes = file.read()
        filename = file.filename
        
        # Process document
        text, is_resume, confidence = document_processor.process_document(filename, file_bytes)
        
        if not is_resume:
            return jsonify({
                'error': 'The uploaded document does not appear to be a resume',
                'confidence': confidence,
                'suggestion': 'Please upload a valid resume document'
            }), 400
        
        # Extract metadata
        metadata = document_processor.get_document_metadata(text)
        
        # Extract skills
        skills = skill_extractor.extract_skills(text)
        
        return jsonify({
            'success': True,
            'resume_text': text,
            'is_resume': is_resume,
            'confidence': confidence,
            'metadata': metadata,
            'skills': skills,
            'message': 'Resume processed successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/resume/analyze', methods=['POST'])
@jwt_required()
def analyze_resume():
    """Comprehensive resume analysis using AI"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text')
        job_description = data.get('job_description')
        
        if not resume_text:
            return jsonify({'error': 'Resume text required'}), 400
        
        # AI Analysis using Gemini
        analysis = gemini_service.analyze_resume(resume_text, job_description)
        
        # Extract skills
        resume_skills = skill_extractor.extract_skills(resume_text)
        
        # Calculate similarity if job description provided
        similarity_score = None
        missing_keywords = []
        
        if job_description:
            similarity_score = calculate_similarity(resume_text, job_description)
            job_skills = skill_extractor.extract_skills(job_description)
            
            # Find missing keywords
            job_technical = set(job_skills['technical'])
            resume_technical = set(resume_skills['technical'])
            missing_keywords = list(job_technical - resume_technical)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'skills': resume_skills,
            'similarity_score': similarity_score,
            'missing_keywords': missing_keywords[:10]
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/resume/personality', methods=['POST'])
@jwt_required()
def predict_personality():
    """Predict personality traits from resume"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text')
        
        if not resume_text:
            return jsonify({'error': 'Resume text required'}), 400
        
        personality = gemini_service.predict_personality(resume_text)
        
        return jsonify({
            'success': True,
            'personality': personality
        }), 200
        
    except Exception as e:
        logger.error(f"Error predicting personality: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ==================== JOB MATCHING ROUTES ====================

@app.route('/api/jobs/match', methods=['POST'])
@jwt_required()
def match_jobs():
    """Find matching jobs from dataset based on resume"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text')
        resume_skills = data.get('resume_skills', [])
        top_n = data.get('top_n', 10)
        
        if not resume_text:
            return jsonify({'error': 'Resume text required'}), 400
        
        # Find matching jobs
        matches = job_matcher.find_matching_jobs(resume_text, resume_skills, top_n=top_n)
        
        return jsonify({
            'success': True,
            'matches': matches,
            'total_matches': len(matches)
        }), 200
        
    except Exception as e:
        logger.error(f"Error matching jobs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/skill-gap/<int:job_id>', methods=['POST'])
@jwt_required()
def skill_gap_analysis(job_id):
    """Analyze skill gaps for specific job"""
    try:
        data = request.get_json()
        resume_skills = data.get('resume_skills', [])
        
        gap_analysis = job_matcher.get_skill_gap_analysis(resume_skills, job_id)
        
        return jsonify({
            'success': True,
            'gap_analysis': gap_analysis
        }), 200
        
    except Exception as e:
        logger.error(f"Error in skill gap analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/validate-skills', methods=['POST'])
@jwt_required()
def validate_skills():
    """Validate resume skills against market data"""
    try:
        data = request.get_json()
        resume_skills = data.get('resume_skills', [])
        target_roles = data.get('target_roles', [])
        
        validation = job_matcher.validate_resume_against_market(resume_skills, target_roles)
        
        return jsonify({
            'success': True,
            'validation': validation
        }), 200
        
    except Exception as e:
        logger.error(f"Error validating skills: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ==================== AI ASSISTANT ROUTES ====================

@app.route('/api/assistant/chat', methods=['POST'])
@jwt_required()
def chatbot():
    """Virtual assistant for resume and career guidance"""
    try:
        data = request.get_json()
        message = data.get('message')
        context = data.get('context', {})
        
        if not message:
            return jsonify({'error': 'Message required'}), 400
        
        response = gemini_service.chatbot_response(message, context)
        
        return jsonify({
            'success': True,
            'response': response
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/assistant/interview-questions', methods=['POST'])
@jwt_required()
def generate_questions():
    """Generate interview questions based on resume"""
    try:
        data = request.get_json()
        resume_text = data.get('resume_text')
        job_role = data.get('job_role')
        
        if not resume_text or not job_role:
            return jsonify({'error': 'Resume text and job role required'}), 400
        
        questions = gemini_service.generate_interview_questions(resume_text, job_role)
        
        return jsonify({
            'success': True,
            'questions': questions
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/assistant/improve-section', methods=['POST'])
@jwt_required()
def improve_section():
    """Improve specific resume section"""
    try:
        data = request.get_json()
        section_text = data.get('section_text')
        section_type = data.get('section_type')
        
        if not section_text or not section_type:
            return jsonify({'error': 'Section text and type required'}), 400
        
        improved = gemini_service.improve_resume_section(section_text, section_type)
        
        return jsonify({
            'success': True,
            'improved_text': improved
        }), 200
        
    except Exception as e:
        logger.error(f"Error improving section: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'document_processor': True,
            'gemini_ai': GEMINI_API_KEY is not None,
            'job_matcher': os.path.exists(JOBS_DATASET_PATH),
            'classifier': resume_classifier is not None
        }
    }), 200

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
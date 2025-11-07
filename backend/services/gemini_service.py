import google.generativeai as genai
import os
from typing import Dict, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    """
    Service for interacting with Google Gemini AI API
    Provides resume analysis, suggestions, and personality insights
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Configure safety settings
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
    def analyze_resume(self, resume_text: str, job_description: str = None) -> Dict:
        """
        Comprehensive resume analysis using Gemini AI
        """
        prompt = self._create_analysis_prompt(resume_text, job_description)
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            # Parse response
            analysis = self._parse_analysis_response(response.text)
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return {
                'error': str(e),
                'summary': 'Unable to generate analysis',
                'strengths': [],
                'weaknesses': [],
                'suggestions': []
            }
    
    def _create_analysis_prompt(self, resume_text: str, job_description: str = None) -> str:
        """Create detailed prompt for resume analysis"""
        base_prompt = f"""
You are an expert HR professional and resume analyst. Analyze the following resume and provide detailed, actionable feedback.

RESUME TEXT:
{resume_text[:8000]}

Please provide analysis in the following JSON format:
{{
    "overall_rating": "Excellent/Good/Average/Poor",
    "rating_score": <number 1-10>,
    "summary": "<2-3 sentence overview>",
    "strengths": [
        "<strength 1>",
        "<strength 2>",
        "<strength 3>"
    ],
    "weaknesses": [
        "<weakness 1>",
        "<weakness 2>",
        "<weakness 3>"
    ],
    "suggestions": [
        "<actionable suggestion 1>",
        "<actionable suggestion 2>",
        "<actionable suggestion 3>"
    ],
    "formatting_issues": [
        "<formatting issue if any>"
    ],
    "missing_sections": [
        "<missing critical section if any>"
    ],
    "keyword_optimization": {{
        "current_keywords": ["<keyword1>", "<keyword2>"],
        "recommended_keywords": ["<keyword1>", "<keyword2>"]
    }},
    "ats_compatibility_score": <number 1-100>
}}
"""
        
        if job_description:
            base_prompt += f"""

JOB DESCRIPTION:
{job_description[:4000]}

Additionally analyze:
- How well does this resume match the job requirements?
- What specific skills or experiences should be highlighted?
- What's missing that the job description requires?
"""
        
        return base_prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
                return analysis
            else:
                # Fallback parsing
                return self._fallback_parse(response_text)
        except json.JSONDecodeError:
            return self._fallback_parse(response_text)
    
    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing when JSON extraction fails"""
        return {
            'overall_rating': 'Good',
            'rating_score': 7,
            'summary': text[:500],
            'strengths': [],
            'weaknesses': [],
            'suggestions': [],
            'ats_compatibility_score': 75
        }
    
    def predict_personality(self, resume_text: str) -> Dict:
        """
        Predict personality traits from resume using NLP and sentiment analysis
        """
        prompt = f"""
As an industrial psychologist, analyze the following resume to predict personality traits and work style.

RESUME:
{resume_text[:6000]}

Provide analysis in JSON format:
{{
    "personality_traits": {{
        "leadership": <score 1-10>,
        "teamwork": <score 1-10>,
        "creativity": <score 1-10>,
        "analytical_thinking": <score 1-10>,
        "attention_to_detail": <score 1-10>,
        "communication": <score 1-10>
    }},
    "work_style": "<description>",
    "cultural_fit_indicators": [
        "<indicator 1>",
        "<indicator 2>"
    ],
    "team_role": "<likely team role>",
    "strengths_summary": "<key personality strengths>",
    "development_areas": "<areas for growth>"
}}
"""
        
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return self._parse_analysis_response(response.text)
        except Exception as e:
            logger.error(f"Personality prediction error: {str(e)}")
            return {'error': str(e)}
    
    def generate_interview_questions(self, resume_text: str, job_role: str) -> List[str]:
        """
        Generate tailored interview questions based on resume and role
        """
        prompt = f"""
Based on the following resume and target job role, generate 10 relevant interview questions.

JOB ROLE: {job_role}

RESUME:
{resume_text[:5000]}

Generate questions that:
1. Test technical skills mentioned in resume
2. Explore work experience and achievements
3. Assess cultural fit and soft skills
4. Challenge the candidate appropriately

Provide questions in JSON format:
{{
    "questions": [
        "Question 1",
        "Question 2",
        ...
    ]
}}
"""
        
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            parsed = self._parse_analysis_response(response.text)
            return parsed.get('questions', [])
        except Exception as e:
            logger.error(f"Question generation error: {str(e)}")
            return []
    
    def improve_resume_section(self, section_text: str, section_type: str) -> str:
        """
        Improve specific resume section (experience, education, skills, etc.)
        """
        prompt = f"""
Improve the following {section_type} section of a resume. Make it more impactful, quantified, and ATS-friendly.

ORIGINAL:
{section_text}

Provide the improved version that:
- Uses strong action verbs
- Includes quantifiable achievements
- Is concise and impactful
- Passes ATS screening
- Follows industry best practices

Return only the improved text without explanations.
"""
        
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Section improvement error: {str(e)}")
            return section_text
    
    def generate_resume_summary(self, resume_text: str, target_role: str) -> str:
        """
        Generate a professional summary/objective for resume
        """
        prompt = f"""
Create a compelling professional summary (2-3 sentences) for this resume targeting a {target_role} position.

RESUME:
{resume_text[:4000]}

The summary should:
- Highlight key qualifications and experience
- Align with the target role
- Be concise and impactful
- Include relevant keywords

Return only the summary text.
"""
        
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return ""
    
    def chatbot_response(self, user_message: str, context: Dict = None) -> str:
        """
        Virtual assistant chatbot for resume and interview guidance
        """
        context_str = ""
        if context:
            if 'resume_summary' in context:
                context_str += f"\nUser's Resume Summary: {context['resume_summary']}"
            if 'target_role' in context:
                context_str += f"\nTarget Role: {context['target_role']}"
        
        prompt = f"""
You are a helpful career guidance assistant specializing in resume building and interview preparation.
{context_str}

User Question: {user_message}

Provide helpful, actionable advice. Be concise but thorough.
"""
        
        try:
            response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Chatbot error: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
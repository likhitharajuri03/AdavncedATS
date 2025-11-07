import re
import spacy
import pandas as pd
from typing import List, Dict, Set
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkillExtractor:
    """
    Advanced skill extraction using NLP and pattern matching
    Extracts both technical and soft skills from resumes
    """
    
    def __init__(self, skills_database_path: str = None):
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Load skills database from dataset
        self.technical_skills = set()
        self.soft_skills = set()
        
        if skills_database_path:
            self._load_skills_from_dataset(skills_database_path)
        else:
            self._initialize_default_skills()
    
    def _load_skills_from_dataset(self, data_path: str):
        """Load skills from the jobs dataset"""
        try:
            df = pd.read_csv(data_path)
            
            # Extract IT Skills
            it_skills = df['IT Skills'].dropna()
            for skills_str in it_skills:
                skills = self._parse_skill_string(skills_str)
                self.technical_skills.update(skills)
            
            # Extract Soft Skills
            soft_skills = df['Soft Skills'].dropna()
            for skills_str in soft_skills:
                skills = self._parse_skill_string(skills_str)
                self.soft_skills.update(skills)
            
            logger.info(f"Loaded {len(self.technical_skills)} technical skills and {len(self.soft_skills)} soft skills")
        except Exception as e:
            logger.error(f"Error loading skills from dataset: {str(e)}")
            self._initialize_default_skills()
    
    def _parse_skill_string(self, skills_str: str) -> Set[str]:
        """Parse comma-separated or otherwise delimited skill strings"""
        if pd.isna(skills_str):
            return set()
        
        # Split by common delimiters
        skills = re.split(r'[,;|/\n]', str(skills_str))
        
        # Clean and normalize
        cleaned_skills = set()
        for skill in skills:
            skill = skill.strip().lower()
            if skill and len(skill) > 1:
                cleaned_skills.add(skill)
        
        return cleaned_skills
    
    def _initialize_default_skills(self):
        """Initialize with comprehensive default skill sets"""
        self.technical_skills = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php',
            'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'perl',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'vue.js', 'node.js', 'express',
            'django', 'flask', 'spring', 'asp.net', 'jquery', 'bootstrap', 'tailwind',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra',
            'dynamodb', 'elasticsearch', 'sqlite', 'mariadb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
            'gitlab', 'ci/cd', 'terraform', 'ansible', 'linux', 'unix',
            
            # Data Science & ML
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'scikit-learn', 'pandas', 'numpy', 'nlp', 'computer vision', 'data mining',
            
            # Mobile Development
            'android', 'ios', 'react native', 'flutter', 'xamarin',
            
            # Tools & Frameworks
            'jira', 'agile', 'scrum', 'rest api', 'graphql', 'microservices',
            'hadoop', 'spark', 'kafka', 'tableau', 'power bi'
        }
        
        self.soft_skills = {
            'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical',
            'critical thinking', 'creativity', 'time management', 'adaptability',
            'collaboration', 'decision making', 'conflict resolution', 'negotiation',
            'presentation', 'interpersonal', 'organizational', 'attention to detail',
            'multitasking', 'strategic thinking', 'emotional intelligence'
        }
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all skills from resume text
        Returns dictionary with 'technical' and 'soft' skill lists
        """
        text_lower = text.lower()
        
        # Extract technical skills
        found_technical = []
        for skill in self.technical_skills:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_technical.append(skill)
        
        # Extract soft skills
        found_soft = []
        for skill in self.soft_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_soft.append(skill)
        
        # Use NER for additional extraction
        doc = self.nlp(text[:100000])  # Limit text length for performance
        
        # Extract entities that might be skills
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'TECH']:
                ent_text = ent.text.lower()
                if ent_text not in found_technical and len(ent_text) > 2:
                    # Check if it matches known patterns
                    if self._is_likely_technical_skill(ent_text):
                        found_technical.append(ent_text)
        
        return {
            'technical': sorted(list(set(found_technical))),
            'soft': sorted(list(set(found_soft)))
        }
    
    def _is_likely_technical_skill(self, text: str) -> bool:
        """Determine if a text fragment is likely a technical skill"""
        tech_indicators = ['js', 'py', '.net', 'db', 'api', 'framework', 'library']
        return any(indicator in text for indicator in tech_indicators)
    
    def extract_experience_years(self, text: str) -> Dict[str, float]:
        """Extract years of experience for different skills/technologies"""
        experience_dict = {}
        
        # Pattern: "X years of Python experience" or "Python (X years)"
        patterns = [
            r'(\d+)[\+]?\s*(?:years?|yrs?).*?(?:of\s+)?([a-z][a-z0-9\+\#\.\s]+?)(?:\s+experience|\s+exp|\,|\.|$)',
            r'([a-z][a-z0-9\+\#\.\s]+?)\s*\(?(\d+)[\+]?\s*(?:years?|yrs?)\)?'
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    if match.group(1).isdigit():
                        years = int(match.group(1))
                        skill = match.group(2).strip()
                    else:
                        skill = match.group(1).strip()
                        years = int(match.group(2))
                    
                    if skill in self.technical_skills and 0 < years < 50:
                        experience_dict[skill] = years
                except:
                    continue
        
        return experience_dict
    
    def match_skills_with_job(self, resume_skills: Dict[str, List[str]], 
                              job_description: str) -> Dict[str, any]:
        """
        Match resume skills with job description requirements
        """
        job_skills = self.extract_skills(job_description)
        
        # Calculate matches
        technical_required = set(job_skills['technical'])
        technical_possessed = set(resume_skills['technical'])
        
        soft_required = set(job_skills['soft'])
        soft_possessed = set(resume_skills['soft'])
        
        technical_match = technical_required.intersection(technical_possessed)
        technical_missing = technical_required - technical_possessed
        
        soft_match = soft_required.intersection(soft_possessed)
        soft_missing = soft_required - soft_possessed
        
        # Calculate match percentages
        tech_match_pct = len(technical_match) / len(technical_required) * 100 if technical_required else 100
        soft_match_pct = len(soft_match) / len(soft_required) * 100 if soft_required else 100
        
        overall_match = (tech_match_pct * 0.7 + soft_match_pct * 0.3)
        
        return {
            'overall_match_percentage': round(overall_match, 2),
            'technical_match_percentage': round(tech_match_pct, 2),
            'soft_match_percentage': round(soft_match_pct, 2),
            'technical_skills_matched': list(technical_match),
            'technical_skills_missing': list(technical_missing),
            'soft_skills_matched': list(soft_match),
            'soft_skills_missing': list(soft_missing),
            'total_skills_required': len(technical_required) + len(soft_required),
            'total_skills_possessed': len(technical_possessed) + len(soft_possessed)
        }
    
    def get_skill_suggestions(self, current_skills: List[str], target_role: str) -> List[str]:
        """Suggest additional skills based on target role"""
        # This would ideally use the dataset to find common skill combinations
        suggestions = []
        
        # Placeholder logic - in production, use ML model or skill graph
        role_lower = target_role.lower()
        
        if 'frontend' in role_lower or 'front-end' in role_lower:
            frontend_skills = ['react', 'vue', 'angular', 'typescript', 'css', 'html', 'javascript']
            suggestions = [s for s in frontend_skills if s not in current_skills]
        elif 'backend' in role_lower or 'back-end' in role_lower:
            backend_skills = ['node.js', 'python', 'java', 'sql', 'mongodb', 'rest api', 'docker']
            suggestions = [s for s in backend_skills if s not in current_skills]
        elif 'data' in role_lower:
            data_skills = ['python', 'sql', 'pandas', 'numpy', 'machine learning', 'tableau']
            suggestions = [s for s in data_skills if s not in current_skills]
        
        return suggestions[:5]  # Return top 5 suggestions
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess_text(text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True, remove_numbers: bool = False) -> str:
    """
    Preprocess text for NLP tasks
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove stop words
        lemmatize: Whether to lemmatize words
        remove_numbers: Whether to remove numbers
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove numbers if specified
    if remove_numbers:
        tokens = [token for token in tokens if not token.isdigit()]
    
    # Remove stopwords if specified
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # Keep important words that are usually stopwords in resume context
        keep_words = {'experience', 'years', 'work', 'skills', 'education'}
        stop_words = stop_words - keep_words
        tokens = [token for token in tokens if token not in stop_words]
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Lemmatize if specified
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join back to string
    processed_text = ' '.join(tokens)
    
    return processed_text


def extract_contact_info(text: str) -> dict:
    """Extract contact information from resume text"""
    contact_info = {
        'emails': [],
        'phones': [],
        'linkedin': None,
        'github': None
    }
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    contact_info['emails'] = re.findall(email_pattern, text)
    
    # Extract phone numbers
    phone_patterns = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',
        r'\b\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        contact_info['phones'].extend(phones)
    
    # Extract LinkedIn
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
    if linkedin_match:
        contact_info['linkedin'] = linkedin_match.group()
    
    # Extract GitHub
    github_pattern = r'github\.com/[\w-]+'
    github_match = re.search(github_pattern, text, re.IGNORECASE)
    if github_match:
        contact_info['github'] = github_match.group()
    
    return contact_info


def extract_education(text: str) -> list:
    """Extract education information from resume"""
    education = []
    
    # Common degree patterns
    degree_patterns = [
        r'\b(bachelor|master|phd|doctorate|associate|diploma|mba|btech|mtech|bsc|msc|ba|ma|be|me)\b',
        r'\b(b\.tech|m\.tech|b\.sc|m\.sc|b\.a|m\.a|b\.e|m\.e)\b'
    ]
    
    text_lower = text.lower()
    for pattern in degree_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            # Get context around the match (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            education.append(context.strip())
    
    return list(set(education))


def extract_experience_years(text: str) -> float:
    """Extract total years of experience from resume"""
    # Patterns for experience mentions
    patterns = [
        r'(\d+)[\+]?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        r'experience\s*:\s*(\d+)[\+]?\s*(?:years?|yrs?)',
        r'(\d+)[\+]?\s*(?:years?|yrs?)\s+in\s+(?:the\s+)?field'
    ]
    
    max_years = 0
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            years = int(match)
            if years > max_years and years < 50:  # Sanity check
                max_years = years
    
    return float(max_years)


def clean_resume_text(text: str) -> str:
    """Clean resume text while preserving important information"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers
    text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove common headers/footers
    text = re.sub(r'resume|curriculum vitae|cv', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def tokenize_for_matching(text: str) -> list:
    """
    Tokenize text specifically for resume-job matching
    Preserves important technical terms
    """
    # Don't lowercase to preserve acronyms like SQL, AWS
    text = clean_resume_text(text)
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation but keep + and # for C++, C#, etc.
    tokens = [token for token in tokens if token not in string.punctuation or token in ['+', '#']]
    
    # Remove very short tokens
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens
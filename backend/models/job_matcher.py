import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobMatcher:
    """
    Matches candidate skills with jobs from dataset
    Validates AI recommendations with real job data
    Enhanced for IT Skills + Soft Skills matching
    """
    
    def __init__(self, dataset_path='data/jobs_dataset_processed.csv'):
        self.dataset_path = dataset_path
        self.jobs_df = None
        self.vectorizer = TfidfVectorizer()
        self.load_dataset()
    
    def load_dataset(self):
        """
        Load job dataset from CSV
        """
        try:
            if os.path.exists(self.dataset_path):
                self.jobs_df = pd.read_csv(self.dataset_path)
                print(f"✅ Loaded {len(self.jobs_df)} jobs from dataset")
                print(f"📊 Columns: {list(self.jobs_df.columns)}")
            else:
                print(f"⚠️ Dataset not found at {self.dataset_path}")
                self.create_directory_and_save_sample()
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            self.create_directory_and_save_sample()
    
    def create_directory_and_save_sample(self):
        """
        Create data directory and save sample dataset
        """
        os.makedirs('data', exist_ok=True)
        
        # Sample data matching your format
        sample_data = {
            'Job Title': [
                'Software Engineer', 'Software Engineer', 'Software Engineer',
                'Data Scientist', 'Data Scientist', 'Data Scientist',
                'Full Stack Developer', 'Full Stack Developer', 'Full Stack Developer',
                'DevOps Engineer', 'DevOps Engineer', 'DevOps Engineer',
                'ML Engineer', 'ML Engineer', 'ML Engineer',
                'Frontend Developer', 'Frontend Developer', 'Frontend Developer',
                'Backend Developer', 'Backend Developer', 'Backend Developer'
            ],
            'Description': [
                'Develop and maintain web applications using modern technologies',
                'Build scalable software solutions for enterprise clients',
                'Create microservices and cloud-native applications',
                'Analyze data and build machine learning models',
                'Develop predictive models and analyze complex datasets',
                'Create ML models for business insights',
                'Build end-to-end web applications',
                'Develop responsive web applications',
                'Create modern web solutions',
                'Manage cloud infrastructure and CI/CD pipelines',
                'Implement and maintain CI/CD workflows',
                'Automate deployment processes',
                'Develop and deploy machine learning models',
                'Build and optimize ML models',
                'Create AI solutions for business problems',
                'Create responsive web interfaces',
                'Build modern web applications',
                'Develop user-friendly interfaces',
                'Design and implement server-side applications',
                'Build scalable APIs and services',
                'Create microservices architecture'
            ],
            'IT Skills': [
                'Python,JavaScript,React,Node.js,SQL',
                'Java,Spring,SQL,Docker,Git',
                'Python,Go,AWS,Kubernetes,Docker',
                'Python,R,SQL,Machine Learning,TensorFlow',
                'Python,Pandas,Scikit-learn,SQL,Spark',
                'Python,TensorFlow,PyTorch,SQL,Statistics',
                'JavaScript,Python,React,Node.js,MongoDB',
                'JavaScript,TypeScript,Angular,Node.js,SQL',
                'JavaScript,Vue.js,Python,Django,PostgreSQL',
                'AWS,Docker,Kubernetes,Jenkins,Linux',
                'Azure,Terraform,Docker,Ansible,Linux',
                'GCP,Docker,Kubernetes,GitLab,Python',
                'Python,TensorFlow,PyTorch,SQL',
                'Python,Keras,Scikit-learn,MLflow',
                'Python,TensorFlow,PyTorch,Kubernetes',
                'HTML,CSS,JavaScript,React',
                'TypeScript,React,Redux,SASS',
                'JavaScript,Vue.js,CSS,Webpack',
                'Python,Java,Node.js,SQL',
                'Go,gRPC,PostgreSQL,Redis',
                'Java,Spring Boot,MongoDB,RabbitMQ'
            ],
            'Soft Skills': [
                'Communication,Problem Solving,Teamwork',
                'Leadership,Problem Solving,Communication',
                'Critical Thinking,Adaptability,Teamwork',
                'Analytical Thinking,Research,Attention to Detail',
                'Data Analysis,Problem Solving,Communication',
                'Research,Mathematical Skills,Teamwork',
                'Teamwork,Time Management,Communication',
                'Problem Solving,Organization,Communication',
                'Adaptability,Technical Skills,Collaboration',
                'Problem Solving,Organization,Collaboration',
                'Communication,Automation Skills,Teamwork',
                'Infrastructure Design,Troubleshooting,Leadership',
                'Critical Thinking,Research,Communication',
                'Algorithm Design,Problem Solving,Teamwork',
                'Research,Mathematical Skills,Communication',
                'Creativity,Attention to Detail,Teamwork',
                'UI/UX Skills,Communication,Organization',
                'Design Skills,Problem Solving,Teamwork',
                'Problem Solving,Communication,Organization',
                'System Design,Technical Skills,Teamwork',
                'Architecture Design,Communication,Leadership'
            ]
        }
        
        self.jobs_df = pd.DataFrame(sample_data)
        self.jobs_df.to_csv(self.dataset_path, index=False)
        print(f"✅ Created sample dataset with {len(self.jobs_df)} jobs at {self.dataset_path}")
    
    def find_matching_jobs(self, candidate_skills, top_n=10):
        """
        Find jobs that match candidate's skills using advanced matching
        """
        if self.jobs_df is None or len(self.jobs_df) == 0:
            return []
        
        # Convert skills to lowercase for matching
        candidate_skills_lower = [skill.lower().strip() for skill in candidate_skills]
        
        matches = []
        
        for idx, row in self.jobs_df.iterrows():
            # Parse IT skills and soft skills
            it_skills = [s.strip().lower() for s in str(row['IT Skills']).split(',')]
            soft_skills = [s.strip().lower() for s in str(row['Soft Skills']).split(',')]
            all_job_skills = it_skills + soft_skills
            
            # Calculate overlap for IT skills
            matching_it_skills = set(candidate_skills_lower) & set(it_skills)
            it_match_score = len(matching_it_skills) / len(it_skills) if it_skills else 0
            
            # Calculate overlap for soft skills
            matching_soft_skills = set(candidate_skills_lower) & set(soft_skills)
            soft_match_score = len(matching_soft_skills) / len(soft_skills) if soft_skills else 0
            
            # Overall match (weighted: IT 70%, Soft 30%)
            overall_match = (it_match_score * 0.7) + (soft_match_score * 0.3)
            
            # Calculate semantic similarity using job description
            text_similarity = self._calculate_text_similarity(
                ' '.join(candidate_skills_lower),
                row['Description'].lower() + ' ' + ' '.join(all_job_skills)
            )
            
            # Final score (combine rule-based and semantic)
            final_score = (overall_match * 0.6) + (text_similarity * 0.4)
            
            if final_score > 0.1:  # Minimum threshold
                matches.append({
                    'title': row['Job Title'],
                    'description': row['Description'],
                    'it_skills': row['IT Skills'],
                    'soft_skills': row['Soft Skills'],
                    'match_score': final_score,
                    'it_match_score': it_match_score,
                    'soft_match_score': soft_match_score,
                    'matching_it_skills': list(matching_it_skills),
                    'matching_soft_skills': list(matching_soft_skills),
                    'missing_it_skills': list(set(it_skills) - set(candidate_skills_lower)),
                    'missing_soft_skills': list(set(soft_skills) - set(candidate_skills_lower))
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matches[:top_n]
    
    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts
        """
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def get_skill_demand_analysis(self):
        """
        Analyze which skills are most in demand
        """
        if self.jobs_df is None:
            return {}
        
        skill_counts = {}
        
        # Count IT skills
        for skills_str in self.jobs_df['IT Skills']:
            skills = [s.strip().lower() for s in str(skills_str).split(',')]
            for skill in skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Count soft skills
        for skills_str in self.jobs_df['Soft Skills']:
            skills = [s.strip().lower() for s in str(skills_str).split(',')]
            for skill in skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Sort by frequency
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_unique_skills': len(skill_counts),
            'top_20_skills': sorted_skills[:20],
            'all_skills': sorted_skills
        }
    
    def get_job_title_distribution(self):
        """
        Get distribution of job titles
        """
        if self.jobs_df is None:
            return {}
        
        title_counts = self.jobs_df['Job Title'].value_counts().to_dict()
        return title_counts
    
    def get_skills_gap_report(self, candidate_skills, target_job_title):
        """
        Generate comprehensive skills gap report
        """
        if self.jobs_df is None:
            return None
        
        # Filter jobs by title
        target_jobs = self.jobs_df[self.jobs_df['Job Title'].str.lower() == target_job_title.lower()]
        
        if target_jobs.empty:
            return None
        
        candidate_skills_lower = [s.lower().strip() for s in candidate_skills]
        
        # Aggregate all skills for this job title
        all_it_skills = set()
        all_soft_skills = set()
        
        for _, job in target_jobs.iterrows():
            it_skills = [s.strip().lower() for s in str(job['IT Skills']).split(',')]
            soft_skills = [s.strip().lower() for s in str(job['Soft Skills']).split(',')]
            all_it_skills.update(it_skills)
            all_soft_skills.update(soft_skills)
        
        # Calculate gaps
        missing_it_skills = all_it_skills - set(candidate_skills_lower)
        missing_soft_skills = all_soft_skills - set(candidate_skills_lower)
        
        matching_it_skills = all_it_skills & set(candidate_skills_lower)
        matching_soft_skills = all_soft_skills & set(candidate_skills_lower)
        
        return {
            'job_title': target_job_title,
            'total_postings': len(target_jobs),
            'required_it_skills': list(all_it_skills),
            'required_soft_skills': list(all_soft_skills),
            'matching_it_skills': list(matching_it_skills),
            'matching_soft_skills': list(matching_soft_skills),
            'missing_it_skills': list(missing_it_skills),
            'missing_soft_skills': list(missing_soft_skills),
            'it_coverage': len(matching_it_skills) / len(all_it_skills) if all_it_skills else 0,
            'soft_coverage': len(matching_soft_skills) / len(all_soft_skills) if all_soft_skills else 0
        }
    
    def recommend_learning_path(self, current_skills, target_job_title):
        """
        Recommend learning path based on skills gap
        """
        gap_report = self.get_skills_gap_report(current_skills, target_job_title)
        
        if not gap_report:
            return None
        
        # Prioritize skills to learn
        skill_demand = self.get_skill_demand_analysis()
        
        # Score missing skills by demand
        missing_skills_scored = []
        for skill in gap_report['missing_it_skills']:
            demand = next((count for s, count in skill_demand['all_skills'] if s == skill), 0)
            missing_skills_scored.append((skill, demand))
        
        missing_skills_scored.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'priority_skills': [skill for skill, _ in missing_skills_scored[:5]],
            'soft_skills_needed': gap_report['missing_soft_skills'][:3],
            'current_coverage': {
                'it_skills': f"{gap_report['it_coverage']:.1%}",
                'soft_skills': f"{gap_report['soft_coverage']:.1%}"
            }
        }
    
    def search_jobs_by_keyword(self, keyword):
        """
        Search jobs by keyword in title or description
        """
        if self.jobs_df is None:
            return []
        
        keyword_lower = keyword.lower()
        
        matches = self.jobs_df[
            self.jobs_df['Job Title'].str.lower().str.contains(keyword_lower, na=False) |
            self.jobs_df['Description'].str.lower().str.contains(keyword_lower, na=False) |
            self.jobs_df['IT Skills'].str.lower().str.contains(keyword_lower, na=False)
        ]
        
        return matches.to_dict('records')
    
    def get_dataset_statistics(self):
        """
        Get comprehensive statistics about the dataset
        """
        if self.jobs_df is None or len(self.jobs_df) == 0:
            return {}
        
        skill_analysis = self.get_skill_demand_analysis()
        title_dist = self.get_job_title_distribution()
        
        return {
            'total_jobs': len(self.jobs_df),
            'unique_job_titles': len(title_dist),
            'job_title_distribution': title_dist,
            'total_unique_skills': skill_analysis['total_unique_skills'],
            'most_demanded_skills': [skill for skill, _ in skill_analysis['top_20_skills'][:10]],
            'average_skills_per_job': (
                self.jobs_df['IT Skills'].str.count(',').mean() + 
                self.jobs_df['Soft Skills'].str.count(',').mean() + 2
            )
        }
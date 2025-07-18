
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv
import zipfile
import io
import base64
import re
from collections import defaultdict, Counter
import ast
import tokenize
from io import StringIO
import subprocess
import tempfile
import shutil

# Load environment variables
load_dotenv()

# WatsonX configuration
WATSONX_AUTH_URL = os.getenv("WATSONX_AUTH_URL")
WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")


# GitHub configuration - Multiple tokens for different users
GITHUB_TOKENS = {
    "nithiyanandham12": os.getenv("GITHUB_TOKEN_NITHIYANANDHAM12"),
    "sbasadeesh": os.getenv("GITHUB_TOKEN_SBASADEESH"), 
    "Srihari-bot": os.getenv("GITHUB_TOKEN_SRIHARI")
}

# Mapping from username to environment variable name for error messages
USERNAME_TO_ENV_VAR = {
    "nithiyanandham12": "GITHUB_TOKEN_NITHIYANANDHAM12",
    "sbasadeesh": "GITHUB_TOKEN_SBASADEESH",
    "Srihari-bot": "GITHUB_TOKEN_SRIHARI"
}

# Predefined GitHub usernames for dropdown
PREDEFINED_USERS = [
    {"value": "nithiyanandham12", "label": "nithiyanandham12"},
    {"value": "sbasadeesh", "label": "sbasadeesh"},
    {"value": "Srihari-bot", "label": "Srihari-bot"}
]

try:
    WATSONX_PARAMETERS = json.loads(os.getenv("WATSONX_PARAMETERS", "{}"))
except json.JSONDecodeError as e:
    st.error(f"Error parsing WATSONX_PARAMETERS: {e}")
    WATSONX_PARAMETERS = {"max_new_tokens": 512, "temperature": 0.7}

# Domain classification keywords
DOMAIN_KEYWORDS = {
    'construction': [
        'construction', 'building', 'contractor', 'architecture', 'civil', 'structural',
        'blueprint', 'CAD', 'engineering', 'project management', 'site management',
        'materials', 'concrete', 'steel', 'foundation', 'renovation', 'infrastructure',
        'crane', 'excavator', 'heavy machinery', 'safety compliance', 'permits'
    ],
    'healthcare': [
        'hospital', 'medical', 'health', 'patient', 'doctor', 'nurse', 'clinic',
        'pharmacy', 'medication', 'diagnosis', 'treatment', 'EHR', 'EMR',
        'healthcare', 'radiology', 'laboratory', 'surgery', 'appointment',
        'medical records', 'HIPAA', 'telemedicine', 'prescription'
    ],
    'finance': [
        'bank', 'finance', 'accounting', 'payment', 'transaction', 'invoice',
        'billing', 'budget', 'financial', 'money', 'currency', 'investment',
        'trading', 'portfolio', 'loan', 'credit', 'debit', 'financial planning',
        'taxation', 'audit', 'compliance', 'risk management'
    ],
    'education': [
        'school', 'education', 'student', 'teacher', 'course', 'learning',
        'curriculum', 'grade', 'exam', 'assessment', 'university', 'college',
        'classroom', 'e-learning', 'LMS', 'academic', 'enrollment', 'degree',
        'certification', 'training', 'workshop', 'tutorial'
    ],
    'ecommerce': [
        'shop', 'store', 'ecommerce', 'retail', 'product', 'cart', 'checkout',
        'payment', 'order', 'inventory', 'catalog', 'marketplace', 'vendor',
        'customer', 'sales', 'discount', 'coupon', 'shipping', 'delivery',
        'warehouse', 'logistics', 'supply chain'
    ],
    'gaming': [
        'game', 'gaming', 'player', 'level', 'score', 'character', 'weapon',
        'unity', 'unreal', 'engine', 'graphics', 'animation', 'physics',
        'multiplayer', 'leaderboard', 'achievement', 'quest', 'RPG',
        'simulation', 'arcade', 'mobile game', 'console'
    ],
    'social_media': [
        'social', 'media', 'post', 'comment', 'like', 'share', 'follow',
        'friend', 'timeline', 'feed', 'chat', 'message', 'notification',
        'profile', 'community', 'forum', 'discussion', 'content',
        'influencer', 'engagement', 'viral', 'trend'
    ],
    'data_analytics': [
        'analytics', 'data', 'dashboard', 'visualization', 'chart', 'graph',
        'report', 'metrics', 'KPI', 'statistics', 'machine learning', 'AI',
        'algorithm', 'model', 'prediction', 'classification', 'regression',
        'clustering', 'neural network', 'deep learning', 'big data'
    ]
}

# File type patterns for different functionalities
FILE_PATTERNS = {
    'frontend': ['.html', '.css', '.js', '.jsx', '.ts', '.tsx', '.vue', '.angular'],
    'backend': ['.py', '.java', '.php', '.rb', '.go', '.cpp', '.c', '.cs', '.rs'],
    'database': ['.sql', '.db', '.sqlite', '.mongodb', '.redis'],
    'config': ['.json', '.yaml', '.yml', '.xml', '.ini', '.cfg', '.env'],
    'documentation': ['.md', '.txt', '.rst', '.pdf', '.doc'],
    'mobile': ['.swift', '.kt', '.dart', '.xaml'],
    'devops': ['dockerfile', '.sh', '.bat', '.ps1', 'jenkins', '.terraform']
}

# Algorithm patterns for code understanding
ALGORITHM_PATTERNS = {
    'sorting': ['sort', 'bubble', 'quick', 'merge', 'heap', 'radix', 'counting'],
    'searching': ['binary', 'linear', 'depth', 'breadth', 'dijkstra', 'astar'],
    'machine_learning': ['train', 'predict', 'classify', 'regression', 'clustering', 'neural'],
    'optimization': ['optimize', 'minimize', 'maximize', 'gradient', 'genetic', 'simulated'],
    'data_structures': ['tree', 'graph', 'linked_list', 'stack', 'queue', 'heap', 'hash']
}

# Page configuration
st.set_page_config(
    page_title="GitHub RAG Explorer",
    page_icon="🐙",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.3);
    }
    .repo-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        padding: 16px;
        margin: 8px 0;
    }
    .repo-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #58a6ff;
        text-decoration: none;
    }
    .repo-description {
        color: #c9d1d9;
        margin: 8px 0;
    }
    .repo-stats {
        display: flex;
        gap: 16px;
        font-size: 0.875rem;
        color: #8b949e;
        flex-wrap: wrap;
    }
    .stat-item {
        display: flex;
        align-items: center;
        gap: 4px;
        color: #8b949e;
    }
    .sidebar-section {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .setup-section {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .chat-input {
        position: sticky;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
    }
    .user-message {
        background-color: rgba(0, 123, 255, 0.1);
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
    }
    .download-section {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .file-item {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 4px;
        padding: 8px;
        margin: 4px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'repos_data' not in st.session_state:
    st.session_state.repos_data = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'file_vector_db' not in st.session_state:
    st.session_state.file_vector_db = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'current_username' not in st.session_state:
    st.session_state.current_username = None
if 'repository_analysis' not in st.session_state:
    st.session_state.repository_analysis = {}
if 'file_contents_cache' not in st.session_state:
    st.session_state.file_contents_cache = {}
if 'api_requests_count' not in st.session_state:
    st.session_state.api_requests_count = 0
if 'file_vector_db' not in st.session_state:
    st.session_state.file_vector_db = None

@st.cache_data(ttl=3600)  # Cache for 1 hour (tokens typically expire after 1 hour)
def get_access_token(api_key):
    """Get Watson access token with automatic refresh"""
    try:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key
        }
        response = requests.post(WATSONX_AUTH_URL, headers=headers, data=data)
        response.raise_for_status()
        token_info = response.json()
        return token_info['access_token']
    except requests.exceptions.RequestException as e:
        st.error(f"Error obtaining access token: {e}")
        raise

def generate_watson_response(prompt, access_token):
    """Generate response using Watson model with token refresh on expiry"""
    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "parameters": WATSONX_PARAMETERS,
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    try:
        response = requests.post(WATSONX_API_URL, headers=headers, json=body)
        
        # Check for authentication error and retry with fresh token
        if response.status_code == 401:
            try:
                # Clear the cached token and get a fresh one
                get_access_token.clear()
                fresh_token = get_access_token(API_KEY)
                
                # Retry with fresh token
                headers["Authorization"] = f"Bearer {fresh_token}"
                response = requests.post(WATSONX_API_URL, headers=headers, json=body)
                response.raise_for_status()
            except Exception as refresh_error:
                st.error(f"Failed to refresh token: {refresh_error}")
                return None
        
        elif response.status_code != 200:
            error_details = response.text
            st.error(f"Watson API Error Details: {error_details}")
            return None
        else:
            response.raise_for_status()
        
        data = response.json()
        
        # Handle chat completion response format
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content'].strip()
        elif 'results' in data and len(data['results']) > 0:
            # Fallback for older response format
            return data['results'][0]['generated_text'].strip()
        else:
            st.error("No results returned from Watson API.")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error generating response: {e}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Response content: {e.response.text}")
        return None

@st.cache_resource
def load_embeddings_model():
    """Load sentence transformer model for embeddings"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embeddings model: {e}")
        return None

def initialize_vector_db():
    """Initialize ChromaDB vector database"""
    try:
        client = chromadb.Client()
        
        # Check if collection already exists and delete it to start fresh
        try:
            existing_collection = client.get_collection("github_repos")
            client.delete_collection("github_repos")
        except:
            pass  # Collection doesn't exist, which is fine
        
        # Create new collection
        collection = client.create_collection(
            name="github_repos",
            metadata={"description": "GitHub repositories collection"}
        )
        return collection
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        return None

def process_repository_content(repo):
    """Process repository data for embedding"""
    content_parts = []
    
    # Basic repository information
    content_parts.append(f"Repository: {repo['name']}")
    content_parts.append(f"Description: {repo.get('description', 'No description')}")
    content_parts.append(f"Language: {repo.get('language', 'Unknown')}")
    content_parts.append(f"Stars: {repo.get('stargazers_count', 0)}")
    content_parts.append(f"Forks: {repo.get('forks_count', 0)}")
    content_parts.append(f"Size: {repo.get('size', 0)} KB")
    content_parts.append(f"Created: {format_date(repo.get('created_at', ''))}")
    content_parts.append(f"Updated: {format_date(repo.get('updated_at', ''))}")
    
    # Topics if available
    if repo.get('topics'):
        content_parts.append(f"Topics: {', '.join(repo['topics'])}")
    
    # License information
    if repo.get('license'):
        content_parts.append(f"License: {repo['license'].get('name', 'Unknown')}")
    
    # Additional metadata
    if repo.get('homepage'):
        content_parts.append(f"Homepage: {repo['homepage']}")
    
    content_parts.append(f"Default branch: {repo.get('default_branch', 'main')}")
    content_parts.append(f"Open issues: {repo.get('open_issues_count', 0)}")
    content_parts.append(f"Watchers: {repo.get('watchers_count', 0)}")
    
    return " | ".join(content_parts)

def embed_repositories(repos, model, vector_db):
    """Embed repository content and store in vector database"""
    try:
        documents = []
        metadatas = []
        ids = []
        
        for i, repo in enumerate(repos):
            content = process_repository_content(repo)
            documents.append(content)
            
            # Store repository metadata with proper handling of None values
            metadata = {
                "name": repo.get('name', 'Unknown'),
                "description": repo.get('description') or 'No description',
                "language": repo.get('language') or 'Unknown',
                "stars": int(repo.get('stargazers_count', 0)),
                "forks": int(repo.get('forks_count', 0)),
                "url": repo.get('html_url', ''),
                "created_at": repo.get('created_at', ''),
                "updated_at": repo.get('updated_at', ''),
                "size": int(repo.get('size', 0)),
                "open_issues": int(repo.get('open_issues_count', 0)),
                "watchers": int(repo.get('watchers_count', 0)),
                "default_branch": repo.get('default_branch') or 'main',
                "is_fork": bool(repo.get('fork', False)),
                "is_private": bool(repo.get('private', False))
            }
            
            # Ensure all values are not None and of correct types
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, (int, float, str, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)
                else:
                    # Provide defaults for None values based on expected type
                    if key in ['stars', 'forks', 'size', 'open_issues', 'watchers']:
                        cleaned_metadata[key] = 0
                    elif key in ['is_fork', 'is_private']:
                        cleaned_metadata[key] = False
                    else:
                        cleaned_metadata[key] = ''
            
            metadatas.append(cleaned_metadata)
            ids.append(f"repo_{i}")
        
        # Generate embeddings
        embeddings = model.encode(documents)
        
        # Store in vector database
        vector_db.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return True
    except Exception as e:
        st.error(f"Error embedding repositories: {e}")
        return False

def search_repositories(query, model, vector_db, top_k=5):
    """Search repositories using semantic similarity"""
    try:
        # Generate query embedding
        query_embedding = model.encode([query])
        
        # Search in vector database
        results = vector_db.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return results
    except Exception as e:
        st.error(f"Error searching repositories: {e}")
        return None

def create_rag_prompt(question, context_repos):
    """Create RAG prompt with context and question"""
    context = "\n".join([
        f"Repository: {repo['name']}\n"
        f"Description: {repo['description']}\n"
        f"Language: {repo['language']}\n"
        f"Stars: {repo['stars']}\n"
        f"URL: {repo['url']}\n"
        for repo in context_repos
    ])
    
    prompt = f"""Based on the following GitHub repository information, please answer the question:

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the repository information above. If the question cannot be answered from the provided context, please indicate that clearly.

ANSWER:"""
    
    return prompt

def fetch_user_repositories(username, token=None):
    """Fetch repositories for a specific user using GitHub API"""
    try:
        url = f"https://api.github.com/users/{username}/repos"
        headers = {}
        
        if token:
            headers['Authorization'] = f'token {token}'
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            repos = response.json()
            return repos, None
        else:
            error_msg = f"Error {response.status_code}: {response.json().get('message', 'Unknown error')}"
            return [], error_msg
            
    except Exception as e:
        return [], str(e)

def fetch_user_info(username, token=None):
    """Fetch user information using GitHub API"""
    try:
        url = f"https://api.github.com/users/{username}"
        headers = {}
        
        if token:
            headers['Authorization'] = f'token {token}'
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_msg = f"Error {response.status_code}: {response.json().get('message', 'Unknown error')}"
            return None, error_msg
            
    except Exception as e:
        return None, str(e)

def format_date(date_string):
    """Format ISO date string to readable format"""
    try:
        date_obj = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return date_obj.strftime('%Y-%m-%d')
    except:
        return date_string

def download_repository_zip(username, repo_name, token=None):
    """Download repository as ZIP file with better error handling"""
    try:
        # Track API request count for rate limiting
        if not token:
            st.session_state.api_requests_count += 1
        
        url = f"https://api.github.com/repos/{username}/{repo_name}/zipball"
        headers = {
            'User-Agent': 'GitHub-RAG-Explorer/1.0'
        }
        
        if token:
            headers['Authorization'] = f'token {token}'
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 200:
            return response.content, None
        elif response.status_code == 403:
            # Rate limit exceeded
            return None, "Rate limit exceeded. Please wait a moment and try again."
        elif response.status_code == 404:
            return None, "Repository not found or access denied."
        elif response.status_code == 401:
            return None, "Authentication required. Please check your GitHub token."
        else:
            error_msg = f"Error {response.status_code}: {response.json().get('message', 'Could not download repository')}"
            return None, error_msg
            
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Please check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def get_repository_contents(username, repo_name, path="", token=None):
    """Get repository contents/files with better error handling"""
    try:
        # Track API request count for rate limiting
        if not token:
            st.session_state.api_requests_count += 1
        
        url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{path}"
        headers = {
            'User-Agent': 'GitHub-RAG-Explorer/1.0'
        }
        
        if token:
            headers['Authorization'] = f'token {token}'
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 403:
            # Rate limit exceeded
            return None, "Rate limit exceeded. Please wait a moment and try again."
        elif response.status_code == 404:
            return None, "Repository not found or access denied."
        elif response.status_code == 401:
            return None, "Authentication required. Please check your GitHub token."
        else:
            error_msg = f"Error {response.status_code}: {response.json().get('message', 'Could not fetch repository contents')}"
            return None, error_msg
            
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Please check your internet connection."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def download_file_content(username, repo_name, file_path, token=None):
    """Download specific file content with better error handling"""
    try:
        url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{file_path}"
        headers = {
            'User-Agent': 'GitHub-RAG-Explorer/1.0'
        }
        
        if token:
            headers['Authorization'] = f'token {token}'
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            file_data = response.json()
            if file_data.get('content'):
                # Decode base64 content
                content = base64.b64decode(file_data['content']).decode('utf-8')
                return content, file_data.get('name', 'file.txt'), None
            else:
                return None, None, "File content not available"
        elif response.status_code == 403:
            return None, None, "Rate limit exceeded. Please wait a moment and try again."
        elif response.status_code == 404:
            return None, None, "File not found or access denied."
        elif response.status_code == 401:
            return None, None, "Authentication required. Please check your GitHub token."
        else:
            error_msg = f"Error {response.status_code}: {response.json().get('message', 'Could not download file')}"
            return None, None, error_msg
            
    except requests.exceptions.Timeout:
        return None, None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, None, "Connection error. Please check your internet connection."
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"

def display_repository_card(repo):
    """Display a repository as a card"""
    description = repo.get('description') or "No description available"
    language = repo.get('language') or "Unknown"
    updated_at = format_date(repo.get('updated_at', ''))
    
    with st.container():
        st.markdown(f"""
        <div class="repo-card">
            <div class="repo-title">
                <a href="{repo['html_url']}" target="_blank" style="text-decoration: none; color: #0366d6;">
                    📁 {repo['name']}
                </a>
            </div>
            <div class="repo-description">{description}</div>
            <div class="repo-stats">
                <span class="stat-item">⭐ {repo.get('stargazers_count', 0)}</span>
                <span class="stat-item">🍴 {repo.get('forks_count', 0)}</span>
                <span class="stat-item">👁️ {repo.get('watchers_count', 0)}</span>
                <span class="stat-item">💻 {language}</span>
                <span class="stat-item">📅 Updated: {updated_at}</span>
                <span class="stat-item">📏 Size: {repo.get('size', 0)} KB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_user_stats(user_info, repos):
    """Display user statistics"""
    if user_info and repos:
        total_stars = sum(repo.get('stargazers_count', 0) for repo in repos)
        total_forks = sum(repo.get('forks_count', 0) for repo in repos)
        languages = [repo.get('language') for repo in repos if repo.get('language')]
        
        # User info section
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Repositories", len(repos))
        with col2:
            st.metric("Total Stars", total_stars)
        with col3:
            st.metric("Total Forks", total_forks)
        with col4:
            st.metric("Public Repos", user_info.get('public_repos', len(repos)))
        
        # User profile info
        st.subheader(f"👤 Profile: {user_info.get('name', user_info.get('login', 'Unknown'))}")
        
        profile_col1, profile_col2 = st.columns([1, 2])
        with profile_col1:
            if user_info.get('avatar_url'):
                st.image(user_info['avatar_url'], width=150)
        with profile_col2:
            if user_info.get('bio'):
                st.write(f"**Bio:** {user_info['bio']}")
            if user_info.get('location'):
                st.write(f"**Location:** {user_info['location']}")
            if user_info.get('company'):
                st.write(f"**Company:** {user_info['company']}")
            if user_info.get('blog'):
                st.write(f"**Website:** {user_info['blog']}")
            st.write(f"**Followers:** {user_info.get('followers', 0)}")
            st.write(f"**Following:** {user_info.get('following', 0)}")
        
        # Language distribution
        if languages:
            language_counts = pd.Series(languages).value_counts()
            if not language_counts.empty:
                st.subheader("🔧 Language Distribution")
                fig = px.bar(
                    x=language_counts.index[:10],
                    y=language_counts.values[:10],
                    labels={'x': 'Programming Language', 'y': 'Number of Repositories'},
                    title="Top 10 Programming Languages"
                )
                st.plotly_chart(fig, use_container_width=True)

def classify_domain(text):
    """Classify text into domain categories based on keywords"""
    text_lower = text.lower()
    domain_scores = {}
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            domain_scores[domain] = score
    
    return domain_scores

def analyze_file_content(content, filename):
    """Analyze file content to understand its purpose and domain"""
    # Use enhanced analysis for code files
    if filename.lower().endswith(('.py', '.js', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs')):
        return enhanced_analyze_file_content(content, filename)
    else:
        # Basic analysis for non-code files
        analysis = {
            'file_type': get_file_category(filename),
            'domains': classify_domain(content),
            'functionality': extract_functionality_keywords(content),
            'size': len(content),
            'lines': content.count('\n') + 1 if content else 0,
            'code_structure': {'functions': [], 'classes': [], 'logic_flow': [], 'algorithms': []},
            'complexity_score': 0,
            'main_purpose': '',
            'key_algorithms': [],
            'data_flow': []
        }
        return analysis

def get_file_category(filename):
    """Categorize file by extension"""
    for category, extensions in FILE_PATTERNS.items():
        if any(filename.lower().endswith(ext) for ext in extensions):
            return category
    return 'other'

def extract_functionality_keywords(content):
    """Extract functionality-related keywords from code content"""
    # Common functionality patterns
    patterns = {
        'authentication': r'(auth|login|password|token|session|jwt|oauth)',
        'database': r'(database|db|query|select|insert|update|delete|sql)',
        'api': r'(api|endpoint|route|request|response|http|rest|graphql)',
        'ui': r'(button|form|input|component|render|display|interface)',
        'payment': r'(payment|billing|invoice|transaction|stripe|paypal)',
        'file_management': r'(file|upload|download|storage|filesystem)',
        'notification': r'(notification|email|sms|alert|message)',
        'search': r'(search|filter|query|find|index)',
        'validation': r'(validate|validation|verify|check|validate)',
        'security': r'(security|encrypt|decrypt|hash|ssl|tls)',
        'logging': r'(log|logger|debug|trace|error|warning)',
        'testing': r'(test|spec|mock|assert|unittest|pytest)'
    }
    
    found_functionalities = {}
    content_lower = content.lower()
    
    for func, pattern in patterns.items():
        matches = len(re.findall(pattern, content_lower))
        if matches > 0:
            found_functionalities[func] = matches
    
    return found_functionalities

def fetch_repository_files(username, repo_name, token=None, max_files=50):
    """Fetch repository files and their contents for analysis"""
    try:
        files_data = []
        
        # Get repository tree to find all files
        contents, error = get_repository_contents(username, repo_name, "", token)
        if error or not contents:
            return files_data, error
        
        files_to_analyze = []
        
        # Prioritize important files
        priority_files = ['README.md', 'package.json', 'requirements.txt', 'pom.xml', 'Cargo.toml']
        
        # Collect files to analyze
        for item in contents:
            if item['type'] == 'file':
                # Always include priority files
                if item['name'] in priority_files:
                    files_to_analyze.insert(0, item)
                # Include code files (skip very large files)
                elif item.get('size', 0) < 100000:  # Skip files > 100KB
                    files_to_analyze.append(item)
        
        # Limit number of files to analyze
        files_to_analyze = files_to_analyze[:max_files]
        
        for file_item in files_to_analyze:
            try:
                content, filename, error = download_file_content(username, repo_name, file_item['path'], token)
                if content and not error:
                    analysis = analyze_file_content(content, filename)
                    files_data.append({
                        'name': filename,
                        'path': file_item['path'],
                        'content': content[:5000],  # Store first 5KB for embedding
                        'analysis': analysis,
                        'size': file_item.get('size', 0)
                    })
            except Exception as e:
                continue  # Skip problematic files
        
        return files_data, None
        
    except Exception as e:
        return [], str(e)

def create_enhanced_repository_content(repo, files_data=None):
    """Create enhanced content for repository embedding including file analysis"""
    content_parts = []
    
    # Basic repository information
    content_parts.append(f"Repository: {repo['name']}")
    content_parts.append(f"Description: {repo.get('description', 'No description')}")
    content_parts.append(f"Language: {repo.get('language', 'Unknown')}")
    
    # Domain classification based on repository info
    repo_text = f"{repo['name']} {repo.get('description', '')}"
    domains = classify_domain(repo_text)
    if domains:
        sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0]
        content_parts.append(f"Primary Domain: {primary_domain}")
        content_parts.append(f"Domain Categories: {', '.join([f'{d}({s})' for d, s in sorted_domains])}")
    
    # File analysis if available
    if files_data:
        file_types = defaultdict(int)
        functionalities = defaultdict(int)
        total_domains = defaultdict(int)
        
        for file_data in files_data:
            analysis = file_data['analysis']
            file_types[analysis['file_type']] += 1
            
            # Aggregate functionality keywords
            for func, count in analysis['functionality'].items():
                functionalities[func] += count
            
            # Aggregate domain keywords from files
            for domain, count in analysis['domains'].items():
                total_domains[domain] += count
        
        # Add file type summary
        if file_types:
            file_summary = ', '.join([f"{ft}: {count}" for ft, count in file_types.items()])
            content_parts.append(f"File Types: {file_summary}")
        
        # Add functionality summary
        if functionalities:
            top_functions = sorted(functionalities.items(), key=lambda x: x[1], reverse=True)[:5]
            func_summary = ', '.join([f"{func}({count})" for func, count in top_functions])
            content_parts.append(f"Key Functionalities: {func_summary}")
        
        # Add file-based domain analysis
        if total_domains:
            sorted_file_domains = sorted(total_domains.items(), key=lambda x: x[1], reverse=True)
            file_domain_summary = ', '.join([f"{domain}({count})" for domain, count in sorted_file_domains[:3]])
            content_parts.append(f"File Content Domains: {file_domain_summary}")
        
        # Add sample file contents
        important_files = [f for f in files_data if f['name'] in ['README.md', 'package.json', 'requirements.txt']]
        for file_data in important_files[:2]:  # Include content from important files
            content_parts.append(f"File {file_data['name']}: {file_data['content'][:500]}")
    
    # Original metadata
    content_parts.append(f"Stars: {repo.get('stargazers_count', 0)}")
    content_parts.append(f"Forks: {repo.get('forks_count', 0)}")
    content_parts.append(f"Size: {repo.get('size', 0)} KB")
    content_parts.append(f"Created: {format_date(repo.get('created_at', ''))}")
    content_parts.append(f"Updated: {format_date(repo.get('updated_at', ''))}")
    
    if repo.get('topics'):
        content_parts.append(f"Topics: {', '.join(repo['topics'])}")
    
    return " | ".join(content_parts)

def initialize_file_vector_db():
    """Initialize separate vector database for individual files"""
    try:
        client = chromadb.Client()
        
        # Check if collection already exists and delete it
        try:
            existing_collection = client.get_collection("github_files")
            client.delete_collection("github_files")
        except:
            pass
        
        # Create new collection for files
        collection = client.create_collection(
            name="github_files",
            metadata={"description": "Individual GitHub repository files"}
        )
        return collection
    except Exception as e:
        st.error(f"Error initializing file vector database: {e}")
        return None

def embed_repository_files(files_data, repo_name, model, vector_db):
    """Embed individual files for detailed search"""
    try:
        # Check if vector_db is available
        if vector_db is None:
            st.warning("File vector database not initialized. Skipping file embedding.")
            return False
            
        documents = []
        metadatas = []
        ids = []
        
        for i, file_data in enumerate(files_data):
            # Create content for embedding
            content_parts = [
                f"Repository: {repo_name}",
                f"File: {file_data['name']}",
                f"Path: {file_data['path']}",
                f"Type: {file_data['analysis']['file_type']}",
                f"Content: {file_data['content']}"
            ]
            
            # Add functionality information
            if file_data['analysis']['functionality']:
                func_list = ', '.join(file_data['analysis']['functionality'].keys())
                content_parts.append(f"Functionalities: {func_list}")
            
            # Add domain information
            if file_data['analysis']['domains']:
                domain_list = ', '.join(file_data['analysis']['domains'].keys())
                content_parts.append(f"Domains: {domain_list}")
            
            # Add code analysis information
            if file_data['analysis'].get('code_structure'):
                code_struct = file_data['analysis']['code_structure']
                
                # Add functions
                if code_struct.get('functions'):
                    func_names = [f['name'] for f in code_struct['functions'][:5]]
                    content_parts.append(f"Functions: {', '.join(func_names)}")
                
                # Add classes
                if code_struct.get('classes'):
                    class_names = [c['name'] for c in code_struct['classes'][:3]]
                    content_parts.append(f"Classes: {', '.join(class_names)}")
                
                # Add logic flow
                if code_struct.get('logic_flow'):
                    flow_summary = ', '.join(code_struct['logic_flow'][:3])
                    content_parts.append(f"Logic Flow: {flow_summary}")
                
                # Add algorithms
                if code_struct.get('algorithms'):
                    algo_summary = ', '.join(code_struct['algorithms'][:3])
                    content_parts.append(f"Algorithms: {algo_summary}")
            
            # Add main purpose
            if file_data['analysis'].get('main_purpose'):
                content_parts.append(f"Purpose: {file_data['analysis']['main_purpose'][:200]}")
            
            documents.append(" | ".join(content_parts))
            
            # Metadata for file
            metadata = {
                "repository": repo_name,
                "filename": file_data['name'],
                "filepath": file_data['path'],
                "file_type": file_data['analysis']['file_type'],
                "size": file_data['size'],
                "lines": file_data['analysis']['lines'],
                "complexity_score": file_data['analysis'].get('complexity_score', 0)
            }
            
            # Add top functionality
            if file_data['analysis']['functionality']:
                top_func = max(file_data['analysis']['functionality'].items(), key=lambda x: x[1])
                metadata["primary_functionality"] = top_func[0]
            
            # Add top domain
            if file_data['analysis']['domains']:
                top_domain = max(file_data['analysis']['domains'].items(), key=lambda x: x[1])
                metadata["primary_domain"] = top_domain[0]
            
            # Add code analysis metadata
            if file_data['analysis'].get('code_structure'):
                code_struct = file_data['analysis']['code_structure']
                metadata["function_count"] = len(code_struct.get('functions', []))
                metadata["class_count"] = len(code_struct.get('classes', []))
                
                # Add top functions
                if code_struct.get('functions'):
                    top_functions = [f['name'] for f in code_struct['functions'][:3]]
                    metadata["top_functions"] = ','.join(top_functions)
                
                # Add top classes
                if code_struct.get('classes'):
                    top_classes = [c['name'] for c in code_struct['classes'][:3]]
                    metadata["top_classes"] = ','.join(top_classes)
                
                # Add algorithms
                if code_struct.get('algorithms'):
                    algorithms = code_struct['algorithms'][:3]
                    metadata["algorithms"] = ','.join(algorithms)
            
            # Add main purpose
            if file_data['analysis'].get('main_purpose'):
                metadata["main_purpose"] = file_data['analysis']['main_purpose'][:100]
            
            metadatas.append(metadata)
            ids.append(f"file_{repo_name}_{i}")
        
        if documents:
            # Generate embeddings
            embeddings = model.encode(documents)
            
            # Store in vector database
            vector_db.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return True
    except Exception as e:
        st.error(f"Error embedding files: {e}")
        return False

def enhanced_embed_repositories(repos, model, vector_db, file_vector_db, username, token=None):
    """Enhanced repository embedding with file analysis"""
    try:
        documents = []
        metadatas = []
        ids = []
        
        for i, repo in enumerate(repos):
            repo_name = repo['name']
            
            # Fetch and analyze repository files
            with st.spinner(f"Analyzing files in {repo_name}..."):
                files_data, error = fetch_repository_files(username, repo_name, token, max_files=30)
                
                if not error and files_data:
                    # Store file analysis in session state
                    st.session_state.repository_analysis[repo_name] = files_data
                    
                    # Embed individual files only if file_vector_db is available
                    if file_vector_db is not None:
                        embed_repository_files(files_data, repo_name, model, file_vector_db)
                else:
                    files_data = []
            
            # Create enhanced repository content
            content = create_enhanced_repository_content(repo, files_data)
            documents.append(content)
            
            # Enhanced metadata with domain classification
            repo_text = f"{repo['name']} {repo.get('description', '')}"
            domains = classify_domain(repo_text)
            
            metadata = {
                "name": repo.get('name', 'Unknown'),
                "description": repo.get('description') or 'No description',
                "language": repo.get('language') or 'Unknown',
                "stars": int(repo.get('stargazers_count', 0)),
                "forks": int(repo.get('forks_count', 0)),
                "url": repo.get('html_url', ''),
                "created_at": repo.get('created_at', ''),
                "updated_at": repo.get('updated_at', ''),
                "size": int(repo.get('size', 0)),
                "file_count": len(files_data),
                "has_analysis": len(files_data) > 0
            }
            
            # Add domain classification
            if domains:
                sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
                metadata["primary_domain"] = sorted_domains[0][0]
                metadata["domain_score"] = sorted_domains[0][1]
                metadata["all_domains"] = ','.join([d for d, s in sorted_domains])
            
            # Add file type summary
            if files_data:
                file_types = defaultdict(int)
                for file_data in files_data:
                    file_types[file_data['analysis']['file_type']] += 1
                metadata["file_types"] = ','.join([f"{ft}:{count}" for ft, count in file_types.items()])
            
            metadatas.append(metadata)
            ids.append(f"repo_{i}")
        
        # Generate embeddings for repositories
        embeddings = model.encode(documents)
        
        # Store in vector database
        vector_db.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return True
    except Exception as e:
        st.error(f"Error in enhanced embedding: {e}")
        return False

def enhanced_search_repositories(query, model, vector_db, file_vector_db=None, top_k=5):
    """Enhanced search that can search both repositories and files"""
    try:
        query_embedding = model.encode([query])
        
        # Search repositories
        repo_results = vector_db.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Search files if available
        file_results = None
        if file_vector_db:
            try:
                file_results = file_vector_db.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=top_k
                )
            except:
                pass  # File search optional
        
        return {
            'repositories': repo_results,
            'files': file_results
        }
    except Exception as e:
        st.error(f"Error in enhanced search: {e}")
        return None

def create_enhanced_rag_prompt(question, repo_results, file_results=None):
    """Create enhanced RAG prompt with repository and file context"""
    
    # Repository context
    repo_context = []
    if repo_results and repo_results.get('metadatas') and repo_results['metadatas']:
        for repo in repo_results['metadatas'][0]:
            repo_info = f"""Repository: {repo['name']}
Description: {repo['description']}
Primary Domain: {repo.get('primary_domain', 'Unknown')}
Language: {repo['language']}
File Types: {repo.get('file_types', 'Unknown')}
Stars: {repo['stars']}
URL: {repo['url']}"""
            repo_context.append(repo_info)
    
    # File context
    file_context = []
    if file_results and file_results.get('metadatas') and file_results['metadatas']:
        for file_data in file_results['metadatas'][0]:
            file_info = f"""File: {file_data['filename']} (in {file_data['repository']})
Path: {file_data['filepath']}
Type: {file_data['file_type']}
Primary Domain: {file_data.get('primary_domain', 'Unknown')}
Primary Functionality: {file_data.get('primary_functionality', 'Unknown')}"""
            file_context.append(file_info)
    
    # Combine contexts
    all_context = []
    if repo_context:
        all_context.append("REPOSITORIES:\n" + "\n\n".join(repo_context))
    if file_context:
        all_context.append("RELEVANT FILES:\n" + "\n\n".join(file_context))
    
    context = "\n\n" + "="*50 + "\n\n".join(all_context)
    
    prompt = f"""You are an expert code analyst and project manager. Based on the following GitHub repository and file information, please answer the question with detailed insights.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide specific repository and file names in your answer
- Group information by domain/project type when relevant
- Explain the reasoning behind classifications
- If the question asks about grouping or categorization, provide clear categories
- Include relevant details like file types, functionalities, and domains
- If you cannot find specific information, clearly state what is missing

ANSWER:"""
    
    return prompt

def analyze_code_structure(content, filename):
    """Analyze code structure and extract functions, classes, and logic flow"""
    analysis = {
        'functions': [],
        'classes': [],
        'imports': [],
        'variables': [],
        'logic_flow': [],
        'algorithms': [],
        'complexity': 'unknown',
        'patterns': []
    }
    
    try:
        # Try to parse as Python code
        if filename.lower().endswith('.py'):
            tree = ast.parse(content)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node) or '',
                        'lines': node.lineno,
                        'complexity': calculate_complexity(node)
                    }
                    analysis['functions'].append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node) or '',
                        'lines': node.lineno
                    }
                    analysis['classes'].append(class_info)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['variables'].append(target.id)
        
        # Extract logic flow patterns
        analysis['logic_flow'] = extract_logic_flow(content)
        
        # Detect algorithms
        analysis['algorithms'] = detect_algorithms(content)
        
        # Detect design patterns
        analysis['patterns'] = detect_design_patterns(content)
        
    except SyntaxError:
        # Handle non-Python files or syntax errors
        analysis['functions'] = extract_functions_generic(content)
        analysis['classes'] = extract_classes_generic(content)
        analysis['logic_flow'] = extract_logic_flow(content)
        analysis['algorithms'] = detect_algorithms(content)
    
    return analysis

def calculate_complexity(node):
    """Calculate cyclomatic complexity of a function"""
    complexity = 1  # Base complexity
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    
    if complexity <= 5:
        return 'low'
    elif complexity <= 10:
        return 'medium'
    else:
        return 'high'

def extract_logic_flow(content):
    """Extract logic flow patterns from code"""
    flow_patterns = []
    
    # Control flow patterns
    if_patterns = re.findall(r'if\s*\([^)]+\)|if\s+[^:]+:', content, re.IGNORECASE)
    loop_patterns = re.findall(r'for\s*\([^)]+\)|while\s*\([^)]+\)|for\s+[^:]+:', content, re.IGNORECASE)
    switch_patterns = re.findall(r'switch\s*\([^)]+\)|case\s+[^:]+:', content, re.IGNORECASE)
    
    # Data flow patterns
    data_flow = re.findall(r'(\w+)\s*=\s*(\w+)|(\w+)\.(\w+)\s*\(', content)
    
    # Error handling
    error_handling = re.findall(r'try\s*\{|catch\s*\(|except\s+[^:]+:', content, re.IGNORECASE)
    
    if if_patterns:
        flow_patterns.append(f"Conditional logic: {len(if_patterns)} conditions")
    if loop_patterns:
        flow_patterns.append(f"Loop structures: {len(loop_patterns)} loops")
    if switch_patterns:
        flow_patterns.append(f"Switch statements: {len(switch_patterns)} cases")
    if error_handling:
        flow_patterns.append(f"Error handling: {len(error_handling)} try-catch blocks")
    
    return flow_patterns

def detect_algorithms(content):
    """Detect algorithms in code"""
    detected_algorithms = []
    content_lower = content.lower()
    
    for category, patterns in ALGORITHM_PATTERNS.items():
        for pattern in patterns:
            if pattern in content_lower:
                detected_algorithms.append(f"{category}: {pattern}")
    
    return detected_algorithms

def detect_design_patterns(content):
    """Detect common design patterns"""
    patterns = []
    content_lower = content.lower()
    
    # Singleton pattern
    if re.search(r'private\s+static.*instance|getInstance', content_lower):
        patterns.append('Singleton')
    
    # Factory pattern
    if re.search(r'create.*factory|factory.*create', content_lower):
        patterns.append('Factory')
    
    # Observer pattern
    if re.search(r'addObserver|removeObserver|notify', content_lower):
        patterns.append('Observer')
    
    # Strategy pattern
    if re.search(r'strategy|algorithm.*interface', content_lower):
        patterns.append('Strategy')
    
    return patterns

def extract_functions_generic(content):
    """Extract function definitions from any programming language"""
    functions = []
    
    # Common function patterns
    patterns = [
        r'function\s+(\w+)\s*\([^)]*\)',  # JavaScript, PHP
        r'def\s+(\w+)\s*\([^)]*\)',      # Python
        r'(\w+)\s+(\w+)\s*\([^)]*\)\s*\{',  # Java, C++, C#
        r'(\w+)\s+(\w+)\s*\([^)]*\)\s*:',   # Python with type hints
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            func_name = match.group(1) if match.group(1) else match.group(2)
            functions.append({
                'name': func_name,
                'pattern': pattern,
                'line': content[:match.start()].count('\n') + 1
            })
    
    return functions

def extract_classes_generic(content):
    """Extract class definitions from any programming language"""
    classes = []
    
    # Common class patterns
    patterns = [
        r'class\s+(\w+)',  # Most languages
        r'public\s+class\s+(\w+)',  # Java
        r'class\s+(\w+)\s*:',  # Python
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            class_name = match.group(1)
            classes.append({
                'name': class_name,
                'pattern': pattern,
                'line': content[:match.start()].count('\n') + 1
            })
    
    return classes

def create_code_analysis_prompt(question, code_analysis, file_content):
    """Create a prompt for analyzing code logic and structure"""
    
    # Extract key information from code analysis
    functions_summary = []
    for func in code_analysis.get('functions', [])[:5]:  # Top 5 functions
        func_desc = f"- {func['name']}({', '.join(func['args'])}) - {func['docstring'][:100]}"
        functions_summary.append(func_desc)
    
    classes_summary = []
    for cls in code_analysis.get('classes', [])[:3]:  # Top 3 classes
        class_desc = f"- {cls['name']} with methods: {', '.join(cls['methods'][:5])}"
        classes_summary.append(class_desc)
    
    logic_flow = code_analysis.get('logic_flow', [])
    algorithms = code_analysis.get('algorithms', [])
    patterns = code_analysis.get('patterns', [])
    
    # Create the analysis prompt
    analysis_text = f"""
CODE ANALYSIS FOR: {question}

FILE CONTENT (first 2000 characters):
{file_content[:2000]}

CODE STRUCTURE:
Functions: {len(code_analysis.get('functions', []))}
Classes: {len(code_analysis.get('classes', []))}
Imports: {', '.join(code_analysis.get('imports', [])[:10])}

KEY FUNCTIONS:
{chr(10).join(functions_summary)}

KEY CLASSES:
{chr(10).join(classes_summary)}

LOGIC FLOW:
{chr(10).join(logic_flow)}

ALGORITHMS DETECTED:
{chr(10).join(algorithms)}

DESIGN PATTERNS:
{chr(10).join(patterns)}

Please analyze this code and explain:
1. How the code works and what it does
2. The main logic flow and algorithms used
3. Key functions and their purposes
4. How it relates to the user's question
5. Any important patterns or architectural decisions

Focus on explaining the actual implementation and logic, not just the structure.
"""
    
    return analysis_text

def enhanced_analyze_file_content(content, filename):
    """Enhanced file content analysis with code structure understanding"""
    analysis = {
        'file_type': get_file_category(filename),
        'domains': classify_domain(content),
        'functionality': extract_functionality_keywords(content),
        'size': len(content),
        'lines': content.count('\n') + 1 if content else 0,
        'code_structure': analyze_code_structure(content, filename),
        'complexity_score': 0,
        'main_purpose': '',
        'key_algorithms': [],
        'data_flow': []
    }
    
    # Calculate complexity score
    functions = analysis['code_structure']['functions']
    if functions:
        complexity_levels = {'low': 1, 'medium': 2, 'high': 3}
        total_complexity = sum(complexity_levels.get(func.get('complexity', 'low'), 1) for func in functions)
        analysis['complexity_score'] = total_complexity / len(functions)
    
    # Extract main purpose from docstrings and comments
    docstring_patterns = re.findall(r'"""([^"]*)"""|\'\'\'([^\']*)\'\'\'|//\s*(.+)|#\s*(.+)', content)
    if docstring_patterns:
        docstrings = [match[0] or match[1] or match[2] or match[3] for match in docstring_patterns]
        analysis['main_purpose'] = ' '.join(docstrings[:3])  # First 3 docstrings
    
    # Extract key algorithms
    analysis['key_algorithms'] = analysis['code_structure']['algorithms']
    
    # Extract data flow patterns
    data_flow_patterns = re.findall(r'(\w+)\s*=\s*(\w+)|(\w+)\.(\w+)\s*\(|(\w+)\s*->\s*(\w+)', content)
    analysis['data_flow'] = [f"{match[0] or match[2] or match[4]} -> {match[1] or match[3] or match[5]}" for match in data_flow_patterns[:10]]
    
    return analysis

def create_enhanced_rag_prompt_with_code(question, repo_results, file_results=None, code_analysis=None):
    """Create enhanced RAG prompt with detailed code analysis"""
    
    # Repository context
    repo_context = []
    if repo_results and repo_results.get('metadatas') and repo_results['metadatas']:
        for repo in repo_results['metadatas'][0]:
            repo_info = f"""Repository: {repo['name']}
Description: {repo['description']}
Primary Domain: {repo.get('primary_domain', 'Unknown')}
Language: {repo['language']}
File Types: {repo.get('file_types', 'Unknown')}
Stars: {repo['stars']}
URL: {repo['url']}"""
            repo_context.append(repo_info)
    
    # File context with code analysis
    file_context = []
    if file_results and file_results.get('metadatas') and file_results['metadatas']:
        for file_data in file_results['metadatas'][0]:
            file_info = f"""File: {file_data['filename']} (in {file_data['repository']})
Path: {file_data['filepath']}
Type: {file_data['file_type']}
Primary Domain: {file_data.get('primary_domain', 'Unknown')}
Primary Functionality: {file_data.get('primary_functionality', 'Unknown')}"""
            
            # Add code analysis if available
            if code_analysis and file_data['filename'] in code_analysis:
                analysis = code_analysis[file_data['filename']]
                
                # Get the actual file content from session state
                file_content = ""
                if st.session_state.repository_analysis.get(file_data['repository']):
                    for repo_file in st.session_state.repository_analysis[file_data['repository']]:
                        if repo_file['name'] == file_data['filename']:
                            file_content = repo_file['content']
                            break
                
                file_info += f"""
Code Analysis:
- Functions: {len(analysis['code_structure']['functions'])}
- Classes: {len(analysis['code_structure']['classes'])}
- Logic Flow: {', '.join(analysis['code_structure']['logic_flow'][:3])}
- Algorithms: {', '.join(analysis['code_structure']['algorithms'][:3])}
- Main Purpose: {analysis['main_purpose'][:200]}

KEY FUNCTIONS:
{chr(10).join([f"- {func['name']}({', '.join(func['args'])}) - {func['docstring'][:100]}" for func in analysis['code_structure']['functions'][:5]])}

KEY CLASSES:
{chr(10).join([f"- {cls['name']} with methods: {', '.join(cls['methods'][:5])}" for cls in analysis['code_structure']['classes'][:3]])}

FILE CONTENT (first 1500 characters):
{file_content[:1500]}"""
            
            file_context.append(file_info)
    
    # Add repository file analysis if available (for simple mode)
    elif st.session_state.repository_analysis:
        for repo_name, files_data in st.session_state.repository_analysis.items():
            if files_data:
                file_context.append(f"Repository: {repo_name}")
                file_context.append(f"Files analyzed: {len(files_data)}")
                
                # Add key files with their content
                for file_data in files_data[:3]:  # Show first 3 files
                    file_context.append(f"""
File: {file_data['name']}
Type: {file_data['analysis']['file_type']}
Content Preview: {file_data['content'][:500]}
Key Functions: {', '.join([f['name'] for f in file_data['analysis']['code_structure']['functions'][:3]])}
""")
    
    # Combine contexts
    all_context = []
    if repo_context:
        all_context.append("REPOSITORIES:\n" + "\n\n".join(repo_context))
    if file_context:
        all_context.append("RELEVANT FILES WITH CODE ANALYSIS:\n" + "\n\n".join(file_context))
    
    context = "\n\n" + "="*50 + "\n\n".join(all_context)
    
    prompt = f"""You are an expert software engineer and code analyst. Based on the following GitHub repository and file information, please provide a detailed technical analysis.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide specific repository and file names in your answer
- Use the actual code content provided to explain implementation details
- Reference specific functions, classes, and code snippets from the file content
- Explain how the code logic works based on the actual implementation
- Describe algorithms and data structures used in the code
- Show the flow of data and control logic using the provided code
- If the question asks about specific functionality, explain how it's implemented using the actual code
- Include code patterns, architectural decisions, and design choices found in the code
- Quote relevant code snippets to support your explanations
- If you cannot find specific information in the provided code, clearly state what is missing
- Focus on technical implementation details from the actual code, not generic descriptions
- IMPORTANT: If you see file content in the context, use it to provide specific answers about the code
- Look for keywords in the file content that match the user's question
- Explain what the code actually does, not what it might do

ANSWER:"""
    
    return prompt

def main():
    # Header
    st.markdown('<h1 class="main-header">🐙 GitHub RAG Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Compact repository setup section
    st.sidebar.markdown("### 🚀 Fetch User Repositories")
    
    # Username dropdown with predefined options
    username = st.sidebar.selectbox(
        "Select GitHub Username:",
        options=[user["value"] for user in PREDEFINED_USERS],
        index=2,  # Default to "Srihari-bot"
        help="Select a username from the predefined list"
    )
    
    # Get the corresponding token for the selected username
    github_token = GITHUB_TOKENS.get(username)
    
    # Show token status
    if github_token:
        st.sidebar.success(f"✅ Token available for {username}")
    else:
        st.sidebar.warning(f"⚠️ No token configured for {username}")
        st.sidebar.info("Add the token to your .env file to avoid rate limits")
    
    setup_button = st.sidebar.button("Fetch Repositories", type="primary", use_container_width=True)
    
    # Show rate limit info
    if not github_token:
        st.sidebar.warning("⚠️ Without a GitHub token, you're limited to 60 requests/hour.")
        st.sidebar.info(f"📊 API Requests: {st.session_state.api_requests_count}/60")
        
        # Show prominent rate limit warning if approaching limit
        if st.session_state.api_requests_count >= 50:
            st.sidebar.error("🚨 Rate limit approaching! Add the token to your .env file.")
        
        # Manual reset button
        if st.session_state.api_requests_count >= 60:
            if st.sidebar.button("🔄 Reset Rate Limit (Wait 1 hour)", help="Click if you've waited an hour"):
                st.session_state.api_requests_count = 0
                st.session_state.last_reset_time = time.time()
                st.sidebar.success("Rate limit reset! You can try again.")
                st.rerun()
        
        # Reset counter every hour
        if 'last_reset_time' not in st.session_state:
            st.session_state.last_reset_time = time.time()
        elif time.time() - st.session_state.last_reset_time > 3600:  # 1 hour
            st.session_state.api_requests_count = 0
            st.session_state.last_reset_time = time.time()
    else:
        st.sidebar.success("✅ GitHub token available - higher rate limits")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Setup repositories and vector database
    if setup_button and username:
        # Clear previous data
        st.session_state.repos_data = []
        st.session_state.vector_db = None
        st.session_state.qa_history = []
        st.session_state.current_username = None
        
        # Check rate limit before making requests
        if not github_token and st.session_state.api_requests_count >= 60:
            st.error("🚨 Rate limit reached! Please add the GitHub token to your .env file or wait an hour.")
            st.info(f"""
            **To continue immediately:**
            1. Add the GitHub token for {username} to your .env file:
               ```
               {USERNAME_TO_ENV_VAR.get(username, f"GITHUB_TOKEN_{username.upper().replace('-', '_')}")}=your_token_here
               ```
            2. Restart the application
            3. Click "Fetch Repositories" again
            """)
            return
        
        with st.spinner(f"Analyzing {username}'s repositories..."):
            repos, error = fetch_user_repositories(username, github_token)
            user_info, user_error = fetch_user_info(username, github_token)
        
        if error:
            if "rate limit exceeded" in error.lower():
                st.error("❌ Rate limit exceeded! Please add the GitHub token to your .env file to continue.")
                st.info(f"""
                **To fix this issue:**
                1. Add the GitHub token for {username} to your .env file:
                   ```
                   {USERNAME_TO_ENV_VAR.get(username, f"GITHUB_TOKEN_{username.upper().replace('-', '_')}")}=your_token_here
                   ```
                2. Restart the application
                3. Click "Fetch Repositories" again
                """)
            else:
                st.error(f"❌ {error}")
        elif repos:
            st.sidebar.success(f"✅ Found {len(repos)} repositories for {username}")
            st.session_state.repos_data = repos
            st.session_state.current_username = username
            
            # Automatically create vector database
            with st.spinner("🧠 Building AI knowledge base..."):
                try:
                    # Initialize models and database
                    if st.session_state.embeddings_model is None:
                        st.session_state.embeddings_model = load_embeddings_model()
                    
                    # Always create a fresh vector database
                    st.session_state.vector_db = initialize_vector_db()
                    
                    if st.session_state.embeddings_model and st.session_state.vector_db:
                        # Enhanced embedding with file analysis
                        success = enhanced_embed_repositories(
                            st.session_state.repos_data,
                            st.session_state.embeddings_model,
                            st.session_state.vector_db,
                            st.session_state.file_vector_db,  # Use session state file vector DB
                            username,
                            github_token
                        )
                        if success:
                            st.sidebar.success("🎉 Ready to chat! Ask me anything about the repositories.")
                        else:
                            st.sidebar.error("❌ Failed to create knowledge base.")
                    else:
                        st.sidebar.error("❌ Failed to initialize AI models.")
                except Exception as e:
                    st.sidebar.error(f"Error creating knowledge base: {e}")
            
            # Show brief user stats
            if user_info:
                with st.sidebar.expander(f"📊 {username}'s Profile Summary", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Repositories", len(repos))
        else:
            st.sidebar.warning(f"No repositories found for user: {username}")
    
    if not st.session_state.repos_data or not st.session_state.vector_db:
        st.sidebar.info("👆 Enter a GitHub username and click Setup to start chatting!")
    else:
        # Chat input
        question = st.chat_input("Ask me anything about the repositories...")
        
        if question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get access token
                    access_token = get_access_token(API_KEY)
                    
                    # Search relevant repositories
                    search_results = enhanced_search_repositories(
                        question,
                        st.session_state.embeddings_model,
                        st.session_state.vector_db,
                        st.session_state.file_vector_db,  # Use session state file vector DB
                        top_k=5
                    )
                    
                    if search_results and search_results.get('repositories'):
                        # Prepare context
                        repo_results = search_results['repositories']
                        
                        # Create enhanced RAG prompt with code analysis
                        rag_prompt = create_enhanced_rag_prompt_with_code(question, repo_results, search_results.get('files'))
                        
                        # Generate answer
                        answer = generate_watson_response(rag_prompt, access_token)
                        
                        if answer:
                            # Store in history
                            st.session_state.qa_history.append({
                                'question': question,
                                'answer': answer,
                                'context_repos': repo_results.get('metadatas', [[]])[0] if repo_results.get('metadatas') else [],
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                        else:
                            st.sidebar.error("Failed to generate answer from Watson API")
                    else:
                        st.sidebar.error("No relevant repositories found for your question")
                
                except Exception as e:
                    st.sidebar.error(f"Error processing question: {e}")
    
    # Display Chat History (ChatGPT style)
    if st.session_state.qa_history:
        st.markdown("### 📜 Conversation")
        
        for i, qa in enumerate(st.session_state.qa_history):
            # User message
            with st.chat_message("user"):
                st.write(qa['question'])
            
            # Side-by-side layout: AI Response (left) | Source Repositories (right)
            col_left, col_right = st.columns([1, 1])
            
            # Left column: AI Response
            with col_left:
                with st.chat_message("assistant"):
                    st.write(qa['answer'])
            
            # Right column: Source Repositories
            with col_right:
                if qa.get('context_repos') and len(qa['context_repos']) > 0:
                    meaningful_repos = [repo for repo in qa['context_repos'][:3] if repo.get('name')]
                    if meaningful_repos:
                        st.markdown("##### 🔗 Source Repositories")
                        
                        # Create horizontal layout for repositories
                        if len(meaningful_repos) == 1:
                            repo_cols = st.columns([1])
                        elif len(meaningful_repos) == 2:
                            repo_cols = st.columns([1, 1])
                        else:
                            repo_cols = st.columns([1, 1, 1])
                        
                        for idx, repo in enumerate(meaningful_repos):
                            repo_name = repo.get('name', 'Unknown')
                            repo_url = repo.get('url', '')
                            if not repo_url:
                                repo_url = f"https://github.com/{st.session_state.current_username}/{repo_name}"
                            
                            with repo_cols[idx]:
                                # Compact repository card
                                st.markdown(f"**📁 {repo_name}**")
                                
                                # Compact action buttons
                                st.link_button("🌐 View", repo_url, help="View Repository")
                                
                                # Pre-fetch zip content for immediate download
                                if f"zip_content_{repo_name}" not in st.session_state:
                                    with st.spinner("..."):
                                        zip_content, error = download_repository_zip(st.session_state.current_username, repo_name, github_token)
                                        if zip_content:
                                            st.session_state[f"zip_content_{repo_name}"] = zip_content
                                        else:
                                            st.session_state[f"zip_content_{repo_name}"] = None
                                            st.session_state[f"zip_error_{repo_name}"] = error
                                
                                # Compact download button
                                if st.session_state.get(f"zip_content_{repo_name}"):
                                    st.download_button(
                                        label="📦 Download ZIP",
                                        data=st.session_state[f"zip_content_{repo_name}"],
                                        file_name=f"{repo_name}.zip",
                                        mime="application/zip",
                                        key=f"download_{repo_name}_{i}",
                                        help="Download ZIP"
                                    )
                                else:
                                    if st.button("📦 Download ZIP", key=f"zip_{repo_name}_{i}", help="Download ZIP"):
                                        with st.spinner("Downloading..."):
                                            zip_content, error = download_repository_zip(st.session_state.current_username, repo_name, github_token)
                                            if zip_content:
                                                st.session_state[f"zip_content_{repo_name}"] = zip_content
                                                st.rerun()
                                            else:
                                                if "rate limit" in error.lower():
                                                    st.error("Rate limit exceeded. Please add the token to your .env file or wait an hour.")
                                                else:
                                                    st.error(f"Failed: {error}")
                                
                                # Compact browse button
                                if st.button("📄 Browse Files", key=f"browse_{repo_name}_{i}", help="Browse Files"):
                                    with st.spinner("Loading..."):
                                        contents, error = get_repository_contents(st.session_state.current_username, repo_name, "", github_token)
                                        if contents:
                                            with st.expander("Files", expanded=True):
                                                for item in contents[:5]:  # Show first 5 items
                                                    if item['type'] == 'file':
                                                        if st.button(f"⬇️ {item['name']}", key=f"file_{item['name']}_{repo_name}_{i}"):
                                                            file_content, file_name, file_error = download_file_content(st.session_state.current_username, repo_name, item['path'], github_token)
                                                            if file_content:
                                                                st.download_button(
                                                                    label=f"💾 {file_name}",
                                                                    data=file_content,
                                                                    file_name=file_name,
                                                                    key=f"dl_{item['name']}_{repo_name}_{i}"
                                                                )
                                                            else:
                                                                st.error("Failed to download file")
                                        else:
                                            if "rate limit" in error.lower():
                                                st.error("Rate limit exceeded. Please add the token to your .env file or wait an hour.")
                                            else:
                                                st.error("Failed to load files")
    
    # Close container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    

if __name__ == "__main__":
    main()


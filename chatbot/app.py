from flask import Flask, render_template, request, jsonify
import pickle
import os
from datetime import datetime
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data if not already present
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# IMPORTANT: Add these class definitions BEFORE loading the model
class TextPreprocessor:
    """Handles text cleaning and preprocessing for FAQ questions"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

class FAQChatbotModel:
    """Main FAQ Chatbot model using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
        self.faq_vectors = None
        self.faq_data = None
        self.preprocessor = TextPreprocessor()
    
    def train(self, faq_dataframe):
        """Train the chatbot model with FAQ data"""
        self.faq_data = faq_dataframe.copy()
        processed_questions = faq_dataframe['processed_question'].tolist()
        self.faq_vectors = self.vectorizer.fit_transform(processed_questions)
        
    def find_best_match(self, user_question, threshold=0.1):
        """Find the best matching FAQ for user question"""
        # Preprocess user question
        processed_user_question = self.preprocessor.preprocess_text(user_question)
        
        # Convert user question to TF-IDF vector
        user_vector = self.vectorizer.transform([processed_user_question])
        
        # Calculate cosine similarity with all FAQ questions
        similarities = cosine_similarity(user_vector, self.faq_vectors).flatten()
        
        # Find best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Return result based on similarity threshold
        if best_similarity > threshold:
            return {
                'question': self.faq_data.iloc[best_match_idx]['question'],
                'answer': self.faq_data.iloc[best_match_idx]['answer'],
                'similarity_score': float(best_similarity),
                'confidence': 'High' if best_similarity > 0.5 else 'Medium'
            }
        else:
            return {
                'question': None,
                'answer': "I'm sorry, I couldn't find a relevant answer to your question. Please contact our support team for assistance.",
                'similarity_score': float(best_similarity),
                'confidence': 'Low'
            }

app = Flask(__name__)

# Global variable to store the chatbot model
chatbot_model = None

def load_chatbot_model():
    """Load the trained FAQ chatbot model"""
    global chatbot_model
    
    try:
        model_path = os.path.join('models', 'faq_chatbot_model.pkl')
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a simple model class to hold the loaded data
        class LoadedChatbot:
            def __init__(self, model_data):
                self.vectorizer = model_data['vectorizer']
                self.faq_vectors = model_data['faq_vectors']
                self.faq_data = model_data['faq_data']
                self.preprocessor = model_data['preprocessor']
                self.model_info = model_data['model_info']
            
            def find_best_match(self, user_question, threshold=0.1):
                """Find the best matching FAQ for user question"""
                # Preprocess user question
                processed_user_question = self.preprocessor.preprocess_text(user_question)
                
                # Convert user question to TF-IDF vector
                user_vector = self.vectorizer.transform([processed_user_question])
                
                # Calculate cosine similarity with all FAQ questions
                similarities = cosine_similarity(user_vector, self.faq_vectors).flatten()
                
                # Find best match
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                # Return result based on similarity threshold
                if best_similarity > threshold:
                    return {
                        'question': self.faq_data.iloc[best_match_idx]['question'],
                        'answer': self.faq_data.iloc[best_match_idx]['answer'],
                        'similarity_score': float(best_similarity),
                        'confidence': 'High' if best_similarity > 0.5 else 'Medium'
                    }
                else:
                    return {
                        'question': None,
                        'answer': "I'm sorry, I couldn't find a relevant answer to your question. Please contact our support team for assistance.",
                        'similarity_score': float(best_similarity),
                        'confidence': 'Low'
                    }
        
        chatbot_model = LoadedChatbot(model_data)
        print("✅ Chatbot model loaded successfully!")
        print(f"📊 Model Info: {chatbot_model.model_info['num_faqs']} FAQs, {chatbot_model.model_info['vocabulary_size']} vocabulary")
        
    except FileNotFoundError:
        print("❌ Model file not found. Please ensure faq_chatbot_model.pkl is in the models/ directory.")
        chatbot_model = None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        chatbot_model = None

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests from the frontend"""
    try:
        # Check if model is loaded
        if chatbot_model is None:
            return jsonify({
                'answer': 'Sorry, the chatbot is currently unavailable. Please try again later.',
                'error': 'Model not loaded'
            }), 500
        
        # Get user question from request
        data = request.get_json()
        user_question = data.get('question', '').strip()
        
        if not user_question:
            return jsonify({
                'answer': 'Please ask a question!',
                'error': 'Empty question'
            }), 400
        
        # Get response from chatbot model
        response = chatbot_model.find_best_match(user_question)
        
        # Return the response
        return jsonify({
            'answer': response['answer'],
            'confidence': response['confidence'],
            'similarity_score': response['similarity_score']
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'answer': 'Sorry, there was an error processing your request. Please try again.',
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': chatbot_model is not None
    }
    
    if chatbot_model:
        status['model_info'] = chatbot_model.model_info
    
    return jsonify(status)

if __name__ == '__main__':
    # Load the chatbot model on startup
    load_chatbot_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

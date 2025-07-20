import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FAQChatbotLoader:
    """Alternative model loader class"""
    
    def __init__(self, model_path='models/faq_chatbot_model.pkl'):
        self.model_path = model_path
        self.vectorizer = None
        self.faq_vectors = None
        self.faq_data = None
        self.preprocessor = None
        self.model_info = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.faq_vectors = model_data['faq_vectors']
            self.faq_data = model_data['faq_data']
            self.preprocessor = model_data['preprocessor']
            self.model_info = model_data['model_info']
            
            print(f"✅ Model loaded: {self.model_info['num_faqs']} FAQs")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, user_question, threshold=0.1):
        """Get chatbot response for user question"""
        # Preprocess user question
        processed_question = self.preprocessor.preprocess_text(user_question)
        
        # Convert to TF-IDF vector
        user_vector = self.vectorizer.transform([processed_question])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.faq_vectors).flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score > threshold:
            return {
                'answer': self.faq_data.iloc[best_idx]['answer'],
                'question': self.faq_data.iloc[best_idx]['question'],
                'confidence': 'High' if best_score > 0.5 else 'Medium',
                'similarity_score': float(best_score)
            }
        else:
            return {
                'answer': "I'm sorry, I couldn't find a relevant answer. Please contact support.",
                'question': None,
                'confidence': 'Low',
                'similarity_score': float(best_score)
            }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1-3: User Representative Movie Selection Module
Select representative movies for users based on refined taste analysis
"""

import os
import json
import argparse
import pandas as pd
import time
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from openai import OpenAI

class UserMovieMemory:
    """Memory module for storing user movie selection patterns and history with learning capabilities"""
    
    def __init__(self):
        """Initialize the memory module"""
        self.user_selections = {}  # user_id -> selection_data
        self.candidate_selections = {}  # user_id -> candidate_items
        self.representative_selections = {}  # user_id -> representative_items
        self.processing_history = []  # List of processing records
        self.selection_patterns = {}  # user_id -> selection_summary
        self.confidence_scores = {}  # user_id -> confidence_score
        
        # Learning and pattern analysis
        self.movie_patterns = {}  # pattern_type -> frequency
        self.successful_selections = []  # List of successful selection patterns
        self.genre_preferences = {}  # user_id -> genre_preferences
        self.rating_patterns = {}  # rating_pattern -> selection_characteristics
        self.learned_insights = []  # List of learned insights
        self.quality_metrics = {}  # user_id -> quality_metrics
        
        print("✅ User Movie Memory initialized with learning capabilities")
    
    def store_user_selection(self, user_id: int, candidate_items: List[Dict], 
                           representative_items: List[Dict], selection_summary: str, 
                           confidence: float = 0.8):
        """Store a user movie selection result
        
        Args:
            user_id: User ID
            candidate_items: Selected candidate items
            representative_items: Selected representative items
            selection_summary: Summary of the selection process
            confidence: Confidence score (0-1)
        """
        self.user_selections[user_id] = {
            'candidate_items': candidate_items,
            'representative_items': representative_items,
            'selection_summary': selection_summary,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self.candidate_selections[user_id] = candidate_items
        self.representative_selections[user_id] = representative_items
        self.selection_patterns[user_id] = selection_summary
        self.confidence_scores[user_id] = confidence
        
        # Add to processing history
        self.processing_history.append({
            'user_id': user_id,
            'timestamp': time.time(),
            'confidence': confidence,
            'candidate_count': len(candidate_items),
            'representative_count': len(representative_items)
        })
        
        # Learn from this experience
        self._learn_from_user_selection(user_id, candidate_items, representative_items, confidence)
    
    def get_user_selection(self, user_id: int) -> Optional[Dict]:
        """Get stored selection information for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with selection information or None if not found
        """
        if user_id in self.user_selections:
            return self.user_selections[user_id]
        return None
    
    def has_user_selection(self, user_id: int) -> bool:
        """Check if user selection is already stored
        
        Args:
            user_id: User ID
            
        Returns:
            True if user selection is stored, False otherwise
        """
        return user_id in self.user_selections
    
    def _learn_from_user_selection(self, user_id: int, candidate_items: List[Dict], 
                                 representative_items: List[Dict], confidence: float):
        """Learn from user movie selection experience
        
        Args:
            user_id: User ID
            candidate_items: Selected candidate items
            representative_items: Selected representative items
            confidence: Confidence score
        """
        # Analyze movie patterns
        self._analyze_movie_patterns(candidate_items, representative_items, confidence)
        
        # Learn from successful patterns
        if confidence > 0.7:
            self._learn_successful_patterns(candidate_items, representative_items)
        
        # Extract insights
        self._extract_insights(user_id, candidate_items, representative_items, confidence)
        
        # Update quality metrics
        self._update_quality_metrics(user_id, confidence)
    
    def _analyze_movie_patterns(self, candidate_items: List[Dict], 
                              representative_items: List[Dict], confidence: float):
        """Analyze movie patterns from selections
        
        Args:
            candidate_items: Selected candidate items
            representative_items: Selected representative items
            confidence: Confidence score
        """
        # Analyze genre patterns
        all_items = candidate_items + representative_items
        for item in all_items:
            movie_name = item.get('movie_name', '').lower()
            
            # Genre patterns
            genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller', 'documentary']
            for genre in genres:
                if genre in movie_name:
                    pattern_key = f"genre_{genre}"
                    self.movie_patterns[pattern_key] = self.movie_patterns.get(pattern_key, 0) + 1
        
        # Rating patterns
        high_rated = [item for item in all_items if item.get('rating', 0) >= 4]
        if len(high_rated) > len(all_items) * 0.7:
            self.movie_patterns['high_rating_preference'] = self.movie_patterns.get('high_rating_preference', 0) + 1
    
    def _learn_successful_patterns(self, candidate_items: List[Dict], representative_items: List[Dict]):
        """Learn from successful movie selection patterns
        
        Args:
            candidate_items: Selected candidate items
            representative_items: Selected representative items
        """
        if len(candidate_items) > 0 and len(representative_items) > 0:
            pattern = {
                'candidate_count': len(candidate_items),
                'representative_count': len(representative_items),
                'selection_ratio': len(representative_items) / len(candidate_items) if len(candidate_items) > 0 else 1.0,
                'timestamp': time.time()
            }
            self.successful_selections.append(pattern)
            
            # Keep only recent successful patterns (last 100)
            if len(self.successful_selections) > 100:
                self.successful_selections = self.successful_selections[-100:]
    
    def _extract_insights(self, user_id: int, candidate_items: List[Dict], 
                         representative_items: List[Dict], confidence: float):
        """Extract insights from movie selection process
        
        Args:
            user_id: User ID
            candidate_items: Selected candidate items
            representative_items: Selected representative items
            confidence: Confidence score
        """
        # Analyze selection patterns
        if len(candidate_items) > 0 and len(representative_items) > 0:
            selection_ratio = len(representative_items) / len(candidate_items)
            
            if selection_ratio > 0.3:  # High selection ratio
                insight = {
                    'type': 'high_selection_ratio',
                    'user_id': user_id,
                    'selection_ratio': selection_ratio,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                self.learned_insights.append(insight)
            
            if confidence > 0.8:
                insight = {
                    'type': 'high_confidence_selection',
                    'user_id': user_id,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                self.learned_insights.append(insight)
        
        # Keep only recent insights (last 50)
        if len(self.learned_insights) > 50:
            self.learned_insights = self.learned_insights[-50:]
    
    def _update_quality_metrics(self, user_id: int, confidence: float):
        """Update quality metrics for user
        
        Args:
            user_id: User ID
            confidence: Confidence score
        """
        if user_id not in self.quality_metrics:
            self.quality_metrics[user_id] = {
                'total_selections': 0,
                'avg_confidence': 0.0,
                'high_confidence_count': 0,
                'last_updated': time.time()
            }
        
        metrics = self.quality_metrics[user_id]
        
        # Ensure all required keys exist
        if 'total_selections' not in metrics:
            metrics['total_selections'] = 0
        if 'avg_confidence' not in metrics:
            metrics['avg_confidence'] = 0.0
        if 'high_confidence_count' not in metrics:
            metrics['high_confidence_count'] = 0
        if 'last_updated' not in metrics:
            metrics['last_updated'] = time.time()
        
        metrics['total_selections'] += 1
        metrics['avg_confidence'] = (metrics['avg_confidence'] * (metrics['total_selections'] - 1) + confidence) / metrics['total_selections']
        
        if confidence > 0.7:
            metrics['high_confidence_count'] += 1
        
        metrics['last_updated'] = time.time()
    
    def get_learned_insights(self) -> List[Dict]:
        """Get learned insights from memory
        
        Returns:
            List of learned insights
        """
        return self.learned_insights.copy()
    
    def get_movie_patterns(self) -> Dict:
        """Get learned movie patterns
        
        Returns:
            Dictionary of movie patterns and their frequencies
        """
        return self.movie_patterns.copy()
    
    def get_successful_selections(self) -> List[Dict]:
        """Get successful selection patterns
        
        Returns:
            List of successful selection patterns
        """
        return self.successful_selections.copy()
    
    def get_quality_metrics(self, user_id: int = None) -> Dict:
        """Get quality metrics
        
        Args:
            user_id: Specific user ID, or None for all users
            
        Returns:
            Quality metrics for user(s)
        """
        if user_id:
            return self.quality_metrics.get(user_id, {})
        else:
            return self.quality_metrics.copy()
    
    def get_memory_learning_summary(self) -> Dict:
        """Get summary of learning progress
        
        Returns:
            Dictionary with learning summary
        """
        return {
            'total_users_processed': len(self.user_selections),
            'learned_patterns': len(self.movie_patterns),
            'successful_selections': len(self.successful_selections),
            'learned_insights': len(self.learned_insights),
            'avg_confidence': sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            'high_confidence_users': sum(1 for conf in self.confidence_scores.values() if conf > 0.7),
            'top_patterns': sorted(self.movie_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def save_to_file(self, memory_file: str):
        """Save complete memory state to file
        
        Args:
            memory_file: Path to save memory file
        """
        try:
            memory_data = {
                'user_selections': self.user_selections,
                'candidate_selections': self.candidate_selections,
                'representative_selections': self.representative_selections,
                'processing_history': self.processing_history,
                'selection_patterns': self.selection_patterns,
                'confidence_scores': self.confidence_scores,
                # Learning data
                'movie_patterns': self.movie_patterns,
                'successful_selections': self.successful_selections,
                'genre_preferences': self.genre_preferences,
                'rating_patterns': self.rating_patterns,
                'learned_insights': self.learned_insights,
                'quality_metrics': self.quality_metrics,
                'metadata': {
                    'total_users': len(self.user_selections),
                    'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'version': '2.0',
                    'learning_enabled': True
                }
            }
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ User Movie Memory saved to {memory_file} ({len(self.user_selections)} users)")
            
        except Exception as e:
            print(f"❌ Error saving user movie memory: {e}")
    
    def load_from_file(self, memory_file: str) -> bool:
        """Load memory state from file
        
        Args:
            memory_file: Path to memory file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(memory_file):
            print(f"⚠️ Memory file {memory_file} does not exist")
            return False
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Load memory data
            self.user_selections = memory_data.get('user_selections', {})
            self.candidate_selections = memory_data.get('candidate_selections', {})
            self.representative_selections = memory_data.get('representative_selections', {})
            self.processing_history = memory_data.get('processing_history', [])
            self.selection_patterns = memory_data.get('selection_patterns', {})
            self.confidence_scores = memory_data.get('confidence_scores', {})
            
            # Load learning data (with backward compatibility)
            self.movie_patterns = memory_data.get('movie_patterns', {})
            self.successful_selections = memory_data.get('successful_selections', [])
            self.genre_preferences = memory_data.get('genre_preferences', {})
            self.rating_patterns = memory_data.get('rating_patterns', {})
            self.learned_insights = memory_data.get('learned_insights', [])
            self.quality_metrics = memory_data.get('quality_metrics', {})
            
            # Convert string keys back to integers
            self.user_selections = {int(k): v for k, v in self.user_selections.items()}
            self.candidate_selections = {int(k): v for k, v in self.candidate_selections.items()}
            self.representative_selections = {int(k): v for k, v in self.representative_selections.items()}
            self.selection_patterns = {int(k): v for k, v in self.selection_patterns.items()}
            self.confidence_scores = {int(k): v for k, v in self.confidence_scores.items()}
            self.genre_preferences = {int(k): v for k, v in self.genre_preferences.items()}
            
            # Convert quality_metrics and ensure complete structure
            self.quality_metrics = {}
            for k, v in memory_data.get('quality_metrics', {}).items():
                user_id = int(k)
                # Ensure quality_metrics has complete structure
                if isinstance(v, dict):
                    self.quality_metrics[user_id] = {
                        'total_selections': v.get('total_selections', 0),
                        'avg_confidence': v.get('avg_confidence', 0.0),
                        'high_confidence_count': v.get('high_confidence_count', 0),
                        'last_updated': v.get('last_updated', time.time())
                    }
                else:
                    # Fallback for incomplete data
                    self.quality_metrics[user_id] = {
                        'total_selections': 0,
                        'avg_confidence': 0.0,
                        'high_confidence_count': 0,
                        'last_updated': time.time()
                    }
            
            metadata = memory_data.get('metadata', {})
            print(f"✅ User Movie Memory loaded from {memory_file}")
            print(f"   - Users: {len(self.user_selections)}")
            print(f"   - Last updated: {metadata.get('last_updated', 'Unknown')}")
            print(f"   - Version: {metadata.get('version', 'Unknown')}")
            print(f"   - Learning enabled: {metadata.get('learning_enabled', False)}")
            if metadata.get('learning_enabled', False):
                print(f"   - Learned patterns: {len(self.movie_patterns)}")
                print(f"   - Learned insights: {len(self.learned_insights)}")
                print(f"   - Successful selections: {len(self.successful_selections)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading user movie memory: {e}")
            return False
    
    def clear_memory(self):
        """Clear all memory data"""
        self.user_selections.clear()
        self.candidate_selections.clear()
        self.representative_selections.clear()
        self.processing_history.clear()
        self.selection_patterns.clear()
        self.confidence_scores.clear()
        # Clear learning data
        self.movie_patterns.clear()
        self.successful_selections.clear()
        self.genre_preferences.clear()
        self.rating_patterns.clear()
        self.learned_insights.clear()
        self.quality_metrics.clear()
        print("🧹 User Movie Memory cleared (including learning data)")
    
    def get_memory_info(self) -> Dict:
        """Get detailed memory information
        
        Returns:
            Dictionary with detailed memory information
        """
        return {
            'total_users': len(self.user_selections),
            'avg_confidence': sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            'processing_history_count': len(self.processing_history),
            'memory_size_mb': sum(len(str(v)) for v in self.user_selections.values()) / (1024 * 1024),
            'recent_users': list(self.user_selections.keys())[-5:] if self.user_selections else [],
            'learned_patterns': len(self.movie_patterns),
            'learned_insights': len(self.learned_insights),
            'successful_selections': len(self.successful_selections)
        }

class MovieSelectionDebateModule:
    """Debate module for movie selection - Multiple expert perspectives"""
    
    def __init__(self, client, parent_generator):
        self.client = client
        self.parent_generator = parent_generator
    
    def debate_candidate_selection(self, user_id: int, profile: Dict, taste: str, interactions: List[Dict]) -> Dict:
        """Conduct debate for candidate item selection
        
        Args:
            user_id: User ID
            profile: User profile
            taste: User taste description
            interactions: User interaction history
            
        Returns:
            Dictionary with debate results
        """
        debate_prompt = f"""You are a movie selection expert. Three specialized agents will debate about selecting candidate movies for user {user_id}.

User Profile:
- User ID: {user_id}
- Demographics: {profile.get('demographics', {})}
- Interaction Count: {len(interactions)}

User Taste Description:
{taste}

User Interaction History (Top 20 by rating):
{self._format_interactions(interactions[:20])}

The three agents are:
1. **Rating Analyst**: Focuses on rating patterns and user satisfaction
2. **Genre Specialist**: Analyzes genre preferences and thematic consistency  
3. **Diversity Expert**: Ensures variety and prevents over-specialization

Please simulate a constructive debate among these three agents and reach a consensus on the best candidate movies. The final consensus should select 5 movies that best represent the user's preferences.

Format your response as:
DEBATE PROCESS: [Describe the debate process]
CONSENSUS: [Final consensus with 5 selected movies]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": debate_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                if "CONSENSUS:" in response_text:
                    consensus = response_text.split("CONSENSUS:")[-1].strip()
                    debate_process = response_text.split("CONSENSUS:")[0].replace("DEBATE PROCESS:", "").strip()
                else:
                    consensus = response_text
                    debate_process = "Debate process not clearly formatted"
                
                self.parent_generator._smart_delay()
                return {
                    'consensus': consensus,
                    'debate_process': debate_process
                }
                
            except Exception as e:
                print(f"❌ Debate attempt {attempt + 1} failed: {e}")
                self.parent_generator._increment_failure_count()
                if attempt < self.parent_generator.retry_attempts - 1:
                    self.parent_generator._smart_delay()
                else:
                    return {
                        'consensus': "Debate failed - using fallback selection",
                        'debate_process': f"Error: {e}"
                    }
    
    def debate_representative_selection(self, user_id: int, profile: Dict, taste: str, 
                                     candidate_items: List[Dict], all_interactions: List[Dict]) -> Dict:
        """Conduct debate for representative item selection
        
        Args:
            user_id: User ID
            profile: User profile
            taste: User taste description
            candidate_items: Selected candidate items
            all_interactions: All user interactions
            
        Returns:
            Dictionary with debate results
        """
        debate_prompt = f"""You are a movie selection expert. Three specialized agents will debate about selecting representative movies for user {user_id}.

User Profile:
- User ID: {user_id}
- Demographics: {profile.get('demographics', {})}
- Total Interactions: {len(all_interactions)}

User Taste Description:
{taste}

Selected Candidate Movies (5):
{self._format_items(candidate_items)}

All User Interactions:
{self._format_interactions(all_interactions)}

The three agents are:
1. **Representativeness Expert**: Ensures selected movies truly represent user preferences
2. **Coverage Specialist**: Analyzes genre and thematic coverage
3. **Quality Assessor**: Evaluates movie quality and user satisfaction correlation

Please simulate a constructive debate among these three agents and reach a consensus on selecting up to 50 representative movies that best capture the user's taste.

Format your response as:
DEBATE PROCESS: [Describe the debate process]
CONSENSUS: [Final consensus with selected representative movies]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": debate_prompt}],
                    max_tokens=600,
                    temperature=0.7
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                if "CONSENSUS:" in response_text:
                    consensus = response_text.split("CONSENSUS:")[-1].strip()
                    debate_process = response_text.split("CONSENSUS:")[0].replace("DEBATE PROCESS:", "").strip()
                else:
                    consensus = response_text
                    debate_process = "Debate process not clearly formatted"
                
                self.parent_generator._smart_delay()
                return {
                    'consensus': consensus,
                    'debate_process': debate_process
                }
                
            except Exception as e:
                print(f"❌ Debate attempt {attempt + 1} failed: {e}")
                self.parent_generator._increment_failure_count()
                if attempt < self.parent_generator.retry_attempts - 1:
                    self.parent_generator._smart_delay()
                else:
                    return {
                        'consensus': "Debate failed - using fallback selection",
                        'debate_process': f"Error: {e}"
                    }
    
    def debate_representative_liked_selection(self, user_id: int, profile: Dict, taste: str, 
                                            liked_interactions: List[Dict]) -> Dict:
        """Conduct debate for representative liked items selection
        
        Args:
            user_id: User ID
            profile: User profile
            taste: User taste description
            liked_interactions: User liked interaction history (rating >= 4)
            
        Returns:
            Dictionary with debate results
        """
        debate_prompt = f"""You are a movie selection expert. Three specialized agents will debate about selecting representative liked movies for user {user_id}.

User Profile:
- User ID: {user_id}
- Demographics: {profile.get('demographics', {})}
- Total Liked Interactions: {len(liked_interactions)}

User Taste Description:
{taste}

User Liked Movies (Rating >= 4):
{self._format_interactions(liked_interactions[:20])}

The three agents are:
1. **Representativeness Expert**: Ensures selected movies truly represent user preferences
2. **Coverage Specialist**: Analyzes genre and thematic coverage
3. **Quality Assessor**: Evaluates movie quality and user satisfaction correlation

Please simulate a constructive debate among these three agents and reach a consensus on selecting up to 50 representative liked movies that best capture the user's taste.

Format your response as:
DEBATE PROCESS: [Describe the debate process]
CONSENSUS: [Final consensus with selected representative liked movies]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": debate_prompt}],
                    max_tokens=600,
                    temperature=0.6
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                if "CONSENSUS:" in response_text:
                    consensus = response_text.split("CONSENSUS:")[-1].strip()
                    debate_process = response_text.split("CONSENSUS:")[0].replace("DEBATE PROCESS:", "").strip()
                else:
                    consensus = response_text
                    debate_process = "Debate process not clearly formatted"
                
                self.parent_generator._smart_delay()
                return {
                    'consensus': consensus,
                    'debate_process': debate_process
                }
                
            except Exception as e:
                print(f"❌ Debate attempt {attempt + 1} failed: {e}")
                self.parent_generator._increment_failure_count()
                if attempt < self.parent_generator.retry_attempts - 1:
                    self.parent_generator._smart_delay()
                else:
                    return {
                        'consensus': "Debate failed - using fallback selection",
                        'debate_process': f"Error: {e}"
                    }
    
    def _format_interactions(self, interactions: List[Dict]) -> str:
        """Format interactions for display"""
        if not interactions:
            return "No interactions available"
        
        formatted = []
        for i, interaction in enumerate(interactions[:20], 1):
            movie_name = interaction.get('movie_name', 'Unknown')
            rating = interaction.get('rating', 0)
            formatted.append(f"{i}. {movie_name} (Rating: {rating})")
        
        return "\n".join(formatted)
    
    def _format_items(self, items: List[Dict]) -> str:
        """Format items for display"""
        if not items:
            return "No items available"
        
        formatted = []
        for i, item in enumerate(items, 1):
            movie_name = item.get('movie_name', 'Unknown')
            rating = item.get('rating', 0)
            formatted.append(f"{i}. {movie_name} (Rating: {rating})")
        
        return "\n".join(formatted)

class MovieSelectionReflectionModule:
    """Reflection module for movie selection - Self-reflection and improvement"""
    
    def __init__(self, client, parent_generator):
        self.client = client
        self.parent_generator = parent_generator
    
    def reflect_candidate_selection(self, user_id: int, profile: Dict, taste: str, 
                                  interactions: List[Dict], debated_result: Dict) -> Dict:
        """Reflect on candidate selection results
        
        Args:
            user_id: User ID
            profile: User profile
            taste: User taste description
            interactions: User interaction history
            debated_result: Results from debate module
            
        Returns:
            Dictionary with reflection results
        """
        reflection_prompt = f"""You are a movie selection expert reflecting on the candidate selection for user {user_id}.

User Profile:
- User ID: {user_id}
- Demographics: {profile.get('demographics', {})}
- Interaction Count: {len(interactions)}

User Taste Description:
{taste}

User Interaction History (Top 20 by rating):
{self._format_interactions(interactions[:20])}

Previous Debate Result:
{debated_result.get('consensus', 'No consensus available')}

Please reflect on the candidate selection and provide a refined, improved selection. Consider:
1. Are the selected movies truly representative of the user's taste?
2. Is there good diversity in the selection?
3. Are there any important preferences that were missed?
4. How can the selection be improved?

Format your response as:
REFLECTION: [Your analysis and insights]
REFINED SELECTION: [Improved candidate selection]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    max_tokens=400,
                    temperature=0.6
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                if "REFINED SELECTION:" in response_text:
                    refined_selection = response_text.split("REFINED SELECTION:")[-1].strip()
                    reflection = response_text.split("REFINED SELECTION:")[0].replace("REFLECTION:", "").strip()
                else:
                    refined_selection = response_text
                    reflection = "Reflection process not clearly formatted"
                
                self.parent_generator._smart_delay()
                return {
                    'refined_selection': refined_selection,
                    'reflection': reflection
                }
                
            except Exception as e:
                print(f"❌ Reflection attempt {attempt + 1} failed: {e}")
                self.parent_generator._increment_failure_count()
                if attempt < self.parent_generator.retry_attempts - 1:
                    self.parent_generator._smart_delay()
                else:
                    return {
                        'refined_selection': debated_result.get('consensus', 'Reflection failed'),
                        'reflection': f"Error: {e}"
                    }
    
    def reflect_representative_selection(self, user_id: int, profile: Dict, taste: str,
                                      candidate_items: List[Dict], all_interactions: List[Dict],
                                      debated_result: Dict) -> Dict:
        """Reflect on representative selection results
        
        Args:
            user_id: User ID
            profile: User profile
            taste: User taste description
            candidate_items: Selected candidate items
            all_interactions: All user interactions
            debated_result: Results from debate module
            
        Returns:
            Dictionary with reflection results
        """
        reflection_prompt = f"""You are a movie selection expert reflecting on the representative selection for user {user_id}.

User Profile:
- User ID: {user_id}
- Demographics: {profile.get('demographics', {})}
- Total Interactions: {len(all_interactions)}

User Taste Description:
{taste}

Selected Candidate Movies (5):
{self._format_items(candidate_items)}

All User Interactions:
{self._format_interactions(all_interactions)}

Previous Debate Result:
{debated_result.get('consensus', 'No consensus available')}

Please reflect on the representative selection and provide a refined, improved selection. Consider:
1. Do the selected movies provide good coverage of the user's preferences?
2. Is the selection diverse enough to avoid over-specialization?
3. Are there any important movies that should be included?
4. How can the selection be optimized for representativeness?

Format your response as:
REFLECTION: [Your analysis and insights]
REFINED SELECTION: [Improved representative selection]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    max_tokens=500,
                    temperature=0.6
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                if "REFINED SELECTION:" in response_text:
                    refined_selection = response_text.split("REFINED SELECTION:")[-1].strip()
                    reflection = response_text.split("REFINED SELECTION:")[0].replace("REFLECTION:", "").strip()
                else:
                    refined_selection = response_text
                    reflection = "Reflection process not clearly formatted"
                
                self.parent_generator._smart_delay()
                return {
                    'refined_selection': refined_selection,
                    'reflection': reflection
                }
                
            except Exception as e:
                print(f"❌ Reflection attempt {attempt + 1} failed: {e}")
                self.parent_generator._increment_failure_count()
                if attempt < self.parent_generator.retry_attempts - 1:
                    self.parent_generator._smart_delay()
                else:
                    return {
                        'refined_selection': debated_result.get('consensus', 'Reflection failed'),
                        'reflection': f"Error: {e}"
                    }
    
    def reflect_representative_liked_selection(self, user_id: int, profile: Dict, taste: str,
                                             liked_interactions: List[Dict], debated_result: Dict) -> Dict:
        """Reflect on representative liked selection results
        
        Args:
            user_id: User ID
            profile: User profile
            taste: User taste description
            liked_interactions: User liked interaction history (rating >= 4)
            debated_result: Results from debate module
            
        Returns:
            Dictionary with reflection results
        """
        reflection_prompt = f"""You are a movie selection expert reflecting on the representative liked selection for user {user_id}.

User Profile:
- User ID: {user_id}
- Demographics: {profile.get('demographics', {})}
- Total Liked Interactions: {len(liked_interactions)}

User Taste Description:
{taste}

User Liked Movies (Rating >= 4):
{self._format_interactions(liked_interactions[:20])}

Previous Debate Result:
{debated_result.get('consensus', 'No consensus available')}

Please reflect on the representative liked selection and provide a refined, improved selection. Consider:
1. Are the selected movies truly representative of the user's taste?
2. Is there good diversity in the selection?
3. Are there any important preferences that were missed?
4. How can the selection be improved?

Format your response as:
REFLECTION: [Your analysis and insights]
REFINED SELECTION: [Improved representative liked selection]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    max_tokens=500,
                    temperature=0.6
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                if "REFINED SELECTION:" in response_text:
                    refined_selection = response_text.split("REFINED SELECTION:")[-1].strip()
                    reflection = response_text.split("REFINED SELECTION:")[0].replace("REFLECTION:", "").strip()
                else:
                    refined_selection = response_text
                    reflection = "Reflection process not clearly formatted"
                
                self.parent_generator._smart_delay()
                return {
                    'refined_selection': refined_selection,
                    'reflection': reflection
                }
                
            except Exception as e:
                print(f"❌ Reflection attempt {attempt + 1} failed: {e}")
                self.parent_generator._increment_failure_count()
                if attempt < self.parent_generator.retry_attempts - 1:
                    self.parent_generator._smart_delay()
                else:
                    return {
                        'refined_selection': debated_result.get('consensus', 'Reflection failed'),
                        'reflection': f"Error: {e}"
                    }
    
    def _format_interactions(self, interactions: List[Dict]) -> str:
        """Format interactions for display"""
        if not interactions:
            return "No interactions available"
        
        formatted = []
        for i, interaction in enumerate(interactions[:20], 1):
            movie_name = interaction.get('movie_name', 'Unknown')
            rating = interaction.get('rating', 0)
            formatted.append(f"{i}. {movie_name} (Rating: {rating})")
        
        return "\n".join(formatted)
    
    def _format_items(self, items: List[Dict]) -> str:
        """Format items for display"""
        if not items:
            return "No items available"
        
        formatted = []
        for i, item in enumerate(items, 1):
            movie_name = item.get('movie_name', 'Unknown')
            rating = item.get('rating', 0)
            formatted.append(f"{i}. {movie_name} (Rating: {rating})")
        
        return "\n".join(formatted)

class UserMovieSelector:
    """User Movie Selector Module - Select representative movies for users"""
    
    def __init__(self, user_profiles_file: str = "user_profiles.json", 
                 user_tastes_file: str = "user_tastes.json",
                 openai_api_key: str = None, memory_file: str = None,
                 enable_debate_reflection: bool = True):
        self.user_profiles_file = user_profiles_file
        self.user_tastes_file = user_tastes_file
        self.memory_file = memory_file
        self.enable_debate_reflection = enable_debate_reflection
        
        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment variable
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️  Warning: No OpenAI API key provided. GPT analysis will be disabled.")
                self.client = None
        
        # Rate limiting configuration
        self.base_delay = 0.5  # Base delay between requests (seconds)
        self.max_delay = 5.0   # Maximum delay between requests (seconds)
        self.requests_per_minute = 60  # Target requests per minute
        self.retry_attempts = 3  # Number of retry attempts for failed requests
        self.consecutive_failures = 0  # Track consecutive failures
        
        # Initialize memory module
        self.memory = UserMovieMemory()
        
        # Load memory if file is provided
        if self.memory_file:
            self.load_memory()
        
        # Load data
        self.user_profiles = self._load_user_profiles()
        self.user_tastes = self._load_user_tastes()
        
        # Initialize modules only if debate and reflection are enabled
        if self.enable_debate_reflection and self.client:
            self.debate_module = MovieSelectionDebateModule(self.client, self)
            self.reflection_module = MovieSelectionReflectionModule(self.client, self)
            print("🧠 Debate and Reflection modules enabled")
        else:
            self.debate_module = None
            self.reflection_module = None
            print("⚡ Simple mode: Direct GPT generation only")
        
        print(f"✅ User Movie Selector initialized with {len(self.user_profiles)} user profiles")
        print(f"✅ Loaded user tastes for {len(self.user_tastes.get('refined_tastes', {}))} users")
        print(f"📊 Rate limiting: {self.requests_per_minute} requests/minute")
        print(f"⏱️  Base delay: {self.base_delay}s, Max delay: {self.max_delay}s")
        if self.memory_file:
            print(f"🧠 Memory file: {self.memory_file}")
        if self.client:
            print(f"✅ OpenAI client initialized for GPT analysis")
        else:
            print(f"⚠️  OpenAI client not available - using fallback selection method")
        
        # Validate user ID consistency
        self._validate_user_consistency()
    
    def get_filtered_users(self, filtered_train_file: str = "filtered_train_set.txt", 
                          rec_save_dict_file: str = "rec_save_dict.csv") -> Set[int]:
        """Get users that exist in both filtered_train_set.txt and rec_save_dict.csv
        
        Args:
            filtered_train_file: Path to filtered train set file
            rec_save_dict_file: Path to rec save dict file
            
        Returns:
            Set of user IDs that exist in both files
        """
        try:
            # Load users from filtered_train_set.txt
            filtered_users = set()
            if os.path.exists(filtered_train_file):
                with open(filtered_train_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()  # Split by whitespace (space or tab)
                            if len(parts) >= 1:
                                try:
                                    user_id = int(parts[0])
                                    filtered_users.add(user_id)
                                except ValueError:
                                    continue
            
            # Load users from rec_save_dict.csv
            rec_users = set()
            if os.path.exists(rec_save_dict_file):
                import pandas as pd
                try:
                    df = pd.read_csv(rec_save_dict_file)
                    if len(df.columns) > 0:
                        rec_users = set(df.iloc[:, 0].astype(int))
                except Exception as e:
                    print(f"⚠️ Error reading {rec_save_dict_file}: {e}")
            
            # Return intersection
            common_users = filtered_users.intersection(rec_users)
            
            print(f"🔍 Filtered Users Analysis:")
            print(f"   - Users in filtered_train_set: {len(filtered_users)}")
            print(f"   - Users in rec_save_dict: {len(rec_users)}")
            print(f"   - Common users: {len(common_users)}")
            
            return common_users
            
        except Exception as e:
            print(f"❌ Error loading filtered users: {e}")
            return set()
    
    def get_filtered_and_rec_users(self, filtered_train_file: str = "filtered_train_set.txt", 
                                  rec_save_dict_file: str = "rec_save_dict.csv") -> Tuple[Set[int], Set[int]]:
        """Get users from filtered_train_set.txt and rec_save_dict.csv separately
        
        Args:
            filtered_train_file: Path to filtered train set file
            rec_save_dict_file: Path to rec save dict file
            
        Returns:
            Tuple of (filtered_users, rec_users) sets
        """
        try:
            # Load users from filtered_train_set.txt
            filtered_users = set()
            if os.path.exists(filtered_train_file):
                with open(filtered_train_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()  # Split by whitespace (space or tab)
                            if len(parts) >= 1:
                                try:
                                    user_id = int(parts[0])
                                    filtered_users.add(user_id)
                                except ValueError:
                                    continue
            
            # Load users from rec_save_dict.csv
            rec_users = set()
            if os.path.exists(rec_save_dict_file):
                import pandas as pd
                try:
                    df = pd.read_csv(rec_save_dict_file)
                    if len(df.columns) > 0:
                        rec_users = set(df.iloc[:, 0].astype(int))
                except Exception as e:
                    print(f"⚠️ Error reading {rec_save_dict_file}: {e}")
            
            print(f"🔍 Filtered Users Analysis:")
            print(f"   - Users in filtered_train_set: {len(filtered_users)}")
            print(f"   - Users in rec_save_dict: {len(rec_users)}")
            
            return filtered_users, rec_users
            
        except Exception as e:
            print(f"❌ Error loading filtered users: {e}")
            return set(), set()
    
    def load_memory(self, memory_file: str = None):
        """Load memory from file
        
        Args:
            memory_file: Path to memory file (optional, uses self.memory_file if not provided)
        """
        if memory_file:
            self.memory_file = memory_file
        
        if self.memory_file:
            success = self.memory.load_from_file(self.memory_file)
            if success:
                print(f"🧠 User Movie Memory loaded successfully from {self.memory_file}")
            else:
                print(f"🆕 No existing memory file found at {self.memory_file}")
                print(f"   This appears to be the first run. Memory will be created after processing.")
        else:
            print("⚠️ No memory file specified")
    
    def save_memory(self, memory_file: str = None):
        """Save memory to file
        
        Args:
            memory_file: Path to save memory file (optional, uses self.memory_file if not provided)
        """
        if memory_file:
            self.memory_file = memory_file
        
        if self.memory_file:
            self.memory.save_to_file(self.memory_file)
        else:
            print("⚠️ No memory file specified for saving")
    
    def clear_memory(self):
        """Clear all memory data"""
        self.memory.clear_memory()
    
    def _smart_delay(self):
        """Implement smart delay with exponential backoff for API rate limiting"""
        if self.consecutive_failures == 0:
            # Normal delay
            delay = self.base_delay
        else:
            # Exponential backoff
            delay = min(self.base_delay * (2 ** self.consecutive_failures), self.max_delay)
        
        # Add some randomness to avoid thundering herd
        delay += random.uniform(0, 0.5)
        
        time.sleep(delay)
    
    def _reset_failure_count(self):
        """Reset consecutive failure count after successful request"""
        self.consecutive_failures = 0
    
    def _increment_failure_count(self):
        """Increment consecutive failure count after failed request"""
        self.consecutive_failures += 1
    
    def _reset_failure_counter_between_stages(self, stage_name: str):
        """Reset failure counter between processing stages
        
        Args:
            stage_name: Name of the stage for logging purposes
        """
        if self.consecutive_failures > 0:
            print(f"🔄 Reset failure counter {stage_name} (was {self.consecutive_failures})")
            self.consecutive_failures = 0
    
    def _load_user_profiles(self) -> Dict[int, Dict]:
        """Load user profiles from JSON file"""
        try:
            with open(self.user_profiles_file, 'r', encoding='utf-8') as f:
                user_profiles = json.load(f)
            
            # Convert string keys back to integers
            return {int(k): v for k, v in user_profiles.items()}
            
        except Exception as e:
            print(f"❌ Error loading user profiles: {e}")
            return {}
    
    def _load_user_tastes(self) -> Dict:
        """Load user tastes from JSON file"""
        try:
            with open(self.user_tastes_file, 'r', encoding='utf-8') as f:
                user_tastes = json.load(f)
            return user_tastes
            
        except Exception as e:
            print(f"❌ Error loading user tastes: {e}")
            return {}
    
    def _validate_user_consistency(self):
        """Validate that user IDs are consistent between profiles and tastes"""
        profile_user_ids = set(self.user_profiles.keys())
        taste_user_ids = set(self.user_tastes.get('refined_tastes', {}).keys())
        
        # Convert taste user IDs to integers for comparison
        taste_user_ids = {int(uid) for uid in taste_user_ids}
        
        # Find intersection and differences
        common_users = profile_user_ids.intersection(taste_user_ids)
        only_in_profiles = profile_user_ids - taste_user_ids
        only_in_tastes = taste_user_ids - profile_user_ids
        
        print(f"🔍 User ID Consistency Check:")
        print(f"  ✅ Common users: {len(common_users)}")
        if only_in_profiles:
            print(f"  ⚠️  Users only in profiles: {len(only_in_profiles)} (will be ignored)")
        if only_in_tastes:
            print(f"  ⚠️  Users only in tastes: {len(only_in_tastes)} (will be ignored)")
        
        if not common_users:
            print(f"  ❌ No common users found! Check your data files.")
        else:
            print(f"  ✅ Ready to process {len(common_users)} users")
    
    def select_candidate_items(self, user_ids: List[int] = None) -> Dict[int, Dict]:
        """Select candidate items using GPT analysis with dual mode support"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())
        
        print(f"🎬 Selecting candidate items for {len(user_ids)} users...")
        print(f"🔧 Mode: {'Full (Debate + Reflection)' if self.enable_debate_reflection else 'Simple (Direct GPT)'}")
        
        candidate_items = {}
        processed_count = 0
        skipped_count = 0
        
        for i, user_id in enumerate(user_ids):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Show progress every 10 users with current user ID
            if i % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(user_ids)} users processed... (Current: User {user_id})")
            
            # Show detailed progress every 50 users
            if i % 50 == 0 and i > 0:
                print(f"📈 Detailed Progress: {i}/{len(user_ids)} users processed ({i/len(user_ids)*100:.1f}%)")
            
            try:
                # Check if already processed in memory
                if self.memory.has_user_selection(user_id):
                    existing_selection = self.memory.get_user_selection(user_id)
                    candidate_items[user_id] = {
                        'candidate_movies': existing_selection['candidate_items'],
                        'total_candidates': len(existing_selection['candidate_items']),
                        'from_memory': True
                    }
                    skipped_count += 1
                    continue
                
                profile = self.user_profiles[user_id]
                interactions = profile['interactions']
                taste = self.user_tastes.get('refined_tastes', {}).get(str(user_id), "")
                
                if self.client and taste:
                    if self.enable_debate_reflection:
                        # Full mode: Debate + Reflection
                        print(f"🔄 Processing User {user_id} (Full Mode: Debate + Reflection)...")
                        candidate_list = self._full_mode_select_candidates(user_id, profile, taste, interactions)
                    else:
                        # Simple mode: Direct GPT
                        print(f"🔄 Processing User {user_id} (Simple Mode: Direct GPT)...")
                        candidate_list = self._simple_mode_select_candidates(user_id, profile, taste, interactions)
                else:
                    # Fallback to rating-based selection
                    print(f"🔄 Processing User {user_id} (Fallback Mode: Rating-based)...")
                    candidate_list = self._fallback_select_candidates(interactions)
                
                candidate_items[user_id] = {
                    'candidate_movies': candidate_list[:5],
                    'total_candidates': len(candidate_list),
                    'from_memory': False
                }
                
                processed_count += 1
                print(f"✅ User {user_id} processed successfully ({len(candidate_list)} candidates selected)")
                
            except Exception as e:
                print(f"❌ Error selecting candidate items for user {user_id}: {e}")
                candidate_items[user_id] = {
                    'candidate_movies': [],
                    'total_candidates': 0,
                    'from_memory': False
                }
                processed_count += 1
        
        print(f"✅ Candidate items selection completed:")
        print(f"   📊 Total users: {len(user_ids)}")
        print(f"   🔄 Processed: {processed_count}")
        print(f"   💾 From memory: {skipped_count}")
        return candidate_items
    
    def _full_mode_select_candidates(self, user_id: int, profile: Dict, taste: str, interactions: List[Dict]) -> List[Dict]:
        """Full mode: Use debate and reflection for candidate selection"""
        try:
            # Step 1: Debate
            debated_result = self.debate_module.debate_candidate_selection(user_id, profile, taste, interactions)
            
            # Step 2: Reflection
            reflected_result = self.reflection_module.reflect_candidate_selection(
                user_id, profile, taste, interactions, debated_result
            )
            
            # Step 3: Parse and return results
            final_selection = reflected_result.get('refined_selection', debated_result.get('consensus', ''))
            candidate_list = self._parse_candidate_selection(final_selection, interactions)
            
            # Store in memory
            self.memory.store_user_selection(
                user_id, candidate_list, [], 
                f"Full mode candidate selection: {debated_result.get('debate_process', '')} | {reflected_result.get('reflection', '')}",
                0.8
            )
            
            self._reset_failure_count()
            return candidate_list
            
        except Exception as e:
            print(f"❌ Full mode candidate selection failed for user {user_id}: {e}")
            self._increment_failure_count()
            return self._fallback_select_candidates(interactions)
    
    def _simple_mode_select_candidates(self, user_id: int, profile: Dict, taste: str, interactions: List[Dict]) -> List[Dict]:
        """Simple mode: Direct GPT for candidate selection"""
        try:
            # Build simple prompt
            prompt = self._build_simple_candidate_prompt(user_id, profile, taste, interactions)
            
            # Call GPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            candidate_list = self._parse_candidate_selection(response_text, interactions)
            
            # Store in memory
            self.memory.store_user_selection(
                user_id, candidate_list, [],
                f"Simple mode candidate selection: {response_text[:100]}...",
                0.7
            )
            
            self._smart_delay()
            self._reset_failure_count()
            return candidate_list
            
        except Exception as e:
            print(f"❌ Simple mode candidate selection failed for user {user_id}: {e}")
            self._increment_failure_count()
            return self._fallback_select_candidates(interactions)
    
    def _build_simple_candidate_prompt(self, user_id: int, profile: Dict, taste: str, interactions: List[Dict]) -> str:
        """Build simple prompt for candidate selection"""
        interactions_text = "\n".join([
            f"- {interaction.get('movie_name', 'Unknown')} (Rating: {interaction.get('rating', 0)})"
            for interaction in interactions[:20]
        ])
        
        return f"""Select 5 candidate movies for user {user_id} based on their taste and interaction history.

User Taste: {taste}

User Interaction History:
{interactions_text}

Please select 5 movies that best represent this user's preferences. Consider rating patterns, genre preferences, and overall taste alignment.

Format your response as a simple list of movie names."""
    
    def _parse_candidate_selection(self, selection_text: str, interactions: List[Dict]) -> List[Dict]:
        """Parse candidate selection text and match with interactions"""
        # Extract movie names from selection text
        lines = selection_text.split('\n')
        selected_movies = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Try to extract movie name
                movie_name = line.split('.')[-1].strip() if '.' in line else line
                movie_name = movie_name.split('(')[0].strip()  # Remove rating info
                
                # Find matching interaction
                for interaction in interactions:
                    if interaction.get('movie_name', '').lower() == movie_name.lower():
                        selected_movies.append(interaction)
                        break
        
        # If we don't have enough, add top-rated movies
        if len(selected_movies) < 5:
            remaining_interactions = [i for i in interactions if i not in selected_movies]
            remaining_interactions.sort(key=lambda x: x.get('rating', 0), reverse=True)
            selected_movies.extend(remaining_interactions[:5-len(selected_movies)])
        
        return selected_movies[:5]
    
    def _gpt_select_candidates(self, user_id: int, profile: Dict, taste: str, interactions: List[Dict]) -> List[Dict]:
        """Use GPT to intelligently select candidate movies based on user taste"""
        try:
            # Prepare prompt for GPT
            prompt = self._build_candidate_selection_prompt(user_id, profile, taste, interactions)
            
            # Call GPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a movie recommendation expert. Analyze the user's profile and taste to select the most representative candidate movies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse GPT response to extract movie IDs
            gpt_response = response.choices[0].message.content
            selected_movies = self._parse_gpt_candidate_response(gpt_response, interactions)
            
            # Rate limiting
            time.sleep(0.1)
            
            return selected_movies
            
        except Exception as e:
            print(f"GPT analysis failed for user {user_id}: {e}, using fallback method")
            return self._fallback_select_candidates(interactions)
    
    def _fallback_select_candidates(self, interactions: List[Dict]) -> List[Dict]:
        """Fallback method: select candidates based on rating only"""
        # Sort interactions by rating (descending)
        sorted_interactions = sorted(interactions, key=lambda x: x['rating'], reverse=True)
        
        # Get highest rated movies (rating 5)
        highest_rated = [item for item in sorted_interactions if item['rating'] == 5]
        
        # Get second highest rated movies (rating 4)
        second_highest = [item for item in sorted_interactions if item['rating'] == 4]
        
        # Select 3 highest rated movies
        selected_highest = highest_rated[:3]
        
        # Select 2 second highest rated movies
        selected_second = second_highest[:2]
        
        # Combine selections
        candidate_list = selected_highest + selected_second
        
        # Ensure we have exactly 5 items
        if len(candidate_list) < 5:
            # If not enough high-rated movies, fill with lower ratings
            remaining = [item for item in sorted_interactions 
                       if item not in candidate_list][:5-len(candidate_list)]
            candidate_list.extend(remaining)
        
        return candidate_list[:5]
    
    def _build_candidate_selection_prompt(self, user_id: int, profile: Dict, taste: str, interactions: List[Dict]) -> str:
        """Build prompt for GPT to select candidate movies"""
        demographics = profile.get('demographics', {})
        gender = demographics.get('gender', 'Unknown')
        age = demographics.get('age', 'Unknown')
        occupation = demographics.get('occupation', 'Unknown')
        
        # Get top rated movies for context
        top_movies = sorted(interactions, key=lambda x: x['rating'], reverse=True)[:10]
        top_movies_text = "\n".join([f"- {m['movie_name']} (Rating: {m['rating']})" for m in top_movies])
        
        prompt = f"""Based on the following user profile and taste analysis, select exactly 5 movies as candidate items:

User ID: {user_id}
Gender: {gender}
Age: {age}
Occupation: {occupation}

User's Refined Taste Analysis:
{taste}

Top Rated Movies (for reference):
{top_movies_text}

Available Interactions (Total: {len(interactions)}):
{chr(10).join([f"- {m['movie_name']} (ID: {m['movie_id']}, Rating: {m['rating']})" for m in interactions[:20]])}

Task: Select exactly 5 movies that best represent the user's taste and preferences. 
- Choose 3 movies with rating 5 (highest rated)
- Choose 2 movies with rating 4 (second highest rated)
- If not enough high-rated movies, fill with the best available ratings
- Focus on movies that align with the user's taste analysis

Output format: Return only the movie IDs separated by commas, e.g., "1234,5678,9012,3456,7890"
"""
        return prompt
    
    def _parse_gpt_candidate_response(self, gpt_response: str, interactions: Dict) -> List[Dict]:
        """Parse GPT response to extract selected movie IDs"""
        try:
            # Extract movie IDs from GPT response
            import re
            movie_ids = re.findall(r'\d+', gpt_response)
            
            # Convert to integers and find corresponding movies
            selected_movies = []
            for movie_id in movie_ids[:5]:  # Take first 5
                movie_id = int(movie_id)
                for movie in interactions:
                    if movie['movie_id'] == movie_id:
                        selected_movies.append(movie)
                        break
            
            # If GPT didn't provide enough movies, fill with fallback
            if len(selected_movies) < 5:
                remaining = self._fallback_select_candidates(interactions)
                selected_movies.extend(remaining[len(selected_movies):5])
            
            return selected_movies[:5]
            
        except Exception as e:
            print(f"Failed to parse GPT response: {e}, using fallback")
            return self._fallback_select_candidates(interactions)
    
    def select_representative_liked_items(self, user_ids: List[int] = None, 
                                        max_items: int = 50) -> Dict[int, Dict]:
        """Select representative liked items using GPT analysis with dual mode support"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())
        
        print(f"🎭 Selecting representative liked items for {len(user_ids)} users...")
        print(f"🔧 Mode: {'Full (Debate + Reflection)' if self.enable_debate_reflection else 'Simple (Direct GPT)'}")
        
        representative_items = {}
        processed_count = 0
        skipped_count = 0
        
        for i, user_id in enumerate(user_ids):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Show progress every 10 users with current user ID
            if i % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(user_ids)} users processed... (Current: User {user_id})")
            
            # Show detailed progress every 50 users
            if i % 50 == 0 and i > 0:
                print(f"📈 Detailed Progress: {i}/{len(user_ids)} users processed ({i/len(user_ids)*100:.1f}%)")
            
            try:
                # Check if already processed in memory
                if self.memory.has_user_selection(user_id):
                    existing_selection = self.memory.get_user_selection(user_id)
                    representative_items[user_id] = {
                        'representative_movies': existing_selection['representative_items'],
                        'total_representative': len(existing_selection['representative_items']),
                        'from_memory': True
                    }
                    skipped_count += 1
                    continue
                
                profile = self.user_profiles[user_id]
                interactions = profile['interactions']
                taste = self.user_tastes.get('refined_tastes', {}).get(str(user_id), "")
                
                # Filter to only liked items (rating >= 4)
                liked_interactions = [item for item in interactions if item.get('rating', 0) >= 4]
                
                if len(liked_interactions) == 0:
                    print(f"⚠️ User {user_id} has no liked items (rating >= 4), skipping...")
                    representative_items[user_id] = {
                        'representative_movies': [],
                        'total_representative': 0,
                        'from_memory': False
                    }
                    continue
                
                if self.client and taste:
                    if self.enable_debate_reflection:
                        # Full mode: Debate + Reflection
                        print(f"🔄 Processing User {user_id} (Full Mode: Debate + Reflection)...")
                        selected_representative = self._full_mode_select_representative_liked(
                            user_id, profile, taste, liked_interactions, max_items
                        )
                    else:
                        # Simple mode: Direct GPT
                        print(f"🔄 Processing User {user_id} (Simple Mode: Direct GPT)...")
                        selected_representative = self._simple_mode_select_representative_liked(
                            user_id, profile, taste, liked_interactions, max_items
                        )
                else:
                    # Fallback to rating-based selection
                    print(f"🔄 Processing User {user_id} (Fallback Mode: Rating-based)...")
                    liked_interactions.sort(
                        key=lambda x: (x['rating'], x['movie_name']), 
                        reverse=True
                    )
                    selected_representative = liked_interactions[:max_items]
                
                representative_items[user_id] = {
                    'representative_movies': selected_representative,
                    'total_representative': len(selected_representative),
                    'excluded_candidates': 0,  # New logic doesn't exclude candidates
                    'from_memory': False
                }
                
                processed_count += 1
                print(f"✅ User {user_id} processed successfully ({len(selected_representative)} representative liked items selected)")
                
            except Exception as e:
                print(f"❌ Error selecting representative liked items for user {user_id}: {e}")
                representative_items[user_id] = {
                    'representative_movies': [],
                    'total_representative': 0,
                    'excluded_candidates': 0,  # New logic doesn't exclude candidates
                    'from_memory': False
                }
        
        print(f"✅ Representative liked items selection completed:")
        print(f"   📊 Total users: {len(user_ids)}")
        print(f"   🔄 Processed: {processed_count}")
        print(f"   💾 From memory: {skipped_count}")
        return representative_items
    
    def select_candidate_items_from_representative(self, user_ids: List[int] = None, 
                                                 representative_items: Dict[int, Dict] = None) -> Dict[int, Dict]:
        """Select candidate items from representative liked items"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())
        
        print(f"🎬 Selecting candidate items from representative items for {len(user_ids)} users...")
        print(f"🔧 Mode: {'Full (Debate + Reflection)' if self.enable_debate_reflection else 'Simple (Direct GPT)'}")
        
        candidate_items = {}
        processed_count = 0
        skipped_count = 0
        
        for i, user_id in enumerate(user_ids):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Show progress every 10 users with current user ID
            if i % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(user_ids)} users processed... (Current: User {user_id})")
            
            # Show detailed progress every 50 users
            if i % 50 == 0 and i > 0:
                print(f"📈 Detailed Progress: {i}/{len(user_ids)} users processed ({i/len(user_ids)*100:.1f}%)")
            
            try:
                # Check if already processed in memory (only for candidate items)
                if self.memory.has_user_selection(user_id):
                    existing_selection = self.memory.get_user_selection(user_id)
                    if 'candidate_items' in existing_selection and existing_selection['candidate_items']:
                        candidate_items[user_id] = {
                            'candidate_movies': existing_selection['candidate_items'],
                            'total_candidates': len(existing_selection['candidate_items']),
                            'from_memory': True
                        }
                        skipped_count += 1
                        continue
                
                # Get representative items for this user
                if not representative_items or user_id not in representative_items:
                    print(f"⚠️ No representative items found for user {user_id}, skipping...")
                    candidate_items[user_id] = {
                        'candidate_movies': [],
                        'total_candidates': 0,
                        'from_memory': False
                    }
                    continue
                
                representative_movies = representative_items[user_id]['representative_movies']
                
                if len(representative_movies) == 0:
                    print(f"⚠️ User {user_id} has no representative movies, skipping...")
                    candidate_items[user_id] = {
                        'candidate_movies': [],
                        'total_candidates': 0,
                        'from_memory': False
                    }
                    continue
                
                # Select candidate items based on rating: 3 rating-5 items + 2 rating-4 items
                candidate_list = self._select_candidates_by_rating(representative_movies)
                
                candidate_items[user_id] = {
                    'candidate_movies': candidate_list,
                    'total_candidates': len(candidate_list),
                    'from_memory': False
                }
                
                # Store candidate items in memory
                self.memory.store_user_selection(
                    user_id, candidate_list, representative_movies,
                    f"New logic: candidates from representative items",
                    0.8
                )
                
                processed_count += 1
                print(f"✅ User {user_id} processed successfully ({len(candidate_list)} candidate items selected)")
                
            except Exception as e:
                print(f"❌ Error selecting candidate items for user {user_id}: {e}")
                candidate_items[user_id] = {
                    'candidate_movies': [],
                    'total_candidates': 0,
                    'from_memory': False
                }
        
        print(f"✅ Candidate items selection completed:")
        print(f"   📊 Total users: {len(user_ids)}")
        print(f"   🔄 Processed: {processed_count}")
        print(f"   💾 From memory: {skipped_count}")
        return candidate_items
    
    def _select_candidates_by_rating(self, representative_movies: List[Dict]) -> List[Dict]:
        """Select candidate items based on rating: 3 rating-5 items + 2 rating-4 items"""
        # Separate movies by rating
        rating_5_movies = [movie for movie in representative_movies if movie.get('rating', 0) == 5]
        rating_4_movies = [movie for movie in representative_movies if movie.get('rating', 0) == 4]
        other_movies = [movie for movie in representative_movies if movie.get('rating', 0) not in [4, 5]]
        
        candidate_list = []
        
        # Select 3 rating-5 movies
        if len(rating_5_movies) >= 3:
            # Randomly select 3 from rating-5 movies
            selected_5 = random.sample(rating_5_movies, 3)
            candidate_list.extend(selected_5)
        else:
            # Use all rating-5 movies if less than 3
            candidate_list.extend(rating_5_movies)
        
        # Select 2 rating-4 movies
        if len(rating_4_movies) >= 2:
            # Randomly select 2 from rating-4 movies
            selected_4 = random.sample(rating_4_movies, 2)
            candidate_list.extend(selected_4)
        else:
            # Use all rating-4 movies if less than 2
            candidate_list.extend(rating_4_movies)
        
        # If we still need more movies to reach 5, fill with other high-rated movies
        if len(candidate_list) < 5:
            remaining_needed = 5 - len(candidate_list)
            # Sort other movies by rating (descending) and take the best ones
            other_movies.sort(key=lambda x: x.get('rating', 0), reverse=True)
            candidate_list.extend(other_movies[:remaining_needed])
        
        # If still not enough, fill with any remaining movies
        if len(candidate_list) < 5:
            remaining_needed = 5 - len(candidate_list)
            all_used = set(movie.get('movie_id') for movie in candidate_list)
            remaining_movies = [movie for movie in representative_movies 
                              if movie.get('movie_id') not in all_used]
            candidate_list.extend(remaining_movies[:remaining_needed])
        
        return candidate_list[:5]  # Ensure exactly 5 items
    
    def _full_mode_select_representative_liked(self, user_id: int, profile: Dict, taste: str,
                                             liked_interactions: List[Dict], max_items: int) -> List[Dict]:
        """Full mode: Use debate and reflection for representative liked items selection"""
        try:
            # Step 1: Debate
            debated_result = self.debate_module.debate_representative_liked_selection(
                user_id, profile, taste, liked_interactions
            )
            
            # Step 2: Reflection
            reflected_result = self.reflection_module.reflect_representative_liked_selection(
                user_id, profile, taste, liked_interactions, debated_result
            )
            
            # Step 3: Parse and return results
            final_selection = reflected_result.get('refined_selection', debated_result.get('consensus', ''))
            representative_list = self._parse_representative_selection(final_selection, liked_interactions, max_items)
            
            # Store in memory
            self.memory.store_user_selection(
                user_id, [], representative_list,
                f"Full mode representative liked selection: {debated_result.get('debate_process', '')} | {reflected_result.get('reflection', '')}",
                0.8
            )
            
            self._reset_failure_count()
            return representative_list
            
        except Exception as e:
            print(f"❌ Full mode representative liked selection failed for user {user_id}: {e}")
            self._increment_failure_count()
            # Fallback to rating-based selection
            liked_interactions.sort(key=lambda x: x.get('rating', 0), reverse=True)
            return liked_interactions[:max_items]
    
    def _simple_mode_select_representative_liked(self, user_id: int, profile: Dict, taste: str,
                                               liked_interactions: List[Dict], max_items: int) -> List[Dict]:
        """Simple mode: Direct GPT for representative liked items selection"""
        try:
            # Build simple prompt
            prompt = self._build_simple_representative_liked_prompt(user_id, profile, taste, liked_interactions, max_items)
            
            # Call GPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            representative_list = self._parse_representative_selection(response_text, liked_interactions, max_items)
            
            # Store in memory
            self.memory.store_user_selection(
                user_id, [], representative_list,
                f"Simple mode representative liked selection: {response_text[:100]}...",
                0.7
            )
            
            self._smart_delay()
            self._reset_failure_count()
            return representative_list
            
        except Exception as e:
            print(f"❌ Simple mode representative liked selection failed for user {user_id}: {e}")
            self._increment_failure_count()
            # Fallback to rating-based selection
            liked_interactions.sort(key=lambda x: x.get('rating', 0), reverse=True)
            return liked_interactions[:max_items]
    
    def _build_simple_representative_liked_prompt(self, user_id: int, profile: Dict, taste: str,
                                                liked_interactions: List[Dict], max_items: int) -> str:
        """Build simple prompt for representative liked items selection"""
        interactions_text = "\n".join([
            f"- {interaction.get('movie_name', 'Unknown')} (Rating: {interaction.get('rating', 0)})"
            for interaction in liked_interactions[:50]  # Limit to top 50 for prompt length
        ])
        
        return f"""Select up to {max_items} representative liked movies for user {user_id} based on their taste and interaction history.

User Taste: {taste}

User Liked Movies (Rating >= 4):
{interactions_text}

Please select up to {max_items} movies that best represent this user's preferences. Consider rating patterns, genre preferences, and overall taste alignment.

Format your response as a simple list of movie names."""
    
    def select_representative_items(self, user_ids: List[int] = None, 
                                   max_items: int = 50, 
                                   candidate_items: Dict[int, Dict] = None) -> Dict[int, Dict]:
        """Select representative items using GPT analysis with dual mode support"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())
        
        print(f"🎭 Selecting representative items for {len(user_ids)} users...")
        print(f"🔧 Mode: {'Full (Debate + Reflection)' if self.enable_debate_reflection else 'Simple (Direct GPT)'}")
        
        representative_items = {}
        processed_count = 0
        skipped_count = 0
        
        for i, user_id in enumerate(user_ids):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Show progress every 10 users with current user ID
            if i % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(user_ids)} users processed... (Current: User {user_id})")
            
            # Show detailed progress every 50 users
            if i % 50 == 0 and i > 0:
                print(f"📈 Detailed Progress: {i}/{len(user_ids)} users processed ({i/len(user_ids)*100:.1f}%)")
            
            try:
                # Check if already processed in memory
                if self.memory.has_user_selection(user_id):
                    existing_selection = self.memory.get_user_selection(user_id)
                    representative_items[user_id] = {
                        'representative_movies': existing_selection['representative_items'],
                        'total_representative': len(existing_selection['representative_items']),
                        'excluded_candidates': 0,
                        'from_memory': True
                    }
                    skipped_count += 1
                    continue
                
                profile = self.user_profiles[user_id]
                interactions = profile['interactions']
                taste = self.user_tastes.get('refined_tastes', {}).get(str(user_id), "")
                
                # Get candidate movie IDs for this user
                candidate_movie_ids = set()
                candidate_movies = []
                if candidate_items and user_id in candidate_items:
                    candidate_movies = candidate_items[user_id]['candidate_movies']
                    candidate_movie_ids = {movie['movie_id'] for movie in candidate_movies}
                
                # Filter out candidate movies from interactions
                representative_interactions = [
                    item for item in interactions 
                    if item['movie_id'] not in candidate_movie_ids
                ]
                
                if self.client and taste:
                    if self.enable_debate_reflection:
                        # Full mode: Debate + Reflection
                        print(f"🔄 Processing User {user_id} (Full Mode: Debate + Reflection)...")
                        selected_representative = self._full_mode_select_representative(
                            user_id, profile, taste, candidate_movies, interactions, max_items
                        )
                    else:
                        # Simple mode: Direct GPT
                        print(f"🔄 Processing User {user_id} (Simple Mode: Direct GPT)...")
                        selected_representative = self._simple_mode_select_representative(
                            user_id, profile, taste, representative_interactions, max_items
                        )
                else:
                    # Fallback to rating-based selection
                    print(f"🔄 Processing User {user_id} (Fallback Mode: Rating-based)...")
                    representative_interactions.sort(
                        key=lambda x: (x['rating'], x['movie_name']), 
                        reverse=True
                    )
                    selected_representative = representative_interactions[:max_items]
                
                representative_items[user_id] = {
                    'representative_movies': selected_representative,
                    'total_representative': len(selected_representative),
                    'excluded_candidates': len(candidate_movie_ids),
                    'from_memory': False
                }
                
                processed_count += 1
                print(f"✅ User {user_id} processed successfully ({len(selected_representative)} representative items selected)")
                
            except Exception as e:
                print(f"❌ Error selecting representative items for user {user_id}: {e}")
                representative_items[user_id] = {
                    'representative_movies': [],
                    'total_representative': 0,
                    'excluded_candidates': 0
                }
        
        print(f"✅ Representative items selection completed:")
        print(f"   📊 Total users: {len(user_ids)}")
        print(f"   🔄 Processed: {processed_count}")
        print(f"   💾 From memory: {skipped_count}")
        return representative_items
    
    def _full_mode_select_representative(self, user_id: int, profile: Dict, taste: str,
                                       candidate_movies: List[Dict], all_interactions: List[Dict],
                                       max_items: int) -> List[Dict]:
        """Full mode: Use debate and reflection for representative selection"""
        try:
            # Step 1: Debate
            debated_result = self.debate_module.debate_representative_selection(
                user_id, profile, taste, candidate_movies, all_interactions
            )
            
            # Step 2: Reflection
            reflected_result = self.reflection_module.reflect_representative_selection(
                user_id, profile, taste, candidate_movies, all_interactions, debated_result
            )
            
            # Step 3: Parse and return results
            final_selection = reflected_result.get('refined_selection', debated_result.get('consensus', ''))
            representative_list = self._parse_representative_selection(final_selection, all_interactions, max_items)
            
            # Update memory with representative items
            if self.memory.has_user_selection(user_id):
                existing_selection = self.memory.get_user_selection(user_id)
                self.memory.store_user_selection(
                    user_id, existing_selection['candidate_items'], representative_list,
                    f"Full mode representative selection: {debated_result.get('debate_process', '')} | {reflected_result.get('reflection', '')}",
                    0.8
                )
            
            self._reset_failure_count()
            return representative_list
            
        except Exception as e:
            print(f"❌ Full mode representative selection failed for user {user_id}: {e}")
            self._increment_failure_count()
            # Fallback to rating-based selection
            all_interactions.sort(key=lambda x: x.get('rating', 0), reverse=True)
            return all_interactions[:max_items]
    
    def _simple_mode_select_representative(self, user_id: int, profile: Dict, taste: str,
                                         representative_interactions: List[Dict], max_items: int) -> List[Dict]:
        """Simple mode: Direct GPT for representative selection"""
        try:
            # Build simple prompt
            prompt = self._build_simple_representative_prompt(user_id, profile, taste, representative_interactions, max_items)
            
            # Call GPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            representative_list = self._parse_representative_selection(response_text, representative_interactions, max_items)
            
            # Update memory with representative items
            if self.memory.has_user_selection(user_id):
                existing_selection = self.memory.get_user_selection(user_id)
                self.memory.store_user_selection(
                    user_id, existing_selection['candidate_items'], representative_list,
                    f"Simple mode representative selection: {response_text[:100]}...",
                    0.7
                )
            
            self._smart_delay()
            self._reset_failure_count()
            return representative_list
            
        except Exception as e:
            print(f"❌ Simple mode representative selection failed for user {user_id}: {e}")
            self._increment_failure_count()
            # Fallback to rating-based selection
            representative_interactions.sort(key=lambda x: x.get('rating', 0), reverse=True)
            return representative_interactions[:max_items]
    
    def _build_simple_representative_prompt(self, user_id: int, profile: Dict, taste: str,
                                          representative_interactions: List[Dict], max_items: int) -> str:
        """Build simple prompt for representative selection"""
        interactions_text = "\n".join([
            f"- {interaction.get('movie_name', 'Unknown')} (Rating: {interaction.get('rating', 0)})"
            for interaction in representative_interactions[:50]  # Limit to top 50 for prompt length
        ])
        
        return f"""Select up to {max_items} representative movies for user {user_id} based on their taste and interaction history.

User Taste: {taste}

Available Movies (excluding candidate movies):
{interactions_text}

Please select movies that best represent this user's preferences. Consider:
1. Rating patterns and user satisfaction
2. Genre diversity and thematic coverage
3. Overall taste alignment

Format your response as a simple list of movie names."""
    
    def _parse_representative_selection(self, selection_text: str, interactions: List[Dict], max_items: int) -> List[Dict]:
        """Parse representative selection text and match with interactions"""
        # Extract movie names from selection text
        lines = selection_text.split('\n')
        selected_movies = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                # Try to extract movie name
                movie_name = line.split('.')[-1].strip() if '.' in line else line
                movie_name = movie_name.split('(')[0].strip()  # Remove rating info
                
                # Find matching interaction
                for interaction in interactions:
                    if interaction.get('movie_name', '').lower() == movie_name.lower():
                        selected_movies.append(interaction)
                        break
        
        # If we don't have enough, add top-rated movies
        if len(selected_movies) < max_items:
            remaining_interactions = [i for i in interactions if i not in selected_movies]
            remaining_interactions.sort(key=lambda x: x.get('rating', 0), reverse=True)
            selected_movies.extend(remaining_interactions[:max_items-len(selected_movies)])
        
        return selected_movies[:max_items]
    
    def _gpt_select_representative(self, user_id: int, profile: Dict, taste: str, 
                                  representative_interactions: List[Dict], max_items: int) -> List[Dict]:
        """Use GPT to intelligently select representative movies based on user taste"""
        try:
            # Prepare prompt for GPT
            prompt = self._build_representative_selection_prompt(
                user_id, profile, taste, representative_interactions, max_items
            )
            
            # Call GPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a movie recommendation expert. Analyze the user's profile and taste to select the most representative movies that showcase their preferences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            # Parse GPT response to extract movie IDs
            gpt_response = response.choices[0].message.content
            selected_movies = self._parse_gpt_representative_response(gpt_response, representative_interactions, max_items)
            
            # Rate limiting
            time.sleep(0.1)
            
            return selected_movies
            
        except Exception as e:
            print(f"GPT analysis failed for user {user_id}: {e}, using fallback method")
            # Fallback to rating-based selection
            representative_interactions.sort(
                key=lambda x: (x['rating'], x['movie_name']), 
                reverse=True
            )
            return representative_interactions[:max_items]
    
    def _build_representative_selection_prompt(self, user_id: int, profile: Dict, taste: str, 
                                             representative_interactions: List[Dict], max_items: int) -> str:
        """Build prompt for GPT to select representative movies"""
        demographics = profile.get('demographics', {})
        gender = demographics.get('gender', 'Unknown')
        age = demographics.get('age', 'Unknown')
        occupation = demographics.get('occupation', 'Unknown')
        
        # Get sample of available interactions for context
        sample_interactions = representative_interactions[:30]  # Limit to avoid too long prompt
        interactions_text = "\n".join([f"- {m['movie_name']} (ID: {m['movie_id']}, Rating: {m['rating']})" for m in sample_interactions])
        
        prompt = f"""Based on the following user profile and taste analysis, select up to {max_items} representative movies:

User ID: {user_id}
Gender: {gender}
Age: {age}
Occupation: {occupation}

User's Refined Taste Analysis:
{taste}

Available Movies (excluding candidate items, Total: {len(representative_interactions)}):
{interactions_text}

Task: Select up to {max_items} movies that best represent the user's taste and preferences.
- Focus on movies that align with the user's taste analysis
- Consider rating, but prioritize taste alignment over pure rating
- Select diverse movies that showcase different aspects of the user's preferences
- Ensure the selection represents the user's unique taste profile

Output format: Return only the movie IDs separated by commas, e.g., "1234,5678,9012,3456,7890"
Select exactly {max_items} movies if possible, or fewer if not enough suitable movies available.
"""
        return prompt
    
    def _parse_gpt_representative_response(self, gpt_response: str, representative_interactions: List[Dict], 
                                         max_items: int) -> List[Dict]:
        """Parse GPT response to extract selected representative movie IDs"""
        try:
            # Extract movie IDs from GPT response
            import re
            movie_ids = re.findall(r'\d+', gpt_response)
            
            # Convert to integers and find corresponding movies
            selected_movies = []
            for movie_id in movie_ids[:max_items]:  # Take up to max_items
                movie_id = int(movie_id)
                for movie in representative_interactions:
                    if movie['movie_id'] == movie_id:
                        selected_movies.append(movie)
                        break
            
            # If GPT didn't provide enough movies, fill with fallback
            if len(selected_movies) < max_items:
                remaining = sorted(representative_interactions, 
                                 key=lambda x: (x['rating'], x['movie_name']), 
                                 reverse=True)
                selected_movies.extend(remaining[len(selected_movies):max_items])
            
            return selected_movies[:max_items]
            
        except Exception as e:
            print(f"Failed to parse GPT response: {e}, using fallback")
            # Fallback to rating-based selection
            representative_interactions.sort(
                key=lambda x: (x['rating'], x['movie_name']), 
                reverse=True
            )
            return representative_interactions[:max_items]
    
    def save_candidate_items(self, candidate_items: Dict[int, Dict], 
                           output_file: str = "candidate_items.json"):
        """Save candidate items to JSON file"""
        print(f"Saving candidate items to {output_file}...")
        
        try:
            # Convert to serializable format
            serializable_data = {}
            for user_id, data in candidate_items.items():
                serializable_data[str(user_id)] = {
                    'candidate_movies': [
                        {
                            'movie_id': movie['movie_id'],
                            'movie_name': movie['movie_name'],
                            'rating': movie['rating']
                        }
                        for movie in data['candidate_movies']
                    ],
                    'total_candidates': data['total_candidates']
                }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"✅ Candidate items saved: {output_file}")
            print(f"   File size: {file_size:.2f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Error saving candidate items: {e}")
            return False
    
    def save_representative_items(self, representative_items: Dict[int, Dict], 
                                output_file: str = "representative_items.json"):
        """Save representative items to JSON file"""
        print(f"Saving representative items to {output_file}...")
        
        try:
            # Convert to serializable format
            serializable_data = {}
            for user_id, data in representative_items.items():
                serializable_data[str(user_id)] = {
                    'representative_movies': [
                        {
                            'movie_id': movie['movie_id'],
                            'movie_name': movie['movie_name'],
                            'rating': movie['rating']
                        }
                        for movie in data['representative_movies']
                    ],
                    'total_representative': data['total_representative'],
                    'excluded_candidates': data['excluded_candidates']
                }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"✅ Representative items saved: {output_file}")
            print(f"   File size: {file_size:.2f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Error saving representative items: {e}")
            return False
    
    def run_full_selection(self, user_ids: List[int] = None, max_representative: int = 50) -> Dict:
        """Run the complete movie selection workflow with dual mode support"""
        if user_ids is None:
            # Get intersection of users from both files
            profile_user_ids = set(self.user_profiles.keys())
            taste_user_ids = set(self.user_tastes.get('refined_tastes', {}).keys())
            # Convert taste user IDs to integers for comparison
            taste_user_ids = {int(uid) for uid in taste_user_ids}
            
            # Only process users that exist in both files
            user_ids = list(profile_user_ids.intersection(taste_user_ids))
            
            print(f"📊 User ID intersection analysis:")
            print(f"  Users in profiles: {len(profile_user_ids)}")
            print(f"  Users in tastes: {len(taste_user_ids)}")
            print(f"  Users in both: {len(user_ids)}")
            print(f"  Users only in profiles: {len(profile_user_ids - taste_user_ids)}")
            print(f"  Users only in tastes: {len(taste_user_ids - profile_user_ids)}")
        
        print("=" * 60)
        print("A1-3: User Movie Selection Workflow")
        print("=" * 60)
        print(f"🔧 Mode: {'Full (Debate + Reflection)' if self.enable_debate_reflection else 'Simple (Direct GPT)'}")
        print(f"📊 Processing {len(user_ids)} users...")
        print(f"🎭 Max representative items per user: {max_representative}")
        if self.memory_file:
            print(f"🧠 Memory file: {self.memory_file}")
        print()
        
        # Step 1: Select representative liked items
        print("🎭 Step 1: Selecting representative liked items...")
        representative_items = self.select_representative_liked_items(user_ids, max_representative)
        
        # Reset failure counter between Step 1 and Step 2
        self._reset_failure_counter_between_stages("between Step 1 and Step 2")
        
        # Step 2: Select candidate items from representative items
        print("\n🎬 Step 2: Selecting candidate items from representative items...")
        candidate_items = self.select_candidate_items_from_representative(user_ids, representative_items)
        
        # Save memory if file is specified
        if self.memory_file:
            print(f"\n💾 Saving memory to {self.memory_file}...")
            self.save_memory()
            self._display_learning_progress()
        
        # Compile results
        results = {
            'workflow_info': {
                'total_users': len(user_ids),
                'max_representative_items': max_representative,
                'workflow_version': '2.0',
                'enable_debate_reflection': self.enable_debate_reflection,
                'mode': 'Full' if self.enable_debate_reflection else 'Simple',
                'memory_file': self.memory_file
            },
            'candidate_items': candidate_items,
            'representative_items': representative_items
        }
        
        print("\n" + "=" * 60)
        print("✅ User Movie Selection Workflow Completed!")
        print("=" * 60)
        
        return results
    
    def _display_learning_progress(self):
        """Display memory learning progress summary"""
        if not self.memory_file:
            return
        
        learning_summary = self.memory.get_memory_learning_summary()
        
        print(f"\n🧠 Memory Learning Progress")
        print(f"=" * 50)
        print(f"📊 Total users processed: {learning_summary['total_users_processed']}")
        print(f"🎯 Learned patterns: {learning_summary['learned_patterns']}")
        print(f"💡 Learned insights: {learning_summary['learned_insights']}")
        print(f"✅ Successful selections: {learning_summary['successful_selections']}")
        print(f"📈 Average confidence: {learning_summary['avg_confidence']:.2f}")
        print(f"🌟 High confidence users: {learning_summary['high_confidence_users']}")
        
        if learning_summary['top_patterns']:
            print(f"🔝 Top patterns:")
            for pattern, count in learning_summary['top_patterns']:
                print(f"   - {pattern}: {count}")
        
        print(f"=" * 50)
    
    def print_selection_summary(self, candidate_items: Dict[int, Dict], 
                              representative_items: Dict[int, Dict]):
        """Print summary of selections"""
        print("\n" + "=" * 60)
        print("📊 Selection Summary")
        print("=" * 60)
        
        # Candidate items summary
        total_candidates = sum(data['total_candidates'] for data in candidate_items.values())
        avg_candidates = total_candidates / len(candidate_items) if candidate_items else 0
        
        print(f"Candidate Items:")
        print(f"  Total users: {len(candidate_items)}")
        print(f"  Total movies selected: {total_candidates}")
        print(f"  Average per user: {avg_candidates:.1f}")
        
        # Representative items summary
        total_representative = sum(data['total_representative'] for data in representative_items.values())
        avg_representative = total_representative / len(representative_items) if representative_items else 0
        
        print(f"\nRepresentative Items:")
        print(f"  Total users: {len(representative_items)}")
        print(f"  Total movies selected: {total_representative}")
        print(f"  Average per user: {avg_representative:.1f}")
        
        # Users with sufficient data
        users_with_candidates = sum(1 for data in candidate_items.values() if data['total_candidates'] >= 5)
        users_with_representative = sum(1 for data in representative_items.values() if data['total_representative'] > 0)
        
        print(f"\nData Quality:")
        print(f"  Users with 5+ candidates: {users_with_candidates}/{len(candidate_items)}")
        print(f"  Users with representative items: {users_with_representative}/{len(representative_items)}")


def main():
    """Main function with dual mode and memory support"""
    parser = argparse.ArgumentParser(description='Select representative movies for users based on refined taste analysis')
    parser.add_argument('--user-profiles', type=str, default='user_profiles.json',
                       help='User profiles JSON file path')
    parser.add_argument('--user-tastes', type=str, default='user_tastes.json',
                       help='User tastes JSON file path')
    parser.add_argument('--candidate-output', type=str, default='candidate_items.json',
                       help='Output file for candidate items')
    parser.add_argument('--representative-output', type=str, default='representative_items.json',
                       help='Output file for representative items')
    parser.add_argument('--max-representative', type=int, default=50,
                       help='Maximum number of representative items per user')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of users to process (None for all users)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                       help='OpenAI API key for GPT analysis')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for sampling')
    
    # Dual mode arguments
    parser.add_argument('--enable-debate-reflection', action='store_true', default=True,
                       help='Enable debate and reflection mode (default: True)')
    parser.add_argument('--simple-mode', action='store_true',
                       help='Use simple mode (direct GPT, disables debate and reflection)')
    parser.add_argument('--disable-debate-reflection', action='store_true',
                       help='Disable debate and reflection mode')
    
    # Memory arguments
    parser.add_argument('--memory-file', type=str, default=None,
                       help='Memory file path for persistent storage')
    parser.add_argument('--load-memory', action='store_true',
                       help='Load memory from file at startup')
    parser.add_argument('--clear-memory', action='store_true',
                       help='Clear memory before processing')
    
    # Filtered users arguments
    parser.add_argument('--use-filtered-users', action='store_true', default=True,
                       help='Only process users that exist in filtered_train_set.txt and rec_save_dict.csv (default: True)')
    parser.add_argument('--no-filtered-users', action='store_true',
                       help='Disable filtered users mode (process all users from user_profiles and user_tastes)')
    parser.add_argument('--filtered-train-file', type=str, default='filtered_train_set.txt',
                       help='Path to filtered train set file')
    parser.add_argument('--rec-save-dict-file', type=str, default='rec_save_dict.csv',
                       help='Path to rec save dict file')
    
    args = parser.parse_args()
    
    # Handle mode selection
    if args.simple_mode:
        args.enable_debate_reflection = False
    elif args.disable_debate_reflection:
        args.enable_debate_reflection = False
    
    # Handle filtered users selection
    if args.no_filtered_users:
        args.use_filtered_users = False
    
    print("=" * 60)
    print("A1-3: User Movie Selection Module")
    print("=" * 60)
    print(f"User profiles file: {args.user_profiles}")
    print(f"User tastes file: {args.user_tastes}")
    print(f"Candidate output: {args.candidate_output}")
    print(f"Representative output: {args.representative_output}")
    print(f"Max representative items: {args.max_representative}")
    print(f"Sample size: {args.sample_size if args.sample_size else 'All users'}")
    print(f"Random seed: {args.random_seed}")
    print(f"OpenAI API key: {'Provided' if args.openai_api_key else 'From environment'}")
    print(f"Mode: {'Full (Debate + Reflection)' if args.enable_debate_reflection else 'Simple (Direct GPT)'}")
    print(f"Memory file: {args.memory_file if args.memory_file else 'None'}")
    print(f"Use filtered users: {args.use_filtered_users}")
    if args.use_filtered_users:
        print(f"Filtered train file: {args.filtered_train_file}")
        print(f"Rec save dict file: {args.rec_save_dict_file}")
    print()
    
    # Create movie selector with dual mode and memory support
    selector = UserMovieSelector(
        user_profiles_file=args.user_profiles,
        user_tastes_file=args.user_tastes,
        openai_api_key=args.openai_api_key,
        memory_file=args.memory_file,
        enable_debate_reflection=args.enable_debate_reflection
    )
    
    if not selector.user_profiles:
        print("❌ Failed to load user profiles. Exiting.")
        return
    
    if not selector.user_tastes:
        print("❌ Failed to load user tastes. Exiting.")
        return
    
    # Handle memory options
    if args.clear_memory:
        print("🧹 Clearing memory as requested...")
        selector.clear_memory()
    
    if args.load_memory and args.memory_file:
        print(f"🧠 Attempting to load memory from {args.memory_file}...")
        selector.load_memory()
    elif args.load_memory and not args.memory_file:
        print("⚠️ --load-memory specified but no --memory-file provided")
    
    # Determine users to process
    if args.use_filtered_users:
        # When using filtered users, get users from filtered_train_set and rec_save_dict
        filtered_users, rec_users = selector.get_filtered_and_rec_users(args.filtered_train_file, args.rec_save_dict_file)
        available_users = set(selector.user_profiles.keys())
        
        if args.sample_size:
            # Sample from filtered_train_set only (not including rec_save_dict)
            import random
            random.seed(args.random_seed)
            
            # Filter to only available users
            available_filtered_users = [uid for uid in filtered_users if uid in available_users]
            
            # Sample from filtered_train_set only
            user_ids = random.sample(available_filtered_users, min(args.sample_size, len(available_filtered_users)))
            
            print(f"🎲 Sampling logic:")
            print(f"   - Sampled {len(user_ids)} users from filtered_train_set only")
            print(f"   - Total users to process: {len(user_ids)}")
        else:
            # Use all users from filtered_train_set only
            user_ids = [uid for uid in filtered_users if uid in available_users]
            print(f"📋 Processing all {len(user_ids)} users from filtered_train_set only")
    else:
        # When not using filtered users, work with all user_profiles
        if args.sample_size:
            import random
            random.seed(args.random_seed)
            # Get intersection of users from both files first
            profile_user_ids = set(selector.user_profiles.keys())
            taste_user_ids = set(selector.user_tastes.get('refined_tastes', {}).keys())
            taste_user_ids = {int(uid) for uid in taste_user_ids}
            intersection_user_ids = list(profile_user_ids.intersection(taste_user_ids))
            
            # Sample from intersection
            user_ids = random.sample(intersection_user_ids, min(args.sample_size, len(intersection_user_ids)))
            print(f"🎲 Randomly sampled {len(user_ids)} users from {len(intersection_user_ids)} common users")
        else:
            user_ids = None
            print("📋 Processing all users that exist in both files")
    
    # Run workflow
    results = selector.run_full_selection(user_ids, args.max_representative)
    
    # Save results
    candidate_success = selector.save_candidate_items(
        results['candidate_items'], args.candidate_output
    )
    
    representative_success = selector.save_representative_items(
        results['representative_items'], args.representative_output
    )
    
    # Print summary
    selector.print_selection_summary(
        results['candidate_items'], 
        results['representative_items']
    )
    
    if candidate_success and representative_success:
        print("\n" + "=" * 60)
        print("✅ User Movie Selection Completed Successfully!")
        print("=" * 60)
        print(f"Candidate items saved to: {args.candidate_output}")
        print(f"Representative items saved to: {args.representative_output}")
        print(f"Total users processed: {len(results['candidate_items'])}")
    else:
        print("\n" + "=" * 60)
        print("⚠️ User Movie Selection Completed with Warnings!")
        print("=" * 60)


if __name__ == "__main__":
    main()

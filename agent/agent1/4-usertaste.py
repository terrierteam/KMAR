#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1-2: User Taste Generation Module
Generate user unique taste descriptions with debate and reflection
"""

import os
import json
import time
import argparse
import random
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from openai import OpenAI

class UserTasteMemory:
    """Memory module for storing user taste generation patterns and history with learning capabilities"""
    
    def __init__(self):
        """Initialize the memory module"""
        self.user_tastes = {}  # user_id -> taste_description
        self.initial_tastes = {}  # user_id -> initial_taste
        self.debated_tastes = {}  # user_id -> debated_taste
        self.refined_tastes = {}  # user_id -> refined_taste
        self.processing_history = []  # List of processing records
        self.debate_patterns = {}  # user_id -> debate_summary
        self.reflection_patterns = {}  # user_id -> reflection_summary
        self.confidence_scores = {}  # user_id -> confidence_score
        
        # Learning and pattern analysis
        self.taste_patterns = {}  # pattern_type -> frequency
        self.successful_prompts = []  # List of successful prompt patterns
        self.user_demographics_patterns = {}  # demographic -> taste_patterns
        self.genre_preferences = {}  # user_id -> genre_preferences
        self.rating_patterns = {}  # rating_pattern -> taste_characteristics
        self.learned_insights = []  # List of learned insights
        self.quality_metrics = {}  # user_id -> quality_metrics
        
        print("✅ User Taste Memory initialized with learning capabilities")
    
    def store_user_taste(self, user_id: int, initial_taste: str, debated_taste: str, 
                        refined_taste: str, debate_summary: str, reflection_summary: str, 
                        confidence: float = 0.8):
        """Store a user taste generation result
        
        Args:
            user_id: User ID
            initial_taste: Initial taste description
            debated_taste: Debated taste description
            refined_taste: Refined taste description
            debate_summary: Summary of the debate process
            reflection_summary: Summary of the reflection process
            confidence: Confidence score (0-1)
        """
        self.user_tastes[user_id] = refined_taste
        self.initial_tastes[user_id] = initial_taste
        self.debated_tastes[user_id] = debated_taste
        self.refined_tastes[user_id] = refined_taste
        self.debate_patterns[user_id] = debate_summary
        self.reflection_patterns[user_id] = reflection_summary
        self.confidence_scores[user_id] = confidence
        
        # Add to processing history
        self.processing_history.append({
            'user_id': user_id,
            'timestamp': time.time(),
            'confidence': confidence,
            'has_initial': bool(initial_taste),
            'has_debated': bool(debated_taste),
            'has_refined': bool(refined_taste)
        })
        
        # Learn from this experience
        self._learn_from_user_taste(user_id, initial_taste, debated_taste, refined_taste, confidence)
    
    def get_user_taste(self, user_id: int) -> Optional[Dict]:
        """Get stored taste information for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with taste information or None if not found
        """
        if user_id in self.user_tastes:
            return {
                'initial_taste': self.initial_tastes.get(user_id, ''),
                'debated_taste': self.debated_tastes.get(user_id, ''),
                'refined_taste': self.refined_tastes.get(user_id, ''),
                'debate_summary': self.debate_patterns.get(user_id, ''),
                'reflection_summary': self.reflection_patterns.get(user_id, ''),
                'confidence': self.confidence_scores.get(user_id, 0.0)
            }
        return None
    
    def has_user_taste(self, user_id: int) -> bool:
        """Check if user taste is already stored
        
        Args:
            user_id: User ID
            
        Returns:
            True if user taste is stored, False otherwise
        """
        return user_id in self.user_tastes
    
    def get_statistics(self) -> Dict:
        """Get memory statistics
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.processing_history:
            return {'total_users': 0, 'avg_confidence': 0.0}
        
        total_users = len(self.processing_history)
        avg_confidence = sum(record['confidence'] for record in self.processing_history) / total_users
        
        return {
            'total_users': total_users,
            'avg_confidence': avg_confidence,
            'recent_users': [record['user_id'] for record in self.processing_history[-5:]]
        }
    
    def save_to_file(self, memory_file: str):
        """Save complete memory state to file
        
        Args:
            memory_file: Path to save memory file
        """
        try:
            memory_data = {
                'user_tastes': self.user_tastes,
                'initial_tastes': self.initial_tastes,
                'debated_tastes': self.debated_tastes,
                'refined_tastes': self.refined_tastes,
                'debate_patterns': self.debate_patterns,
                'reflection_patterns': self.reflection_patterns,
                'confidence_scores': self.confidence_scores,
                'processing_history': self.processing_history,
                # Learning data
                'taste_patterns': self.taste_patterns,
                'successful_prompts': self.successful_prompts,
                'user_demographics_patterns': self.user_demographics_patterns,
                'genre_preferences': self.genre_preferences,
                'rating_patterns': self.rating_patterns,
                'learned_insights': self.learned_insights,
                'quality_metrics': self.quality_metrics,
                'metadata': {
                    'total_users': len(self.user_tastes),
                    'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'version': '2.0',
                    'learning_enabled': True
                }
            }
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ User Taste Memory saved to {memory_file} ({len(self.user_tastes)} users)")
            
        except Exception as e:
            print(f"❌ Error saving user taste memory: {e}")
    
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
            self.user_tastes = memory_data.get('user_tastes', {})
            self.initial_tastes = memory_data.get('initial_tastes', {})
            self.debated_tastes = memory_data.get('debated_tastes', {})
            self.refined_tastes = memory_data.get('refined_tastes', {})
            self.debate_patterns = memory_data.get('debate_patterns', {})
            self.reflection_patterns = memory_data.get('reflection_patterns', {})
            self.confidence_scores = memory_data.get('confidence_scores', {})
            self.processing_history = memory_data.get('processing_history', [])
            
            # Load learning data (with backward compatibility)
            self.taste_patterns = memory_data.get('taste_patterns', {})
            self.successful_prompts = memory_data.get('successful_prompts', [])
            self.user_demographics_patterns = memory_data.get('user_demographics_patterns', {})
            self.genre_preferences = memory_data.get('genre_preferences', {})
            self.rating_patterns = memory_data.get('rating_patterns', {})
            self.learned_insights = memory_data.get('learned_insights', [])
            self.quality_metrics = memory_data.get('quality_metrics', {})
            
            # Convert string keys back to integers
            self.user_tastes = {int(k): v for k, v in self.user_tastes.items()}
            self.initial_tastes = {int(k): v for k, v in self.initial_tastes.items()}
            self.debated_tastes = {int(k): v for k, v in self.debated_tastes.items()}
            self.refined_tastes = {int(k): v for k, v in self.refined_tastes.items()}
            self.debate_patterns = {int(k): v for k, v in self.debate_patterns.items()}
            self.reflection_patterns = {int(k): v for k, v in self.reflection_patterns.items()}
            self.confidence_scores = {int(k): v for k, v in self.confidence_scores.items()}
            self.genre_preferences = {int(k): v for k, v in self.genre_preferences.items()}
            self.quality_metrics = {int(k): v for k, v in self.quality_metrics.items()}
            
            metadata = memory_data.get('metadata', {})
            print(f"✅ User Taste Memory loaded from {memory_file}")
            print(f"   - Users: {len(self.user_tastes)}")
            print(f"   - Last updated: {metadata.get('last_updated', 'Unknown')}")
            print(f"   - Version: {metadata.get('version', 'Unknown')}")
            print(f"   - Learning enabled: {metadata.get('learning_enabled', False)}")
            if metadata.get('learning_enabled', False):
                print(f"   - Learned patterns: {len(self.taste_patterns)}")
                print(f"   - Learned insights: {len(self.learned_insights)}")
                print(f"   - Successful prompts: {len(self.successful_prompts)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading user taste memory: {e}")
            return False
    
    def clear_memory(self):
        """Clear all memory data"""
        self.user_tastes.clear()
        self.initial_tastes.clear()
        self.debated_tastes.clear()
        self.refined_tastes.clear()
        self.debate_patterns.clear()
        self.reflection_patterns.clear()
        self.confidence_scores.clear()
        self.processing_history.clear()
        # Clear learning data
        self.taste_patterns.clear()
        self.successful_prompts.clear()
        self.user_demographics_patterns.clear()
        self.genre_preferences.clear()
        self.rating_patterns.clear()
        self.learned_insights.clear()
        self.quality_metrics.clear()
        print("🧹 User Taste Memory cleared (including learning data)")
    
    def get_memory_info(self) -> Dict:
        """Get detailed memory information
        
        Returns:
            Dictionary with detailed memory information
        """
        return {
            'total_users': len(self.user_tastes),
            'avg_confidence': sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            'processing_history_count': len(self.processing_history),
            'memory_size_mb': sum(len(str(v)) for v in self.user_tastes.values()) / (1024 * 1024),
            'recent_users': list(self.user_tastes.keys())[-5:] if self.user_tastes else [],
            'learned_patterns': len(self.taste_patterns),
            'learned_insights': len(self.learned_insights),
            'successful_prompts': len(self.successful_prompts)
        }
    
    def _learn_from_user_taste(self, user_id: int, initial_taste: str, debated_taste: str, 
                              refined_taste: str, confidence: float):
        """Learn from user taste generation experience
        
        Args:
            user_id: User ID
            initial_taste: Initial taste description
            debated_taste: Debated taste description
            refined_taste: Refined taste description
            confidence: Confidence score
        """
        # Analyze taste patterns
        self._analyze_taste_patterns(refined_taste, confidence)
        
        # Learn from successful patterns
        if confidence > 0.7:
            self._learn_successful_patterns(initial_taste, debated_taste, refined_taste)
        
        # Extract insights
        self._extract_insights(user_id, initial_taste, debated_taste, refined_taste, confidence)
        
        # Update quality metrics
        self._update_quality_metrics(user_id, confidence)
    
    def _analyze_taste_patterns(self, taste_description: str, confidence: float):
        """Analyze taste patterns from descriptions
        
        Args:
            taste_description: Taste description text
            confidence: Confidence score
        """
        # Extract common taste patterns
        taste_lower = taste_description.lower()
        
        # Genre patterns
        genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller', 'documentary']
        for genre in genres:
            if genre in taste_lower:
                pattern_key = f"genre_{genre}"
                self.taste_patterns[pattern_key] = self.taste_patterns.get(pattern_key, 0) + 1
        
        # Rating patterns
        if 'high' in taste_lower and 'rating' in taste_lower:
            self.taste_patterns['high_rating_preference'] = self.taste_patterns.get('high_rating_preference', 0) + 1
        if 'low' in taste_lower and 'rating' in taste_lower:
            self.taste_patterns['low_rating_preference'] = self.taste_patterns.get('low_rating_preference', 0) + 1
        
        # Demographic patterns
        if 'young' in taste_lower or 'teen' in taste_lower:
            self.taste_patterns['young_demographic'] = self.taste_patterns.get('young_demographic', 0) + 1
        if 'adult' in taste_lower or 'mature' in taste_lower:
            self.taste_patterns['adult_demographic'] = self.taste_patterns.get('adult_demographic', 0) + 1
    
    def _learn_successful_patterns(self, initial_taste: str, debated_taste: str, refined_taste: str):
        """Learn from successful taste generation patterns
        
        Args:
            initial_taste: Initial taste description
            debated_taste: Debated taste description
            refined_taste: Refined taste description
        """
        # Store successful prompt patterns
        if len(initial_taste) > 50 and len(refined_taste) > 50:
            pattern = {
                'initial_length': len(initial_taste),
                'refined_length': len(refined_taste),
                'improvement_ratio': len(refined_taste) / len(initial_taste) if len(initial_taste) > 0 else 1.0,
                'timestamp': time.time()
            }
            self.successful_prompts.append(pattern)
            
            # Keep only recent successful patterns (last 100)
            if len(self.successful_prompts) > 100:
                self.successful_prompts = self.successful_prompts[-100:]
    
    def _extract_insights(self, user_id: int, initial_taste: str, debated_taste: str, 
                         refined_taste: str, confidence: float):
        """Extract insights from taste generation process
        
        Args:
            user_id: User ID
            initial_taste: Initial taste description
            debated_taste: Debated taste description
            refined_taste: Refined taste description
            confidence: Confidence score
        """
        # Analyze improvement patterns
        if len(initial_taste) > 0 and len(refined_taste) > 0:
            improvement = len(refined_taste) / len(initial_taste)
            
            if improvement > 1.5:
                insight = {
                    'type': 'significant_improvement',
                    'user_id': user_id,
                    'improvement_ratio': improvement,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                self.learned_insights.append(insight)
            
            if confidence > 0.8:
                insight = {
                    'type': 'high_confidence_generation',
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
                'total_generations': 0,
                'avg_confidence': 0.0,
                'high_confidence_count': 0,
                'last_updated': time.time()
            }
        
        metrics = self.quality_metrics[user_id]
        metrics['total_generations'] += 1
        metrics['avg_confidence'] = (metrics['avg_confidence'] * (metrics['total_generations'] - 1) + confidence) / metrics['total_generations']
        
        if confidence > 0.7:
            metrics['high_confidence_count'] += 1
        
        metrics['last_updated'] = time.time()
    
    def get_learned_insights(self) -> List[Dict]:
        """Get learned insights from memory
        
        Returns:
            List of learned insights
        """
        return self.learned_insights.copy()
    
    def get_taste_patterns(self) -> Dict:
        """Get learned taste patterns
        
        Returns:
            Dictionary of taste patterns and their frequencies
        """
        return self.taste_patterns.copy()
    
    def get_successful_prompts(self) -> List[Dict]:
        """Get successful prompt patterns
        
        Returns:
            List of successful prompt patterns
        """
        return self.successful_prompts.copy()
    
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
            'total_users_processed': len(self.user_tastes),
            'learned_patterns': len(self.taste_patterns),
            'successful_prompts': len(self.successful_prompts),
            'learned_insights': len(self.learned_insights),
            'avg_confidence': sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            'high_confidence_users': sum(1 for conf in self.confidence_scores.values() if conf > 0.7),
            'top_patterns': sorted(self.taste_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        }

class UserTasteGenerator:
    """User Taste Generator Module - Generate unique taste descriptions with debate and reflection"""
    
    def __init__(self, openai_api_key: str, user_profiles_file: str = "user_profiles.json", 
                 memory_file: str = None, enable_debate_reflection: bool = True):
        self.openai_api_key = openai_api_key
        self.user_profiles_file = user_profiles_file
        self.memory_file = memory_file
        self.enable_debate_reflection = enable_debate_reflection
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Load user profiles
        self.user_profiles = self._load_user_profiles()
        
        # Rate limiting configuration
        self.base_delay = 0.5  # Base delay between requests (seconds)
        self.max_delay = 5.0   # Maximum delay between requests (seconds)
        self.requests_per_minute = 60  # Target requests per minute
        self.retry_attempts = 3  # Number of retry attempts for failed requests
        self.consecutive_failures = 0  # Track consecutive failures
        
        # Initialize memory module
        self.memory = UserTasteMemory()
        
        # Load memory if file is provided
        if self.memory_file:
            self.load_memory()
        
        # Initialize modules only if debate and reflection are enabled
        if self.enable_debate_reflection:
            self.debate_module = DebateModule(self.client, self)
            self.reflection_module = ReflectionModule(self.client, self)
            print("🧠 Debate and Reflection modules enabled")
        else:
            self.debate_module = None
            self.reflection_module = None
            print("⚡ Simple mode: Direct GPT generation only")
        
        print(f"✅ User Taste Generator initialized with {len(self.user_profiles)} user profiles")
        print(f"📊 Rate limiting: {self.requests_per_minute} requests/minute")
        print(f"⏱️  Base delay: {self.base_delay}s, Max delay: {self.max_delay}s")
        if self.memory_file:
            print(f"🧠 Memory file: {self.memory_file}")
    
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
                print(f"🧠 User Taste Memory loaded successfully from {self.memory_file}")
            else:
                print(f"🆕 No existing memory file found at {self.memory_file}")
                print(f"   This appears to be the first run. Memory will be created after processing.")
        else:
            print("⚠️ No memory file specified")
    
    def save_memory(self, memory_file: str = None):
        """Save memory to file
        
        Args:
            memory_file: Path to memory file (optional, uses self.memory_file if not provided)
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
    
    def get_memory_info(self) -> Dict:
        """Get memory information
        
        Returns:
            Dictionary with memory information
        """
        return self.memory.get_memory_info()
    
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
    
    def _smart_delay(self):
        """Implement smart delay with exponential backoff"""
        if self.consecutive_failures == 0:
            # Normal operation - use base delay with small random variation
            delay = self.base_delay + random.uniform(0, 0.2)
        else:
            # Exponential backoff with jitter
            delay = min(self.base_delay * (2 ** self.consecutive_failures), self.max_delay)
            delay += random.uniform(0, delay * 0.1)  # Add jitter
        
        time.sleep(delay)
    
    def _handle_api_error(self, error: Exception) -> bool:
        """Handle API errors and determine if retry is needed
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if retry is recommended, False otherwise
        """
        error_str = str(error).lower()
        
        # Rate limit errors - definitely retry
        if any(phrase in error_str for phrase in ['rate limit', 'too many requests', 'quota exceeded']):
            print(f"⚠️  Rate limit hit, increasing delay...")
            self.consecutive_failures += 1
            return True
        
        # Server errors - retry with backoff
        elif any(phrase in error_str for phrase in ['server error', 'internal error', 'service unavailable']):
            print(f"⚠️  Server error, will retry...")
            self.consecutive_failures += 1
            return True
        
        # Authentication errors - don't retry
        elif any(phrase in error_str for phrase in ['authentication', 'unauthorized', 'invalid api key']):
            print(f"❌ Authentication error, skipping...")
            return False
        
        # Other errors - retry once
        else:
            print(f"⚠️  Unexpected error, will retry once...")
            self.consecutive_failures += 1
            return True
    
    def _reset_failure_counter_between_stages(self, stage_name: str):
        """Reset failure counter between processing stages
        
        Args:
            stage_name: Name of the stage for logging purposes
        """
        if self.consecutive_failures > 0:
            print(f"🔄 Reset failure counter {stage_name} (was {self.consecutive_failures})")
            self.consecutive_failures = 0
    
    def get_filtered_users(self, filtered_train_file: str = "filtered_train_set.txt", 
                          rec_save_dict_file: str = "rec_save_dict.csv") -> Set[int]:
        """Get users that exist in both filtered_train_set.txt and rec_save_dict.csv
        
        Args:
            filtered_train_file: Path to filtered training set file
            rec_save_dict_file: Path to recommendation save dict file
            
        Returns:
            Set of user IDs that should be processed
        """
        print(f"🔍 Loading user IDs from {filtered_train_file} and {rec_save_dict_file}...")
        
        filtered_users = set()
        rec_users = set()
        
        # Load users from filtered_train_set.txt
        try:
            with open(filtered_train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        user_id = int(parts[0])
                        filtered_users.add(user_id)
            print(f"✅ Loaded {len(filtered_users)} users from {filtered_train_file}")
        except Exception as e:
            print(f"❌ Error loading {filtered_train_file}: {e}")
            return set()
        
        # Load users from rec_save_dict.csv
        try:
            df = pd.read_csv(rec_save_dict_file, header=None)
            for user_id in df.iloc[:, 0]:  # First column contains user IDs
                rec_users.add(int(user_id))
            print(f"✅ Loaded {len(rec_users)} users from {rec_save_dict_file}")
        except Exception as e:
            print(f"❌ Error loading {rec_save_dict_file}: {e}")
            return set()
        
        # Combine both sets
        target_users = filtered_users.union(rec_users)
        print(f"🎯 Total target users to process: {len(target_users)}")
        print(f"   - Users in filtered_train_set: {len(filtered_users)}")
        print(f"   - Users in rec_save_dict: {len(rec_users)}")
        print(f"   - Union of both sets: {len(target_users)}")
        
        return target_users
    
    def get_filtered_and_rec_users(self, filtered_train_file: str = "filtered_train_set.txt", 
                                  rec_save_dict_file: str = "rec_save_dict.csv") -> Tuple[Set[int], Set[int]]:
        """Get users from filtered_train_set.txt and rec_save_dict.csv separately
        
        Args:
            filtered_train_file: Path to filtered training set file
            rec_save_dict_file: Path to recommendation save dict file
            
        Returns:
            Tuple of (filtered_users_set, rec_users_set)
        """
        print(f"🔍 Loading user IDs from {filtered_train_file} and {rec_save_dict_file}...")
        
        filtered_users = set()
        rec_users = set()
        
        # Load users from filtered_train_set.txt
        try:
            with open(filtered_train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        user_id = int(parts[0])
                        filtered_users.add(user_id)
            print(f"✅ Loaded {len(filtered_users)} users from {filtered_train_file}")
        except Exception as e:
            print(f"❌ Error loading {filtered_train_file}: {e}")
            return set(), set()
        
        # Load users from rec_save_dict.csv
        try:
            df = pd.read_csv(rec_save_dict_file, header=None)
            for user_id in df.iloc[:, 0]:  # First column contains user IDs
                rec_users.add(int(user_id))
            print(f"✅ Loaded {len(rec_users)} users from {rec_save_dict_file}")
        except Exception as e:
            print(f"❌ Error loading {rec_save_dict_file}: {e}")
            return set(), set()
        
        print(f"🎯 User sets loaded:")
        print(f"   - Users in filtered_train_set: {len(filtered_users)}")
        print(f"   - Users in rec_save_dict: {len(rec_users)}")
        print(f"   - Union of both sets: {len(filtered_users.union(rec_users))}")
        
        return filtered_users, rec_users
    
    def generate_initial_tastes(self, user_ids: List[int] = None) -> Dict[int, str]:
        """Generate initial taste descriptions for users with API frequency control"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())
        
        print(f"Generating initial taste descriptions for {len(user_ids)} users...")
        
        # Estimate processing time
        estimated_time_per_user = (self.base_delay + 0.1)  # Base delay + API call time
        estimated_total_time = len(user_ids) * estimated_time_per_user / 60  # Convert to minutes
        print(f"⏱️  Estimated time: {estimated_total_time:.1f} minutes")
        print(f"📊 Target rate: {self.requests_per_minute} requests/minute")
        
        initial_tastes = {}
        start_time = time.time()
        
        for i, user_id in enumerate(user_ids):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Progress update with time estimation
            if i % 10 == 0 or i == len(user_ids) - 1:
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_user = elapsed_time / i
                    remaining_users = len(user_ids) - i
                    estimated_remaining = remaining_users * avg_time_per_user / 60
                    print(f"Processing user {i+1}/{len(user_ids)}: User ID {user_id}")
                    print(f"   ⏱️  Elapsed: {elapsed_time/60:.1f}m, Remaining: {estimated_remaining:.1f}m")
                else:
                    print(f"Processing user {i+1}/{len(user_ids)}: User ID {user_id}")
            
            # Generate taste description with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    user_prompt = self._build_user_prompt(self.user_profiles[user_id])
                    
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=200,
                        temperature=0.7
                    )
                    
                    taste_description = response.choices[0].message.content.strip()
                    initial_tastes[user_id] = taste_description
                    
                    # Success - reset failure counter and apply smart delay
                    if self.consecutive_failures > 0:
                        print(f"✅ Recovered from {self.consecutive_failures} consecutive failures")
                        self.consecutive_failures = 0
                    
                    self._smart_delay()
                    break
                    
                except Exception as e:
                    print(f"GPT analysis attempt {attempt + 1} failed for user {user_id}: {e}")
                    
                    if attempt < self.retry_attempts - 1 and self._handle_api_error(e):
                        print(f"Retrying in {self.base_delay * (2 ** self.consecutive_failures):.1f}s...")
                        self._smart_delay()
                        continue
                    else:
                        print(f"All retry attempts failed, using fallback description")
                        initial_tastes[user_id] = "This user's taste characteristics need further analysis"
                        break
        
        total_time = time.time() - start_time
        print(f"✅ Initial taste descriptions generated: {len(initial_tastes)} users")
        print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
        print(f"📊 Average time per user: {total_time/len(user_ids):.2f} seconds")
        
        return initial_tastes
    
    def generate_simple_tastes(self, user_ids: List[int] = None) -> Dict[int, str]:
        """Generate simple taste descriptions directly using GPT (without debate and reflection)"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())
        
        print(f"Generating simple taste descriptions for {len(user_ids)} users...")
        
        # Estimate processing time
        estimated_time_per_user = (self.base_delay + 0.1)  # Base delay + API call time
        estimated_total_time = len(user_ids) * estimated_time_per_user / 60  # Convert to minutes
        print(f"⏱️  Estimated time: {estimated_total_time:.1f} minutes")
        print(f"📊 Target rate: {self.requests_per_minute} requests/minute")
        
        simple_tastes = {}
        start_time = time.time()
        
        for i, user_id in enumerate(user_ids):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Progress update with time estimation
            if i % 10 == 0 or i == len(user_ids) - 1:
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_user = elapsed_time / i
                    remaining_users = len(user_ids) - i
                    estimated_remaining = remaining_users * avg_time_per_user / 60
                    print(f"Processing user {i+1}/{len(user_ids)}: User ID {user_id}")
                    print(f"   ⏱️  Elapsed: {elapsed_time/60:.1f}m, Remaining: {estimated_remaining:.1f}m")
                else:
                    print(f"Processing user {i+1}/{len(user_ids)}: User ID {user_id}")
            
            # Generate taste description with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    user_prompt = self._build_simple_user_prompt(self.user_profiles[user_id])
                    
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": self._get_simple_system_prompt()},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=300,
                        temperature=0.7
                    )
                    
                    taste_description = response.choices[0].message.content.strip()
                    simple_tastes[user_id] = taste_description
                    
                    # Success - reset failure counter and apply smart delay
                    if self.consecutive_failures > 0:
                        print(f"✅ Recovered from {self.consecutive_failures} consecutive failures")
                        self.consecutive_failures = 0
                    
                    self._smart_delay()
                    break
                    
                except Exception as e:
                    print(f"GPT analysis attempt {attempt + 1} failed for user {user_id}: {e}")
                    
                    if attempt < self.retry_attempts - 1 and self._handle_api_error(e):
                        print(f"Retrying in {self.base_delay * (2 ** self.consecutive_failures):.1f}s...")
                        self._smart_delay()
                        continue
                    else:
                        print(f"All retry attempts failed, using fallback description")
                        simple_tastes[user_id] = "This user's taste characteristics need further analysis"
                        break
        
        total_time = time.time() - start_time
        print(f"✅ Simple taste descriptions generated: {len(simple_tastes)} users")
        print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
        print(f"📊 Average time per user: {total_time/len(user_ids):.2f} seconds")
        
        return simple_tastes
    
    def _build_simple_user_prompt(self, profile: Dict) -> str:
        """Build user prompt for simple taste generation"""
        user_id = profile['user_id']
        interactions = profile['interactions']
        stats = profile['statistics']
        demographics = profile.get('demographics', {})
        
        # Sort interactions by rating
        high_rated = [item for item in interactions if item['rating'] >= 4]
        medium_rated = [item for item in interactions if 3 <= item['rating'] < 4]
        low_rated = [item for item in interactions if item['rating'] <= 2]
        
        prompt = f"User ID: {user_id}\n\n"
        
        # Add demographic information if available
        if demographics:
            prompt += f"Demographics:\n"
            prompt += f"- Gender: {demographics.get('gender', 'Unknown')}\n"
            prompt += f"- Age: {demographics.get('age', 'Unknown')}\n"
            prompt += f"- Occupation: {demographics.get('occupation', 'Unknown')}\n\n"
        
        prompt += f"User Statistics:\n"
        prompt += f"- Total movies watched: {stats.get('total_movies', 0)}\n"
        prompt += f"- Rating distribution: {stats.get('rating_distribution', {})}\n\n"
        
        prompt += "User viewing history (sorted by rating):\n\n"
        
        if high_rated:
            prompt += "High-rated movies (4-5 stars):\n"
            for item in high_rated[:8]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        if medium_rated:
            prompt += "Medium-rated movies (3 stars):\n"
            for item in medium_rated[:5]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        if low_rated:
            prompt += "Low-rated movies (1-2 stars):\n"
            for item in low_rated[:3]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        prompt += """Based on the above user profile, please analyze and describe this user's unique taste characteristics in movies. Consider:
1. Genre preferences (based on high-rated movies)
2. Rating patterns and viewing behavior
3. Demographic influences on taste
4. Unique characteristics that distinguish this user

Please provide a concise but comprehensive description of this user's taste profile in 2-3 sentences."""
        
        return prompt
    
    def _get_simple_system_prompt(self) -> str:
        """Get system prompt for simple taste generation"""
        return """You are an expert movie taste analyst. Your task is to analyze user movie viewing data and generate concise, accurate descriptions of their unique taste characteristics.

Guidelines:
- Focus on the most distinctive aspects of the user's taste
- Consider genre preferences, rating patterns, and demographic factors
- Be specific and avoid generic descriptions
- Keep the description concise but informative (2-3 sentences)
- Highlight what makes this user's taste unique compared to others"""
    
    def _build_user_prompt(self, profile: Dict) -> str:
        """Build user prompt for taste generation"""
        user_id = profile['user_id']
        interactions = profile['interactions']
        stats = profile['statistics']
        demographics = profile.get('demographics', {})
        
        # Sort interactions by rating
        high_rated = [item for item in interactions if item['rating'] >= 4]
        medium_rated = [item for item in interactions if 3 <= item['rating'] < 4]
        low_rated = [item for item in interactions if item['rating'] <= 2]
        
        prompt = f"User ID: {user_id}\n\n"
        
        # Add demographic information if available
        if demographics:
            prompt += f"Demographics:\n"
            prompt += f"- Gender: {demographics.get('gender', 'Unknown')}\n"
            prompt += f"- Age: {demographics.get('age', 'Unknown')}\n"
            prompt += f"- Occupation: {demographics.get('occupation', 'Unknown')}\n\n"
        
        prompt += f"User Statistics:\n"
        prompt += f"- Total movies watched: {stats.get('total_movies', 0)}\n"
        prompt += f"- Rating distribution: {stats.get('rating_distribution', {})}\n\n"
        
        prompt += "User viewing history (sorted by rating):\n\n"
        
        if high_rated:
            prompt += "High-rated movies (4-5 stars):\n"
            for item in high_rated[:8]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        if medium_rated:
            prompt += "Medium-rated movies (3-4 stars):\n"
            for item in medium_rated[:5]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        if low_rated:
            prompt += "Low-rated movies (1-2 stars):\n"
            for item in low_rated[:3]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        prompt += "Please analyze this user's unique taste characteristics based on the above comprehensive data. Focus on their preferences, dislikes, and what makes their taste profile distinctive."
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for taste generation"""
        return """You are an expert movie taste analyst. Your task is to analyze user viewing history and demographic information to identify their unique taste characteristics.

Focus on:
1. Genre preferences and dislikes
2. Rating patterns and what they indicate
3. Demographic factors that might influence preferences
4. Unique combinations of preferences that make this user distinctive

Provide clear, concise taste descriptions that capture the essence of what makes this user's preferences unique."""
    
    def run_full_workflow(self, user_ids: List[int] = None, max_users: int = None, 
                         use_filtered_users: bool = False, 
                         filtered_train_file: str = "filtered_train_set.txt",
                         rec_save_dict_file: str = "rec_save_dict.csv") -> Dict:
        """Run the complete taste generation workflow"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())
        
        # Note: User filtering and sampling logic is now handled in main() function
        # This method now receives the final user list to process
        
        # Apply max_users limit if specified
        if max_users and len(user_ids) > max_users:
            print(f"⚠️ Limiting processing to {max_users} users (from {len(user_ids)} available)")
            user_ids = user_ids[:max_users]
        
        print("=" * 60)
        print("A1-2: User Taste Generation Workflow")
        print("=" * 60)
        print(f"Processing {len(user_ids)} users...")
        print()
        
        # Check memory for cached results and filter out already processed users
        cached_users = []
        new_users = []
        
        for user_id in user_ids:
            if self.memory.has_user_taste(user_id):
                cached_users.append(user_id)
            else:
                new_users.append(user_id)
        
        print(f"📊 Memory status:")
        print(f"   - Cached users: {len(cached_users)}")
        print(f"   - New users to process: {len(new_users)}")
        
        # Load cached results
        initial_tastes = {}
        debated_tastes = {}
        refined_tastes = {}
        
        for user_id in cached_users:
            cached_data = self.memory.get_user_taste(user_id)
            if cached_data:
                initial_tastes[user_id] = cached_data['initial_taste']
                debated_tastes[user_id] = cached_data['debated_taste']
                refined_tastes[user_id] = cached_data['refined_taste']
        
        print(f"✅ Loaded {len(cached_users)} cached results from memory")
        
        # Process new users only
        if new_users:
            print(f"\n🔄 Processing {len(new_users)} new users...")
            
            if self.enable_debate_reflection:
                # Full mode: Debate and Reflection
                print("🧠 Using full mode: Debate and Reflection enabled")
                
                # Step 1: Generate initial tastes for new users
                print("Step 1: Generating initial taste descriptions...")
                new_initial_tastes = self.generate_initial_tastes(new_users)
                initial_tastes.update(new_initial_tastes)
                
                # Reset failure counter between Step 1 and Step 2
                self._reset_failure_counter_between_stages("between Step 1 and Step 2")
                
                # Step 2: Multi-agent debate for new users
                print("\nStep 2: Conducting multi-agent debate...")
                new_debated_tastes = self.debate_module.debate_user_tastes(
                    {uid: self.user_profiles[uid] for uid in new_users}, 
                    new_initial_tastes
                )
                debated_tastes.update(new_debated_tastes)
                
                # Reset failure counter between Step 2 and Step 3
                self._reset_failure_counter_between_stages("between Step 2 and Step 3")
                
                # Step 3: Reflection and refinement for new users
                print("\nStep 3: Reflecting and refining taste descriptions...")
                new_refined_tastes = self.reflection_module.refine_tastes(
                    {uid: self.user_profiles[uid] for uid in new_users}, 
                    new_debated_tastes
                )
                refined_tastes.update(new_refined_tastes)
                
                # Store new results in memory
                for user_id in new_users:
                    self.memory.store_user_taste(
                        user_id=user_id,
                        initial_taste=new_initial_tastes.get(user_id, ''),
                        debated_taste=new_debated_tastes.get(user_id, ''),
                        refined_taste=new_refined_tastes.get(user_id, ''),
                        debate_summary=f"Debate completed for user {user_id}",
                        reflection_summary=f"Reflection completed for user {user_id}",
                        confidence=0.8
                    )
                
                print(f"✅ Stored {len(new_users)} new results in memory")
                
            else:
                # Simple mode: Direct GPT generation only
                print("⚡ Using simple mode: Direct GPT generation only")
                
                # Generate simple tastes directly
                print("Step 1: Generating simple taste descriptions...")
                new_simple_tastes = self.generate_simple_tastes(new_users)
                
                # For simple mode, use the same taste for all stages
                for user_id in new_users:
                    taste = new_simple_tastes.get(user_id, '')
                    initial_tastes[user_id] = taste
                    debated_tastes[user_id] = taste
                    refined_tastes[user_id] = taste
                
                # Store new results in memory
                for user_id in new_users:
                    self.memory.store_user_taste(
                        user_id=user_id,
                        initial_taste=new_simple_tastes.get(user_id, ''),
                        debated_taste=new_simple_tastes.get(user_id, ''),
                        refined_taste=new_simple_tastes.get(user_id, ''),
                        debate_summary=f"Simple mode - no debate for user {user_id}",
                        reflection_summary=f"Simple mode - no reflection for user {user_id}",
                        confidence=0.7
                    )
                
                print(f"✅ Stored {len(new_users)} new results in memory")
        else:
            print("✅ All users already processed, using cached results")
        
        # Reset failure counter before final reflection
        self._reset_failure_counter_between_stages("before final reflection")
        
        # Step 4: Final reflection (only for full mode)
        if self.enable_debate_reflection:
            print("\nStep 4: Performing final reflection...")
            final_reflection = self.reflection_module.final_reflection(
                {uid: self.user_profiles[uid] for uid in user_ids}, 
                refined_tastes
            )
        else:
            print("\nStep 4: Simple mode - skipping final reflection")
            final_reflection = {
                'final_reflection': f"Simple mode completed for {len(user_ids)} users. No debate or reflection was performed.",
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Compile results
        results = {
            'workflow_info': {
                'total_users': len(user_ids),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'workflow_version': '1.2',
                'use_filtered_users': use_filtered_users,
                'filtered_train_file': filtered_train_file if use_filtered_users else None,
                'rec_save_dict_file': rec_save_dict_file if use_filtered_users else None,
                'enable_debate_reflection': self.enable_debate_reflection,
                'mode': 'Full Mode (Debate + Reflection)' if self.enable_debate_reflection else 'Simple Mode (Direct GPT)',
                'cached_users': len(cached_users),
                'new_users': len(new_users)
            },
            'initial_tastes': initial_tastes,
            'debated_tastes': debated_tastes,
            'refined_tastes': refined_tastes,
            'final_reflection': final_reflection
        }
        
        print("\n" + "=" * 60)
        print("✅ User Taste Generation Workflow Completed!")
        print("=" * 60)
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "user_tastes.json"):
        """Save results to JSON file"""
        print(f"Saving results to {output_file}...")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"✅ Results saved: {output_file}")
            print(f"   File size: {file_size:.2f} MB")
            
            # Save memory state
            if self.memory_file:
                self.save_memory()
                # Display learning progress
                self._display_learning_progress()
            else:
                print("⚠️ No memory file specified, memory not saved")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def _display_learning_progress(self):
        """Display learning progress from memory"""
        if not self.memory:
            return
        
        learning_summary = self.memory.get_memory_learning_summary()
        
        print("\n" + "=" * 60)
        print("🧠 Memory Learning Progress")
        print("=" * 60)
        print(f"📊 Total users processed: {learning_summary['total_users_processed']}")
        print(f"🎯 Learned patterns: {learning_summary['learned_patterns']}")
        print(f"💡 Learned insights: {learning_summary['learned_insights']}")
        print(f"✅ Successful prompts: {learning_summary['successful_prompts']}")
        print(f"📈 Average confidence: {learning_summary['avg_confidence']:.3f}")
        print(f"🌟 High confidence users: {learning_summary['high_confidence_users']}")
        
        if learning_summary['top_patterns']:
            print(f"\n🔍 Top learned patterns:")
            for pattern, count in learning_summary['top_patterns']:
                print(f"   - {pattern}: {count} occurrences")
        
        print("=" * 60)


class DebateModule:
    """Multi-agent debate module for refining user taste descriptions"""
    
    def __init__(self, client: OpenAI, parent_generator):
        self.client = client
        self.parent_generator = parent_generator
    
    def debate_user_tastes(self, user_profiles: Dict[int, Dict], initial_tastes: Dict[int, str]) -> Dict[int, str]:
        """Conduct multi-agent debate on user tastes with API frequency control"""
        debated_tastes = {}
        user_ids = list(user_profiles.keys())
        
        print(f"Conducting multi-agent debate for {len(user_ids)} users...")
        
        # Estimate processing time
        estimated_time_per_user = (self.parent_generator.base_delay + 0.1) * 4  # 4 API calls per user
        estimated_total_time = len(user_ids) * estimated_time_per_user / 60
        print(f"⏱️  Estimated debate time: {estimated_total_time:.1f} minutes")
        
        start_time = time.time()
        
        for i, (user_id, profile) in enumerate(user_profiles.items()):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.parent_generator.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.parent_generator.consecutive_failures})")
                    self.parent_generator.consecutive_failures = 0
            
            # Progress update
            if i % 5 == 0 or i == len(user_ids) - 1:
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_user = elapsed_time / i
                    remaining_users = len(user_ids) - i
                    estimated_remaining = remaining_users * avg_time_per_user / 60
                    print(f"  Debating user {i+1}/{len(user_ids)}: User ID {user_id}")
                    print(f"     ⏱️  Elapsed: {elapsed_time/60:.1f}m, Remaining: {estimated_remaining:.1f}m")
                else:
                    print(f"  Debating user {i+1}/{len(user_ids)}: User ID {user_id}")
            
            try:
                # Generate individual agent analyses
                agent_analyses = self._generate_agent_analyses(profile, initial_tastes.get(user_id, ""))
                
                # Conduct debate
                debate_result = self._conduct_debate(profile, agent_analyses)
                
                debated_tastes[user_id] = debate_result['consensus']
                
                # Apply smart delay
                self.parent_generator._smart_delay()
                
            except Exception as e:
                print(f"  Error in debate for user {user_id}: {e}")
                debated_tastes[user_id] = initial_tastes.get(user_id, "Debate failed, using initial analysis")
        
        total_time = time.time() - start_time
        print(f"✅ Multi-agent debate completed: {len(debated_tastes)} users")
        print(f"⏱️  Total debate time: {total_time/60:.1f} minutes")
        
        return debated_tastes
    
    def _generate_agent_analyses(self, profile: Dict, initial_taste: str) -> Dict[str, str]:
        """Generate analyses from three different agent perspectives with API frequency control"""
        agent_analyses = {}
        
        # Agent 1: Genre Analyst
        agent1_prompt = f"""As a Genre Analyst, analyze this user's genre preferences:

User Profile: {profile}
Initial Taste Analysis: {initial_taste}

Focus on:
1. What movie genres does this user prefer?
2. Are there any genres they consistently avoid?
3. How do genre preferences relate to their ratings?

Provide your analysis in 2-3 sentences."""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": agent1_prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                agent_analyses['genre_analyst'] = response.choices[0].message.content.strip()
                self.parent_generator._smart_delay()
                break
            except Exception as e:
                if attempt < self.parent_generator.retry_attempts - 1 and self.parent_generator._handle_api_error(e):
                    self.parent_generator._smart_delay()
                    continue
                else:
                    agent_analyses['genre_analyst'] = f"Genre analysis failed: {e}"
                    break
        
        # Agent 2: Behavioral Analyst
        agent2_prompt = f"""As a Behavioral Analyst, analyze this user's viewing behavior:

User Profile: {profile}
Initial Taste Analysis: {initial_taste}

Focus on:
1. What are their rating patterns?
2. How do they interact with different types of content?
3. What viewing behaviors indicate their preferences?

Provide your analysis in 2-3 sentences."""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": agent2_prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                agent_analyses['behavioral_analyst'] = response.choices[0].message.content.strip()
                self.parent_generator._smart_delay()
                break
            except Exception as e:
                if attempt < self.parent_generator.retry_attempts - 1 and self.parent_generator._handle_api_error(e):
                    self.parent_generator._smart_delay()
                    continue
                else:
                    agent_analyses['behavioral_analyst'] = f"Behavioral analysis failed: {e}"
                    break
        
        # Agent 3: Content Analyst
        agent3_prompt = f"""As a Content Analyst, analyze this user's content preferences:

User Profile: {profile}
Initial Taste Analysis: {initial_taste}

Focus on:
1. What themes or styles do they prefer?
2. What content characteristics appeal to them?
3. How do movie qualities relate to their preferences?

Provide your analysis in 2-3 sentences."""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": agent3_prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                agent_analyses['content_analyst'] = response.choices[0].message.content.strip()
                self.parent_generator._smart_delay()
                break
            except Exception as e:
                if attempt < self.parent_generator.retry_attempts - 1 and self.parent_generator._handle_api_error(e):
                    self.parent_generator._smart_delay()
                    continue
                else:
                    agent_analyses['content_analyst'] = f"Content analysis failed: {e}"
                    break
        
        return agent_analyses
    
    def _conduct_debate(self, profile: Dict, agent_analyses: Dict[str, str]) -> Dict:
        """Conduct debate among agents to reach consensus with API frequency control"""
        debate_prompt = f"""Three agents have analyzed the same user data. Now they need to debate and reach a consensus.

User Profile: {profile}

Agent Analyses:
1. Genre Analyst: {agent_analyses.get('genre_analyst', 'N/A')}
2. Behavioral Analyst: {agent_analyses.get('behavioral_analyst', 'N/A')}
3. Content Analyst: {agent_analyses.get('content_analyst', 'N/A')}

Please simulate a constructive debate among these three agents and reach a consensus on the user's taste characteristics. The final consensus should be a refined, comprehensive description that incorporates insights from all three perspectives.

Format your response as:
DEBATE PROCESS: [Describe the debate process]
CONSENSUS: [Final consensus description]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": debate_prompt}],
                    max_tokens=400,
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
                if attempt < self.parent_generator.retry_attempts - 1 and self.parent_generator._handle_api_error(e):
                    self.parent_generator._smart_delay()
                    continue
                else:
                    return {
                        'consensus': f"Debate failed: {e}",
                        'debate_process': "Error occurred during debate"
                    }


class ReflectionModule:
    """Reflection module for continuous improvement and refinement"""
    
    def __init__(self, client: OpenAI, parent_generator):
        self.client = client
        self.parent_generator = parent_generator
    
    def refine_tastes(self, user_profiles: Dict[int, Dict], debated_tastes: Dict[int, str]) -> Dict[int, str]:
        """Refine taste descriptions through reflection with API frequency control"""
        refined_tastes = {}
        user_ids = list(user_profiles.keys())
        
        print(f"Reflecting and refining tastes for {len(user_ids)} users...")
        
        # Estimate processing time
        estimated_time_per_user = (self.parent_generator.base_delay + 0.1) * 2  # 2 API calls per user
        estimated_total_time = len(user_ids) * estimated_time_per_user / 60
        print(f"⏱️  Estimated reflection time: {estimated_total_time:.1f} minutes")
        
        start_time = time.time()
        
        for i, (user_id, profile) in enumerate(user_profiles.items()):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.parent_generator.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.parent_generator.consecutive_failures})")
                    self.parent_generator.consecutive_failures = 0
            
            # Progress update
            if i % 5 == 0 or i == len(user_ids) - 1:
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_user = elapsed_time / i
                    remaining_users = len(user_ids) - i
                    estimated_remaining = remaining_users * avg_time_per_user / 60
                    print(f"  Reflecting on user {i+1}/{len(user_ids)}: User ID {user_id}")
                    print(f"     ⏱️  Elapsed: {elapsed_time/60:.1f}m, Remaining: {estimated_remaining:.1f}m")
                else:
                    print(f"  Reflecting on user {i+1}/{len(user_ids)}: User ID {user_id}")
            
            try:
                reflection_result = self._reflect_on_taste_analysis(profile, debated_tastes.get(user_id, ""))
                
                refined_tastes[user_id] = reflection_result['refined_taste']
                
                # Apply smart delay
                self.parent_generator._smart_delay()
                
            except Exception as e:
                print(f"  Error in reflection for user {user_id}: {e}")
                refined_tastes[user_id] = debated_tastes.get(user_id, "Reflection failed, using debated analysis")
        
        total_time = time.time() - start_time
        print(f"✅ Reflection and refinement completed: {len(refined_tastes)} users")
        print(f"⏱️  Total reflection time: {total_time/60:.1f} minutes")
        
        return refined_tastes
    
    def _reflect_on_taste_analysis(self, profile: Dict, current_taste: str) -> Dict:
        """Reflect on and refine the current taste analysis with API frequency control"""
        reflection_prompt = f"""Please reflect on this user taste analysis and suggest improvements:

User Profile: {profile}
Current Taste Analysis: {current_taste}

Reflection Questions:
1. Is this analysis consistent with the user's viewing data?
2. Are there any patterns or preferences that might have been overlooked?
3. Could this analysis be more specific or nuanced?
4. Are there any contradictions between the analysis and the data?

Please provide:
1. Your reflection insights
2. A refined taste description that addresses any issues you identified

Format your response as:
REFLECTION INSIGHTS: [Your insights]
REFINED TASTE: [Improved taste description]"""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse response
                if "REFINED TASTE:" in response_text:
                    refined_taste = response_text.split("REFINED TASTE:")[-1].strip()
                    reflection_insights = response_text.split("REFINED TASTE:")[0].replace("REFLECTION INSIGHTS:", "").strip()
                else:
                    refined_taste = response_text
                    reflection_insights = "Reflection insights not clearly formatted"
                
                self.parent_generator._smart_delay()
                return {
                    'reflection_insights': reflection_insights,
                    'refined_taste': refined_taste
                }
                
            except Exception as e:
                if attempt < self.parent_generator.retry_attempts - 1 and self.parent_generator._handle_api_error(e):
                    self.parent_generator._smart_delay()
                    continue
                else:
                    return {
                        'reflection_insights': f"Reflection failed: {e}",
                        'refined_taste': current_taste
                    }
    
    def final_reflection(self, user_profiles: Dict[int, Dict], refined_tastes: Dict[int, str]) -> Dict:
        """Perform final reflection on the entire process with API frequency control"""
        final_reflection_prompt = f"""Please provide a final reflection on the user taste analysis process:

Process Summary:
- Number of users processed: {len(user_profiles)}
- Taste analysis method: Multi-agent debate + reflection
- Analysis approach: Initial generation → Debate → Reflection → Refinement

Questions for reflection:
1. How effective was the multi-agent debate approach?
2. Did the reflection process improve the quality of taste descriptions?
3. Are there any systematic patterns in the results?
4. What could be improved in future iterations?

Please provide a comprehensive reflection on the process quality and outcomes."""

        for attempt in range(self.parent_generator.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": final_reflection_prompt}],
                    max_tokens=400,
                    temperature=0.7
                )
                
                self.parent_generator._smart_delay()
                return {
                    'final_reflection': response.choices[0].message.content.strip(),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                if attempt < self.parent_generator.retry_attempts - 1 and self.parent_generator._handle_api_error(e):
                    self.parent_generator._smart_delay()
                    continue
                else:
                    return {
                        'final_reflection': f"Final reflection failed: {e}",
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate user taste descriptions with debate and reflection')
    parser.add_argument('--user-profiles', type=str, default='user_profiles.json',
                       help='User profiles JSON file path')
    parser.add_argument('--output', type=str, default='user_tastes.json',
                       help='Output JSON file path')
    parser.add_argument('--openai-key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of users to sample randomly (None for all users)')
    parser.add_argument('--max-users', type=int, default=None,
                       help='Maximum number of users to process (None for all users)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--use-filtered-users', action='store_true',
                       help='Only process users that exist in filtered_train_set.txt and rec_save_dict.csv')
    parser.add_argument('--filtered-train-file', type=str, default='filtered_train_set.txt',
                       help='Path to filtered training set file')
    parser.add_argument('--rec-save-dict-file', type=str, default='rec_save_dict.csv',
                       help='Path to recommendation save dict file')
    parser.add_argument('--base-delay', type=float, default=0.5,
                       help='Base delay between requests in seconds (default: 0.5)')
    parser.add_argument('--max-delay', type=float, default=5.0,
                       help='Maximum delay between requests in seconds (default: 5.0)')
    parser.add_argument('--requests-per-minute', type=int, default=60,
                       help='Target requests per minute (default: 60)')
    parser.add_argument('--memory-file', type=str, default=None,
                       help='Path to memory file for loading/saving user taste generation history')
    parser.add_argument('--load-memory', action='store_true',
                       help='Load memory from previous run (requires --memory-file)')
    parser.add_argument('--clear-memory', action='store_true',
                       help='Clear memory before starting (useful for fresh start)')
    parser.add_argument('--enable-debate-reflection', action='store_true', default=True,
                       help='Enable debate and reflection mechanisms (default: True)')
    parser.add_argument('--simple-mode', action='store_true',
                       help='Use simple mode: direct GPT generation only (disables debate and reflection)')
    
    args = parser.parse_args()
    
    # Handle mode selection
    if args.simple_mode:
        args.enable_debate_reflection = False
    
    print("=" * 60)
    print("A1-2: User Taste Generation Module")
    print("=" * 60)
    print(f"User profiles file: {args.user_profiles}")
    print(f"Output file: {args.output}")
    print(f"Sample size: {args.sample_size if args.sample_size else 'All users'}")
    print(f"Max users: {args.max_users if args.max_users else 'No limit'}")
    print(f"Random seed: {args.random_seed}")
    print(f"Use filtered users: {args.use_filtered_users}")
    if args.use_filtered_users:
        print(f"Filtered train file: {args.filtered_train_file}")
        print(f"Rec save dict file: {args.rec_save_dict_file}")
    print(f"Base delay: {args.base_delay}s, Max delay: {args.max_delay}s")
    print(f"Target rate: {args.requests_per_minute} requests/minute")
    print(f"Memory file: {args.memory_file if args.memory_file else 'Not specified'}")
    print(f"Mode: {'Simple Mode (Direct GPT)' if args.simple_mode else 'Full Mode (Debate + Reflection)'}")
    print(f"Debate & Reflection: {'Enabled' if args.enable_debate_reflection else 'Disabled'}")
    print()
    
    # Create taste generator with memory file and mode settings
    generator = UserTasteGenerator(args.openai_key, args.user_profiles, args.memory_file, args.enable_debate_reflection)
    
    if not generator.user_profiles:
        print("❌ Failed to load user profiles. Exiting.")
        return
    
    # Override rate limiting settings if specified
    if args.base_delay != 0.5:
        generator.base_delay = args.base_delay
        print(f"📊 Custom base delay: {generator.base_delay}s")
    if args.max_delay != 5.0:
        generator.max_delay = args.max_delay
        print(f"📊 Custom max delay: {generator.max_delay}s")
    if args.requests_per_minute != 60:
        generator.requests_per_minute = args.requests_per_minute
        print(f"📊 Custom rate limit: {generator.requests_per_minute} requests/minute")
    
    # Handle memory options
    if args.clear_memory:
        print("🧹 Clearing memory as requested...")
        generator.clear_memory()
    
    if args.load_memory and args.memory_file:
        print(f"🧠 Attempting to load memory from {args.memory_file}...")
        generator.load_memory()
    elif args.load_memory and not args.memory_file:
        print("⚠️ --load-memory specified but no --memory-file provided")
    
    # Determine users to process
    if args.use_filtered_users:
        # When using filtered users, get users from filtered_train_set and rec_save_dict
        filtered_users, rec_users = generator.get_filtered_and_rec_users(args.filtered_train_file, args.rec_save_dict_file)
        available_users = set(generator.user_profiles.keys())
        
        if args.sample_size:
            # Sample from filtered_train_set + all rec_save_dict users
            import random
            random.seed(args.random_seed)
            
            # Filter to only available users
            available_filtered_users = [uid for uid in filtered_users if uid in available_users]
            available_rec_users = [uid for uid in rec_users if uid in available_users]
            
            # Sample from filtered_train_set
            sampled_filtered = random.sample(available_filtered_users, min(args.sample_size, len(available_filtered_users)))
            
            # Combine sampled filtered users with all rec_save_dict users
            user_ids = list(set(sampled_filtered + available_rec_users))
            
            print(f"🎲 Sampling logic:")
            print(f"   - Sampled {len(sampled_filtered)} users from filtered_train_set")
            print(f"   - Added {len(available_rec_users)} users from rec_save_dict")
            print(f"   - Total users to process: {len(user_ids)}")
        else:
            # Use all users from filtered_train_set OR rec_save_dict (union)
            target_users = filtered_users.union(rec_users)
            user_ids = [uid for uid in target_users if uid in available_users]
            print(f"📋 Processing all {len(user_ids)} users from union of filtered_train_set and rec_save_dict")
    else:
        # When not using filtered users, work with all user_profiles
        if args.sample_size:
            import random
            random.seed(args.random_seed)
            all_user_ids = list(generator.user_profiles.keys())
            user_ids = random.sample(all_user_ids, min(args.sample_size, len(all_user_ids)))
            print(f"🎲 Randomly sampled {len(user_ids)} users from {len(all_user_ids)} user_profiles")
        else:
            user_ids = None
            print("📋 Processing all users from user_profiles")
    
    # Run workflow
    results = generator.run_full_workflow(
        user_ids=user_ids, 
        max_users=args.max_users,
        use_filtered_users=args.use_filtered_users,
        filtered_train_file=args.filtered_train_file,
        rec_save_dict_file=args.rec_save_dict_file
    )
    
    # Save results
    success = generator.save_results(results, args.output)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ User Taste Generation Completed Successfully!")
        print("=" * 60)
        print(f"Results saved to: {args.output}")
        print(f"Total users processed: {len(results['refined_tastes'])}")
    else:
        print("\n" + "=" * 60)
        print("⚠️ User Taste Generation Completed with Warnings!")
        print("=" * 60)


if __name__ == "__main__":
    main()

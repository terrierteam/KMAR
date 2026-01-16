import os
import json
import time
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from openai import OpenAI
from collections import Counter
import re

class ImprovedRecommendationMemory:
    """Enhanced memory module with advanced learning capabilities"""
    
    def __init__(self):
        """Initialize the enhanced memory module"""
        self.user_adjustments = {}
        self.adjustment_patterns = {}
        self.confidence_scores = {}
        self.processing_history = []
        self.debate_patterns = {}
        self.reflection_patterns = {}
        
        # Enhanced learning capabilities
        self.adjustment_patterns_learned = {}
        self.successful_adjustments = []
        self.user_preference_patterns = {}
        self.quality_metrics = {}
        self.learned_insights = []
        
        # New: Ranking strategy learning
        self.ranking_strategies = {}  # user_id -> successful_strategy
        self.item_preference_scores = {}  # user_id -> item_id -> score
        self.diversity_patterns = {}  # user_id -> diversity_preference
        self.relevance_thresholds = {}  # user_id -> relevance_threshold
        
        print("✅ Improved Recommendation Memory initialized with advanced learning")
    
    def store_adjustment(self, user_id: int, original_list: List[int], adjusted_list: List[int], 
                        reasoning: str, confidence: float, strategy_used: str, 
                        ranking_scores: Dict[int, float] = None, mode: str = "improved"):
        """Store adjustment with enhanced learning"""
        self.user_adjustments[user_id] = {
            'original_list': original_list,
            'adjusted_list': adjusted_list,
            'reasoning': reasoning,
            'confidence': confidence,
            'strategy_used': strategy_used,
            'ranking_scores': ranking_scores or {},
            'mode': mode,
            'timestamp': time.time()
        }
        
        # Enhanced learning
        self._learn_ranking_strategy(user_id, original_list, adjusted_list, strategy_used, confidence)
        self._learn_item_preferences(user_id, adjusted_list, ranking_scores)
        self._learn_diversity_patterns(user_id, original_list, adjusted_list)
        
        # Add to processing history
        self.processing_history.append({
            'user_id': user_id,
            'timestamp': time.time(),
            'confidence': confidence,
            'strategy': strategy_used,
            'adjustment_count': len(adjusted_list)
        })
    
    def _learn_ranking_strategy(self, user_id: int, original_list: List[int], 
                               adjusted_list: List[int], strategy: str, confidence: float):
        """Learn which ranking strategies work best for each user"""
        if confidence > 0.7:
            if user_id not in self.ranking_strategies:
                self.ranking_strategies[user_id] = {}
            
            if strategy not in self.ranking_strategies[user_id]:
                self.ranking_strategies[user_id][strategy] = {'count': 0, 'avg_confidence': 0.0}
            
            strategy_data = self.ranking_strategies[user_id][strategy]
            strategy_data['count'] += 1
            strategy_data['avg_confidence'] = (strategy_data['avg_confidence'] * (strategy_data['count'] - 1) + confidence) / strategy_data['count']
    
    def _learn_item_preferences(self, user_id: int, adjusted_list: List[int], 
                               ranking_scores: Dict[int, float]):
        """Learn user's item preference patterns"""
        if user_id not in self.item_preference_scores:
            self.item_preference_scores[user_id] = {}
        
        for i, item_id in enumerate(adjusted_list):
            # Higher position = higher preference
            position_score = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25, ...
            if item_id in ranking_scores:
                position_score *= ranking_scores[item_id]
            
            if item_id not in self.item_preference_scores[user_id]:
                self.item_preference_scores[user_id][item_id] = []
            self.item_preference_scores[user_id][item_id].append(position_score)
    
    def _learn_diversity_patterns(self, user_id: int, original_list: List[int], 
                                 adjusted_list: List[int]):
        """Learn user's diversity preferences"""
        # Calculate diversity metrics
        original_diversity = len(set(original_list)) / len(original_list) if original_list else 0
        adjusted_diversity = len(set(adjusted_list)) / len(adjusted_list) if adjusted_list else 0
        
        if user_id not in self.diversity_patterns:
            self.diversity_patterns[user_id] = []
        
        self.diversity_patterns[user_id].append({
            'original_diversity': original_diversity,
            'adjusted_diversity': adjusted_diversity,
            'diversity_change': adjusted_diversity - original_diversity,
            'timestamp': time.time()
        })
        
        # Keep only recent patterns
        if len(self.diversity_patterns[user_id]) > 20:
            self.diversity_patterns[user_id] = self.diversity_patterns[user_id][-20:]
    
    def get_best_strategy_for_user(self, user_id: int) -> str:
        """Get the best ranking strategy for a user based on history"""
        if user_id not in self.ranking_strategies:
            return "semantic_relevance"  # Default strategy
        
        strategies = self.ranking_strategies[user_id]
        best_strategy = max(strategies.keys(), key=lambda s: strategies[s]['avg_confidence'])
        return best_strategy
    
    def get_user_item_preference(self, user_id: int, item_id: int) -> float:
        """Get user's historical preference score for an item"""
        if user_id not in self.item_preference_scores or item_id not in self.item_preference_scores[user_id]:
            return 0.5  # Neutral preference
        
        scores = self.item_preference_scores[user_id][item_id]
        return sum(scores) / len(scores) if scores else 0.5
    
    def get_user_diversity_preference(self, user_id: int) -> float:
        """Get user's diversity preference (0-1, higher = more diverse)"""
        if user_id not in self.diversity_patterns or not self.diversity_patterns[user_id]:
            return 0.5  # Neutral diversity preference
        
        recent_patterns = self.diversity_patterns[user_id][-5:]  # Last 5 adjustments
        avg_diversity_change = sum(p['diversity_change'] for p in recent_patterns) / len(recent_patterns)
        
        # Convert to 0-1 scale
        return max(0.0, min(1.0, 0.5 + avg_diversity_change))
    
    def has_user_adjustment(self, user_id: int) -> bool:
        """Check if user has existing adjustment"""
        return user_id in self.user_adjustments
    
    def get_user_adjustment(self, user_id: int) -> Optional[Dict]:
        """Get user adjustment data"""
        return self.user_adjustments.get(user_id)
    
    def save_to_file(self, memory_file: str):
        """Save enhanced memory state to file"""
        try:
            memory_data = {
                'user_adjustments': self.user_adjustments,
                'adjustment_patterns': self.adjustment_patterns,
                'confidence_scores': self.confidence_scores,
                'processing_history': self.processing_history,
                'debate_patterns': self.debate_patterns,
                'reflection_patterns': self.reflection_patterns,
                'adjustment_patterns_learned': self.adjustment_patterns_learned,
                'successful_adjustments': self.successful_adjustments,
                'user_preference_patterns': self.user_preference_patterns,
                'quality_metrics': self.quality_metrics,
                'learned_insights': self.learned_insights,
                # Enhanced learning data
                'ranking_strategies': self.ranking_strategies,
                'item_preference_scores': self.item_preference_scores,
                'diversity_patterns': self.diversity_patterns,
                'relevance_thresholds': self.relevance_thresholds,
                'metadata': {
                    'total_users': len(self.user_adjustments),
                    'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'version': '2.0_improved'
                }
            }
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Enhanced memory saved to {memory_file} ({len(self.user_adjustments)} users)")
            
        except Exception as e:
            print(f"❌ Error saving enhanced memory: {e}")
    
    def load_from_file(self, memory_file: str) -> bool:
        """Load enhanced memory state from file"""
        if not os.path.exists(memory_file):
            print(f"⚠️ Memory file {memory_file} does not exist")
            return False
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Load basic memory data
            self.user_adjustments = memory_data.get('user_adjustments', {})
            self.adjustment_patterns = memory_data.get('adjustment_patterns', {})
            self.confidence_scores = memory_data.get('confidence_scores', {})
            self.processing_history = memory_data.get('processing_history', [])
            self.debate_patterns = memory_data.get('debate_patterns', {})
            self.reflection_patterns = memory_data.get('reflection_patterns', {})
            self.adjustment_patterns_learned = memory_data.get('adjustment_patterns_learned', {})
            self.successful_adjustments = memory_data.get('successful_adjustments', [])
            self.user_preference_patterns = memory_data.get('user_preference_patterns', {})
            self.quality_metrics = memory_data.get('quality_metrics', {})
            self.learned_insights = memory_data.get('learned_insights', [])
            
            # Load enhanced learning data
            self.ranking_strategies = memory_data.get('ranking_strategies', {})
            self.item_preference_scores = memory_data.get('item_preference_scores', {})
            self.diversity_patterns = memory_data.get('diversity_patterns', {})
            self.relevance_thresholds = memory_data.get('relevance_thresholds', {})
            
            # Convert string keys back to integers
            self.user_adjustments = {int(k): v for k, v in self.user_adjustments.items()}
            self.quality_metrics = {int(k): v for k, v in self.quality_metrics.items()}
            self.ranking_strategies = {int(k): v for k, v in self.ranking_strategies.items()}
            self.item_preference_scores = {int(k): v for k, v in self.item_preference_scores.items()}
            self.diversity_patterns = {int(k): v for k, v in self.diversity_patterns.items()}
            self.relevance_thresholds = {int(k): v for k, v in self.relevance_thresholds.items()}
            
            metadata = memory_data.get('metadata', {})
            print(f"✅ Enhanced memory loaded from {memory_file}")
            print(f"   - Users: {len(self.user_adjustments)}")
            print(f"   - Version: {metadata.get('version', '1.0')}")
            print(f"   - Last updated: {metadata.get('last_updated', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading enhanced memory: {e}")
            return False
    
    def clear_memory(self):
        """Clear all memory data"""
        self.user_adjustments.clear()
        self.adjustment_patterns.clear()
        self.confidence_scores.clear()
        self.processing_history.clear()
        self.debate_patterns.clear()
        self.reflection_patterns.clear()
        self.adjustment_patterns_learned.clear()
        self.successful_adjustments.clear()
        self.user_preference_patterns.clear()
        self.quality_metrics.clear()
        self.learned_insights.clear()
        # Clear enhanced data
        self.ranking_strategies.clear()
        self.item_preference_scores.clear()
        self.diversity_patterns.clear()
        self.relevance_thresholds.clear()
        print("🧹 Enhanced memory cleared")


class ImprovedRecommendationAdjuster:
    """Enhanced GPT-based recommendation list adjuster with advanced strategies"""
    
    def __init__(self, user_tastes_file: str = "user_tastes.json", 
                 item_summaries_file: str = "item_summaries.json",
                 rec_save_dict_file: str = "rec_save_dict.csv",
                 openai_api_key: str = None, memory_file: str = None, 
                 enable_debate_reflection: bool = True):
        """Initialize the Improved Recommendation Adjuster"""
        self.enable_debate_reflection = enable_debate_reflection
        self.user_tastes_file = user_tastes_file
        self.item_summaries_file = item_summaries_file
        self.rec_save_dict_file = rec_save_dict_file
        
        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️  Warning: No OpenAI API key provided. GPT analysis will be disabled.")
                self.client = None
        
        # Enhanced rate limiting configuration
        self.base_delay = 0.3  # Reduced base delay for better performance
        self.max_delay = 3.0   # Reduced max delay
        self.requests_per_minute = 80  # Increased requests per minute
        self.retry_attempts = 3
        self.consecutive_failures = 0
        
        # Initialize enhanced memory module
        self.memory = ImprovedRecommendationMemory()
        self.memory_file = memory_file
        
        # Load memory if file is provided
        if self.memory_file:
            self.load_memory()
        
        # Load data
        self.user_tastes = self.load_user_tastes()
        self.item_summaries = self.load_item_summaries()
        self.rec_save_dict = self.load_rec_save_dict()
        
        # Enhanced configuration
        self.diversity_weight = 0.3  # Weight for diversity in ranking
        self.relevance_weight = 0.7  # Weight for relevance in ranking
        self.min_confidence_threshold = 0.6  # Minimum confidence for adjustments
        
        if self.client:
            print("✅ Enhanced OpenAI client initialized for advanced GPT analysis")
            print(f"📊 Enhanced rate limiting: {self.requests_per_minute} requests/minute")
            print(f"⏱️  Optimized delays: base={self.base_delay}s, max={self.max_delay}s")
            print(f"🎯 Enhanced strategies: diversity_weight={self.diversity_weight}, relevance_weight={self.relevance_weight}")
            if self.enable_debate_reflection:
                print("🧠 Full mode: Advanced Debate and Reflection enabled")
            else:
                print("⚡ Simple mode: Enhanced Direct GPT generation")
        else:
            print("⚠️  OpenAI client not available - using enhanced fallback adjustment method")
    
    def load_user_tastes(self) -> Dict:
        """Load user tastes from JSON file"""
        try:
            with open(self.user_tastes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded user tastes: {len(data.get('initial_tastes', {}))} users")
            return data
        except Exception as e:
            print(f"❌ Error loading user tastes: {e}")
            return {}
    
    def load_item_summaries(self) -> Dict:
        """Load item summaries from JSON file"""
        try:
            with open(self.item_summaries_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert list format to dictionary format if needed
            if isinstance(data, list):
                item_summaries_dict = {}
                for item in data:
                    if 'item_id' in item and 'item_summary' in item:
                        item_summaries_dict[str(item['item_id'])] = item['item_summary']
                print(f"✅ Loaded item summaries: {len(item_summaries_dict)} items (converted from list)")
                return item_summaries_dict
            elif isinstance(data, dict):
                print(f"✅ Loaded item summaries: {len(data)} items")
                return data
            else:
                print(f"⚠️  Unexpected data format for item summaries")
                return {}
                
        except Exception as e:
            print(f"❌ Error loading item summaries: {e}")
            return {}
    
    def load_rec_save_dict(self) -> pd.DataFrame:
        """Load LightGCN recommendation CSV file"""
        try:
            df = pd.read_csv(self.rec_save_dict_file)
            print(f"✅ Loaded recommendation data: {len(df)} users")
            return df
        except Exception as e:
            print(f"❌ Error loading recommendation data: {e}")
            return pd.DataFrame()
    
    def _smart_delay(self):
        """Enhanced smart delay with adaptive backoff"""
        if self.consecutive_failures == 0:
            # Normal operation - use base delay with small random variation
            delay = self.base_delay + random.uniform(0, 0.1)
        else:
            # Adaptive backoff with jitter
            delay = min(self.base_delay * (1.5 ** self.consecutive_failures), self.max_delay)
            delay += random.uniform(0, delay * 0.05)  # Reduced jitter
        
        time.sleep(delay)
    
    def _handle_api_error(self, error: Exception) -> bool:
        """Enhanced API error handling"""
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
    
    def load_memory(self, memory_file: str = None):
        """Load memory from file"""
        if memory_file:
            self.memory_file = memory_file
        
        if self.memory_file:
            self.memory.load_from_file(self.memory_file)
    
    def save_memory(self, memory_file: str = None):
        """Save memory to file"""
        if memory_file:
            self.memory_file = memory_file
        
        if self.memory_file:
            self.memory.save_to_file(self.memory_file)
    
    def clear_memory(self):
        """Clear memory"""
        self.memory.clear_memory()
    
    def adjust_recommendations(self, user_ids: List[int] = None, max_users: int = None) -> Dict:
        """Enhanced recommendation adjustment with advanced strategies"""
        if user_ids is None:
            # Get users from rec_save_dict that have user tastes
            available_users = set(self.rec_save_dict.iloc[:, 0].astype(int))
            taste_users = set(int(uid) for uid in self.user_tastes.get('initial_tastes', {}).keys())
            user_ids = list(available_users.intersection(taste_users))
        
        # Apply max_users limit if specified
        if max_users and len(user_ids) > max_users:
            print(f"⚠️ Limiting processing to {max_users} users (from {len(user_ids)} available)")
            user_ids = user_ids[:max_users]
        
        print("=" * 60)
        print("A3-2: Enhanced Recommendation List Adjustment")
        print("=" * 60)
        print(f"🔧 Mode: {'Enhanced Full (Advanced Debate + Reflection)' if self.enable_debate_reflection else 'Enhanced Simple (Advanced Direct GPT)'}")
        print(f"📊 Processing {len(user_ids)} users...")
        print(f"🎯 Strategy: Multi-criteria optimization with learning")
        if self.memory_file:
            print(f"🧠 Enhanced memory file: {self.memory_file}")
        print()
        
        # Check memory for cached results
        cached_users = []
        new_users = []
        
        for user_id in user_ids:
            if self.memory.has_user_adjustment(user_id):
                cached_users.append(user_id)
            else:
                new_users.append(user_id)
        
        print(f"📊 Enhanced memory status:")
        print(f"   - Cached users: {len(cached_users)}")
        print(f"   - New users to process: {len(new_users)}")
        
        # Load cached results
        adjusted_recommendations = {}
        
        for user_id in cached_users:
            cached_data = self.memory.get_user_adjustment(user_id)
            if cached_data:
                adjusted_recommendations[user_id] = cached_data['adjusted_list']
        
        print(f"✅ Loaded {len(cached_users)} cached results from enhanced memory")
        
        # Process new users with enhanced strategies
        if new_users:
            print(f"\n🔄 Processing {len(new_users)} new users with enhanced strategies...")
            
            # Reset failure counter before processing
            self.consecutive_failures = 0
            
            # Process users with enhanced methods
            new_adjustments = self._process_users_enhanced(new_users)
            adjusted_recommendations.update(new_adjustments)
        
        # Save enhanced memory
        if self.memory_file:
            print(f"\n💾 Saving enhanced memory to {self.memory_file}...")
            self.save_memory()
        
        # Compile enhanced results
        results = {
            'workflow_info': {
                'total_users': len(user_ids),
                'cached_users': len(cached_users),
                'new_users': len(new_users),
                'workflow_version': '2.0_improved',
                'enable_debate_reflection': self.enable_debate_reflection,
                'mode': 'Enhanced Full' if self.enable_debate_reflection else 'Enhanced Simple',
                'memory_file': self.memory_file,
                'enhancements': [
                    'Advanced prompt engineering',
                    'Multi-criteria ranking',
                    'Learning-based strategies',
                    'Diversity-relevance balance'
                ]
            },
            'adjusted_recommendations': adjusted_recommendations
        }
        
        print("\n" + "=" * 60)
        print("✅ Enhanced Recommendation Adjustment Completed!")
        print("=" * 60)
        
        return results
    
    def _process_users_enhanced(self, user_ids: List[int]) -> Dict[int, List[int]]:
        """Process users with enhanced strategies"""
        adjusted_recommendations = {}
        start_time = time.time()
        
        for i, user_id in enumerate(user_ids):
            # Reset failure counter every 10 users
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at user {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Progress update
            if i % 50 == 0 or i == len(user_ids) - 1:
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_user = elapsed_time / i
                    remaining_users = len(user_ids) - i
                    estimated_remaining = remaining_users * avg_time_per_user / 60
                    print(f"Processing user {i+1}/{len(user_ids)}: User ID {user_id}")
                    print(f"   ⏱️  Elapsed: {elapsed_time/60:.1f}m, Remaining: {estimated_remaining:.1f}m")
                else:
                    print(f"Processing user {i+1}/{len(user_ids)}: User ID {user_id}")
            
            try:
                # Get user data
                user_taste = self.user_tastes.get('initial_tastes', {}).get(str(user_id), "")
                original_list = self._get_user_recommendations(user_id)
                
                if not user_taste or not original_list:
                    print(f"  ⚠️  Skipping user {user_id}: missing taste or recommendations")
                    continue
                
                # Use enhanced adjustment with learning-based strategy selection
                adjusted_list, strategy_used, ranking_scores = self._enhanced_adjust_recommendations(
                    user_id, user_taste, original_list
                )
                
                adjusted_recommendations[user_id] = adjusted_list
                
            except Exception as e:
                print(f"❌ Error processing user {user_id}: {e}")
                # Use original list as fallback
                original_list = self._get_user_recommendations(user_id)
                adjusted_recommendations[user_id] = original_list if original_list else []
        
        return adjusted_recommendations
    
    def _get_user_recommendations(self, user_id: int) -> List[int]:
        """Get original recommendation list for user"""
        try:
            user_row = self.rec_save_dict[self.rec_save_dict.iloc[:, 0] == user_id]
            if not user_row.empty:
                # Get the 10 recommendation items (columns 1-10)
                recommendations = user_row.iloc[0, 1:11].tolist()
                return [int(x) for x in recommendations if pd.notna(x)]
        except Exception as e:
            print(f"  ⚠️  Error getting recommendations for user {user_id}: {e}")
        return []
    
    def _enhanced_adjust_recommendations(self, user_id: int, user_taste: str, 
                                       original_list: List[int]) -> Tuple[List[int], str, Dict[int, float]]:
        """Enhanced recommendation adjustment with multiple strategies"""
        try:
            # Get best strategy for this user based on learning
            best_strategy = self.memory.get_best_strategy_for_user(user_id)
            
            # Calculate item relevance scores
            relevance_scores = self._calculate_item_relevance_scores(user_id, user_taste, original_list)
            
            # Calculate diversity scores
            diversity_scores = self._calculate_diversity_scores(original_list)
            
            # Get user's diversity preference
            diversity_preference = self.memory.get_user_diversity_preference(user_id)
            
            # Combine scores with adaptive weights
            combined_scores = self._combine_scores_with_learning(
                relevance_scores, diversity_scores, diversity_preference, user_id
            )
            
            # Generate enhanced prompt
            prompt = self._build_enhanced_adjustment_prompt(
                user_id, user_taste, original_list, combined_scores, best_strategy
            )
            
            # Call GPT with enhanced prompt
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self._get_enhanced_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # Increased token limit
                temperature=0.2  # Lower temperature for more consistent results
            )
            
            response_text = response.choices[0].message.content.strip()
            adjusted_list = self._parse_enhanced_adjustment_response(response_text, original_list)
            
            # Calculate confidence based on response quality
            confidence = self._calculate_adjustment_confidence(response_text, adjusted_list, original_list)
            
            # Store in enhanced memory
            self.memory.store_adjustment(
                user_id, original_list, adjusted_list, 
                response_text, confidence, best_strategy, 
                combined_scores, "enhanced"
            )
            
            self._smart_delay()
            self.consecutive_failures = 0
            
            return adjusted_list, best_strategy, combined_scores
            
        except Exception as e:
            print(f"  ❌ Enhanced adjustment failed for user {user_id}: {e}")
            self.consecutive_failures += 1
            return original_list, "fallback", {}  # Return original list as fallback
    
    def _calculate_item_relevance_scores(self, user_id: int, user_taste: str, 
                                       original_list: List[int]) -> Dict[int, float]:
        """Calculate relevance scores for items based on user taste"""
        relevance_scores = {}
        
        # Extract key terms from user taste
        taste_terms = self._extract_taste_keywords(user_taste)
        
        for item_id in original_list:
            score = 0.0
            
            # Get item summary
            item_summary = self.item_summaries.get(str(item_id), "")
            if not item_summary:
                score = 0.5  # Neutral score for items without summaries
            else:
                # Calculate semantic similarity between taste and item summary
                score = self._calculate_semantic_similarity(taste_terms, item_summary)
            
            # Apply historical preference if available
            historical_preference = self.memory.get_user_item_preference(user_id, item_id)
            if historical_preference > 0.5:
                score = (score + historical_preference) / 2  # Blend with historical data
            
            relevance_scores[item_id] = score
        
        return relevance_scores
    
    def _extract_taste_keywords(self, user_taste: str) -> List[str]:
        """Extract key terms from user taste profile"""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w+\b', user_taste.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Return top keywords (limit to avoid noise)
        return keywords[:20]
    
    def _calculate_semantic_similarity(self, taste_terms: List[str], item_summary: str) -> float:
        """Calculate semantic similarity between taste terms and item summary"""
        if not taste_terms or not item_summary:
            return 0.5
        
        # Simple keyword matching - can be enhanced with embeddings
        item_words = re.findall(r'\b\w+\b', item_summary.lower())
        
        matches = 0
        for term in taste_terms:
            if any(term in word or word in term for word in item_words):
                matches += 1
        
        # Normalize by number of taste terms
        similarity = matches / len(taste_terms) if taste_terms else 0.0
        
        # Apply sigmoid function to make scores more distinct
        return 1 / (1 + np.exp(-5 * (similarity - 0.5)))
    
    def _calculate_diversity_scores(self, original_list: List[int]) -> Dict[int, float]:
        """Calculate diversity scores for items"""
        diversity_scores = {}
        
        # Simple diversity based on position (higher position = more diverse)
        for i, item_id in enumerate(original_list):
            # Items at different positions get different diversity scores
            diversity_scores[item_id] = 1.0 - (i / len(original_list)) * 0.5
        
        return diversity_scores
    
    def _combine_scores_with_learning(self, relevance_scores: Dict[int, float], 
                                    diversity_scores: Dict[int, float], 
                                    diversity_preference: float, user_id: int) -> Dict[int, float]:
        """Combine relevance and diversity scores with learning-based weights"""
        combined_scores = {}
        
        # Adaptive weights based on user preference
        relevance_weight = self.relevance_weight
        diversity_weight = self.diversity_weight * diversity_preference
        
        # Normalize weights
        total_weight = relevance_weight + diversity_weight
        relevance_weight /= total_weight
        diversity_weight /= total_weight
        
        for item_id in relevance_scores:
            if item_id in diversity_scores:
                combined_score = (relevance_weight * relevance_scores[item_id] + 
                                diversity_weight * diversity_scores[item_id])
                combined_scores[item_id] = combined_score
            else:
                combined_scores[item_id] = relevance_scores[item_id]
        
        return combined_scores
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt for GPT"""
        return """You are an expert recommendation system that adjusts recommendation lists based on user preferences and item characteristics. 

Your task is to:
1. Analyze user taste profiles and item summaries
2. Consider both relevance and diversity in recommendations
3. Reorder and select items from the original list to create a better recommendation
4. Maintain the same number of items (10) as the original list
5. Only use items from the provided original list

You will receive:
- User taste profile with detailed preferences
- Original recommendation list with 10 items
- Item summaries with detailed information
- Calculated relevance and diversity scores for each item

Provide a well-reasoned adjustment that balances user preferences with recommendation diversity."""
    
    def _build_enhanced_adjustment_prompt(self, user_id: int, user_taste: str, 
                                        original_list: List[int], 
                                        combined_scores: Dict[int, float], 
                                        strategy: str) -> str:
        """Build enhanced prompt for recommendation adjustment"""
        # Get item summaries for the original list
        item_summaries_text = ""
        for i, item_id in enumerate(original_list):
            if str(item_id) in self.item_summaries:
                summary = self.item_summaries[str(item_id)]
                score = combined_scores.get(item_id, 0.5)
                item_summaries_text += f"Item {item_id} (Score: {score:.3f}): {summary}\n"
        
        # Extract key taste preferences
        taste_keywords = self._extract_taste_keywords(user_taste)
        taste_highlights = ", ".join(taste_keywords[:10])  # Top 10 keywords
        
        prompt = f"""Based on the user's detailed taste profile and item analysis, adjust the recommendation list to better match the user's preferences.

User ID: {user_id}
User Taste Profile: {user_taste}
Key Preferences: {taste_highlights}

Original Recommendation List: {original_list}

Item Analysis with Scores:
{item_summaries_text}

Task: Create an adjusted recommendation list that:
- Prioritizes items with higher relevance scores that match the user's taste profile
- Maintains appropriate diversity in recommendations
- Uses ONLY items from the original list: {original_list}
- Keeps exactly 10 items
- Considers the calculated scores for each item
- Balances user preferences with recommendation variety

Strategy: {strategy}

Output format: Return only the adjusted item IDs as a comma-separated list, e.g., "1234,5678,9012,3456,7890,1111,2222,3333,4444,5555"

Reasoning: Briefly explain your ranking decisions based on the scores and user preferences."""
        
        return prompt
    
    def _parse_enhanced_adjustment_response(self, response_text: str, original_list: List[int]) -> List[int]:
        """Parse enhanced GPT response to extract adjusted recommendation list"""
        try:
            # Extract numbers from response
            numbers = re.findall(r'\d+', response_text)
            potential_items = [int(num) for num in numbers]
            
            # Filter to only include items from the original list
            adjusted_list = [item for item in potential_items if item in original_list]
            
            # Remove duplicates while preserving order
            seen = set()
            adjusted_list = [item for item in adjusted_list if not (item in seen or seen.add(item))]
            
            # Ensure we have exactly 10 items
            if len(adjusted_list) < 10:
                # Fill with remaining original items if needed
                remaining = [item for item in original_list if item not in adjusted_list]
                adjusted_list.extend(remaining[:10-len(adjusted_list)])
            elif len(adjusted_list) > 10:
                # Truncate if too many
                adjusted_list = adjusted_list[:10]
            
            # Final validation: ensure all items are from original list
            adjusted_list = [item for item in adjusted_list if item in original_list]
            
            return adjusted_list
            
        except Exception as e:
            print(f"  ⚠️  Error parsing enhanced adjustment response: {e}")
            return original_list
    
    def _calculate_adjustment_confidence(self, response_text: str, adjusted_list: List[int], 
                                       original_list: List[int]) -> float:
        """Calculate confidence score for the adjustment"""
        confidence = 0.5  # Base confidence
        
        # Check if response contains reasoning
        if any(word in response_text.lower() for word in ['because', 'reason', 'prefer', 'match', 'score']):
            confidence += 0.2
        
        # Check if all items are from original list
        if all(item in original_list for item in adjusted_list):
            confidence += 0.2
        
        # Check if list has good length
        if len(adjusted_list) == 10:
            confidence += 0.1
        
        # Check if there's some reordering (not identical to original)
        if adjusted_list != original_list:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def save_adjusted_recommendations(self, adjusted_recommendations: Dict[int, List[int]], 
                                    output_file: str = "final_list_improved.csv"):
        """Save adjusted recommendations to CSV file"""
        print(f"Saving enhanced adjusted recommendations to {output_file}...")
        
        try:
            # Create DataFrame
            data = []
            for user_id, recommendations in adjusted_recommendations.items():
                row = [user_id] + recommendations
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, header=False)
            
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"✅ Enhanced adjusted recommendations saved: {output_file}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Total users: {len(adjusted_recommendations)}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving enhanced adjusted recommendations: {e}")
            return False


def main():
    """Main function for enhanced recommendation adjuster"""
    parser = argparse.ArgumentParser(description='Enhanced recommendation list adjuster with advanced strategies')
    parser.add_argument('--user-tastes', type=str, default='user_tastes.json',
                       help='User tastes JSON file path')
    parser.add_argument('--item-summaries', type=str, default='item_summaries.json',
                       help='Item summaries JSON file path')
    parser.add_argument('--rec-save-dict', type=str, default='rec_save_dict.csv',
                       help='LightGCN recommendation CSV file path')
    parser.add_argument('--output', type=str, default='final_list_improved.csv',
                       help='Output CSV file path for enhanced adjusted recommendations')
    parser.add_argument('--max-users', type=int, default=None,
                       help='Maximum number of users to process (None for all users)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                       help='OpenAI API key for GPT analysis')
    parser.add_argument('--base-delay', type=float, default=0.3,
                       help='Base delay between requests in seconds (default: 0.3)')
    parser.add_argument('--max-delay', type=float, default=3.0,
                       help='Maximum delay between requests in seconds (default: 3.0)')
    parser.add_argument('--requests-per-minute', type=int, default=80,
                       help='Target requests per minute (default: 80)')
    parser.add_argument('--memory-file', type=str, default=None,
                       help='Path to enhanced memory file for loading/saving adjustment history')
    parser.add_argument('--load-memory', action='store_true',
                       help='Load memory from previous run (requires --memory-file)')
    parser.add_argument('--clear-memory', action='store_true',
                       help='Clear memory before starting (useful for fresh start)')
    parser.add_argument('--enable-debate-reflection', action='store_true', default=True,
                       help='Enable enhanced debate and reflection mode (default: True)')
    parser.add_argument('--simple-mode', action='store_true',
                       help='Use enhanced simple mode (direct GPT, no debate/reflection)')
    parser.add_argument('--diversity-weight', type=float, default=0.3,
                       help='Weight for diversity in ranking (default: 0.3)')
    parser.add_argument('--relevance-weight', type=float, default=0.7,
                       help='Weight for relevance in ranking (default: 0.7)')
    
    args = parser.parse_args()
    
    # Determine mode
    enable_debate_reflection = args.enable_debate_reflection and not args.simple_mode
    
    # Create enhanced recommendation adjuster
    adjuster = ImprovedRecommendationAdjuster(
        user_tastes_file=args.user_tastes,
        item_summaries_file=args.item_summaries,
        rec_save_dict_file=args.rec_save_dict,
        openai_api_key=args.openai_api_key,
        memory_file=args.memory_file,
        enable_debate_reflection=enable_debate_reflection
    )
    
    # Override configuration if specified
    if args.base_delay != 0.3:
        adjuster.base_delay = args.base_delay
        print(f"📊 Custom base delay: {adjuster.base_delay}s")
    if args.max_delay != 3.0:
        adjuster.max_delay = args.max_delay
        print(f"📊 Custom max delay: {adjuster.max_delay}s")
    if args.requests_per_minute != 80:
        adjuster.requests_per_minute = args.requests_per_minute
        print(f"📊 Custom rate limit: {adjuster.requests_per_minute} requests/minute")
    if args.diversity_weight != 0.3:
        adjuster.diversity_weight = args.diversity_weight
        print(f"🎯 Custom diversity weight: {adjuster.diversity_weight}")
    if args.relevance_weight != 0.7:
        adjuster.relevance_weight = args.relevance_weight
        print(f"🎯 Custom relevance weight: {adjuster.relevance_weight}")
    
    # Handle memory options
    if args.clear_memory:
        print("🧹 Clearing enhanced memory as requested...")
        adjuster.clear_memory()
    
    if args.load_memory and args.memory_file:
        print(f"🧠 Attempting to load enhanced memory from {args.memory_file}...")
        adjuster.load_memory()
    elif args.load_memory and not args.memory_file:
        print("⚠️ --load-memory specified but no --memory-file provided")
    
    # Run enhanced adjustment workflow
    results = adjuster.adjust_recommendations(max_users=args.max_users)
    
    # Save results
    success = adjuster.save_adjusted_recommendations(
        results['adjusted_recommendations'], args.output
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Enhanced Recommendation Adjustment Completed Successfully!")
        print("=" * 60)
        print(f"Enhanced adjusted recommendations saved to: {args.output}")
        print(f"Total users processed: {len(results['adjusted_recommendations'])}")
        print(f"Enhancements applied:")
        for enhancement in results['workflow_info']['enhancements']:
            print(f"  - {enhancement}")
    else:
        print("\n" + "=" * 60)
        print("⚠️ Enhanced Recommendation Adjustment Completed with Warnings!")
        print("=" * 60)


if __name__ == "__main__":
    main()

import os
import json
import argparse
import time
import random
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

class TripletMemory:
    """Memory module for storing triplet selection patterns and history with learning capabilities"""
    
    def __init__(self):
        """Initialize the memory module"""
        self.item_selections = {}  # item_id -> selected_triplet
        self.selection_patterns = {}  # item_id -> selection_reasoning
        self.confidence_scores = {}  # item_id -> confidence_score
        self.processing_history = []  # List of processing records
        self.debate_patterns = {}  # item_id -> debate_summary
        
        # Learning and pattern analysis
        self.triplet_patterns = {}  # pattern_type -> frequency
        self.successful_selections = []  # List of successful selection patterns
        self.genre_preferences = {}  # item_id -> genre_preferences
        self.relation_patterns = {}  # relation_pattern -> selection_characteristics
        self.learned_insights = []  # List of learned insights
        self.quality_metrics = {}  # item_id -> quality_metrics
        
        print("✅ Triplet Memory initialized with learning capabilities")
    
    def store_selection(self, item_id: int, selected_triplet: Dict, reasoning: str, 
                       confidence: float, debate_summary: str, mode: str = "full"):
        """Store a triplet selection result with learning
        
        Args:
            item_id: Item ID
            selected_triplet: Selected triplet information
            reasoning: Reasoning for the selection
            confidence: Confidence score (0-1)
            debate_summary: Summary of the debate process
            mode: Processing mode ("full" or "simple")
        """
        self.item_selections[item_id] = selected_triplet
        self.selection_patterns[item_id] = reasoning
        self.confidence_scores[item_id] = confidence
        self.debate_patterns[item_id] = debate_summary
        
        # Add to processing history
        self.processing_history.append({
            'item_id': item_id,
            'timestamp': time.time(),
            'confidence': confidence,
            'triplet_count': len(selected_triplet.get('triplet_texts', [])),
            'method': mode
        })
        
        # Learn from this selection
        self._learn_from_selection(item_id, selected_triplet, reasoning, confidence, mode)
    
    def get_selection_pattern(self, item_id: int) -> Optional[Dict]:
        """Get selection pattern for an item
        
        Args:
            item_id: Item ID
            
        Returns:
            Selection pattern if exists, None otherwise
        """
        if item_id in self.item_selections:
            return {
                'selected_triplet': self.item_selections[item_id],
                'reasoning': self.selection_patterns[item_id],
                'confidence': self.confidence_scores[item_id],
                'debate_summary': self.debate_patterns[item_id]
            }
        return None
    
    def get_statistics(self) -> Dict:
        """Get memory statistics
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.processing_history:
            return {'total_items': 0, 'avg_confidence': 0.0}
        
        total_items = len(self.processing_history)
        avg_confidence = sum(record['confidence'] for record in self.processing_history) / total_items
        
        return {
            'total_items': total_items,
            'avg_confidence': avg_confidence,
            'recent_items': self.processing_history[-5:] if len(self.processing_history) >= 5 else self.processing_history
        }
    
    def save_to_file(self, memory_file: str):
        """Save complete memory state to file
        
        Args:
            memory_file: Path to save memory file
        """
        try:
            memory_data = {
                'item_selections': self.item_selections,
                'selection_patterns': self.selection_patterns,
                'confidence_scores': self.confidence_scores,
                'debate_patterns': self.debate_patterns,
                'processing_history': self.processing_history,
                # Learning data
                'triplet_patterns': self.triplet_patterns,
                'successful_selections': self.successful_selections,
                'genre_preferences': self.genre_preferences,
                'relation_patterns': self.relation_patterns,
                'learned_insights': self.learned_insights,
                'quality_metrics': self.quality_metrics,
                'metadata': {
                    'total_items': len(self.item_selections),
                    'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'version': '2.0'
                }
            }
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Memory saved to {memory_file} ({len(self.item_selections)} items)")
            
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
    
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
            self.item_selections = memory_data.get('item_selections', {})
            self.selection_patterns = memory_data.get('selection_patterns', {})
            self.confidence_scores = memory_data.get('confidence_scores', {})
            self.debate_patterns = memory_data.get('debate_patterns', {})
            self.processing_history = memory_data.get('processing_history', [])
            
            # Load learning data (with backward compatibility)
            self.triplet_patterns = memory_data.get('triplet_patterns', {})
            self.successful_selections = memory_data.get('successful_selections', [])
            self.genre_preferences = memory_data.get('genre_preferences', {})
            self.relation_patterns = memory_data.get('relation_patterns', {})
            self.learned_insights = memory_data.get('learned_insights', [])
            self.quality_metrics = memory_data.get('quality_metrics', {})
            
            # Convert string keys back to integers for item_selections
            self.item_selections = {int(k): v for k, v in self.item_selections.items()}
            self.selection_patterns = {int(k): v for k, v in self.selection_patterns.items()}
            self.confidence_scores = {int(k): v for k, v in self.confidence_scores.items()}
            self.debate_patterns = {int(k): v for k, v in self.debate_patterns.items()}
            self.quality_metrics = {int(k): v for k, v in self.quality_metrics.items()}
            
            metadata = memory_data.get('metadata', {})
            print(f"✅ Memory loaded from {memory_file}")
            print(f"   - Items: {len(self.item_selections)}")
            print(f"   - Last updated: {metadata.get('last_updated', 'Unknown')}")
            print(f"   - Version: {metadata.get('version', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading memory: {e}")
            return False
    
    def clear_memory(self):
        """Clear all memory data"""
        self.item_selections.clear()
        self.selection_patterns.clear()
        self.confidence_scores.clear()
        self.debate_patterns.clear()
        self.processing_history.clear()
        # Clear learning data
        self.triplet_patterns.clear()
        self.successful_selections.clear()
        self.genre_preferences.clear()
        self.relation_patterns.clear()
        self.learned_insights.clear()
        self.quality_metrics.clear()
        print("🧹 Memory cleared")
    
    def _learn_from_selection(self, item_id: int, selected_triplet: Dict, reasoning: str, 
                             confidence: float, mode: str):
        """Learn from a triplet selection
        
        Args:
            item_id: Item ID
            selected_triplet: Selected triplet information
            reasoning: Reasoning for the selection
            confidence: Confidence score
            mode: Processing mode
        """
        # Analyze triplet patterns
        self._analyze_triplet_patterns(item_id, selected_triplet, confidence)
        
        # Learn successful patterns
        if confidence > 0.7:
            self._learn_successful_patterns(item_id, selected_triplet, reasoning, mode)
        
        # Extract insights
        self._extract_insights(item_id, selected_triplet, reasoning, confidence)
        
        # Update quality metrics
        self._update_quality_metrics(item_id, confidence)
    
    def _analyze_triplet_patterns(self, item_id: int, selected_triplet: Dict, confidence: float):
        """Analyze patterns in triplet selections"""
        if 'relation' in selected_triplet:
            relation = selected_triplet['relation']
            if relation not in self.relation_patterns:
                self.relation_patterns[relation] = {'count': 0, 'avg_confidence': 0.0}
            
            pattern = self.relation_patterns[relation]
            pattern['count'] += 1
            pattern['avg_confidence'] = (pattern['avg_confidence'] * (pattern['count'] - 1) + confidence) / pattern['count']
    
    def _learn_successful_patterns(self, item_id: int, selected_triplet: Dict, reasoning: str, mode: str):
        """Learn from successful selections"""
        successful_pattern = {
            'item_id': item_id,
            'triplet': selected_triplet,
            'reasoning': reasoning,
            'mode': mode,
            'timestamp': time.time()
        }
        self.successful_selections.append(successful_pattern)
        
        # Keep only recent successful patterns
        if len(self.successful_selections) > 100:
            self.successful_selections = self.successful_selections[-100:]
    
    def _extract_insights(self, item_id: int, selected_triplet: Dict, reasoning: str, confidence: float):
        """Extract insights from selection process"""
        if confidence > 0.8:
            insight = {
                'type': 'high_confidence_selection',
                'item_id': item_id,
                'relation': selected_triplet.get('relation', 'unknown'),
                'reasoning': reasoning,
                'timestamp': time.time()
            }
            self.learned_insights.append(insight)
            
            # Keep only recent insights
            if len(self.learned_insights) > 50:
                self.learned_insights = self.learned_insights[-50:]
    
    def _update_quality_metrics(self, item_id: int, confidence: float):
        """Update quality metrics for item"""
        if item_id not in self.quality_metrics:
            self.quality_metrics[item_id] = {
                'total_selections': 0,
                'avg_confidence': 0.0,
                'high_confidence_count': 0,
                'last_updated': time.time()
            }
        
        metrics = self.quality_metrics[item_id]
        metrics['total_selections'] += 1
        metrics['avg_confidence'] = (metrics['avg_confidence'] * (metrics['total_selections'] - 1) + confidence) / metrics['total_selections']
        
        if confidence > 0.7:
            metrics['high_confidence_count'] += 1
        
        metrics['last_updated'] = time.time()
    
    def get_memory_info(self) -> Dict:
        """Get detailed memory information
        
        Returns:
            Dictionary with detailed memory information
        """
        return {
            'total_items': len(self.item_selections),
            'avg_confidence': sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            'processing_history_count': len(self.processing_history),
            'memory_size_mb': sum(len(str(v)) for v in self.item_selections.values()) / (1024 * 1024),
            'recent_items': list(self.item_selections.keys())[-5:] if self.item_selections else [],
            # Learning data info
            'successful_selections_count': len(self.successful_selections),
            'learned_insights_count': len(self.learned_insights),
            'relation_patterns_count': len(self.relation_patterns),
            'quality_metrics_count': len(self.quality_metrics)
        }

class TripletSelector:
    """GPT-based triplet selector with debate and reflection mechanisms"""
    
    def __init__(self, openai_api_key: str = None, memory_file: str = None, 
                 enable_debate_reflection: bool = True):
        """Initialize the Triplet Selector
        
        Args:
            openai_api_key: OpenAI API key for GPT analysis
            memory_file: Path to memory file for loading/saving
            enable_debate_reflection: Enable debate and reflection mode (True) or simple mode (False)
        """
        self.enable_debate_reflection = enable_debate_reflection
        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment variable
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
        self.memory = TripletMemory()
        self.memory_file = memory_file
        
        # Load memory if file is provided
        if self.memory_file:
            self.load_memory()
        
        if self.client:
            print("✅ OpenAI client initialized for GPT analysis")
            print(f"📊 Rate limiting: {self.requests_per_minute} requests/minute")
            print(f"⏱️  Base delay: {self.base_delay}s, Max delay: {self.max_delay}s")
            if self.enable_debate_reflection:
                print("🧠 Full mode: Debate and Reflection enabled")
            else:
                print("⚡ Simple mode: Direct GPT generation only")
        else:
            print("⚠️  OpenAI client not available - using fallback selection method")
    
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
                print(f"🧠 Memory loaded successfully from {self.memory_file}")
            else:
                print(f"⚠️ Failed to load memory from {self.memory_file}")
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
    
    def _call_gpt(self, messages: List[Dict], max_tokens: int = 500) -> str:
        """Call GPT with error handling and retry logic
        
        Args:
            messages: List of messages for GPT
            max_tokens: Maximum tokens for response
            
        Returns:
            GPT response text
        """
        if not self.client:
            return "GPT not available"
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                
                result = response.choices[0].message.content.strip()
                
                # Success - reset failure counter and apply smart delay
                if self.consecutive_failures > 0:
                    print(f"✅ Recovered from {self.consecutive_failures} consecutive failures")
                    self.consecutive_failures = 0
                
                self._smart_delay()
                return result
                
            except Exception as e:
                print(f"GPT call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_attempts - 1 and self._handle_api_error(e):
                    print(f"Retrying in {self.base_delay * (2 ** self.consecutive_failures):.1f}s...")
                    self._smart_delay()
                    continue
                else:
                    print(f"All retry attempts failed")
                    break
        
        return "GPT call failed after all retry attempts"
    
    def _generate_three_opinions(self, item_data: Dict) -> List[str]:
        """Generate three different opinions about triplet relevance
        
        Args:
            item_data: Item data with triplets and summary
            
        Returns:
            List of three opinion strings
        """
        if not self.client:
            return ["GPT not available", "GPT not available", "GPT not available"]
        
        # Build prompt for three different perspectives
        prompt = f"""You are analyzing movie triplets for relevance. For the movie "{item_data['item_text']}" with summary "{item_data['item_summary']}", provide THREE different perspectives on which triplet is most relevant.

Available triplets:
"""
        
        for i, triplet in enumerate(item_data['triplet_texts'], 1):
            prompt += f"{i}. {triplet['text_format']}\n"
        
        prompt += """
Please provide THREE different expert opinions:

OPINION 1 (Content Relevance Expert): Focus on which triplet best represents the movie's core content, plot, or themes.

OPINION 2 (Character/Performance Expert): Focus on which triplet highlights the most important characters, actors, or performances.

OPINION 3 (Genre/Context Expert): Focus on which triplet best captures the movie's genre, cultural context, or unique characteristics.

Format your response as:
OPINION 1: [expert analysis and recommendation]
OPINION 2: [expert analysis and recommendation]  
OPINION 3: [expert analysis and recommendation]
"""
        
        messages = [
            {"role": "system", "content": "You are a movie analysis expert providing multiple perspectives on triplet relevance."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_gpt(messages, max_tokens=800)
        
        # Parse the three opinions
        opinions = []
        lines = response.split('\n')
        current_opinion = ""
        
        for line in lines:
            if line.startswith('OPINION 1:') or line.startswith('OPINION 2:') or line.startswith('OPINION 3:'):
                if current_opinion:
                    opinions.append(current_opinion.strip())
                current_opinion = line
            elif current_opinion:
                current_opinion += " " + line
        
        if current_opinion:
            opinions.append(current_opinion.strip())
        
        # Ensure we have exactly 3 opinions
        while len(opinions) < 3:
            opinions.append("No opinion available")
        
        return opinions[:3]
    
    def _debate_opinions(self, item_data: Dict, opinions: List[str]) -> str:
        """Debate the three opinions and reach a consensus
        
        Args:
            item_data: Item data with triplets and summary
            opinions: List of three opinions
            
        Returns:
            Debate summary and consensus
        """
        if not self.client:
            return "GPT not available for debate"
        
        prompt = f"""You are moderating a debate between three movie experts about the most relevant triplet for "{item_data['item_text']}".

Movie Summary: {item_data['item_summary']}

Available triplets:
"""
        
        for i, triplet in enumerate(item_data['triplet_texts'], 1):
            prompt += f"{i}. {triplet['text_format']}\n"
        
        prompt += f"""
Expert Opinions:
{opinions[0]}
{opinions[1]}
{opinions[2]}

As the debate moderator, analyze these three perspectives, identify areas of agreement and disagreement, and reach a consensus on which triplet is most relevant. Consider:
1. Which arguments are most compelling?
2. What are the strengths and weaknesses of each perspective?
3. Which triplet best serves the goal of movie recommendation?

Provide your analysis and final recommendation in this format:
DEBATE ANALYSIS: [summary of the debate]
CONSENSUS: [final recommendation with reasoning]
"""
        
        messages = [
            {"role": "system", "content": "You are an expert debate moderator specializing in movie analysis and recommendation systems."},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_gpt(messages, max_tokens=600)
    
    def _final_reflection(self, item_data: Dict, debate_result: str) -> Tuple[Dict, str, float]:
        """Final reflection and confidence assessment
        
        Args:
            item_data: Item data with triplets and summary
            debate_result: Result from the debate process
            
        Returns:
            Tuple of (selected_triplet, reasoning, confidence_score)
        """
        if not self.client:
            # Fallback selection
            if item_data['triplet_texts']:
                selected = item_data['triplet_texts'][0]
                return selected, "Fallback selection", 0.5
            return {}, "No triplets available", 0.0
        
        prompt = f"""You are conducting a final review of the triplet selection process for "{item_data['item_text']}".

Movie Summary: {item_data['item_summary']}

Available triplets:
"""
        
        for i, triplet in enumerate(item_data['triplet_texts'], 1):
            prompt += f"{i}. {triplet['text_format']}\n"
        
        prompt += f"""
Debate Result: {debate_result}

Please provide your final assessment:
1. Which triplet number (1-{len(item_data['triplet_texts'])}) is most relevant?
2. What is your confidence level (0-1) in this selection?
3. Provide a brief reasoning for your final choice.

Format your response as:
SELECTED_TRIPLET: [number]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
        
        messages = [
            {"role": "system", "content": "You are a final reviewer ensuring the quality of triplet selection for movie recommendation systems."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_gpt(messages, max_tokens=400)
        
        # Parse the response
        selected_triplet = None
        confidence = 0.5
        reasoning = "No reasoning provided"
        
        lines = response.split('\n')
        for line in lines:
            if line.startswith('SELECTED_TRIPLET:'):
                try:
                    triplet_num = int(line.split(':')[1].strip())
                    if 1 <= triplet_num <= len(item_data['triplet_texts']):
                        selected_triplet = item_data['triplet_texts'][triplet_num - 1]
                except (ValueError, IndexError):
                    pass
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':')[1].strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except (ValueError, IndexError):
                    pass
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        # Fallback if parsing failed
        if not selected_triplet and item_data['triplet_texts']:
            selected_triplet = item_data['triplet_texts'][0]
            reasoning = "Fallback selection due to parsing error"
        
        return selected_triplet, reasoning, confidence
    
    def select_triplet_with_debate(self, item_data: Dict) -> Dict:
        """Select the most relevant triplet using debate and reflection
        
        Args:
            item_data: Item data with triplets and summary
            
        Returns:
            Dictionary with selection results
        """
        item_id = item_data['item_id']
        print(f"  🔍 Analyzing {len(item_data['triplet_texts'])} triplets for item {item_id}")
        
        # Check if already processed
        existing_pattern = self.memory.get_selection_pattern(item_id)
        if existing_pattern:
            print(f"  📋 Using cached result for item {item_id}")
            return {
                'item_id': item_id,
                'selected_triplet': existing_pattern['selected_triplet'],
                'reasoning': existing_pattern['reasoning'],
                'confidence': existing_pattern['confidence'],
                'debate_summary': existing_pattern['debate_summary'],
                'cached': True
            }
        
        # Generate three opinions
        print(f"  💭 Generating three expert opinions...")
        opinions = self._generate_three_opinions(item_data)
        
        # Debate the opinions
        print(f"  🗣️  Conducting debate...")
        debate_result = self._debate_opinions(item_data, opinions)
        
        # Final reflection
        print(f"  🤔 Final reflection and selection...")
        selected_triplet, reasoning, confidence = self._final_reflection(item_data, debate_result)
        
        # Store in memory
        self.memory.store_selection(item_id, selected_triplet, reasoning, confidence, debate_result)
        
        result = {
            'item_id': item_id,
            'selected_triplet': selected_triplet,
            'reasoning': reasoning,
            'confidence': confidence,
            'debate_summary': debate_result,
            'expert_opinions': opinions,
            'cached': False
        }
        
        print(f"  ✅ Selected triplet with confidence {confidence:.2f}")
        return result
    
    def select_triplet_simple(self, item_data: Dict) -> Dict:
        """Select the most relevant triplet using direct GPT analysis (no debate)
        
        Args:
            item_data: Item data with triplets and summary
            
        Returns:
            Dictionary with selection results
        """
        item_id = item_data['item_id']
        
        # Check if already processed in memory
        existing_pattern = self.memory.get_selection_pattern(item_id)
        if existing_pattern:
            print(f"  📋 Using cached result for item {item_id}")
            return {
                'item_id': item_id,
                'selected_triplet': existing_pattern['selected_triplet'],
                'reasoning': existing_pattern['reasoning'],
                'confidence': existing_pattern['confidence'],
                'debate_summary': existing_pattern['debate_summary'],
                'cached': True
            }
        
        # Direct GPT analysis
        print(f"  🤖 Direct GPT analysis...")
        selected_triplet, reasoning, confidence = self._direct_gpt_selection(item_data)
        
        # Store in memory
        self.memory.store_selection(item_id, selected_triplet, reasoning, confidence, 
                                   "Simple mode - direct GPT analysis", "simple")
        
        result = {
            'item_id': item_id,
            'selected_triplet': selected_triplet,
            'reasoning': reasoning,
            'confidence': confidence,
            'debate_summary': "Simple mode - direct GPT analysis",
            'cached': False
        }
        
        print(f"  ✅ Selected triplet with confidence {confidence:.2f}")
        return result
    
    def _direct_gpt_selection(self, item_data: Dict) -> Tuple[Dict, str, float]:
        """Direct GPT selection without debate
        
        Args:
            item_data: Item data with triplets and summary
            
        Returns:
            Tuple of (selected_triplet, reasoning, confidence_score)
        """
        if not self.client:
            return self._fallback_selection(item_data)
        
        prompt = f"""You are a movie recommendation expert. Analyze the following movie and select the most relevant knowledge graph triplet for recommendation purposes.

Movie: {item_data['item_text']}
Movie Summary: {item_data['item_summary']}

Available triplets:
"""
        
        for i, triplet in enumerate(item_data['triplet_texts'], 1):
            prompt += f"{i}. {triplet['text_format']}\n"
        
        prompt += f"""
Please select the most relevant triplet (1-{len(item_data['triplet_texts'])}) for movie recommendation and provide:
1. The triplet number
2. Your confidence level (0-1)
3. Brief reasoning for your choice

Format your response as:
SELECTION: [number]
CONFIDENCE: [0.0-1.0]
REASONING: [your reasoning]
"""
        
        messages = [
            {"role": "system", "content": "You are an expert in movie recommendation systems and knowledge graphs."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._call_gpt(messages, max_tokens=300)
            
            # Parse response
            lines = response.strip().split('\n')
            selection_num = None
            confidence = 0.7
            reasoning = "Direct GPT analysis"
            
            for line in lines:
                if line.startswith('SELECTION:'):
                    try:
                        selection_num = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            if selection_num is None or selection_num < 1 or selection_num > len(item_data['triplet_texts']):
                selection_num = 1
            
            selected_triplet = item_data['triplet_texts'][selection_num - 1]
            return selected_triplet, reasoning, confidence
            
        except Exception as e:
            print(f"  ⚠️  GPT analysis failed: {e}")
            return self._fallback_selection(item_data)
    
    def _fallback_selection(self, item_data: Dict) -> Tuple[Dict, str, float]:
        """Fallback selection method when GPT is not available"""
        if item_data['triplet_texts']:
            selected_triplet = item_data['triplet_texts'][0]
            return selected_triplet, "Fallback selection - first triplet", 0.5
        else:
            return {}, "No triplets available", 0.0
    
    def load_prepared_data(self, prepared_data_file: str) -> List[Dict]:
        """Load prepared data from a2-2-1-make.py
        
        Args:
            prepared_data_file: Path to prepared_data.json
            
        Returns:
            List of prepared item data
        """
        print(f"Loading prepared data from {prepared_data_file}...")
        
        try:
            with open(prepared_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded {len(data)} items from prepared data")
            return data
            
        except Exception as e:
            print(f"❌ Error loading prepared data: {e}")
            return []
    
    def process_all_items(self, prepared_data: List[Dict], max_items: Optional[int] = None) -> List[Dict]:
        """Process all items to select relevant triplets
        
        Args:
            prepared_data: List of prepared item data
            max_items: Maximum number of items to process
            
        Returns:
            List of selection results
        """
        # Filter items with triplets
        items_with_triplets = [item for item in prepared_data if not item.get('no_triplets', False) and item.get('triplet_texts')]
        
        if max_items:
            items_with_triplets = items_with_triplets[:max_items]
        
        total_items = len(items_with_triplets)
        print(f"Processing {total_items} items with triplets...")
        
        # Estimate processing time based on mode
        if self.enable_debate_reflection:
            estimated_time_per_item = (self.base_delay * 4 + 0.5)  # 4 GPT calls + processing time
            print(f"🔧 Mode: Full (Debate + Reflection)")
        else:
            estimated_time_per_item = (self.base_delay * 1 + 0.2)  # 1 GPT call + processing time
            print(f"🔧 Mode: Simple (Direct GPT)")
        
        estimated_total_time = total_items * estimated_time_per_item / 60  # Convert to minutes
        print(f"⏱️  Estimated time: {estimated_total_time:.1f} minutes")
        print(f"📊 Target rate: {self.requests_per_minute} requests/minute")
        
        results = []
        start_time = time.time()
        
        for i, item_data in enumerate(items_with_triplets):
            # Reset failure counter every 10 items
            if i % 10 == 0 and i > 0:
                if self.consecutive_failures > 0:
                    print(f"🔄 Reset failure counter at item {i+1} (was {self.consecutive_failures})")
                    self.consecutive_failures = 0
            
            # Progress update with time estimation
            if i % 5 == 0 or i == total_items - 1:
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_item = elapsed_time / i
                    remaining_items = total_items - i
                    estimated_remaining = remaining_items * avg_time_per_item / 60
                    print(f"Processing item {i+1}/{total_items}: {item_data['item_text']}")
                    print(f"   ⏱️  Elapsed: {elapsed_time/60:.1f}m, Remaining: {estimated_remaining:.1f}m")
                else:
                    print(f"Processing item {i+1}/{total_items}: {item_data['item_text']}")
            
            try:
                if self.enable_debate_reflection:
                    result = self.select_triplet_with_debate(item_data)
                else:
                    result = self.select_triplet_simple(item_data)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing item {item_data['item_id']}: {e}")
                # Add error result
                results.append({
                    'item_id': item_data['item_id'],
                    'error': str(e),
                    'cached': False
                })
        
        total_time = time.time() - start_time
        print(f"✅ Processed {len(results)} items")
        print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
        print(f"📊 Average time per item: {total_time/total_items:.2f} seconds")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save selection results to output file
        
        Args:
            results: List of selection results
            output_file: Output file path
        """
        print(f"Saving results to {output_file}...")
        
        try:
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(results)} selection results to {output_file}")
            
            # Save complete memory state
            if self.memory_file:
                self.save_memory()
            else:
                # Fallback: save memory statistics
                memory_stats = self.memory.get_statistics()
                memory_file = output_file.replace('.json', '_memory.json')
                with open(memory_file, 'w', encoding='utf-8') as f:
                    json.dump(memory_stats, f, ensure_ascii=False, indent=2)
                print(f"✅ Saved memory statistics to {memory_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def run_full_workflow(self, prepared_data_file: str, output_file: str, 
                         max_items: Optional[int] = None, memory_file: str = None):
        """Run the complete triplet selection workflow
        
        Args:
            prepared_data_file: Path to prepared_data.json
            output_file: Output file path
            max_items: Maximum number of items to process
            memory_file: Path to memory file for loading/saving
        """
        print("=" * 60)
        print("A2-2-2: GPT Triplet Selection Module")
        print("=" * 60)
        print(f"Prepared data file: {prepared_data_file}")
        print(f"Output file: {output_file}")
        print(f"Max items: {max_items if max_items else 'All items'}")
        print(f"Memory file: {memory_file if memory_file else 'Not specified'}")
        print(f"OpenAI API: {'Available' if self.client else 'Not available'}")
        print(f"Mode: {'Full (Debate + Reflection)' if self.enable_debate_reflection else 'Simple (Direct GPT)'}")
        print()
        
        # Set memory file if provided
        if memory_file:
            self.memory_file = memory_file
            self.load_memory()
        
        # Load prepared data
        prepared_data = self.load_prepared_data(prepared_data_file)
        if not prepared_data:
            print("❌ No prepared data loaded. Exiting.")
            return
        
        # Reset failure counter before processing
        self._reset_failure_counter_between_stages("before processing items")
        
        # Process all items
        results = self.process_all_items(prepared_data, max_items)
        
        # Save results
        self.save_results(results, output_file)
        
        print("\n" + "=" * 60)
        print("✅ Triplet Selection Complete!")
        print("=" * 60)
        
        # Print summary statistics
        successful_items = [r for r in results if 'error' not in r]
        error_items = [r for r in results if 'error' in r]
        cached_items = [r for r in successful_items if r.get('cached', False)]
        
        print(f"\n📊 Summary:")
        print(f"   Total items processed: {len(results)}")
        print(f"   Successful: {len(successful_items)}")
        print(f"   Cached results: {len(cached_items)}")
        print(f"   Errors: {len(error_items)}")
        
        if successful_items:
            avg_confidence = sum(r.get('confidence', 0) for r in successful_items) / len(successful_items)
            print(f"   Average confidence: {avg_confidence:.2f}")
        
        if error_items:
            print(f"\n❌ Items with errors:")
            for item in error_items[:5]:  # Show first 5 errors
                print(f"   Item {item['item_id']}: {item['error']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Select most relevant triplets using GPT with debate and reflection')
    parser.add_argument('--prepared-data', type=str, default='prepared_data.json',
                       help='Path to prepared_data.json file')
    parser.add_argument('--output', type=str, default='selected_triplets.json',
                       help='Output file path for selected triplets')
    parser.add_argument('--max-items', type=int, default=None,
                       help='Maximum number of items to process (None for all)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                       help='OpenAI API key for GPT analysis')
    parser.add_argument('--base-delay', type=float, default=0.5,
                       help='Base delay between requests in seconds (default: 0.5)')
    parser.add_argument('--max-delay', type=float, default=5.0,
                       help='Maximum delay between requests in seconds (default: 5.0)')
    parser.add_argument('--requests-per-minute', type=int, default=60,
                       help='Target requests per minute (default: 60)')
    parser.add_argument('--memory-file', type=str, default=None,
                       help='Path to memory file for loading/saving triplet selection history')
    parser.add_argument('--load-memory', action='store_true',
                       help='Load memory from previous run (requires --memory-file)')
    parser.add_argument('--clear-memory', action='store_true',
                       help='Clear memory before starting (useful for fresh start)')
    parser.add_argument('--enable-debate-reflection', action='store_true', default=True,
                       help='Enable debate and reflection mode (default: True)')
    parser.add_argument('--simple-mode', action='store_true',
                       help='Use simple mode (direct GPT, no debate/reflection)')
    
    args = parser.parse_args()
    
    # Determine mode
    enable_debate_reflection = args.enable_debate_reflection and not args.simple_mode
    
    # Create triplet selector with memory file and mode
    selector = TripletSelector(args.openai_api_key, args.memory_file, enable_debate_reflection)
    
    # Override default rate limiting if specified
    if args.base_delay != 0.5:
        selector.base_delay = args.base_delay
        print(f"📊 Custom base delay: {selector.base_delay}s")
    if args.max_delay != 5.0:
        selector.max_delay = args.max_delay
        print(f"📊 Custom max delay: {selector.max_delay}s")
    if args.requests_per_minute != 60:
        selector.requests_per_minute = args.requests_per_minute
        print(f"📊 Custom rate limit: {selector.requests_per_minute} requests/minute")
    
    # Handle memory options
    if args.clear_memory:
        print("🧹 Clearing memory as requested...")
        selector.clear_memory()
    
    if args.load_memory and args.memory_file:
        print(f"🧠 Loading memory from {args.memory_file}...")
        selector.load_memory()
    elif args.load_memory and not args.memory_file:
        print("⚠️ --load-memory specified but no --memory-file provided")
    
    # Run workflow
    selector.run_full_workflow(args.prepared_data, args.output, args.max_items, args.memory_file)

if __name__ == "__main__":
    main()


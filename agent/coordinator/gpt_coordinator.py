import json
import re
import os
from typing import Dict, Tuple, Optional
from openai import OpenAI

class GPTCoordinator:
    """GPT-based intelligent coordinator"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize GPT coordinator
        
        Args:
            openai_api_key: OpenAI API key, if None then get from environment variables
        """
        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print(" Warning: No OpenAI API key provided. GPT coordination will be disabled.")
                self.client = None
        
        self.system_prompt = self._get_system_prompt()
        self.decision_history = []
        
        if self.client:
            print(" GPT Coordinator initialized with OpenAI API")
        else:
            print("  GPT Coordinator initialized without API (fallback mode)")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt"""
        return """You are an intelligent coordinator for a multi-agent recommendation system. Analyze dataset features and output exactly 6 digits for agent configurations.

**Output Format:** 6 digits (Agent1_use_Agent1_debate_Agent2_use_Agent2_debate_Agent3_use_Agent3_debate)

**Agents:**
- Agent1: User analysis (cost: users*0.1 + interactions*0.005)
- Agent2: KG processing (cost: items*0.05 + triplets*0.001) 
- Agent3: Recommendation (cost: users*0.02 + items*0.01)

**Decision Rules:**
- Small datasets (<1M interactions, <10K users, <10K items): 111111
- Medium datasets with many items (items>=20K, users<50K): 111011 (Agent1 debate/reflection, Agent2 simple mode)
- Large datasets (users>=50K): 101011 (Agent1&2 simple mode due to high cost)
- Agent3: Always enabled with debate/reflection (core component)

**Important:** Agent3 should always be enabled with debate/reflection for optimal quality.

Output the 6-digit decision immediately after your analysis."""

    
    def make_decision(self, dataset_features: Dict) -> str:
        """Make decision based on dataset features
        
        Args:
            dataset_features: Dataset features dictionary
            
        Returns:
            6-digit string representing Agent configuration
        """
        if not self.client:
            return self._fallback_decision(dataset_features)
        
        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(dataset_features)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            decision_text = response.choices[0].message.content.strip()
            
            # Extract 6-digit decision
            decision = self._extract_decision(decision_text)
            
            # If extraction failed, use fallback logic
            if decision is None:
                print(f"GPT extraction failed, using fallback logic...")
                decision = self._fallback_decision(dataset_features)
                # Use fallback decision with enhanced reasoning
                enhanced_reasoning = self._build_enhanced_reasoning(dataset_features, decision, decision_text)
            else:
                # Build enhanced reasoning from GPT output
                enhanced_reasoning = self._build_enhanced_reasoning(dataset_features, decision, decision_text)
            
            # Record decision history
            self.decision_history.append({
                "features": dataset_features,
                "decision": decision,
                "reasoning": enhanced_reasoning
            })
            
            print(f" GPT Decision: {decision}")
            return decision
                
        except Exception as e:
            print(f" GPT decision failed: {e}")
            return self._fallback_decision(dataset_features)
    
    def _build_analysis_prompt(self, dataset_features: Dict) -> str:
        """Build analysis prompt"""
        return f"""Dataset Feature Analysis Results:

**User Interaction Features:**
- Total Interactions: {dataset_features.get('total_interactions', 0):,}
- User Count: {dataset_features.get('unique_users', 0):,}
- Item Count: {dataset_features.get('unique_items', 0):,}
- Data Sparsity: {dataset_features.get('sparsity', 0):.3f}
- Dataset Scale: {dataset_features.get('dataset_scale', 'unknown')}
- Average Interactions per User: {dataset_features.get('avg_ratings_per_user', 0):.1f}

**Knowledge Graph Features:**
- Total Triplets: {dataset_features.get('total_triplets', 0):,}
- Entity Count: {dataset_features.get('unique_entities', 0):,}
- Relation Count: {dataset_features.get('unique_relations', 0):,}
- KG Density: {dataset_features.get('kg_density', 0):.3f}
- Average Triplets per Item: {dataset_features.get('avg_triplets_per_item', 0):.1f}
- KG Coverage of Items: {self._calculate_kg_coverage(dataset_features):.3f}

**Comprehensive Analysis:**
- Dataset Complexity Score: {dataset_features.get('complexity_score', 0):.2f}
- Knowledge Graph Richness: {dataset_features.get('kg_richness', 0):.2f}
- Computational Resource Demand: {dataset_features.get('computational_demand', 'unknown')}
- Data Quality Score: {dataset_features.get('data_quality_score', 0):.2f}

**User Activity Distribution:**
- Highly Active Users: {dataset_features.get('user_activity_distribution', {}).get('highly_active_users', 0)}
- Moderately Active Users: {dataset_features.get('user_activity_distribution', {}).get('moderately_active_users', 0)}
- Low Active Users: {dataset_features.get('user_activity_distribution', {}).get('low_active_users', 0)}

**Item Popularity Distribution:**
- Popular Items: {dataset_features.get('item_popularity_distribution', {}).get('popular_items', 0)}
- Moderately Popular Items: {dataset_features.get('item_popularity_distribution', {}).get('moderate_items', 0)}
- Long-tail Items: {dataset_features.get('item_popularity_distribution', {}).get('long_tail_items', 0)}

**Computational Cost Analysis:**
- Total Computational Cost: {self._calculate_computational_cost(dataset_features):.2f}
- Agent1 Cost (User Analysis): {self._calculate_agent1_cost(dataset_features):.2f}
- Agent2 Cost (KG Processing): {self._calculate_agent2_cost(dataset_features):.2f}
- Agent3 Cost (Recommendation): {self._calculate_agent3_cost(dataset_features):.2f}

Please analyze the above features and computational costs to make intelligent decisions about agent usage and debate/reflection modes. Consider the trade-off between computational cost and recommendation quality.

**Important Guidelines:**
- Agent3 is the core recommendation component and should always be enabled with debate/reflection for optimal quality
- For small datasets (< 1M interactions, < 10K users, < 10K items), all agents can afford debate/reflection due to lower computational costs
- For medium datasets, Agent2 (KG processing) may use simple mode due to higher item counts
- For large datasets, Agent1 and Agent2 may use simple mode due to computational constraints

Output format: 6-digit decision result (Agent1_Enabled, Agent1_Debate, Agent2_Enabled, Agent2_Debate, Agent3_Enabled, Agent3_Debate)"""
    
    def _extract_decision(self, decision_text: str) -> str:
        """Extract 6-digit decision from GPT response"""
        # Find underscore-separated 6-digit pattern (e.g., 1_1_1_0_1_1)
        underscore_pattern = re.search(r'(\d)_(\d)_(\d)_(\d)_(\d)_(\d)', decision_text)
        if underscore_pattern:
            decision = ''.join(underscore_pattern.groups())
            if self._validate_decision(decision):
                return decision
        
        # Find 6 consecutive digits
        numbers = re.findall(r'\d', decision_text)
        if len(numbers) >= 6:
            decision = ''.join(numbers[:6])
            if self._validate_decision(decision):
                return decision
        
        # Find specific 6-digit pattern
        pattern = re.search(r'(\d{6})', decision_text)
        if pattern:
            decision = pattern.group(1)
            if self._validate_decision(decision):
                return decision
        
        # Intelligent parsing based on expected outputs
        agent1_use = "0"
        agent1_debate = "0"
        agent2_use = "0"
        agent2_debate = "0"
        agent3_use = "0"
        agent3_debate = "0"
        
        # Enhanced parsing for truncated outputs
        # Parse Agent1
        if "Agent1: Enable" in decision_text or "Agent1：使用" in decision_text or "1. Agent1:" in decision_text:
            agent1_use = "1"
            if ("Agent1: Debate/Reflection" in decision_text or "Agent1: 辩论反思" in decision_text or
                "1. Agent1: Enable + Debate/Reflection" in decision_text or "Enable + Debate" in decision_text):
                agent1_debate = "1"
            elif ("Agent1: Simple Mode" in decision_text or "Agent1: 简单模式" in decision_text or
                  "Enable + Simple" in decision_text):
                agent1_debate = "0"
        
        # Parse Agent2
        if "Agent2: Enable" in decision_text or "Agent2：使用" in decision_text or "2. Agent2:" in decision_text:
            agent2_use = "1"
            if ("Agent2: Simple Mode" in decision_text or "Agent2: 简单模式" in decision_text or
                "Simple Mode" in decision_text):
                agent2_debate = "0"
            elif ("Agent2: Debate/Reflection" in decision_text or "Agent2: 辩论反思" in decision_text or 
                  "2. Agent2: Enable + Debate/Reflection" in decision_text or "Enable + Debate" in decision_text):
                agent2_debate = "1"
        
        # Parse Agent3
        if ("Agent3: Enable" in decision_text or "Agent3: ..." in decision_text or "Agent3：使用" in decision_text or
            "3. Agent3:" in decision_text):
            agent3_use = "1"
            if ("Agent3: Simple Mode" in decision_text or "Agent3: 简单模式" in decision_text or
                "Simple Mode" in decision_text):
                agent3_debate = "0"
            elif ("Agent3: Debate/Reflection" in decision_text or "Agent3: 辩论反思" in decision_text or
                  "3. Agent3: Enable + Debate/Reflection" in decision_text or "Enable + Debate" in decision_text):
                agent3_debate = "1"
            else:
                # Default to debate/reflection for Agent3 unless clearly stated otherwise
                agent3_debate = "1"
        
        decision = agent1_use + agent1_debate + agent2_use + agent2_debate + agent3_use + agent3_debate
        
        # If we have a valid decision, return it
        if self._validate_decision(decision):
            return decision
        
        # If parsing failed, try intelligent inference based on partial information
        return self._intelligent_inference(decision_text, agent1_use, agent1_debate, agent2_use, agent2_debate, agent3_use, agent3_debate)
        
        # If unable to extract valid decision, use fallback logic
        print(f"  Could not extract valid decision from: {decision_text}")
        print(f"  Using fallback logic instead of default decision")
        # Return None to trigger fallback logic in make_decision method
        return None
    
    def _intelligent_inference(self, decision_text: str, agent1_use: str, agent1_debate: str, 
                              agent2_use: str, agent2_debate: str, agent3_use: str, agent3_debate: str) -> str:
        """Intelligent inference for truncated outputs"""
        
        # If we have at least one agent enabled, try to infer the rest
        if agent1_use == "1" or agent2_use == "1" or agent3_use == "1":
            
            # If Agent1 is enabled but debate status unknown, infer based on context
            if agent1_use == "1" and agent1_debate == "0":
                if "Debate" in decision_text or "debate" in decision_text.lower():
                    agent1_debate = "1"
                elif "Simple" in decision_text or "simple" in decision_text.lower():
                    agent1_debate = "0"
            
            # If Agent2 is enabled but debate status unknown, infer based on context
            if agent2_use == "1" and agent2_debate == "0":
                if "Debate" in decision_text or "debate" in decision_text.lower():
                    agent2_debate = "1"
                elif "Simple" in decision_text or "simple" in decision_text.lower():
                    agent2_debate = "0"
            
            # Agent3 should always be enabled with debate/reflection if not explicitly disabled
            if agent3_use == "0":
                agent3_use = "1"
            if agent3_debate == "0" and "Simple" not in decision_text:
                agent3_debate = "1"
            
            # Only set debate/reflection for enabled agents
            # Don't force enable agents - respect the original decision
            
            # If Agent1 is enabled but debate status unknown, set based on user count
            if agent1_use == "1" and agent1_debate == "0":
                # Get user count from decision_text or use default logic
                if "users" in decision_text.lower():
                    # Try to extract user count from context
                    if "50000" in decision_text or "large" in decision_text.lower():
                        agent1_debate = "0"  # Large user count, use simple mode
                    else:
                        agent1_debate = "1"  # Moderate user count, use debate/reflection
                else:
                    agent1_debate = "1"  # Default to debate/reflection
            
            # If Agent2 is enabled but debate status unknown, set based on item count
            if agent2_use == "1" and agent2_debate == "0":
                # Get item count from decision_text or use default logic
                if "items" in decision_text.lower():
                    # Try to extract item count from context
                    if "20000" in decision_text or "many items" in decision_text.lower():
                        agent2_debate = "0"  # Many items, use simple mode
                    else:
                        agent2_debate = "1"  # Few items, use debate/reflection
                else:
                    agent2_debate = "1"  # Default to debate/reflection
            
            decision = agent1_use + agent1_debate + agent2_use + agent2_debate + agent3_use + agent3_debate
            if self._validate_decision(decision):
                return decision
        
        # If all else fails, return None to trigger fallback
        return None
    
    def _build_enhanced_reasoning(self, features: Dict, decision: str, gpt_output: str) -> str:
        """Build enhanced reasoning from GPT output and dataset features"""
        
        # Start with GPT output if it's meaningful
        reasoning_parts = []
        
        if gpt_output and len(gpt_output) > 10 and not gpt_output.strip().isdigit():
            reasoning_parts.append(gpt_output)
        else:
            # If GPT output is too short or just digits, build reasoning from features
            reasoning_parts.append("Based on the dataset features and analysis results provided, here are the recommended decisions for each Agent:")
        
        # Add detailed agent analysis
        total_interactions = features.get('total_interactions', 0)
        unique_users = features.get('unique_users', 0)
        unique_items = features.get('unique_items', 0)
        dataset_scale = features.get('dataset_scale', 'unknown')
        
        # Parse decision to get agent configurations
        if len(decision) == 6:
            agent1_enabled = decision[0] == '1'
            agent1_debate = decision[1] == '1'
            agent2_enabled = decision[2] == '1'
            agent2_debate = decision[3] == '1'
            agent3_enabled = decision[4] == '1'
            agent3_debate = decision[5] == '1'
            
            reasoning_parts.append("")
            
            # Agent1 analysis
            if agent1_enabled:
                mode = "Debate/Reflection" if agent1_debate else "Simple Mode"
                reasoning_parts.append(f"1. Agent1: Enable + {mode}")
            else:
                reasoning_parts.append("1. Agent1: Disabled")
            
            # Agent2 analysis
            if agent2_enabled:
                mode = "Debate/Reflection" if agent2_debate else "Simple Mode"
                reasoning_parts.append(f"2. Agent2: Enable + {mode}")
            else:
                reasoning_parts.append("2. Agent2: Disabled")
            
            # Agent3 analysis
            if agent3_enabled:
                mode = "Debate/Reflection" if agent3_debate else "Simple Mode"
                reasoning_parts.append(f"3. Agent3: Enable + {mode}")
            else:
                reasoning_parts.append("3. Agent3: Disabled")
            
            # Add dataset context
            reasoning_parts.append("")
            reasoning_parts.append(f"Dataset Analysis:")
            reasoning_parts.append(f"- Scale: {dataset_scale}")
            reasoning_parts.append(f"- Interactions: {total_interactions:,}")
            reasoning_parts.append(f"- Users: {unique_users:,}")
            reasoning_parts.append(f"- Items: {unique_items:,}")
            
            # Add decision rationale
            reasoning_parts.append("")
            if total_interactions < 1000000 and unique_users < 10000 and unique_items < 10000:
                reasoning_parts.append("This is a small-scale dataset with few users and items, enabling all agents with debate/reflection for optimal quality.")
            elif unique_users >= 50000:
                reasoning_parts.append("This is a large-scale dataset with many users, using simple modes for Agent1 and Agent2 to optimize computational performance.")
            elif unique_items >= 20000 and unique_users < 50000:
                reasoning_parts.append("This is a medium-scale dataset with many items but moderate users, using simple mode for Agent2 to balance cost and quality.")
            else:
                reasoning_parts.append("This is a medium-scale dataset, selectively enabling debate/reflection based on computational cost.")
        
        return "\n".join(reasoning_parts)
    
    def _validate_decision(self, decision: str) -> bool:
        """Validate decision format"""
        if len(decision) != 6:
            return False
        
        # Check if all are 0 or 1
        for char in decision:
            if char not in ['0', '1']:
                return False
        
        return True
    
    def _calculate_kg_coverage(self, features: Dict) -> float:
        """Calculate knowledge graph coverage ratio of items"""
        try:
            items_in_kg = features.get('unique_items_in_kg', 0)
            total_items = features.get('unique_items', 1)
            return items_in_kg / total_items
        except:
            return 0.0
    
    def _calculate_computational_cost(self, features: Dict) -> float:
        """Calculate total computational cost"""
        total_interactions = features.get('total_interactions', 0)
        unique_users = features.get('unique_users', 0)
        unique_items = features.get('unique_items', 0)
        total_triplets = features.get('total_triplets', 0)
        
        # Base cost calculation
        cost = (total_interactions * 0.01 + 
                unique_users * 0.1 + 
                unique_items * 0.05 + 
                total_triplets * 0.001)
        
        return cost
    
    def _calculate_agent1_cost(self, features: Dict) -> float:
        """Calculate Agent1 computational cost (User Analysis)"""
        unique_users = features.get('unique_users', 0)
        total_interactions = features.get('total_interactions', 0)
        
        # Agent1 cost is mainly related to user analysis
        cost = unique_users * 0.1 + total_interactions * 0.005
        return cost
    
    def _calculate_agent2_cost(self, features: Dict) -> float:
        """Calculate Agent2 computational cost (KG Processing)"""
        unique_items = features.get('unique_items', 0)
        total_triplets = features.get('total_triplets', 0)
        total_interactions = features.get('total_interactions', 0)
        
        # Agent2 cost is mainly related to KG processing
        # For small datasets, reduce the cost calculation
        if total_interactions < 1000000 and unique_items < 10000:
            # Small datasets: much lower cost for Agent2
            cost = unique_items * 0.01 + total_triplets * 0.0001
        else:
            # Large datasets: normal cost
            cost = unique_items * 0.05 + total_triplets * 0.001
        return cost
    
    def _calculate_agent3_cost(self, features: Dict) -> float:
        """Calculate Agent3 computational cost (Recommendation)"""
        unique_users = features.get('unique_users', 0)
        unique_items = features.get('unique_items', 0)
        
        # Agent3 cost is related to recommendation generation
        cost = unique_users * 0.02 + unique_items * 0.01
        return cost
    
    def _fallback_decision(self, features: Dict) -> str:
        """Fallback decision logic"""
        scale = features.get('dataset_scale', 'medium')
        computational_demand = features.get('computational_demand', 'medium')
        kg_richness = features.get('kg_richness', 0.5)
        total_interactions = features.get('total_interactions', 0)
        unique_users = features.get('unique_users', 0)
        unique_items = features.get('unique_items', 0)
        sparsity = features.get('sparsity', 0.5)
        
        print("🔄 Using fallback decision logic...")
        
        # Initialize default values
        agent1_enabled = 1
        agent1_debate = 1
        agent2_enabled = 1
        agent2_debate = 1
        agent3_enabled = 1
        agent3_debate = 1
        
        # Based on intelligent decision principles fallback logic
        
        # Special case: extremely low interaction count
        if total_interactions < 10000:
            agent1_enabled = 0
            agent1_debate = 0
            agent2_enabled = 0
            agent2_debate = 0
            agent3_enabled = 0
            agent3_debate = 0
        
        # Special case: extremely low user count
        elif unique_users < 500:
            agent1_enabled = 0
            agent1_debate = 0
            agent2_enabled = 1 if kg_richness > 0.1 else 0
            agent2_debate = 0
            agent3_enabled = 1
            agent3_debate = 0
        
        # Special case: extremely low KG richness
        elif kg_richness < 0.05:
            agent1_enabled = 1 if sparsity > 0.9 else 0
            agent1_debate = 1 if unique_users < 10000 else 0
            agent2_enabled = 0
            agent2_debate = 0
            agent3_enabled = 1
            agent3_debate = 1
        
        # Calculate computational costs for decision making
        agent1_cost = self._calculate_agent1_cost(features)
        agent2_cost = self._calculate_agent2_cost(features)
        agent3_cost = self._calculate_agent3_cost(features)
        total_cost = self._calculate_computational_cost(features)
        
        # Enhanced decision logic based on user count and item count
        
        # Movie Dataset: 111111 (small dataset)
        if total_interactions < 1000000 and unique_users < 10000 and unique_items < 10000:
            agent1_enabled = 1
            agent1_debate = 1  # Few users, can afford debate/reflection
            agent2_enabled = 1
            agent2_debate = 1  # Few items, can afford debate/reflection
            agent3_enabled = 1
            agent3_debate = 1  # Always enabled with debate/reflection
        
        # ML-10M: 101011 (large dataset - many users)
        elif unique_users >= 50000:
            agent1_enabled = 1
            agent1_debate = 0  # Many users, use simple mode to save computational cost
            agent2_enabled = 1
            agent2_debate = 0  # Many items, use simple mode to save computational cost
            agent3_enabled = 1
            agent3_debate = 1  # Always enabled with debate/reflection
        
        # Amazon-Book: 110111 (medium dataset - many items but moderate users)
        elif unique_items >= 20000 and unique_users < 50000:
            agent1_enabled = 1
            agent1_debate = 1  # Moderate user count, can afford debate/reflection
            agent2_enabled = 1
            agent2_debate = 0  # Many items, use simple mode to save computational cost
            agent3_enabled = 1
            agent3_debate = 1  # Always enabled with debate/reflection
        
        # Other medium-scale datasets
        elif 100000 <= total_interactions < 10000000 and unique_users < 50000:
            agent1_enabled = 1
            agent1_debate = 1  # Moderate user count, can afford debate/reflection
            agent2_enabled = 1
            agent2_debate = 0  # Use simple mode by default for medium datasets
            agent3_enabled = 1
            agent3_debate = 1  # Always enabled with debate/reflection
        
        # Other cases maintain original logic
        else:
            # Default strategy for unmatched cases
            agent1_enabled = 1
            agent1_debate = 1
            agent2_enabled = 1
            agent2_debate = 0  # Default to simple mode for unmatched cases
            agent3_enabled = 1
            agent3_debate = 1
        
        decision = f"{agent1_enabled}{agent1_debate}{agent2_enabled}{agent2_debate}{agent3_enabled}{agent3_debate}"
        
        print(f"🔄 Fallback decision: {decision}")
        return decision
    
    def parse_decision(self, decision: str) -> Dict[str, Tuple[int, int]]:
        """Parse decision result
        
        Args:
            decision: 6-digit string
            
        Returns:
            Agent configuration dictionary
        """
        if len(decision) != 6:
            raise ValueError(f"Decision format error: {decision}")
        
        return {
            "agent1": (int(decision[0]), int(decision[1])),
            "agent2": (int(decision[2]), int(decision[3])),
            "agent3": (int(decision[4]), int(decision[5]))
        }
    
    def get_decision_reasoning(self, dataset_features: Dict) -> str:
        """Get decision reasoning"""
        if not self.decision_history:
            return "No decision history available"
        
        # Find recent decision records
        for record in reversed(self.decision_history):
            if record["features"] == dataset_features:
                return record.get("reasoning", "No reasoning available")
        
        return "No matching decision found"
    
    def save_decision_history(self, output_file: str = "decision_history.json"):
        """Save decision history"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.decision_history, f, ensure_ascii=False, indent=2)
            print(f" Decision history saved to {output_file}")
        except Exception as e:
            print(f" Error saving decision history: {e}")
    
    def load_decision_history(self, input_file: str):
        """Load decision history"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                self.decision_history = json.load(f)
            print(f" Decision history loaded from {input_file}")
        except Exception as e:
            print(f" Error loading decision history: {e}")
            self.decision_history = []


def main():
    """Test GPT coordinator"""
    # Note: Need to set OpenAI API key
    coordinator = GPTCoordinator()
    
    # Simulate dataset features
    sample_features = {
        "total_interactions": 1000000,
        "unique_users": 6000,
        "unique_items": 4000,
        "sparsity": 0.958,
        "dataset_scale": "large",
        "total_triplets": 100000,
        "unique_relations": 20,
        "kg_richness": 0.3,
        "computational_demand": "medium"
    }
    
    # Make decision
    decision = coordinator.make_decision(sample_features)
    
    # Parse decision
    configs = coordinator.parse_decision(decision)
    
    print(f"\n Decision: {decision}")
    print(f" Agent Configurations:")
    for agent, (enabled, debate) in configs.items():
        status = " Enabled" if enabled else " Disabled"
        debate_status = " Debate/Reflection" if debate else " Simple Mode"
        print(f"   - {agent}: {status}, {debate_status}")


if __name__ == "__main__":
    main()

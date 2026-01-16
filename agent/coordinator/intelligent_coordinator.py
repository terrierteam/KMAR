import json
import os
import argparse
from typing import Dict, Tuple, Optional
from dataset_analyzer import DatasetAnalyzer
from gpt_coordinator import GPTCoordinator

class IntelligentCoordinator:
    """Main intelligent coordinator class"""
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 config_file: Optional[str] = None):
        """Initialize intelligent coordinator
        
        Args:
            openai_api_key: OpenAI API key
            config_file: Configuration file path
        """
        self.analyzer = DatasetAnalyzer()
        self.gpt_coordinator = GPTCoordinator(openai_api_key)
        self.config_file = config_file
        self.config = self._load_config() if config_file else {}
        
        print("Intelligent Coordinator initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Configuration loaded from {self.config_file}")
            return config
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            return {}
    
    def coordinate_agents(self, ratings_file: str, kg_file: str, 
                         dataset_name: Optional[str] = None,
                         save_features: bool = True,
                         save_decision: bool = True) -> Dict[str, Tuple[int, int]]:
        """Coordinate agent usage strategies
        
        Args:
            ratings_file: User interaction data file path
            kg_file: Knowledge graph triplets file path
            dataset_name: Dataset name
            save_features: Whether to save feature analysis results
            save_decision: Whether to save decision results
            
        Returns:
            Agent configuration dictionary
        """
        print("Starting intelligent coordination process...")
        print("=" * 60)
        
        # 1. Analyze dataset features
        print("Step 1: Analyzing dataset features...")
        features = self.analyzer.analyze_dataset(ratings_file, kg_file)
        
        # 2. Save feature analysis results
        if save_features:
            features_file = f"{dataset_name}_features.json" if dataset_name else "dataset_features.json"
            self.analyzer.save_features(features, features_file)
        
        # 3. GPT intelligent decision making
        print("\nStep 2: GPT intelligent decision making...")
        decision = self.gpt_coordinator.make_decision(features)
        
        # 4. Parse decision results
        print("\nStep 3: Parsing decision results...")
        agent_configs = self.gpt_coordinator.parse_decision(decision)
        
        # 5. Save decision results
        if save_decision:
            decision_file = f"{dataset_name}_decision.json" if dataset_name else "agent_decision.json"
            self._save_decision_result(features, decision, agent_configs, decision_file)
        
        # 6. Output results
        print("\nStep 4: Final results...")
        self._print_results(features, decision, agent_configs, dataset_name)
        
        return agent_configs
    
    def coordinate_from_config(self, dataset_name: str) -> Dict[str, Tuple[int, int]]:
        """Coordinate agents from configuration file"""
        if not self.config:
            raise ValueError("No configuration loaded")
        
        dataset_config = self.config.get("datasets", {}).get(dataset_name)
        if not dataset_config:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        
        ratings_file = dataset_config.get("ratings_file")
        kg_file = dataset_config.get("kg_file")
        
        if not ratings_file or not kg_file:
            raise ValueError(f"Missing file paths for dataset {dataset_name}")
        
        return self.coordinate_agents(ratings_file, kg_file, dataset_name)
    
    def batch_coordinate(self, datasets_config: Dict[str, Dict]) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """Batch coordinate multiple datasets
        
        Args:
            datasets_config: Dataset configuration dictionary
            
        Returns:
            Agent configurations for all datasets
        """
        print("Starting batch coordination...")
        results = {}
        
        for dataset_name, config in datasets_config.items():
            print(f"\nProcessing dataset: {dataset_name}")
            try:
                ratings_file = config["ratings_file"]
                kg_file = config["kg_file"]
                
                agent_configs = self.coordinate_agents(ratings_file, kg_file, dataset_name)
                results[dataset_name] = agent_configs
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                results[dataset_name] = {}
        
        # Save batch results
        self._save_batch_results(results)
        
        return results
    
    def _save_decision_result(self, features: Dict, decision: str, 
                            agent_configs: Dict, output_file: str):
        """Save decision results"""
        result = {
            "dataset_features": self._make_json_serializable(features),
            "gpt_decision": decision,
            "agent_configurations": {
                agent: {"enabled": int(config[0]), "debate_reflection": int(config[1])}
                for agent, config in agent_configs.items()
            },
            "decision_reasoning": self.gpt_coordinator.get_decision_reasoning(features),
            "timestamp": self._get_timestamp()
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Decision result saved to {output_file}")
        except Exception as e:
            print(f"Error saving decision result: {e}")
    
    def _save_batch_results(self, results: Dict):
        """Save batch coordination results"""
        batch_result = {
            "batch_results": results,
            "summary": self._generate_batch_summary(results),
            "timestamp": self._get_timestamp()
        }
        
        try:
            with open("batch_coordination_results.json", 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, ensure_ascii=False, indent=2)
            print("Batch results saved to batch_coordination_results.json")
        except Exception as e:
            print(f"Error saving batch results: {e}")
    
    def _generate_batch_summary(self, results: Dict) -> Dict:
        """Generate batch results summary"""
        summary = {
            "total_datasets": len(results),
            "successful_datasets": len([r for r in results.values() if r]),
            "agent_usage_stats": {
                "agent1_enabled": 0,
                "agent2_enabled": 0,
                "agent3_enabled": 0,
                "debate_reflection_usage": 0
            }
        }
        
        for dataset_result in results.values():
            if dataset_result:
                for agent, (enabled, debate) in dataset_result.items():
                    if enabled:
                        summary["agent_usage_stats"][f"{agent}_enabled"] += 1
                    if debate:
                        summary["agent_usage_stats"]["debate_reflection_usage"] += 1
        
        return summary
    
    def _print_results(self, features: Dict, decision: str, 
                      agent_configs: Dict, dataset_name: Optional[str]):
        """Print coordination results"""
        print("\n" + "=" * 60)
        print("INTELLIGENT COORDINATION RESULTS")
        print("=" * 60)
        
        if dataset_name:
            print(f"Dataset: {dataset_name}")
        
        print(f"\nDataset Features Summary:")
        print(f"   - Scale: {features.get('dataset_scale', 'unknown')}")
        print(f"   - Users: {features.get('unique_users', 0):,}")
        print(f"   - Items: {features.get('unique_items', 0):,}")
        print(f"   - Interactions: {features.get('total_interactions', 0):,}")
        print(f"   - KG Triplets: {features.get('total_triplets', 0):,}")
        print(f"   - Complexity Score: {features.get('complexity_score', 0):.2f}")
        print(f"   - KG Richness: {features.get('kg_richness', 0):.2f}")
        print(f"   - Computational Demand: {features.get('computational_demand', 'unknown')}")
        
        print(f"\nGPT Decision: {decision}")
        
        print(f"\nAgent Configurations:")
        for agent, (enabled, debate) in agent_configs.items():
            status = "Enabled" if enabled else "Disabled"
            debate_status = "Debate/Reflection" if debate else "Simple Mode"
            print(f"   - {agent.upper()}: {status}, {debate_status}")
        
        # Print decision reasoning
        reasoning = self.gpt_coordinator.get_decision_reasoning(features)
        print(f"\n💭 Decision Reasoning:")
        if reasoning and reasoning != "No decision history available" and reasoning != "No reasoning available":
            # Show full reasoning if available
            print(f"   {reasoning}")
        else:
            # Fallback: show just the decision if no reasoning available
            print(f"   {decision}")
        
        print("=" * 60)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_agent_command_line_args(self, agent_configs: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
        """Generate agent command line arguments
        
        Args:
            agent_configs: Agent configuration dictionary
            
        Returns:
            Command line arguments dictionary
        """
        commands = {}
        
        for agent, (enabled, debate) in agent_configs.items():
            if enabled:
                # Build command line arguments
                args = []
                
                if agent == "agent1":
                    args.extend([
                        "--enable-agent",
                        "--debate-reflection" if debate else "--simple-mode"
                    ])
                elif agent == "agent2":
                    args.extend([
                        "--enable-agent",
                        "--debate-reflection" if debate else "--simple-mode"
                    ])
                elif agent == "agent3":
                    args.extend([
                        "--enable-agent",
                        "--enable-debate-reflection" if debate else "--simple-mode"
                    ])
                
                commands[agent] = " ".join(args)
            else:
                commands[agent] = "--disable-agent"
        
        return commands
    
    def generate_execution_script(self, agent_configs: Dict[str, Tuple[int, int]], 
                                output_file: str = "run_agents.sh"):
        """Generate agent execution script
        
        Args:
            agent_configs: Agent configuration dictionary
            output_file: Output script file name
        """
        commands = self.get_agent_command_line_args(agent_configs)
        
        script_content = "#!/bin/bash\n"
        script_content += "# Auto-generated agent execution script\n"
        script_content += "# Generated by Intelligent Coordinator\n\n"
        
        for agent, command in commands.items():
            script_content += f"# Run {agent.upper()}\n"
            script_content += f"echo 'Running {agent.upper()}...'\n"
            script_content += f"python {agent}_main.py {command}\n\n"
        
        try:
            with open(output_file, 'w') as f:
                f.write(script_content)
            print(f"Execution script saved to {output_file}")
        except Exception as e:
            print(f"Error saving execution script: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Intelligent Coordinator for Multi-Agent Recommendation System')
    parser.add_argument('--ratings-file', type=str, required=True,
                       help='User interaction data file path')
    parser.add_argument('--kg-file', type=str, required=True,
                       help='Knowledge graph triplets file path')
    parser.add_argument('--dataset-name', type=str, default=None,
                       help='Dataset name for output files')
    parser.add_argument('--openai-api-key', type=str, default=None,
                       help='OpenAI API key for GPT coordination')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--output-script', type=str, default='run_agents.sh',
                       help='Output execution script filename')
    parser.add_argument('--save-features', action='store_true', default=True,
                       help='Save dataset features analysis')
    parser.add_argument('--save-decision', action='store_true', default=True,
                       help='Save decision results')
    
    args = parser.parse_args()
    
    # Create intelligent coordinator
    coordinator = IntelligentCoordinator(
        openai_api_key=args.openai_api_key,
        config_file=args.config_file
    )
    
    # Coordinate agents
    agent_configs = coordinator.coordinate_agents(
        ratings_file=args.ratings_file,
        kg_file=args.kg_file,
        dataset_name=args.dataset_name,
        save_features=args.save_features,
        save_decision=args.save_decision
    )
    
    # Generate execution script
    coordinator.generate_execution_script(agent_configs, args.output_script)
    
    print(f"\nCoordination completed successfully!")
    print(f"Agent configurations: {agent_configs}")
    print(f"Execution script: {args.output_script}")


if __name__ == "__main__":
    main()

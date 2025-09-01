#!/usr/bin/env python3
"""
Main entry point for the Deep Reinforcement Learning Heuristic (DRLH) PDPTW solver.

This script provides a command-line interface to train and test the DRL agent
on Pickup and Delivery Problem with Time Windows (PDPTW) instances.

Usage Examples:
    # Train a new agent (requires dataset path)
    python main.py --mode train --episodes 1000 --dataset data/pdptw/pdptw-training-192

    # Train with time limit (useful for development)
    python main.py --mode train --time-limit 30 --dataset data/pdptw/pdptw-training-96

    # Test a trained agent
    python main.py --mode test --model logs/pdptw_experiment_1/models/checkpoint_1000.pt --dataset data/pdptw/pdptw-test

    # Solve specific instances
    python main.py --mode solve --dataset data/pdptw/pdptw-test --instances 5

NOTE: Unlike CVRP, PDPTW requires actual instance files for training and testing
      since generate_instance() is not implemented for PDPTW.
"""

import os
import argparse
import sys
import time
from datetime import datetime
import numpy as np
import torch

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import DRL_Agent
from problems.pdptw import PDPTW


class PDPTWConfig:
    """Configuration class to hold all PDPTW solver parameters."""
    
    def __init__(self, args):
        # Core parameters
        self.mode = args.mode
        self.dataset = args.dataset
        self.episodes = args.episodes
        self.instances = args.instances
        self.time_limit = args.time_limit  # Time limit in minutes
        
        # Model parameters
        self.model_path = args.model_path
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        
        # PDPTW specific parameters
        self.T_f = args.T_f
        self.state_rep = args.state_rep
        self.reward_func = args.reward_func
        
        # Logging and output
        self.logdir = args.logdir
        self.save_model = args.save_model
        self.save_interval = args.save_interval
        self.verbose = args.verbose
        
        # Device
        self.use_cuda = args.cuda and torch.cuda.is_available()


class PDPTWSolver:
    """Main PDPTW solver class that handles training, testing, and solving."""
    
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for logging and model saving."""
        if self.config.logdir:
            os.makedirs(self.config.logdir, exist_ok=True)
            os.makedirs(os.path.join(self.config.logdir, "models"), exist_ok=True)
    
    def create_problem_from_file(self, instance_file):
        """Load a PDPTW problem instance from a file."""
        if not os.path.exists(instance_file):
            raise FileNotFoundError(f"Instance file not found: {instance_file}")
            
        # Create a placeholder PDPTW instance and load the specific instance
        pdptw = PDPTW(
            n_nodes=1,  # Will be overridden by loaded instance
            n_calls=1,  # Will be overridden by loaded instance 
            n_vehicles=1,  # Will be overridden by loaded instance
            calls=[],
            vehicles=[],
            vehicles_compatibility=[],
            calls_compatibility=[],
            dist_matrix=[],
            cost_matrix=[],
            wait_times=[],
            toll_costs=[],
            T_f=self.config.T_f
        )
        
        # Load the actual instance data
        instance_data = pdptw.load_instance(instance_file, T_f=self.config.T_f)
        
        # Create the proper PDPTW instance with loaded data
        pdptw = PDPTW(
            n_nodes=instance_data.n_nodes,
            n_calls=instance_data.n_calls,
            n_vehicles=instance_data.n_vehicles,
            calls=instance_data.calls,
            vehicles=instance_data.vehicles,
            vehicles_compatibility=instance_data.vehicles_compatibility,
            calls_compatibility=instance_data.calls_compatibility,
            dist_matrix=instance_data.dist_matrix,
            cost_matrix=instance_data.cost_matrix,
            wait_times=instance_data.wait_times,
            toll_costs=instance_data.toll_costs,
            T_f=self.config.T_f,
            state_rep=self.config.state_rep,
            reward_func=self.config.reward_func
        )
        
        # Calculate call difficulty after the instance is properly initialized
        from utils.utils.utils import pdptw_calculate_call_difficulty
        pdptw.call_difficulty = pdptw_calculate_call_difficulty(pdptw)
        
        return pdptw
    
    def get_instance_files(self, dataset_path, max_files=None):
        """Get list of instance files from dataset directory."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
            
        # Get all .txt files in the directory
        instance_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
        instance_files.sort()  # Sort for consistent ordering
        
        if max_files:
            instance_files = instance_files[:max_files]
            
        if not instance_files:
            raise ValueError(f"No .txt instance files found in {dataset_path}")
            
        return [os.path.join(dataset_path, f) for f in instance_files]
    
    def train(self):
        """Train the DRL agent on PDPTW instances."""
        print("🚛 Training PDPTW Agent")
        print("=" * 50)
        
        if not self.config.dataset:
            raise ValueError("Training requires --dataset path to PDPTW instance files")
        
        # Get training instance files
        instance_files = self.get_instance_files(self.config.dataset, self.config.instances)
        print(f"Training on {len(instance_files)} instance files from {self.config.dataset}")
        
        # Load first instance to get problem dimensions for agent initialization
        first_instance = self.create_problem_from_file(instance_files[0])
        print(f"Problem dimensions: {first_instance.n_nodes} nodes, {first_instance.n_calls} calls, {first_instance.n_vehicles} vehicles")
        
        # Create agent
        agent = DRL_Agent(
            problem=first_instance,
            alpha=self.config.learning_rate,
            gamma=self.config.gamma,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            logdir=self.config.logdir,
            use_cuda=self.config.use_cuda
        )
        
        # Training loop
        start_time = time.time()
        time_limit_seconds = self.config.time_limit * 60 if self.config.time_limit else float('inf')
        
        episode = 0
        while episode < self.config.episodes:
            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit_seconds:
                print(f"\n⏰ Time limit reached ({self.config.time_limit} minutes)")
                break
                
            # Select instance file (cycle through available files)
            instance_file = instance_files[episode % len(instance_files)]
            
            # Load instance
            problem = self.create_problem_from_file(instance_file)
            
            if self.config.verbose and episode % 10 == 0:
                remaining_time = (time_limit_seconds - elapsed_time) / 60 if self.config.time_limit else float('inf')
                print(f"Episode {episode + 1}/{self.config.episodes} - Instance: {os.path.basename(instance_file)} - Time remaining: {remaining_time:.1f}min")
            
            # Train on this instance
            agent.problem = problem  # Update agent's problem
            agent.train(
                max_samples_train=1,  # One episode per instance
                learning_rate=self.config.learning_rate,
                logging=False,  # Disable logging to avoid TensorBoard issues
                save_model=self.config.save_model and (episode + 1) % self.config.save_interval == 0,
                verbose=self.config.verbose,
                print_every_n_step=50
            )
            
            episode += 1
            
            # Save model periodically and at end
            if self.config.save_model and (episode % self.config.save_interval == 0 or episode == self.config.episodes):
                # Ensure models directory exists
                models_dir = os.path.join(self.config.logdir, "models")
                os.makedirs(models_dir, exist_ok=True)
                
                model_path = os.path.join(models_dir, f"checkpoint_{episode}.pt")
                agent.save_model(model_path)
                if self.config.verbose:
                    print(f"Model saved to {model_path}")
        
        # Save final model
        if self.config.save_model:
            # Ensure models directory exists
            models_dir = os.path.join(self.config.logdir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, f"final_model.pt")
            agent.save_model(model_path)
            print(f"Final model saved to {model_path}")
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Episodes completed: {episode}")
        if episode > 0:
            print(f"Average time per episode: {training_time/episode:.2f} seconds")
    
    def generate_results_summary_table(self, results, total_time, model_path, dataset_path):
        """Generate a formatted results summary table as text."""
        avg_cost = np.mean([r['cost'] for r in results])
        avg_time = total_time / len(results)
        
        # Create the summary table
        summary_lines = []
        summary_lines.append("📊 Final Results Summary")
        summary_lines.append("=" * 100)
        summary_lines.append("")
        summary_lines.append(f"Model: {model_path}")
        summary_lines.append(f"Dataset: {dataset_path}")
        summary_lines.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Calculate column widths for proper alignment
        max_instance_width = max(len(result['instance']) for result in results)
        max_instance_width = max(max_instance_width, len("Instance"))  # At least as wide as header
        
        max_cost_width = max(len(f"{result['cost']:,.2f}") for result in results)
        max_cost_width = max(max_cost_width, len("Cost"))  # At least as wide as header
        
        max_time_width = max(len(f"{result['time']:.2f}s") for result in results)
        max_time_width = max(max_time_width, len("Time (seconds)"))  # At least as wide as header
        
        # Calculate problem size width
        problem_sizes = []
        for result in results:
            instance_name = result['instance']
            problem_size = "N/A"
            if "Call_" in instance_name and "_Vehicle_" in instance_name:
                parts = instance_name.replace(".txt", "").split("_")
                if len(parts) >= 4:
                    calls = parts[1]
                    vehicles = parts[3]
                    problem_size = f"{calls} calls, {vehicles} vehicles"
            problem_sizes.append(problem_size)
        
        max_size_width = max(len(size) for size in problem_sizes)
        max_size_width = max(max_size_width, len("Problem Size"))  # At least as wide as header
        
        # Add some padding
        instance_width = max_instance_width + 2
        cost_width = max_cost_width + 2
        time_width = max_time_width + 2
        size_width = max_size_width + 2
        
        # Table header with proper alignment
        header = f"| {'Instance':<{instance_width}} | {'Cost':>{cost_width}} | {'Time (seconds)':>{time_width}} | {'Problem Size':<{size_width}} |"
        separator = f"|{'-' * (instance_width + 2)}|{'-' * (cost_width + 2)}|{'-' * (time_width + 2)}|{'-' * (size_width + 2)}|"
        
        summary_lines.append(header)
        summary_lines.append(separator)
        
        # Table rows with proper alignment
        for i, result in enumerate(results):
            instance_name = result['instance']
            cost = result['cost']
            solve_time = result['time']
            problem_size = problem_sizes[i]
            
            cost_str = f"{cost:,.2f}"
            time_str = f"{solve_time:.2f}s"
            
            row = f"| {instance_name:<{instance_width}} | {cost_str:>{cost_width}} | {time_str:>{time_width}} | {problem_size:<{size_width}} |"
            summary_lines.append(row)
        
        summary_lines.append("")
        summary_lines.append("📈 Overall Performance")
        summary_lines.append("-" * 50)
        summary_lines.append(f"Average cost: {avg_cost:,.2f}")
        summary_lines.append(f"Average time: {avg_time:.2f} seconds per instance")
        summary_lines.append(f"Total instances: {len(results)}")
        summary_lines.append(f"Total test time: {total_time:.2f} seconds")
        
        return "\n".join(summary_lines)

    def test(self):
        """Test a trained DRL agent on PDPTW instances."""
        print("🧪 Testing PDPTW Agent")
        print("=" * 50)
        
        if not self.config.model_path:
            raise ValueError("Testing requires --model-path to trained model")
        if not self.config.dataset:
            raise ValueError("Testing requires --dataset path to PDPTW instance files")
        
        # Get test instance files
        instance_files = self.get_instance_files(self.config.dataset, self.config.instances)
        print(f"Testing on {len(instance_files)} instances from {self.config.dataset}")
        
        # Load first instance to create agent
        first_instance = self.create_problem_from_file(instance_files[0])
        
        # Create agent and load model
        agent = DRL_Agent(
            problem=first_instance,
            logdir=self.config.logdir + "/",  # Ensure trailing slash for agent's file paths
            use_cuda=self.config.use_cuda
        )
        agent.load_model(self.config.model_path)
        print(f"Loaded model from {self.config.model_path}")
        
        # Test on all instances
        results = []
        total_time = 0
        
        for i, instance_file in enumerate(instance_files):
            print(f"\nTesting instance {i+1}/{len(instance_files)}: {os.path.basename(instance_file)}")
            
            # Load instance
            problem = self.create_problem_from_file(instance_file)
            agent.problem = problem
            
            # Solve instance
            start_time = time.time()
            solution, cost = agent.solve(
                problem=problem,
                logging=True,
                verbose=self.config.verbose
            )
            solve_time = time.time() - start_time
            total_time += solve_time
            
            results.append({
                'instance': os.path.basename(instance_file),
                'cost': cost,
                'time': solve_time,
                'solution': solution
            })
            
            print(f"Cost: {cost:.2f}, Time: {solve_time:.2f}s")
            
            # Note: All original output files (Result.txt, ACTION_SEQUENCE.txt, 
            # DIST_SEQUENCE.txt, MIN_DIST_SEQUENCE.txt, TIME.txt, BEST_SOL.txt) 
            # are automatically generated by agent.solve() when logging=True
        
        # Print summary
        print(f"\n📊 Test Results Summary")
        print("-" * 30)
        avg_cost = np.mean([r['cost'] for r in results])
        avg_time = total_time / len(results)
        print(f"Average cost: {avg_cost:.2f}")
        print(f"Average time: {avg_time:.2f} seconds")
        print(f"Total instances: {len(results)}")
        
        # Save results summary to file
        summary_table = self.generate_results_summary_table(results, total_time, self.config.model_path, self.config.dataset)
        
        # Create output filename based on model and dataset
        model_name = os.path.splitext(os.path.basename(self.config.model_path))[0]
        dataset_name = os.path.basename(self.config.dataset)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"test_results_{model_name}_{dataset_name}_{timestamp}.txt"
        output_path = os.path.join(self.config.logdir, output_filename)
        
        # Ensure log directory exists
        os.makedirs(self.config.logdir, exist_ok=True)
        
        # Write summary to file
        with open(output_path, 'w') as f:
            f.write(summary_table)
        
        print(f"\n💾 Results summary saved to: {output_path}")
    
    def solve(self):
        """Solve specific PDPTW instances without training."""
        print("🔧 Solving PDPTW Instances")
        print("=" * 50)
        
        if not self.config.dataset:
            raise ValueError("Solving requires --dataset path to PDPTW instance files")
        
        # Get instance files
        instance_files = self.get_instance_files(self.config.dataset, self.config.instances)
        print(f"Solving {len(instance_files)} instances from {self.config.dataset}")
        
        for i, instance_file in enumerate(instance_files):
            print(f"\n🎯 Solving instance {i+1}/{len(instance_files)}: {os.path.basename(instance_file)}")
            
            # Create problem
            problem = self.create_problem_from_file(instance_file)
            
            # Create agent
            agent = DRL_Agent(
                problem=problem,
                alpha=0.001,
                use_cuda=self.config.use_cuda
            )
            
            # Quick training and solve
            print("Quick training...")
            agent.train(
                max_samples_train=5,
                learning_rate=0.001,
                logging=False,
                save_model=False,
                verbose=False
            )
            
            print("Solving...")
            solution, cost = agent.solve(
                problem=problem,
                logging=False,
                verbose=True
            )
            
            print(f"✅ Solution found with cost: {cost:.2f}")


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning Heuristic for PDPTW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on PDPTW dataset
  python main.py --mode train --episodes 500 --dataset data/pdptw/pdptw-training-192
  
  # Train with 30-minute time limit (useful for development)
  python main.py --mode train --time-limit 30 --dataset data/pdptw/pdptw-training-96
  
  # Test trained model
  python main.py --mode test --model logs/pdptw_exp/models/final_model.pt --dataset data/pdptw/pdptw-test
  
  # Solve specific instances
  python main.py --mode solve --dataset data/pdptw/pdptw-test --instances 5
        """
    )
    
    # Core arguments
    parser.add_argument('--mode', choices=['train', 'test', 'solve'], required=True,
                       help='Mode: train new agent, test existing model, or solve instances')
    parser.add_argument('--dataset', required=True,
                       help='Path to directory containing PDPTW instance files (.txt)')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--instances', type=int, default=None,
                       help='Maximum number of instances to use (default: all)')
    parser.add_argument('--time-limit', dest='time_limit', type=float, default=None,
                       help='Time limit for training in minutes (default: no limit)')
    
    # Model parameters
    parser.add_argument('--model-path', dest='model_path',
                       help='Path to saved model (required for test mode)')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.0003,
                       help='Learning rate (default: 0.0003)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--epochs', dest='n_epochs', type=int, default=10,
                       help='Number of epochs per update (default: 10)')
    
    # PDPTW specific parameters
    parser.add_argument('--T-f', dest='T_f', type=float, default=0.05,
                       help='Final temperature for simulated annealing (default: 0.05)')
    parser.add_argument('--state-rep', dest='state_rep', 
                       default="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",
                       help='State representation string')
    parser.add_argument('--reward-func', dest='reward_func', default="5310",
                       help='Reward function identifier (default: 5310)')
    
    # Output and logging
    parser.add_argument('--logdir', default=f"logs/pdptw_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Directory for logs and models')
    parser.add_argument('--save-model', dest='save_model', action='store_true', default=True,
                       help='Save model during training (default: True)')
    parser.add_argument('--save-interval', dest='save_interval', type=int, default=50,
                       help='Save model every N episodes (default: 50)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    # Hardware
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available (default: True)')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = PDPTWConfig(args)
    
    # Validate arguments
    if config.mode == 'test' and not config.model_path:
        parser.error("Test mode requires --model-path")
    
    print("🚛 DRLH PDPTW Solver")
    print("=" * 40)
    print(f"Mode: {config.mode}")
    print(f"Dataset: {config.dataset}")
    print(f"Device: {'CUDA' if config.use_cuda else 'CPU'}")
    print(f"Log directory: {config.logdir}")
    
    try:
        # Create solver and run
        solver = PDPTWSolver(config)
        
        if config.mode == 'train':
            solver.train()
        elif config.mode == 'test':
            solver.test()
        elif config.mode == 'solve':
            solver.solve()
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

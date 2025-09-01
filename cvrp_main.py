#!/usr/bin/env python3
"""
Main entry point for the Deep Reinforcement Learning Heuristic (DRLH) CVRP solver.

This script provides a command-line interface to train and test the DRL agent
on Capacitated Vehicle Routing Problem (CVRP) instances.

Usage Examples:
    # Train a new agent
    python cvrp_main.py --mode train --episodes 1000 --dataset data/cvrp/cvrp_50_training

    # Test a trained agent
    python cvrp_main.py --mode test --model logs/experiment_1/models/checkpoint_1000.pt --dataset data/cvrp/cvrp_50_testing

    # Generate and solve random instances
    python cvrp_main.py --mode solve --size 50 --instances 10
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

from DRLH.agents import DRL_Agent
from DRLH.problems.cvrp import CVRP


class CVRPSolver:
    """Main class for CVRP solving using Deep Reinforcement Learning."""
    
    def __init__(self, config):
        """Initialize the CVRP solver with configuration."""
        self.config = config
        self.setup_device()
        self.setup_logging()
        
    def setup_device(self):
        """Set up CUDA device if available."""
        self.use_cuda = torch.cuda.is_available() and self.config.cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        print(f"Using device: {self.device}")
        
    def setup_logging(self):
        """Set up logging directory."""
        if self.config.logdir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.logdir = f"logs/experiment_{timestamp}"
        
        if not os.path.exists(self.config.logdir):
            os.makedirs(self.config.logdir)
        print(f"Logging to: {self.config.logdir}")
    
    def create_problem(self):
        """Create CVRP problem environment."""
        problem_config = {
            'size': self.config.size,
            'max_steps': self.config.max_steps,
            'T_0': self.config.T_0,
            'T_f': self.config.T_f,
            'cs': self.config.cs,
            'dataset': self.config.dataset,
            'state_rep': self.config.state_representation,
            'reward_func': self.config.reward_function,
            'acceptance_func': self.config.acceptance_function,
            'warmup_steps': self.config.warmup_steps,
            'pos_deltas_target': self.config.pos_deltas_target,
            'n_iterations_per_instance': self.config.n_iterations_per_instance,
            'num_cold_start_solves': self.config.num_cold_start_solves
        }
        
        return CVRP(**problem_config)
    
    def create_agent(self, problem):
        """Create DRL agent."""
        agent_config = {
            'gamma': self.config.gamma,
            'alpha': self.config.learning_rate,
            'gae_lambda': self.config.gae_lambda,
            'policy_clip': self.config.policy_clip,
            'batch_size': self.config.batch_size,
            'n_epochs': self.config.n_epochs,
            'logdir': self.config.logdir,
            'normalization_func': self.config.normalization,
            'use_cuda': self.use_cuda,
            'n_steps_look_into_future': self.config.n_steps_lookahead,
            'last_100k_size': self.config.last_100k_size
        }
        
        return DRL_Agent(problem, **agent_config)
    
    def train(self):
        """Train the DRL agent."""
        print("=== Training Mode ===")
        print(f"Training episodes: {self.config.episodes}")
        print(f"Problem size: {self.config.size}")
        print(f"Dataset: {self.config.dataset}")
        print(f"State representation: {self.config.state_representation}")
        print(f"Reward function: {self.config.reward_function}")
        
        # Create problem and agent
        problem = self.create_problem()
        agent = self.create_agent(problem)
        
        # Set random seeds for reproducibility
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            agent.seed(self.config.seed)
        
        # Start training
        start_time = time.time()
        agent.train(
            max_samples_train=self.config.episodes,
            learning_rate=self.config.learning_rate,
            logging=True,
            log_single_interval=self.config.log_interval,
            save_model=True,
            save_model_interval=self.config.save_interval,
            baseline_results=self.config.baseline_results,
            resume=self.config.resume,
            verbose=self.config.verbose,
            print_every_n_step=self.config.print_interval
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Average time per episode: {training_time/self.config.episodes:.2f} seconds")
    
    def test(self):
        """Test a trained agent."""
        print("=== Testing Mode ===")
        print(f"Model path: {self.config.model_path}")
        print(f"Test dataset: {self.config.dataset}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        # Create problem and agent
        problem = self.create_problem()
        agent = self.create_agent(problem)
        
        # Load trained model
        model_dir = os.path.dirname(self.config.model_path)
        model_file = os.path.basename(self.config.model_path)
        epoch = int(model_file.split('_')[-1].split('.')[0])
        
        agent.load_model(logdir=model_dir, i=epoch)
        
        # Run testing
        start_time = time.time()
        agent.solve(
            problem=problem,
            logging=True,
            baseline_results=self.config.baseline_results,
            log_single_interval=self.config.log_interval,
            verbose=self.config.verbose
        )
        
        testing_time = time.time() - start_time
        print(f"\nTesting completed in {testing_time:.2f} seconds")
    
    def solve_random(self):
        """Solve randomly generated instances."""
        print("=== Solve Random Instances Mode ===")
        print(f"Problem size: {self.config.size}")
        print(f"Number of instances: {self.config.instances}")
        
        # Create problem without dataset (generates random instances)
        problem_config = {
            'size': self.config.size,
            'max_steps': self.config.max_steps,
            'T_0': self.config.T_0,
            'T_f': self.config.T_f,
            'cs': self.config.cs,
            'state_rep': self.config.state_representation,
            'reward_func': self.config.reward_function,
            'acceptance_func': self.config.acceptance_function,
            'num_cold_start_solves': self.config.instances
        }
        
        problem = CVRP(**problem_config)
        
        if self.config.model_path:
            # Use trained agent
            agent = self.create_agent(problem)
            model_dir = os.path.dirname(self.config.model_path)
            model_file = os.path.basename(self.config.model_path)
            epoch = int(model_file.split('_')[-1].split('.')[0])
            agent.load_model(logdir=model_dir, i=epoch)
        else:
            # Train agent on the fly
            agent = self.create_agent(problem)
            print("No model specified, training agent first...")
            agent.train(
                max_samples_train=min(100, self.config.instances),
                learning_rate=self.config.learning_rate,
                logging=False,
                verbose=False
            )
        
        # Solve instances
        start_time = time.time()
        agent.solve(
            problem=problem,
            logging=True,
            verbose=self.config.verbose
        )
        
        solving_time = time.time() - start_time
        print(f"\nSolving completed in {solving_time:.2f} seconds")
        print(f"Average time per instance: {solving_time/self.config.instances:.2f} seconds")


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Deep Reinforcement Learning Heuristic (DRLH) CVRP Solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main execution mode
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'solve'], 
                       required=True, help='Execution mode')
    
    # Problem configuration
    parser.add_argument('--size', type=int, default=50, 
                       help='Problem size (number of customers)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset directory')
    parser.add_argument('--instances', type=int, default=10,
                       help='Number of random instances to solve (solve mode)')
    
    # Agent configuration
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                       help='Learning rate for neural networks')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--policy-clip', type=float, default=0.2,
                       help='PPO policy clipping parameter')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of PPO epochs per update')
    parser.add_argument('--n-steps-lookahead', type=int, default=10,
                       help='Number of steps to look ahead for advantage calculation')
    
    # Problem environment configuration
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--T-0', type=float, default=None,
                       help='Initial temperature for simulated annealing')
    parser.add_argument('--T-f', type=float, default=0.05,
                       help='Final temperature for simulated annealing')
    parser.add_argument('--cs', type=float, default=None,
                       help='Cooling schedule parameter')
    parser.add_argument('--warmup-steps', type=int, default=None,
                       help='Number of warmup steps for dynamic temperature')
    parser.add_argument('--pos-deltas-target', type=int, default=None,
                       help='Target number of positive deltas for warmup')
    
    # State representation and reward function
    parser.add_argument('--state-representation', type=str, 
                       default='reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen',
                       choices=[
                           'reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen',
                           'reduced_dist___dist_from_min___dist___min_dist___no_improvement___index_step___was_changed___unseen',
                           'reduced_dist___dist_from_min___temp___cs___no_improvement___index_step___was_changed___unseen',
                           'reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen'
                       ],
                       help='State representation function')
    parser.add_argument('--reward-function', type=str, default='5310',
                       choices=['5310', '10310', 'pm', 'pzm', 'delta_change', 
                               'delta_change_scaled', 'new_best', 'new_best_p1', 'min_distance'],
                       help='Reward function')
    parser.add_argument('--acceptance-function', type=str, default='simulated_annealing_ac',
                       choices=['simulated_annealing_ac', 'record_to_record_ac'],
                       help='Acceptance criteria function')
    
    # Normalization
    parser.add_argument('--normalization', type=str, default='last_100k_normalize',
                       choices=['max_normalize', 'last_100k_normalize', 'no_normalization'],
                       help='State normalization method')
    parser.add_argument('--last-100k-size', type=int, default=1000000,
                       help='Buffer size for last 100k normalization')
    
    # Training and logging
    parser.add_argument('--logdir', type=str, default=None,
                       help='Logging directory')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=-1,
                       help='Model save interval (-1 for end only)')
    parser.add_argument('--print-interval', type=int, default=100,
                       help='Print progress interval')
    parser.add_argument('--baseline-results', type=str, default=None,
                       help='Path to baseline results for comparison')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    # Execution configuration
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                       help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                       help='Quiet mode')
    
    # Additional parameters
    parser.add_argument('--n-iterations-per-instance', type=int, default=1,
                       help='Number of iterations per instance')
    parser.add_argument('--num-cold-start-solves', type=int, default=1,
                       help='Number of cold start solves')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Print configuration
    print("=== DRLH CVRP Solver ===")
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if args.cuda and torch.cuda.is_available() else 'CPU'}")
    
    # Create solver and run
    solver = CVRPSolver(args)
    
    try:
        if args.mode == 'train':
            solver.train()
        elif args.mode == 'test':
            if args.model_path is None:
                raise ValueError("Model path required for testing mode")
            solver.test()
        elif args.mode == 'solve':
            solver.solve_random()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()

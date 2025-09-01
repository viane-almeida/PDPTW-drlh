#!/usr/bin/env python3
"""
Simple example script demonstrating how to use the DRLH CVRP solver.

This script shows basic usage patterns for training and testing the DRL agent.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DRLH.agents import DRL_Agent
from DRLH.problems.cvrp import CVRP


def example_train_small():
    """Example: Train agent on a small problem."""
    print("=== Training Example ===")
    
    # Create a small CVRP problem (20 customers for faster training)
    problem = CVRP(
        size=20,
        max_steps=500,
        T_f=0.1,
        state_rep="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",
        reward_func="5310",
        num_cold_start_solves=50  # Train on 50 random instances
    )
    
    # Create agent with basic configuration
    agent = DRL_Agent(
        problem=problem,
        alpha=0.001,  # Slightly higher LR for faster learning
        gamma=0.95,
        batch_size=32,
        n_epochs=5,
        logdir="logs/example_small",
        normalization_func="no_normalization",  # Simpler for small problems
        use_cuda=False  # CPU-only for compatibility
    )
    
    print(f"Problem size: {problem.size} customers")
    print(f"Action space: {problem.action_space.n} heuristics")
    print(f"State space: {problem.observation_space.shape}")
    
    # Train for a small number of episodes
    agent.train(
        max_samples_train=50,
        learning_rate=0.001,
        logging=False,  # Disable logging to avoid TensorBoard issues
        save_model=True,
        save_model_interval=25,
        verbose=True,
        print_every_n_step=50
    )
    
    print("Training completed! Model saved to logs/example_small/")


def example_test_dataset():
    """Example: Test agent on dataset instances."""
    print("\n=== Dataset Testing Example ===")
    
    # Check if test dataset exists
    test_dataset = "data/cvrp/cvrp_50_testing"
    if not os.path.exists(test_dataset):
        print(f"Test dataset not found at {test_dataset}")
        print("Generating random instances instead...")
        return example_solve_random()
    
    # Create problem with test dataset
    problem = CVRP(
        size=50,
        dataset=test_dataset,
        max_steps=1000,
        T_f=0.05,
        state_rep="reduced_dist___dist_from_min___dist___min_dist___no_improvement___index_step___was_changed___unseen",
        reward_func="5310"
    )
    
    # Create agent
    agent = DRL_Agent(
        problem=problem,
        logdir="logs/example_test",
        normalization_func="last_100k_normalize"
    )
    
    print(f"Testing on {len(problem)} instances")
    
    # Check if we have a trained model
    model_path = "logs/example_small/models/checkpoint_50.pt"
    if os.path.exists(model_path):
        print("Loading trained model...")
        agent.load_model(logdir="logs/example_small", i=50)
    else:
        print("No trained model found, using random policy...")
    
    # Test on a few instances
    results = []
    for i in range(min(5, len(problem))):  # Test first 5 instances
        print(f"\nSolving instance {i}...")
        observation = problem.reset()
        agent.reset(problem.start_distance, problem.max_steps)
        
        done = False
        steps = 0
        while not done and steps < 100:  # Limit steps for demo
            action, _, _, _ = agent.choose_action(observation)
            observation, reward, done, info = problem.step(action)
            steps += 1
        
        result = info["min_distance"]
        results.append(result)
        print(f"Instance {i}: Best distance = {result:.4f} in {steps} steps")
    
    print(f"\nAverage best distance: {np.mean(results):.4f}")
    print(f"Standard deviation: {np.std(results):.4f}")


def example_solve_random():
    """Example: Solve randomly generated instances."""
    print("\n=== Random Instance Solving Example ===")
    
    # Create problem that generates random instances
    problem = CVRP(
        size=30,
        max_steps=500,
        T_f=0.1,
        state_rep="reduced_dist___dist_from_min___temp___cs___no_improvement___index_step___was_changed___unseen",
        reward_func="delta_change",
        num_cold_start_solves=10
    )
    
    # Create agent
    agent = DRL_Agent(
        problem=problem,
        alpha=0.0005,
        normalization_func="max_normalize",
        logdir="logs/example_random"
    )
    
    print(f"Solving {len(problem)} random instances of size {problem.size}")
    
    # Quick training on first few instances
    print("Quick training...")
    agent.train(
        max_samples_train=5,
        learning_rate=0.0005,
        logging=False,
        save_model=False,
        verbose=False
    )
    
    # Solve all instances
    print("Solving instances...")
    agent.solve(
        problem=problem,
        logging=False,  # Disable logging to avoid TensorBoard issues
        verbose=True
    )
    
    print("Results saved to logs/example_random/")


def compare_heuristics():
    """Example: Compare different heuristic strategies."""
    print("\n=== Heuristic Comparison Example ===")
    
    # Create a simple test instance
    problem = CVRP(
        size=15,
        max_steps=100,
        num_cold_start_solves=1
    )
    
    # Test different reward functions
    reward_functions = ["5310", "delta_change", "new_best"]
    results = {}
    
    for reward_func in reward_functions:
        print(f"\nTesting reward function: {reward_func}")
        
        # Create problem with specific reward function
        test_problem = CVRP(
            size=15,
            max_steps=100,
            reward_func=reward_func,
            num_cold_start_solves=1
        )
        
        # Create agent
        agent = DRL_Agent(
            problem=test_problem,
            alpha=0.002,
            batch_size=16,
            n_epochs=3,
            normalization_func="no_normalization"
        )
        
        # Quick training
        agent.train(
            max_samples_train=10,
            learning_rate=0.002,
            logging=False,
            save_model=False,
            verbose=False
        )
        
        # Test solve
        observation = test_problem.reset()
        agent.reset(test_problem.start_distance, test_problem.max_steps)
        
        best_distance = test_problem.start_distance
        for _ in range(50):
            action, _, _, _ = agent.choose_action(observation)
            observation, reward, done, info = test_problem.step(action)
            if info["min_distance"] < best_distance:
                best_distance = info["min_distance"]
            if done:
                break
        
        results[reward_func] = best_distance
        print(f"Best distance achieved: {best_distance:.4f}")
    
    print("\n=== Comparison Results ===")
    for reward_func, distance in results.items():
        print(f"{reward_func}: {distance:.4f}")


def main():
    """Run all examples."""
    print("DRLH CVRP Solver - Examples")
    print("=" * 40)
    
    try:
        # Run examples
        example_train_small()
        example_test_dataset()
        example_solve_random()
        compare_heuristics()
        
        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        print("\nTo run individual examples:")
        print("python -c \"from example import example_train_small; example_train_small()\"")
        print("python -c \"from example import example_solve_random; example_solve_random()\"")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed and data is available")


if __name__ == "__main__":
    main()

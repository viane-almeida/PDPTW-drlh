#!/usr/bin/env python3
"""
Example script for solving a single CVRP instance with the DRLH solver.

This demonstrates different ways to solve a single CVRP test instance:
1. Using a trained model
2. Training from scratch on a single instance
3. Loading a specific test instance from the dataset

Usage:
    python cvrp_solve_single_instance.py
"""

import sys
import os
sys.path.insert(0, '.')

from DRLH.agents import DRL_Agent
from DRLH.problems.cvrp import CVRP
import numpy as np


def solve_with_trained_model(instance_file=None):
    """Solve using a previously trained model."""
    print("=== Solving with Trained Model ===")
    
    # Create problem environment
    if instance_file:
        problem = CVRP(
            size=20,  # Match the training size for model compatibility
            dataset=None,  # We'll load manually
            max_steps=1000,
            T_f=0.05,
            state_rep="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",  # Match training state
            reward_func="5310"
        )
        
        # Load specific instance
        print(f"Loading instance from: {instance_file}")
        problem.instance = problem.load_instance(instance_file)
    else:
        # Generate a random instance
        problem = CVRP(
            size=20,
            max_steps=500,
            T_f=0.05,
            state_rep="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",
            reward_func="5310",
            num_cold_start_solves=1
        )
    
    # Create agent
    agent = DRL_Agent(
        problem=problem,
        alpha=0.0003,
        normalization_func="last_100k_normalize",
        use_cuda=False
    )
    
    # Try to load a trained model
    model_path = "logs/example_small/models/checkpoint_50.pt"
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        agent.load_model(logdir="logs/example_small", i=50)
    else:
        print("No trained model found, using random policy")
    
    # Solve the instance
    print(f"Problem size: {problem.size} customers")
    print("Starting optimization...")
    
    observation = problem.reset()
    agent.reset(problem.start_distance, problem.max_steps)
    
    done = False
    step = 0
    improvements = []
    
    print(f"Initial solution distance: {problem.start_distance:.4f}")
    
    while not done and step < 500:  # Limit steps for demo
        action, prob, val, dist_ = agent.choose_action(observation)
        observation_, reward, done, info = problem.step(action)
        
        step += 1
        observation = observation_
        
        # Record improvements
        improvements.append(info["min_distance"])
        
        # Print progress every 50 steps
        if step % 50 == 0:
            print(f"Step {step:3d}: Best distance = {info['min_distance']:.4f}, "
                  f"Improvement = {problem.start_distance - info['min_distance']:.4f}")
    
    print(f"\n=== Final Result ===")
    print(f"Initial distance: {problem.start_distance:.4f}")
    print(f"Final best distance: {info['min_distance']:.4f}")
    print(f"Total improvement: {problem.start_distance - info['min_distance']:.4f}")
    print(f"Improvement percentage: {100 * (problem.start_distance - info['min_distance']) / problem.start_distance:.1f}%")
    print(f"Best solution: {info['best_solution']}")
    
    return info


def solve_from_scratch():
    """Train an agent from scratch on a single instance."""
    print("\n=== Training from Scratch ===")
    
    # Create a small problem for quick training
    problem = CVRP(
        size=15,  # Smaller for faster training
        max_steps=300,
        T_f=0.1,
        state_rep="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",
        reward_func="5310",
        num_cold_start_solves=1
    )
    
    # Create agent
    agent = DRL_Agent(
        problem=problem,
        alpha=0.002,  # Higher learning rate for faster learning
        batch_size=16,
        n_epochs=3,
        normalization_func="no_normalization",
        use_cuda=False
    )
    
    print(f"Problem size: {problem.size} customers")
    print("Training agent on this specific instance...")
    
    # Quick training (just a few episodes)
    agent.train(
        max_samples_train=5,
        learning_rate=0.002,
        logging=False,
        save_model=False,
        verbose=False
    )
    
    # Now solve with the trained agent
    observation = problem.reset()
    agent.reset(problem.start_distance, problem.max_steps)
    
    done = False
    step = 0
    
    print(f"Initial solution distance: {problem.start_distance:.4f}")
    print("Solving with trained agent...")
    
    while not done and step < 200:
        action, prob, val, dist_ = agent.choose_action(observation)
        observation_, reward, done, info = problem.step(action)
        
        step += 1
        observation = observation_
        
        if step % 25 == 0:
            print(f"Step {step:3d}: Best distance = {info['min_distance']:.4f}")
    
    print(f"\n=== Training Result ===")
    print(f"Initial distance: {problem.start_distance:.4f}")
    print(f"Final best distance: {info['min_distance']:.4f}")
    print(f"Total improvement: {problem.start_distance - info['min_distance']:.4f}")
    print(f"Improvement percentage: {100 * (problem.start_distance - info['min_distance']) / problem.start_distance:.1f}%")
    
    return info


def solve_test_instance():
    """Solve a specific instance from the test dataset."""
    print("\n=== Solving Test Dataset Instance ===")
    
    # Check if test dataset exists
    test_instance = "data/cvrp/cvrp_50_testing/instance_0.txt"
    if not os.path.exists(test_instance):
        print(f"Test instance not found: {test_instance}")
        print("Generating random instance instead...")
        return solve_with_trained_model()
    
    return solve_with_trained_model(test_instance)


def interactive_solve():
    """Interactive mode to choose what to solve."""
    print("\n" + "="*50)
    print("DRLH CVRP Single Instance Solver")
    print("="*50)
    
    print("\nChoose solving method:")
    print("1. Use trained model on random instance")
    print("2. Train from scratch on small instance")
    print("3. Solve test dataset instance")
    print("4. Solve all methods")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        solve_with_trained_model()
    elif choice == "2":
        solve_from_scratch()
    elif choice == "3":
        solve_test_instance()
    elif choice == "4":
        solve_with_trained_model()
        solve_from_scratch()
        solve_test_instance()
    else:
        print("Invalid choice, running all methods...")
        solve_with_trained_model()
        solve_from_scratch()


def analyze_solution(info):
    """Analyze the solution quality and provide insights."""
    print(f"\n=== Solution Analysis ===")
    
    # Basic metrics
    start_dist = info["start_distance"]
    final_dist = info["min_distance"]
    improvement = start_dist - final_dist
    improvement_pct = 100 * improvement / start_dist
    
    print(f"Solution Quality:")
    print(f"  Initial distance: {start_dist:.4f}")
    print(f"  Final distance: {final_dist:.4f}")
    print(f"  Absolute improvement: {improvement:.4f}")
    print(f"  Relative improvement: {improvement_pct:.1f}%")
    
    # Performance assessment
    if improvement_pct > 50:
        quality = "Excellent"
    elif improvement_pct > 30:
        quality = "Good"
    elif improvement_pct > 10:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"  Quality assessment: {quality}")
    
    # Additional metrics
    print(f"\nOptimization Process:")
    print(f"  Steps to best solution: {info['min_step']}")
    print(f"  Total improvements: {info['num_improvements']}")
    print(f"  New best solutions found: {info['num_best_improvements']}")
    print(f"  Unique solutions explored: {info['num_seen_solutions']}")
    
    # Most used heuristics
    print(f"\nHeuristic Usage:")
    action_counts = info["action_counter"]
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (action_name, count) in enumerate(sorted_actions[:5]):
        if count > 0:
            print(f"  {i+1}. {action_name}: {count} times")


if __name__ == "__main__":
    # For direct execution, run interactive mode
    interactive_solve()

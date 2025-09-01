#!/usr/bin/env python3
"""
Solve a custom CVRP instance with your own data.

Usage:
    python cvrp_solve_custom.py
"""

import sys
import os
sys.path.insert(0, '.')

from DRLH.agents import DRL_Agent
from DRLH.problems.cvrp import CVRP
import json


def create_custom_instance():
    """Create a custom instance with your own customer locations and demands."""
    
    # Example: 10 customers with custom locations and demands
    instance_data = {
        "size": 10,
        "locations": [
            [0.5, 0.5],    # Depot at center
            [0.1, 0.1],    # Customer 1
            [0.9, 0.1],    # Customer 2  
            [0.1, 0.9],    # Customer 3
            [0.9, 0.9],    # Customer 4
            [0.3, 0.7],    # Customer 5
            [0.7, 0.3],    # Customer 6
            [0.2, 0.5],    # Customer 7
            [0.8, 0.5],    # Customer 8
            [0.5, 0.2],    # Customer 9
            [0.5, 0.8],    # Customer 10
        ],
        "demand": [0, 5, 3, 4, 6, 2, 7, 3, 4, 5, 3],  # Depot has 0 demand
        "max_capacity": 15.0,  # Vehicle capacity
        "T_f": 0.05
    }
    
    # Save to file
    with open("my_instance.json", "w") as f:
        json.dump(instance_data, f, indent=2)
    
    print("✅ Custom instance created: my_instance.json")
    return "my_instance.json"


def solve_custom_instance(instance_file):
    """Solve your custom instance."""
    
    print(f"🎯 Solving custom instance: {instance_file}")
    
    # Create problem
    problem = CVRP(
        size=10,  # Will be overridden by loaded instance
        max_steps=300,
        T_f=0.05,
        state_rep="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",
        reward_func="5310"
    )
    
    # Load your instance
    problem.instance = problem.load_instance(instance_file)
    
    # Create agent
    agent = DRL_Agent(
        problem=problem,
        alpha=0.001,
        normalization_func="no_normalization",
        use_cuda=False
    )
    
    # Try to load trained model
    model_path = "logs/example_small/models/checkpoint_50.pt"
    if os.path.exists(model_path):
        try:
            agent.load_model(logdir="logs/example_small", i=50)
            print("✅ Using trained model")
        except:
            print("⚠️  Model size mismatch, using random policy")
    
    # Solve
    observation = problem.reset()
    agent.reset(problem.start_distance, problem.max_steps)
    
    print(f"🚗 Initial distance: {problem.start_distance:.4f}")
    
    done = False
    step = 0
    
    while not done and step < 150:
        action, _, _, _ = agent.choose_action(observation)
        observation, reward, done, info = problem.step(action)
        step += 1
        
        if step % 25 == 0:
            improvement = problem.start_distance - info["min_distance"]
            print(f"Step {step}: Best = {info['min_distance']:.4f}, "
                  f"Improved = {improvement:.4f}")
    
    # Results
    improvement = problem.start_distance - info["min_distance"]
    improvement_pct = 100 * improvement / problem.start_distance
    
    print(f"\n🎉 RESULTS:")
    print(f"Initial: {problem.start_distance:.4f}")
    print(f"Final: {info['min_distance']:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
    print(f"Best routes: {info['best_solution']}")


if __name__ == "__main__":
    # Create and solve custom instance
    instance_file = create_custom_instance()
    solve_custom_instance(instance_file)

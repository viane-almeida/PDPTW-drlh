#!/usr/bin/env python3
"""
Simple script to solve a single CVRP instance.

This demonstrates the most straightforward way to solve a single CVRP instance.

Usage:
    python cvrp_solve_simple.py
"""

import sys
import os
sys.path.insert(0, '.')

from DRLH.agents import DRL_Agent
from DRLH.problems.cvrp import CVRP


def solve_single_instance():
    """Solve a single CVRP instance step by step."""
    
    print("🚛 DRLH CVRP Single Instance Solver")
    print("=" * 40)
    
    # Step 1: Create the problem
    print("\n📋 Step 1: Creating CVRP problem...")
    problem = CVRP(
        size=20,                    # 20 customers (manageable size)
        max_steps=500,              # Maximum optimization steps
        T_f=0.05,                   # Final temperature for simulated annealing
        state_rep="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",
        reward_func="5310",         # Reward function encouraging improvements
        num_cold_start_solves=1     # Generate 1 random instance
    )
    
    print(f"✅ Problem created: {problem.size} customers, {problem.action_space.n} heuristics available")
    
    # Step 2: Create the agent
    print("\n🤖 Step 2: Creating DRL agent...")
    agent = DRL_Agent(
        problem=problem,
        alpha=0.001,                # Learning rate
        normalization_func="no_normalization",  # Simple for demo
        use_cuda=False              # CPU-only for compatibility
    )
    
    print(f"✅ Agent created with {problem.observation_space.shape[0]}-dimensional state space")
    
    # Step 3: Try to load a trained model (optional)
    print("\n📁 Step 3: Checking for trained model...")
    model_path = "logs/example_small/models/checkpoint_50.pt"
    if os.path.exists(model_path):
        try:
            print(f"🔄 Loading trained model: {model_path}")
            agent.load_model(logdir="logs/example_small", i=50)
            print("✅ Trained model loaded successfully!")
            trained = True
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            print("🎲 Using random policy instead")
            trained = False
    else:
        print("🎲 No trained model found, using random policy")
        trained = False
    
    # Step 4: Generate/reset the problem instance
    print("\n🎯 Step 4: Generating problem instance...")
    observation = problem.reset()
    agent.reset(problem.start_distance, problem.max_steps)
    
    print(f"✅ Instance generated:")
    print(f"   📍 {problem.size} customers to visit")
    print(f"   🚗 Initial solution distance: {problem.start_distance:.4f}")
    print(f"   🎯 Goal: Minimize total travel distance")
    
    # Step 5: Solve the instance
    print(f"\n🔄 Step 5: Optimizing solution...")
    print("   (Each step selects a heuristic operation to improve the solution)")
    
    done = False
    step = 0
    best_distances = []
    
    while not done and step < 200:  # Limit steps for demo
        # Agent chooses which heuristic to apply
        action, prob, val, dist_ = agent.choose_action(observation)
        
        # Apply the chosen heuristic
        observation_, reward, done, info = problem.step(action)
        
        step += 1
        observation = observation_
        best_distances.append(info["min_distance"])
        
        # Print progress every 25 steps
        if step % 25 == 0:
            improvement = problem.start_distance - info["min_distance"]
            improvement_pct = 100 * improvement / problem.start_distance
            print(f"   Step {step:3d}: Best = {info['min_distance']:.4f}, "
                  f"Improved = {improvement:.4f} ({improvement_pct:.1f}%)")
    
    # Step 6: Show results
    print(f"\n🎉 Step 6: Optimization complete!")
    
    final_distance = info["min_distance"]
    initial_distance = problem.start_distance
    improvement = initial_distance - final_distance
    improvement_pct = 100 * improvement / initial_distance
    
    print(f"\n📊 RESULTS:")
    print(f"   🏁 Initial distance:  {initial_distance:.4f}")
    print(f"   🎯 Final distance:    {final_distance:.4f}")
    print(f"   📈 Improvement:       {improvement:.4f} ({improvement_pct:.1f}%)")
    print(f"   ⚡ Steps taken:       {step}")
    print(f"   🧠 Model used:        {'Trained' if trained else 'Random'}")
    
    # Show solution quality assessment
    if improvement_pct > 50:
        quality = "🏆 Excellent"
    elif improvement_pct > 30:
        quality = "🥈 Good"
    elif improvement_pct > 10:
        quality = "🥉 Fair"
    else:
        quality = "❌ Poor"
    
    print(f"   ⭐ Solution quality:  {quality}")
    
    # Step 7: Show the actual solution
    print(f"\n🗺️  SOLUTION DETAILS:")
    print(f"   📍 Best route found: {info['best_solution']}")
    print(f"   🔄 Total improvements made: {info['num_improvements']}")
    print(f"   ⏰ Step where best was found: {info['min_step']}")
    
    # Show most effective heuristics
    print(f"\n🔧 HEURISTIC USAGE:")
    action_counts = info["action_counter"]
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("   Most used heuristics:")
    for i, (action_name, count) in enumerate(sorted_actions[:3]):
        if count > 0:
            heuristic_name = action_name.replace("cvrp_", "").replace("_and_", " + ")
            print(f"   {i+1}. {heuristic_name}: {count} times")
    
    return info


def explain_the_process():
    """Explain what the algorithm is doing."""
    print("\n" + "="*60)
    print("🧠 HOW THE DRLH ALGORITHM WORKS")
    print("="*60)
    print()
    print("1. 🎯 PROBLEM: Given customers at different locations with demands,")
    print("   find the shortest routes for vehicles to visit all customers.")
    print()
    print("2. 🧠 AGENT: A neural network (PPO) learns to select which")
    print("   optimization heuristic to apply at each step.")
    print()
    print("3. 🔧 HEURISTICS: Classical operations like:")
    print("   • Remove customers from routes")
    print("   • Insert customers in better positions")
    print("   • Swap customers between routes")
    print()
    print("4. 🎯 GOAL: The agent learns which heuristics work best in")
    print("   different situations to minimize total distance.")
    print()
    print("5. 🏆 RESULT: Intelligent combination of AI learning +")
    print("   classical optimization = better solutions!")


if __name__ == "__main__":
    # Solve an instance
    result = solve_single_instance()
    
    # Explain the process
    explain_the_process()
    
    print(f"\n✨ Ready to solve more instances!")
    print(f"💡 Tip: Run this script multiple times to see different random instances")

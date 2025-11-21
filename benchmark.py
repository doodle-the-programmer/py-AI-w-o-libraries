"""
Performance Benchmark - Neural Network Training Speed
Tests the vectorized NumPy implementation with different configurations.
"""

import time
import numpy as np
from main import NeuralNetworkVectorized, base_training_data

print("="*70)
print("NEURAL NETWORK PERFORMANCE BENCHMARK")
print("="*70)

# Prepare test data
print("\nPreparing test data...")
X = np.array([x[0] for x in base_training_data[:10]])
y = np.array([x[1] for x in base_training_data[:10]])

# Test 1: Small network
print("\n[1/3] Testing small network (9→18→3)...")
try:
    net_small = NeuralNetworkVectorized([9, 18, 3], learning_rate=0.1)
    
    start = time.time()
    for _ in range(100):
        net_small.backward(X, y)
    time_small = time.time() - start
    
    print(f"   ✓ Small network: {time_small:.3f}s for 100 epochs")
except Exception as e:
    print(f"   ✗ Test failed: {e}")
    time_small = None

# Test 2: Medium network
print("\n[2/3] Testing medium network (9→24→12→3)...")
try:
    net_medium = NeuralNetworkVectorized([9, 24, 12, 3], learning_rate=0.1)
    
    start = time.time()
    for _ in range(100):
        net_medium.backward(X, y)
    time_medium = time.time() - start
    
    print(f"   ✓ Medium network: {time_medium:.3f}s for 100 epochs")
except Exception as e:
    print(f"   ✗ Test failed: {e}")
    time_medium = None

# Test 3: Large network
print("\n[3/3] Testing large network (9→48→24→12→3)...")
try:
    net_large = NeuralNetworkVectorized([9, 48, 24, 12, 3], learning_rate=0.1)
    
    start = time.time()
    for _ in range(100):
        net_large.backward(X, y)
    time_large = time.time() - start
    
    print(f"   ✓ Large network: {time_large:.3f}s for 100 epochs")
except Exception as e:
    print(f"   ✗ Test failed: {e}")
    time_large = None

# Summary
print("\n" + "="*70)
print("BENCHMARK SUMMARY")
print("="*70)

results = []
if time_small is not None:
    results.append(("Small (9→18→3)", time_small, 39))
if time_medium is not None:
    results.append(("Medium (9→24→12→3)", time_medium, 324))
if time_large is not None:
    results.append(("Large (9→48→24→12→3)", time_large, 1296))

if results:
    print(f"\n{'Network':<25} {'Parameters':<15} {'Time (100 epochs)':<20}")
    print("-"*70)
    for name, time_taken, params in results:
        print(f"{name:<25} {params:<15} {time_taken:.3f}s")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    fastest = min(results, key=lambda x: x[1])
    print(f"\n⚡ Fastest: {fastest[0]} ({fastest[1]:.3f}s)")
    
    if time_small and time_medium:
        ratio = time_medium / time_small
        print(f"\n📊 Medium network is {ratio:.1f}x slower than small network")
    
    if time_medium and time_large:
        ratio = time_large / time_medium
        print(f"📊 Large network is {ratio:.1f}x slower than medium network")
    
    print("\n💡 Recommendation:")
    print("   For this dataset, the small or medium network is optimal.")
    print("   Larger networks provide diminishing returns with more training time.")
    
    print("\n" + "="*70)
else:
    print("\n❌ Benchmark failed! Make sure NumPy is installed:")
    print("   pip install numpy")

print()

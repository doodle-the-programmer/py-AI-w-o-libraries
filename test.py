#!/usr/bin/env python3
"""
Quick test of the neural network with sample predictions.
Run this to verify the network is working correctly.
"""

from main import *

def test_shapes():
    """Test the network on each shape type."""
    print("="*60)
    print("QUICK NETWORK TEST")
    print("="*60)
    
    # Create and train a small network
    print("\nTraining small network for testing...")
    ensemble = AcceleratedEnsemble(2, [9, 18, 3], learning_rate=0.1)
    ensemble.train(base_training_data, epochs=10000, batch_size=160, validation_split=0.2)
    
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    shape_names = {0: "L", 1: "O", 2: ">"}
    
    # Test cases
    test_cases = [
        ("L-shape", np.array([1, 0, 0, 1, 0, 0, 1, 1, 1]), 0),
        ("O-shape", np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]), 1),
        (">-shape", np.array([1, 0, 0, 0, 1, 0, 1, 0, 0]), 2),
    ]
    
    correct = 0
    for name, grid, expected in test_cases:
        print(f"\nTesting {name}:")
        visualize_grid(grid)
        
        output = ensemble.forward(grid)
        predicted_class = int(np.argmax(output))
        confidence = float(output[predicted_class]) * 100
        
        status = "✓" if predicted_class == expected else "✗"
        print(f"{status} Predicted: {shape_names[predicted_class]} (confidence: {confidence:.1f}%)")
        print(f"  Probabilities: L={float(output[0])*100:.1f}%, O={float(output[1])*100:.1f}%, >={float(output[2])*100:.1f}%")
        
        if predicted_class == expected:
            correct += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.0f}%)")
    print("="*60)
    
    if correct == len(test_cases):
        print("\n🎉 Perfect score! Network is working great!")
    elif correct >= len(test_cases) * 0.6:
        print("\n✓ Network is learning! Try training longer for better accuracy.")
    else:
        print("\n⚠ Network needs more training. This is normal for such a small dataset.")
    
    print()

if __name__ == "__main__":
    test_shapes()

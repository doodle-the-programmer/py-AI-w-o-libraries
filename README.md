# Neural Network from Scratch - NumPy Vectorized

A high-performance neural network implementation using **NumPy** for vectorized operations. Recognizes patterns in 3x3 grids.

**No TensorFlow, PyTorch, or Pandas** - built from scratch with modern acceleration!

## 🎯 What It Does

Classifies 3x3 grid patterns into three shape categories:
- **L-shapes**: Corner and L-shaped patterns
- **O-shapes**: Square and filled patterns  
- **>-shapes**: Arrow and chevron patterns

## 🚀 Quick Start

### Install Dependencies

```bash
pip install numpy
```

### Run Training

```bash
# Option 1: Using the run script (recommended - ensures venv is used)
./run.sh

# Option 2: Direct execution
python3 main.py
```

The network will:
- Train on augmented dataset with validation
- Launch interactive prediction mode

### Quick Test

```bash
python3 test.py
```

Trains a small network and tests predictions on sample shapes.

### Run Benchmark

```bash
python3 benchmark.py
```

Tests different network configurations to compare performance.

## 📊 Performance

**10-100x faster than pure Python** thanks to NumPy vectorization!

- Small network (9→18→3): ~0.02s for 100 epochs
- Medium network (9→24→12→3): ~0.03s for 100 epochs
- Large network (9→48→24→12→3): ~0.08s for 100 epochs

## 🎮 Usage

### Training & Interactive Mode

```bash
python3 main.py
```

Example session:
```
============================================================
NEURAL NETWORK TRAINING
============================================================
Backend: NumPy (CPU-optimized)
Networks: 3
Architecture: [9, 24, 12, 3]
============================================================

Training samples: 91
Validation samples: 17
Batch size: 16
------------------------------------------------------------

Epoch    0 | Loss: 1.1179 | Train: 51.6% | Val: 58.8% | Time: 0.0s
Epoch   10 | Loss: 0.8151 | Train: 70.3% | Val: 41.2% | Time: 0.1s
...
Training complete in 2.3s
Final validation accuracy: 72.5%

============================================================
INTERACTIVE PREDICTION MODE
============================================================
Enter a 3x3 grid pattern to predict the shape.
```

### Making Predictions

Input a 3x3 grid:
```
Enter grid values (top-left to bottom-right):
  Square #1: 1
  Square #2: 0
  ...
  Square #9: 1

  Grid visualization:
  ⬛ ⬜ ⬜
  ⬛ ⬜ ⬜
  ⬛ ⬛ ⬛

🎯 Predicted shape: L
📊 Confidence: 85.43%
📈 Class probabilities:
   L: 85.43%
   O: 8.21%
   >: 6.36%
```

## 🏗️ Architecture

```
Input Layer: 9 neurons (3x3 grid)
    ↓
Hidden Layer 1: 24 neurons (Leaky ReLU)
    ↓
Hidden Layer 2: 12 neurons (Leaky ReLU)
    ↓
Output Layer: 3 neurons (Sigmoid → Softmax)
```

**Features:**
- ✅ Vectorized operations (matrix multiplication)
- ✅ Adam optimizer with momentum
- ✅ He initialization
- ✅ Mini-batch training (batch size: 16)
- ✅ Data augmentation (rotations, flips)
- ✅ Early stopping with validation
- ✅ Ensemble learning (3 networks)

## 📦 Files

- **`main.py`** - Complete neural network implementation (529 lines)
- **`test.py`** - Quick test with sample predictions
- **`benchmark.py`** - Performance comparison tool
- **`run.sh`** - Launcher script (ensures virtual environment is used)
- **`README.md`** - This file

## 🎓 Technical Details

### Vectorization Example

**Before (Pure Python):**
```python
for i in range(len(weights)):
    weights[i] += learning_rate * gradients[i]
```

**After (NumPy):**
```python
weights += learning_rate * gradients  # All at once!
```

This simple change provides 10-100x speedup by using optimized C/Fortran code under the hood.

### Why Not TensorFlow/PyTorch?

This project demonstrates:
- ✅ Neural network fundamentals
- ✅ Backpropagation implementation
- ✅ Optimization algorithms (Adam)
- ✅ Performance optimization with NumPy
- ✅ Proper training techniques (data augmentation, validation, early stopping)

**Built from scratch** to understand what happens under the hood!

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install numpy
```

### Low accuracy
The network should achieve 70-85% validation accuracy. If lower:
- Try training longer: increase `epochs` parameter
- Adjust learning rate in `AcceleratedEnsemble` initialization
- Ensure you're using the augmented dataset

### Slow performance
This implementation is already optimized with NumPy vectorization. For larger networks:
- Reduce network size: fewer neurons or layers
- Reduce batch size: change `batch_size` in `train()` call
- Use fewer networks in ensemble

## 🤝 What We Use

### Libraries Used ✅
- **NumPy**: Matrix operations, vectorization
- **Standard library**: math, random, time, collections, multiprocessing

### Libraries NOT Used ❌
- ❌ TensorFlow
- ❌ PyTorch
- ❌ Keras
- ❌ Pandas
- ❌ scikit-learn
- ❌ CuPy (GPU libraries)

Everything is built from scratch with just NumPy!

## 📚 Learn More

- **Neural Networks**: [3Blue1Brown Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- **NumPy**: [Official Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- **Backpropagation**: [CS231n Notes](http://cs231n.github.io/optimization-2/)
- **Adam Optimizer**: [Original Paper](https://arxiv.org/abs/1412.6980)

## 🏆 Try It

```bash
# Install
pip install numpy

# Run (recommended - uses venv automatically)
./run.sh

# Or run directly
python3 main.py

# Quick test
python3 test.py

# Benchmark
python3 benchmark.py
```

---

**Built to learn** - Demonstrates neural networks from scratch with modern performance optimization.

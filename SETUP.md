# Quick Setup Guide

## For First-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/doodle-the-programmer/py-AI-w-o-libraries.git
cd py-AI-w-o-libraries

# 2. Create virtual environment
python3 -m venv .venv

# 3. Install dependencies
.venv/bin/pip install -r requirements.txt

# OR if you prefer activating:
source .venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
deactivate
```

## Running the Code

```bash
# Option 1: Direct execution (recommended)
./run.sh

# Option 2: With activation
source .venv/bin/activate
python3 main.py
deactivate

# Option 3: Direct venv usage
.venv/bin/python3 main.py
```

## Quick Commands

```bash
# Run tests
.venv/bin/python3 test.py

# Run benchmarks
.venv/bin/python3 benchmark.py

# Update dependencies
.venv/bin/pip install --upgrade -r requirements.txt
```

## Troubleshooting

**No .venv directory?**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Permission denied for run.sh?**
```bash
chmod +x run.sh
```

**Want to reset everything?**
```bash
rm -rf .venv __pycache__
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

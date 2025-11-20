which python3
python3 -V

python3 - << 'EOF'
import sys
print("Python exe:", sys.executable)
try:
    import executorch
    print("executorch imported from:", executorch.__file__)
except ImportError as e:
    print("ImportError:", e)
EOF
import torch, torchvision, numpy as np
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy      :", np.__version__)
print("tensor ok  :", torch.randn(1).dtype)

# which buck2
# buck2 --version
# # (Keep the output—you’ll compare after the switch.)

# /home/yug/.cargo/bin/buck2
# buck2 686538840a321c0e86fd0eeaa12a3e8c976f13cf <build-id>
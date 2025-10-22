import torch
from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True)
model = model.net2
# Remove the last layer
# model = torch.nn.Sequential(*list(model.children())[:])

# write out the model's state_dict keys and their shapes
PATH = "/home/monstalinux/final-project/dual_stream_det_and_cls/classification/ianpan_mammoscreen_net2_architecture.txt"
with open(PATH, "w") as f:
    for key, value in model.state_dict().items():
        f.write(f"{key}: {value.shape}\n")

# Save the modified model
PATH = "/home/monstalinux/final-project/dual_stream_det_and_cls/classification/ianpan_mammoscreen_net2.pth"
torch.save(model, PATH)


print(f"Model saved to {PATH}")
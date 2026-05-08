import torch
try:
    state_dict = torch.load("best_swin_model.pth", map_location="cpu")
    print("Keys in state_dict (first 10):", list(state_dict.keys())[:10])
    if "classifier.weight" in state_dict:
        print("Classifier weight shape:", state_dict["classifier.weight"].shape)
    elif "classifier.0.weight" in state_dict:
        print("Classifier.0 weight shape:", state_dict["classifier.0.weight"].shape)
    elif "head.weight" in state_dict:
        print("Head weight shape:", state_dict["head.weight"].shape)
except Exception as e:
    print(f"Error: {e}")

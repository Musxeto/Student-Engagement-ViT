from transformers import SwinConfig, SwinForImageClassification
import torch

def test_load():
    try:
        print("Loading config...")
        config = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k", num_labels=3)
        print("Initializing model from config (no weights download)...")
        model = SwinForImageClassification(config)
        print("Loading local weights...")
        model.load_state_dict(torch.load("best_swin_model.pth", map_location="cpu"))
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_load()

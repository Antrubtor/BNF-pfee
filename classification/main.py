import os
import sys

# this is necessary to ensure that the current directory is in the Python path, so that relative imports work correctly when running this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
from dataset import load_data
from evaluate import evaluate_model
from model import create_convnext_model
from train import train_model

# ==========================================
# CONFIG
# ==========================================
EPOCHS = 2
BATCH_SIZE = 32  # optimized for GPU with 8GB VRAM, adjust if needed
LEARNING_RATE = 1e-4
WEIGHTS_PATH = "convnext_weights.pth"
RESULTS_TXT = "resultats_evaluation.txt"

# Optional: Limit the number of samples for quick testing
MAX_TRAIN = None
MAX_VAL = None


def main():
    global EPOCHS, MAX_TRAIN, MAX_VAL

    # if fast mode is specified in command line arguments, override the default settings
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["--fast", "fast", "test"]:
        EPOCHS = 2
        MAX_TRAIN = 100
        MAX_VAL = 50
        print(
            "[INFO] Fast mode active (fast) : 2 epochs on reduced sample to verify the pipeline."
        )

    # automatically select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Selected device : {device}")
    if device.type == "cuda":
        print(f"[Hardware] GPU detected : {torch.cuda.get_device_name(0)}")

    # load the data and get the DataLoaders for training and validation
    train_loader, val_loader, class_names = load_data(
        batch_size=BATCH_SIZE, max_train_samples=MAX_TRAIN, max_val_samples=MAX_VAL
    )

    # create convnext model with the number of classes detected in the dataset
    model = create_convnext_model(
        num_classes=len(class_names), pretrained=True, device=device
    )

    # train the model and get the total training time
    total_train_time = train_model(
        model=model,
        train_loader=train_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        save_path=WEIGHTS_PATH,
    )

    # evaluate the model on the validation set and save the results to a text file
    evaluate_model(
        model=model,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        txt_output_path=RESULTS_TXT,
        total_train_time=total_train_time,
    )

    print("\n[Done] Pipeline completed successfully!")


if __name__ == "__main__":
    main()

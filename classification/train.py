import os
import time

import torch
from torch import nn


def train_model(
    model,
    train_loader,
    epochs=2,
    lr=1e-4,
    device=None,
    save_path="convnext_weights.pth",
):
    """
    Trains the given model using the provided training data loader.
    Saves the model's weights after each epoch to the specified path.

    TODO: when we have a good model convention name, we can add a parameter to save the model with a name that includes the model type and the date of training.
    """
    if device is None:
        device = next(model.parameters()).device

    print("\n" + "=" * 60)
    print(f"STARTING TRAINING ({epochs} EPOCHS)")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time_total = time.perf_counter()

    model.train()
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            running_loss = 0.0
            total_batches = len(train_loader)

            for batch_idx, (images, labels) in enumerate(train_loader, 1):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # display progress every 50 batches or at the end of the epoch
                if batch_idx % 50 == 0 or batch_idx == total_batches:
                    current_avg = running_loss / batch_idx
                    print(
                        f"Epoch [{epoch:02d}/{epochs:02d}] - "
                        f"Batch [{batch_idx:04d}/{total_batches:04d}] - "
                        f"Loss: {current_avg:.4f}",
                        end="\r",
                    )

            print()
            epoch_duration = time.perf_counter() - epoch_start
            epoch_avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
            print(
                f"[INFO] Epoch [{epoch:02d}/{epochs:02d}] finished in {epoch_duration:.2f}s - "
                f"Average loss: {epoch_avg_loss:.4f}"
            )

            # save model weights after each epoch
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(
                f"[Save] Epoch {epoch}/{epochs} weights saved to : {save_path}"
            )

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training stopped by user.")
        print(
            f"[INFO] Weights from the last completed epoch remain saved in : {save_path}"
        )

    total_train_time = time.perf_counter() - start_time_total

    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = total_train_time % 60

    print("\n" + "=" * 60)
    print("TOTAL TRAINING TIME")
    print("=" * 60)
    if hours > 0:
        time_str = (
            f"{hours}h {minutes}m {seconds:.2f}s ({total_train_time:.2f} seconds)"
        )
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.2f}s ({total_train_time:.2f} seconds)"
    else:
        time_str = f"{seconds:.2f} seconds"
    print(f"[TIME] Total training time : {time_str}")

    return total_train_time

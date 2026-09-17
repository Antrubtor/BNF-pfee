import os
import time
from collections import defaultdict

import torch


def evaluate_model(
    model,
    val_loader,
    class_names,
    device=None,
    # TODO: change this according to a unique model training session
    txt_output_path="train-classsif/resultats_evaluation.txt",
    total_train_time=None,
):

    # if device is None, use the device of the model's parameters
    if device is None:
        device = next(model.parameters()).device

    print("\n" + "=" * 60)
    print("EVALUATION ON VALIDATION DATASET")
    print("=" * 60)

    model.eval()

    total_samples = 0
    correct_samples = 0
    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    eval_start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 1):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct_samples += (preds == labels).sum().item()

            for true_lbl, pred_lbl in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                class_total[true_lbl] += 1
                if true_lbl == pred_lbl:
                    class_correct[true_lbl] += 1

            if batch_idx % 50 == 0 or batch_idx == len(val_loader):
                print(
                    f"Evaluation : Batch [{batch_idx:04d}/{len(val_loader):04d}] processed",
                    end="\r",
                )

    print()  # newline after the last progress print
    eval_total_time = time.perf_counter() - eval_start_time

    overall_acc = (
        (correct_samples / total_samples * 100.0) if total_samples > 0 else 0.0
    )

    # formatting the evaluation report
    lines = []
    lines.append("=" * 70)
    lines.append("               CONVNEXT MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Date / Time                : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Device used                : {device}")
    if total_train_time is not None:
        lines.append(f"Total training time        : {total_train_time:.2f} s")
    lines.append(f"Total evaluation time      : {eval_total_time:.2f} s")
    lines.append("-" * 70)
    lines.append(f"Total validation images    : {total_samples}")
    lines.append(f"Correct predictions        : {correct_samples}")
    lines.append(f"Overall accuracy (Top-1)   : {overall_acc:.2f} %")
    lines.append("-" * 70)
    lines.append("PER-CLASS DETAILS :")
    lines.append(
        f"{'ID':<4} {'Class name':<20} {'Correct / Total':<20} {'Accuracy':<10}"
    )
    lines.append("-" * 70)

    for class_id in sorted(class_names.keys()):
        name = class_names[class_id]
        total_cls = class_total[class_id]
        correct_cls = class_correct[class_id]
        acc_cls = (correct_cls / total_cls * 100.0) if total_cls > 0 else 0.0
        lines.append(
            f"{class_id:<4} {name:<20} {f'{correct_cls} / {total_cls}':<20} {acc_cls:>6.2f} %"
        )

    lines.append("=" * 70)

    report = "\n".join(lines)

    # log the evaluation report to a text file
    os.makedirs(os.path.dirname(os.path.abspath(txt_output_path)), exist_ok=True)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + "=" * 60)
    print(f"RESULTS SAVED IN {txt_output_path}")
    print("=" * 60)
    print(report)

    return overall_acc

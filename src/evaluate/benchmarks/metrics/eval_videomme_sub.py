"""same as eval_videomme.py
"""
import json
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    return matches[0] if matches else ""


def calculate_accuracy(results, key):
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in results:
        value = item.get(key, "Unknown")
        pred = extract_characters_regex(item.get("predicted", ""))
        gt = extract_characters_regex(item.get("GT", ""))
        if pred == gt:
            counts[value]["correct"] += 1
        counts[value]["total"] += 1

    accuracies = {}
    for k, v in counts.items():
        total = v["total"]
        correct = v["correct"]
        accuracies[k] = correct / total if total > 0 else 0.0
    return accuracies


def save_to_excel(all_accuracies, output_path, overall_acc):
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")

    for key, acc_dict in all_accuracies.items():
        df = pd.DataFrame(
            [(k, f"{v:.2%}") for k, v in sorted(acc_dict.items())],
            columns=[key, "Accuracy"]
        )
        df.to_excel(writer, sheet_name=key, index=False)

    pd.DataFrame([{"Overall Accuracy": f"{overall_acc:.2%}"}]).to_excel(writer, sheet_name="Overall", index=False)
    writer.close()
    print(f"Excel saved to: {output_path}")


def save_plots(all_accuracies, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for key, acc_dict in all_accuracies.items():
        labels = list(acc_dict.keys())
        values = [acc_dict[k] * 100 for k in labels]

        plt.figure(figsize=(10, 5))
        bars = plt.barh(labels, values, color="skyblue")
        plt.xlabel("Accuracy (%)")
        plt.title(f"Accuracy by {key}")
        for bar, value in zip(bars, values):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{value:.1f}%", va='center')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"accuracy_by_{key}.png")
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.close()


def save_error_samples(data, output_path):
    errors = [item for item in data if extract_characters_regex(item.get("predicted", "")) != extract_characters_regex(item.get("GT", ""))]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"Error samples saved to: {output_path} ({len(errors)} items)")


def draw_confusion_matrix(data, output_dir):
    y_true = []
    y_pred = []
    for item in data:
        gt = extract_characters_regex(item.get("GT", ""))
        pred = extract_characters_regex(item.get("predicted", ""))
        if gt and pred:
            y_true.append(gt)
            y_pred.append(pred)

    labels = sorted(set(y_true + y_pred))
    matrix = pd.crosstab(pd.Series(y_true, name="Actual"),
                         pd.Series(y_pred, name="Predicted"),
                         rownames=["Actual"], colnames=["Predicted"],
                         dropna=False).reindex(index=labels, columns=labels, fill_value=0)

    matrix_percent = matrix.div(matrix.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=matrix_percent.applymap(lambda x: f"{x:.1f}%"), fmt='', cmap="Blues", cbar=True,
                     xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix with Percentages")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def main(args):
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = sum(1 for item in data if extract_characters_regex(item.get("predicted", "")) == extract_characters_regex(item.get("GT", "")))
    overall_acc = correct / total if total > 0 else 0.0

    all_accuracies = {}
    for key in ["domain", "sub_category", "task_type", "duration"]:
        print(f"\nAccuracy by {key}:")
        acc_by_key = calculate_accuracy(data, key)
        all_accuracies[key] = acc_by_key
        for k, acc in sorted(acc_by_key.items()):
            print(f"  {k:<20} : {acc:.2%}")
    print(f"\nOverall Accuracy: {overall_acc:.2%} ({correct}/{total})\n")

    base_dir = os.path.splitext(args.json_path)[0]
    output_dir = base_dir + "_analyze"
    os.makedirs(output_dir, exist_ok=True)

    save_to_excel(all_accuracies, os.path.join(output_dir, "accuracy.xlsx"), overall_acc)
    save_plots(all_accuracies, output_dir)
    save_error_samples(data, os.path.join(output_dir, "errors.json"))
    draw_confusion_matrix(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="outputs/output.json", help="Path to output.json")

    # Set all options to default=True
    parser.set_defaults(save_excel=True, save_plot=True, save_errors=True, confusion_matrix=True)

    args = parser.parse_args()
    main(args)


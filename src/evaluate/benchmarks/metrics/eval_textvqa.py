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
import string

import warnings
warnings.filterwarnings("ignore")


def process_pred(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.strip().split())
    return text


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


def save_rights_errors_samples(data, output_dir):
    rights, errors = [], []
    for item in data:
        GT_list = item.get("GT", [])
        predicted = item.get("predicted", "")
        cur_item_flag = False
        for GT in GT_list:
            if process_pred(predicted) == process_pred(GT):
                rights.append(item)
                cur_item_flag = True
                break
        if cur_item_flag:
            pass
        else:
            errors.append(item)

    with open(os.path.join(output_dir, "rights.json"), "w", encoding="utf-8") as f:
        json.dump(rights, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"Right and Error samples saved to: {output_dir} (rights: {len(rights)} items) (erros: {len(errors)} items)")


def main(args):
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    for item in data:
        GT_list = item.get("GT", [])
        predicted = item.get("predicted", "")
        for GT in GT_list:
            if process_pred(predicted) == process_pred(GT):
                correct += 1
                break
    overall_acc = correct / total if total > 0 else 0.0

    all_accuracies = {}
    print(f"\nOverall Accuracy: {overall_acc:.2%} ({correct}/{total})\n")

    base_dir = os.path.splitext(args.json_path)[0]
    output_dir = base_dir + "_analyze"
    os.makedirs(output_dir, exist_ok=True)

    save_to_excel(all_accuracies, os.path.join(output_dir, "accuracy.xlsx"), overall_acc)
    save_rights_errors_samples(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="outputs/output.json", help="Path to output.json")

    # Set all options to default=True
    parser.set_defaults(save_excel=True, save_plot=True, save_errors=True, confusion_matrix=True)

    args = parser.parse_args()
    main(args)


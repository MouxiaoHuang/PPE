import torch
import argparse
import os
import sys
import logging
import json
from datetime import datetime
import time
import warnings
from tqdm import tqdm
import random
import numpy as np
import re
import shutil
import subprocess
from PIL import Image

from src.evaluate.benchmarks_config import TASK_DATASET_MAPPING, DATASET_CONFIG

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) 
sys.path.append(ROOT_DIR)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils2 import process_vision_info
from training.monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward

warnings.filterwarnings("ignore")

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "GPU")
if DEVICE_TYPE == "NPU":
    try:
        import torch_npu
        from torch_npu.npu import amp
        from torch_npu.contrib import transfer_to_npu
        print("[INFO] torch_npu detected and imported successfully.")
        DEVICE_TYPE = "NPU"
        torch.npu.set_device(0)
    except ImportError:
        if torch.cuda.is_available():
            DEVICE_TYPE = "GPU"
            print(f"[INFO] GPU detected: {torch.cuda.device_count()} GPU(s) available.")
        else:
            DEVICE_TYPE = "CPU"
            print("[WARN] No NPU or GPU detected, falling back to CPU.")



def setup_logger(save_dir, local_rank):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"output_{local_rank}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logger initialized. Log file: {log_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="videomme", 
                        help="Video or image tasks, see in ./src/evaluate/datasets")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--nframes", type=str, default='64')
    parser.add_argument("--image_min_pixels", type=int, default=512*28*28)
    parser.add_argument("--image_max_pixels", type=int, default=1280*28*28)
    parser.add_argument("--video_min_pixels", type=int, default=128*28*28)
    parser.add_argument("--video_max_pixels", type=int, default=768*28*28)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--ppe_config", type=str, default=None, help="PPE setting, dense token if None")
    return parser.parse_args()

def split_dataset(dataset, num_splits, local_rank):
    return dataset[local_rank::num_splits]

def merge_results(save_dir, num_gpus):
    merged = []
    for i in range(num_gpus):
        fname = os.path.join(save_dir, f"output_{i}.json")
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as fp:
                merged.extend(json.load(fp))
            os.remove(fname)

    merged.sort(key=lambda x: x["original_idx"])
    for item in merged:
        item.pop("original_idx", None)

    merged_path = os.path.join(save_dir, "output.json")
    with open(merged_path, "w", encoding="utf-8") as fp:
        json.dump(merged, fp, indent=2, ensure_ascii=False)
    logging.info(f"Merged output saved to {merged_path}")

    log_list = []
    for i in range(num_gpus):
        log_fname = os.path.join(save_dir, f"output_{i}.log")
        if os.path.exists(log_fname):
            log_list.append(f"===== Logs from output_{i}.log =====\n")
            with open(log_fname, encoding="utf-8") as lf:
                log_list.append(lf.read())
            log_list.append("\n")
            os.remove(log_fname)
    combined_log = os.path.join(save_dir, "combined.log")
    with open(combined_log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_list))
    logging.info(f"Combined log saved to {combined_log}")

def wait_for_all_flags(save_dir, num_gpus, timeout=600):
    while True:
        flags = [os.path.join(save_dir, f"finished_{i}.flag") for i in range(num_gpus)]
        if all(os.path.exists(f) for f in flags):
            break
        time.sleep(10)
    for f in flags:
        os.remove(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_evaluation(output_json, task):
    time.sleep(2)

    module_name = f"src.evaluate.benchmarks.metrics.eval_{task}"
    cmd = f"python -m {module_name} --json_path={output_json}"

    logging.info(f"[RUNNING EVALUATION]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def parse_srt_timestamps(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.findall(r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)\n(?=\d+\n|\Z)', content, flags=re.DOTALL)
    subtitles = []
    for _, start, end, text in blocks:
        start_sec = timestamp_to_seconds(start)
        end_sec = timestamp_to_seconds(end)
        text_clean = re.sub(r'<[^>]+>', '', text.strip())
        subtitles.append((start_sec, end_sec, text_clean))
    return subtitles

def timestamp_to_seconds(ts):
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def main():
    args = parse_args()

    if args.timestamp.lower() == None:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    set_seed(seed=int(args.seed))

    if args.save_dir is None:
        base_dir = os.path.join(args.model_path, "infer_results", args.task)
    else:
        base_dir = os.path.join(args.save_dir, args.task)
    args.save_dir = os.path.join(base_dir, args.timestamp)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.nframes.lower() == 'none':
        args.nframes = None
    else:
        args.nframes = int(args.nframes)

    if args.ppe_config and os.path.exists(args.ppe_config):
        with open(args.ppe_config, "r") as f:
            ppe_config = json.load(f)

    config_save_dir = os.path.join(args.save_dir, "inference_config")
    os.makedirs(config_save_dir, exist_ok=True)
    output_dict = {"args": vars(args)}
    with open(os.path.join(config_save_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    if args.local_rank == 0:
        if args.script_path and os.path.exists(args.script_path):
            shutil.copy(args.script_path, os.path.join(config_save_dir,"script.sh"))
        if args.ppe_config and os.path.exists(args.ppe_config):
            shutil.copy(args.ppe_config, os.path.join(config_save_dir,"ppe_config.json"))

    setup_logger(args.save_dir, args.local_rank)

    replace_qwen2_5_with_mixed_modality_forward()

    logging.info(f"Loading model from {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if DEVICE_TYPE == "GPU" else "sdpa",
        device_map="auto",
    )

    model = model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    task = args.task.lower()
    if task not in TASK_DATASET_MAPPING:
        raise NotImplementedError(f"Task {task} not supported.")

    dataset_cls = TASK_DATASET_MAPPING[task]
    dataset_kwargs = DATASET_CONFIG.get(task, {})

    dataset = dataset_cls(**dataset_kwargs)
    full_dataset = list(dataset)

    if args.num_gpus > 1:
        for idx, item in enumerate(full_dataset):
            item["original_idx"] = idx
        dataset = split_dataset(full_dataset, args.num_gpus, args.local_rank)
        logging.info(f"Multi-{DEVICE_TYPE} mode: {DEVICE_TYPE} {args.local_rank} processing {len(dataset)} samples.")
        output_json = os.path.join(args.save_dir, f"output_{args.local_rank}.json")
    else:
        dataset = full_dataset
        output_json = os.path.join(args.save_dir, "output.json")

    results = []
    for i in tqdm(range(len(dataset)), desc=f"Running Inference on [{task}] ({DEVICE_TYPE} {args.local_rank})"):
        data = dataset[i]
        if getattr(dataset_cls, "modality", None) == "image":
            if data["image"] is None:
                image_pil = Image.open(data["image_path"]).convert('RGB')
            else:
                image_pil = data["image"]
        else:
            video_path = data["video_path"]
        prompt_data = data["prompt"]
        gt_answer = data["GT"]

        logging.info(f"\n>>> Processing sample #{i}")
        if getattr(dataset_cls, "modality", None) == "image":
            logging.info(f"Image: {image_pil}")
        else:
            logging.info(f"Video: {video_path}")
        logging.info(f"Prompt: {prompt_data}")
        logging.info(f"GT answer: {gt_answer}")

        if getattr(dataset_cls, "modality", None) == "image":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_pil,
                            "max_pixels": args.image_max_pixels,
                            "min_pixels": args.image_min_pixels,
                        },
                        {"type": "text", "text": prompt_data},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": args.video_max_pixels,
                            **({"nframes": args.nframes} if args.nframes else {"fps": args.fps})
                        },
                        {"type": "text", "text": prompt_data},
                    ],
                }
            ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        # videomme_sub
        if "subtitle_path" in data:
            assert args.task.lower() == "videomme_sub"
            sampled_fps = video_kwargs['fps'][0]
            nframes = video_inputs[0].shape[0]
            frame_times = [i / sampled_fps for i in range(nframes)]
            subtitle_list = parse_srt_timestamps(data["subtitle_path"])

            matched_subs = []
            for t in frame_times:
                matched = next((txt for start, end, txt in subtitle_list if start <= t <= end), "")
                matched_subs.append(matched)

            used_subs = list(filter(None, matched_subs))
            deduped_subs = []
            seen = set()
            for s in used_subs:
                if s not in seen:
                    deduped_subs.append(s)
                    seen.add(s)

            subtitle_text = "\n".join(deduped_subs)

            updated_prompt = f"This video's subtitles are listed below:\n{subtitle_text}\n{prompt_data}"
            messages[0]["content"][1]["text"] = updated_prompt

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
            ppe_config=ppe_config
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predicted_answer = output_text[0].strip()

        logging.info(f"Predicted: {predicted_answer}")

        result = dict(data)
        if getattr(dataset_cls, "modality", None) == "image":
            del result['image']
        result["predicted"] = predicted_answer
        results.append(result)


        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n Done. All results written to {output_json}")

    flag_file = os.path.join(args.save_dir, f"finished_{args.local_rank}.flag")
    with open(flag_file, "w") as f:
        f.write("finished")

    if args.num_gpus > 1 and args.local_rank == 0:
        logging.info("Local rank 0 waiting for all processes to finish...")
        wait_for_all_flags(args.save_dir, args.num_gpus)
        logging.info("All processes finished. Starting merge...")
        merge_results(args.save_dir, args.num_gpus)

    if args.local_rank == 0:
        run_evaluation(os.path.join(args.save_dir, "output.json"), task=args.task.lower())


if __name__ == "__main__":

    main()
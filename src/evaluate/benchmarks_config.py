from src.evaluate.benchmarks.videomme_sub import VideoMMESubDataset
from src.evaluate.benchmarks.mvbench import MVBenchDataset
from src.evaluate.benchmarks.textvqa import TextVQADataset

TASK_DATASET_MAPPING = {
    "videomme_sub": VideoMMESubDataset,
    "mvbench": MVBenchDataset,
    "textvqa": TextVQADataset,
}


DATASET_CONFIG = {
    "textvqa": {
        "image_path": "/cache/image_benchmark/textvqa/val",
        "anno_path": "/cache/image_benchmark/textvqa/TextVQA_0.5.1_val.json",
        "pre_prompt": "",
        "post_prompt": "\nAnswer the question using a single word or phrase. Use only lowercase letters and no punctuation. Do not provide any explanation."
    },
    "chartqa": {
        "image_path": "/cache/image_benchmark/chartqa/test",
        "anno_path": "/cache/image_benchmark/chartqa/test/test_chartqa.json",
        "pre_prompt": "",
        "post_prompt": ""
    },
    "docvqa": {
        "image_path": "/cache/image_benchmark/docvqa/val",
        "anno_path": "/cache/image_benchmark/docvqa/val/val_v1.0.json",
        "pre_prompt": "",
        "post_prompt": ""
    },
    "flickr": {
        "image_path": "/cache/image_benchmark/flickr30k/flickr30k_images",
        "anno_path": "/cache/image_benchmark/flickr30k/karpathy_splits/test_annotations.json",
        "pre_prompt": "",
        "post_prompt": "Provide a one-sentence caption for the provided image. Do not provide any explanation."
    },
    "mmbench_cn": {
        "image_path": "/cache/image_benchmark/mmbench",
        "anno_path": "/cache/image_benchmark/mmbench/mmbench_dev_cn_20231003.tsv",
        "pre_prompt": "",
        "post_prompt": "\n请直接使用所提供的选项字母作为答案回答。"
    },
    "mmbench_en": {
        "image_path": "/cache/image_benchmark/mmbench",
        "anno_path": "/cache/image_benchmark/mmbench/mmbench_dev_en_20231003.tsv",
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly."
    },
    "ocrbench": {
        "image_path": "/cache/image_benchmark/ocrbench/images",
        "anno_path": "/cache/image_benchmark/ocrbench/GT_eval/OCRbench.json",
        "pre_prompt": "",
        "post_prompt": ""
    },
    "sqa": {
        "image_path": "/cache/image_benchmark/sqa",
        "anno_path": "/cache/image_benchmark/sqa",
        "split": "test",
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly."
    },
    "videomme": {
        "video_path": "/cache/video_benchmark/Video-MME/videos/data",
        "anno_path": "/cache/video_benchmark/Video-MME/annotation_test.json",
        "option_prompt": "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.",
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly."
    },
    "videomme_sub": {
        "video_path": "/cache/video_benchmark/Video-MME/videos/data",
        "anno_path": "/cache/video_benchmark/Video-MME/annotation_test.json",
        "subtitles_path": "/cache/video_benchmark/Video-MME/subtitle",
        "option_prompt": "Select the best answer to the following multiple-choice question based on the video and subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly."
    },
    "mvbench": {
        "video_path": "/cache/video_benchmark/MVBench/videos",
        "anno_path": "/cache/video_benchmark/MVBench/json",
        "option_prompt": "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D or E) of the correct option.",
        "pre_prompt": "",
        "post_prompt": "Answer with the option's letter from the given choices directly."
    },
    "seedbench_video": {
        "video_path": "/cache/video_benchmark/SEED-Bench-video/v1_video",
        "anno_path": "/cache/video_benchmark/SEED-Bench-video/SEED-Bench.json",
        "option_prompt": "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D or E) of the correct option.",
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly."
    },
    "nextqa_mc": {
        "video_path": "/cache/video_benchmark/NExTQA/videos",
        "anno_path": "/cache/video_benchmark/NExTQA/MC/annotation_test.json",
        "option_prompt": "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D or E) of the correct option.",
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly."
    },
    "nextqa_oe": {
        "video_path": "/cache/video_benchmark/NExTQA/videos",
        "anno_path": "/cache/video_benchmark/NExTQA/OE/annotation_test.json",
        "option_prompt": "",
        "pre_prompt": "",
        "post_prompt": "\nAnswer the question using a single word or phrase. Use only lowercase letters and no punctuation. Do not provide any explanation."
    },
}

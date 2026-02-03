import base64
import io
import json
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class VideoMMESubDataset:
    modality = "video"
    
    def __init__(self,
                 video_path="/cache/vid_benchmark/Video-MME/videos/data",
                 anno_path="/cache/vid_benchmark/Video-MME/annotation_test.json",
                 subtitles_path="/cache/vid_benchmark/Video-MME/subtitle",
                 option_prompt="Select the best answer to the following multiple-choice question based on the video and subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
                 pre_prompt="",
                 post_prompt="\nAnswer with the option's letter from the given choices directly."):
        self.video_path = video_path
        self.anno_path = anno_path
        self.subtitles_path = subtitles_path

        with open(self.anno_path, "r") as f:
            self.data  = json.load(f)

        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.option_prompt = option_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        videoID = data_dict["videoID"]
        question = data_dict["question"]
        options = data_dict["options"]
        answer = data_dict["answer"]

        question = question + "\n" + "".join(options)
        full_prompt = "\n" + self.option_prompt + "\n" + question + "\n" + self.post_prompt

        video_path = os.path.join(self.video_path, videoID + ".mp4")

        return_data = data_dict.copy()

        return_data.update({
            "index": idx,
            "prompt": full_prompt,
            "video_path": video_path,
            "GT": answer
        })

        # only add subtitle_path when the subtitle file exists
        subtitle_path = os.path.join(self.subtitles_path, videoID + ".srt")
        if os.path.exists(subtitle_path):
            return_data["subtitle_path"] = subtitle_path

        return return_data





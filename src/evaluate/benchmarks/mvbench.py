import os
import json
import cv2
from tqdm import tqdm
from collections import defaultdict
import shutil
from pathlib import Path
import re


class MVBenchDataset:
    modality = "video"
    
    def __init__(self,
                 video_path="/cache/video_benchmark/MVBench/videos",
                 anno_path="/cache/video_benchmark/MVBench/json",
                 option_prompt="Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, D or E) of the correct option.",
                 pre_prompt="",
                 post_prompt="Answer with the option's letter from the given choices directly."):

        self.video_path = video_path
        self.anno_path = anno_path
        self.data = {}
        self.index_map = []

        self.process_frame_folders()

        self.alter_data0613()

        anno_list = os.listdir(self.anno_path)
        anno_list.sort()
        for filename in anno_list:
            if filename.endswith(".json"):
                file_path = os.path.join(self.anno_path, filename)
                with open(file_path, "r") as f:
                    try:
                        json_data = json.load(f)
                        key = os.path.splitext(filename)[0]
                        self.data[key] = json_data
                        if isinstance(json_data, list):
                            for i in range(len(json_data)):
                                self.index_map.append((key, i))
                        elif isinstance(json_data, dict):
                            self.index_map.append((key, None))
                    except json.JSONDecodeError as e:
                        print(f"read {filename} wrong {e}")

        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.option_prompt = option_prompt

        self.DATA_LIST = {
            "action_sequence": "star/Charades_v1_480/", # has start and end -> need segment
            "action_prediction": "star/Charades_v1_480/", # has start and end -> need segment
            "action_antonym": "ssv2_video/",
            "fine_grained_action": "Moments_in_Time_Raw/videos/",
            "unexpected_action": "FunQA_test/test/",
            "object_existence": "clevrer/video_validation/",
            "object_interaction": "star/Charades_v1_480/", # has start and end -> need segment
            "object_shuffle": "perception/videos/",
            "moving_direction": "clevrer/video_validation/",
            "action_localization": "sta/sta_video/", # has start and end -> need segment
            "scene_transition": "scene_qa/video/",
            "action_count": "perception/videos/",
            "moving_count": "clevrer/video_validation/",
            "moving_attribute": "clevrer/video_validation/",
            "state_change": "perception/videos/",
            "fine_grained_pose": "nturgbd/",
            "character_order": "perception/videos/",
            "egocentric_navigation": "vlnqa/",
            "episodic_reasoning": "tvqa/frames_fps3_hq/", # frames to mp4 first, has start and end
            "counterfactual_inference": "clevrer/video_validation/",
        }
        self.index2ABCDE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        self.option_mapping = ["A", "B", "C", "D", "E"]

        self.segment_json_list = [
            'action_sequence',
            'action_prediction',
            'object_interaction',
            'action_localization'
        ]
        self.segment_subsets()

    def process_frame_folders(self):
        frames_root = os.path.join(self.video_path, "tvqa/frames_fps3_hq/")
        output_root = os.path.join(self.video_path, "tvqa/videos/")
        os.makedirs(output_root, exist_ok=True) 

        for folder in os.listdir(frames_root):
            folder_path = os.path.join(frames_root, folder)
            output_video_path = os.path.join(output_root, f"{folder}.mp4")

            if os.path.exists(output_video_path):
                continue  

            if os.path.isdir(folder_path):
                self.frames_to_video(folder_path, output_video_path)

    def frames_to_video(self, frames_folder, output_video_path, fps=3):
        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg'))])
        if not frame_files:
            print(f"Folder {frames_folder} no image")
            return

        first_frame_path = os.path.join(frames_folder, frame_files[0])
        frame = cv2.imread(first_frame_path)
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)

        video_writer.release()
        print(f"Convert frames [{frames_folder}] --> to --> [{output_video_path}]")

    def alter_data0613(self):
        validation_src = Path(self.video_path) / "data0613/clevrer/video_validation"
        validation_dst = Path(self.video_path) / "clevrer/video_validation"
        
        charades_src = Path(self.video_path) / "data0613/star/Charades_v1_480"
        charades_dst = Path(self.video_path) / "star/Charades_v1_480"

        validation_dst.mkdir(parents=True, exist_ok=True)
        charades_dst.mkdir(parents=True, exist_ok=True)

        for src_file in validation_src.glob("*.mp4"):
            shutil.copy(src_file, validation_dst / src_file.name)

        for src_file in charades_src.glob("*.mp4"):
            shutil.copy(src_file, charades_dst / src_file.name)

    def segment_subsets(self):
        for segment_json in self.segment_json_list:
            json_file_path = os.path.join(self.anno_path, f"{segment_json}.json")
            video_dir = os.path.join(self.video_path, self.DATA_LIST[segment_json])
            subname = os.path.basename(self.DATA_LIST[segment_json].rstrip(os.sep))
            # update self.DATA_LIST
            self.DATA_LIST[segment_json] = self.DATA_LIST[segment_json].replace(subname, f"{subname}_segment")
            output_dir = os.path.join(
                self.video_path,
                self.DATA_LIST[segment_json]
            )
            os.makedirs(output_dir, exist_ok=True) 
            self.segment_videos_from_json(json_file_path, video_dir, output_dir)

    def segment_videos_from_json(self, json_file_path, video_dir, output_dir):
        with open(json_file_path, 'r') as f:
            video_data = json.load(f)
        
        for video_info in video_data:
            video_filename = video_info['video']
            start = video_info['start']
            end = video_info['end']
            input_video_path = os.path.join(video_dir, video_filename)
            output_video_path = os.path.join(output_dir, f"{video_filename}_{str(start)}_{str(end)}.mp4")
            if os.path.exists(output_video_path):
                continue  
            self._segment_video(input_video_path, output_video_path , start, end)

    def _segment_video(self, input_video_path, output_video_path , start, end):
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame, end_frame = int(start * fps), int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while cap.isOpened() and frame_count <= (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
        print(f"Segment [{input_video_path}] --> to --> [{output_video_path}]")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        filename, inner_idx = self.index_map[idx]
        data_dict = self.data[filename]

        if inner_idx is not None:
            data_dict = data_dict[inner_idx]

        videoID = data_dict["video"]
        question = data_dict["question"]

        options = "\n".join([
            f"{self.index2ABCDE[i]}. {cand}" 
            for i, cand in enumerate(data_dict.get('candidates', []))
        ])

        answer_index = data_dict.get('candidates', []).index(data_dict.get('answer'))
        answer = self.option_mapping[answer_index]

        option_letters = self.option_mapping[:len(data_dict.get('candidates', []))]
        option_str = " or ".join([", ".join(option_letters[:-1]), option_letters[-1]] 
                                if len(option_letters) > 1 else option_letters)
        pattern = r'\(?[A-E](?:, [A-E])*(?: or [A-E])?\)?'
        self.option_prompt = re.sub(pattern, f"({option_str})", self.option_prompt)

        full_prompt = (
            f"\n{self.option_prompt}\n"
            f"{question}\n"
            f"{options}\n"
            f"{self.post_prompt}"
        )

        file_key = os.path.splitext(filename)[0]
        video_prefix = self.DATA_LIST.get(file_key, "")

        if file_key == 'episodic_reasoning':
            video_path = os.path.join(self.video_path, f"tvqa/videos/{videoID}.mp4")
        elif file_key in self.segment_json_list:
            try:
                start, end = str(data_dict['start']), str(data_dict['end'])
            except:
                print('Should contain start and end keys!')
                raise KeyError
            video_path = os.path.join(
                self.video_path, 
                video_prefix,
                f"{videoID}_{start}_{end}.mp4"
            )
        else:
            video_path = os.path.join(self.video_path, video_prefix, videoID)

        return_data = data_dict.copy()
        
        return_data.update({
            "index": idx,
            "prompt": full_prompt,
            "video_path": video_path,
            "GT": answer
        })

        return return_data

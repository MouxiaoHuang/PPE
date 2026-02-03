import json
from PIL import Image
import os
import re


class TextVQADataset(object):
    modality = "image"
    
    def __init__(
        self,
        image_path="/cache/image_benchmark/textvqa/val",
        anno_path="/cache/image_benchmark/textvqa/TextVQA_0.5.1_val.json",
        pre_prompt="",
        post_prompt="\nAnswer the question using a single word or phrase. \
                     Use only lowercase letters and no punctuation. \
                     Do not provide any explanation."
    ):
        self.anno_file = json.load(open(anno_path, "r"))["data"]
        self.data_root = image_path
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt

    def __len__(self):
        return len(self.anno_file)

    def capitalize_sentences(self, text):
        sentences = re.split('([.!?] *)', text)
        capitalized_sentences = [sentence.capitalize() for sentence in sentences]
        capitalized_text = ''.join(capitalized_sentences).strip()
        if not capitalized_text.endswith('?'):
            capitalized_text += '?'
        return capitalized_text

    def __getitem__(self, idx):
        data = self.anno_file[idx]

        image_name, question, question_id = (
            data['image_id'], 
            data['question'], 
            data['question_id']
        )

        question = self.capitalize_sentences(text=question)
        full_prompt = f"{self.pre_prompt}{question}{self.post_prompt}"

        image_path = os.path.join(self.data_root.rstrip("/"), f"{image_name}.jpg")
        image = Image.open(image_path).convert('RGB')

        return {
            "image": image,
            "prompt": full_prompt,
            "index": question_id,
            "GT": data["answers"],
            "image_path": image_path,
        }

import time
import os
import json
import argparse

from huggingface_hub import HfApi
from datasets import Dataset, Features, Value, Image, Sequence, Array2D


def check_completion(folder_path, repo_id, split):
    log_file = f"{folder_path}/log.txt"
    img_url_prefix = f"https://huggingface.co/datasets/{repo_id}/blob/main"
    while True:
        if os.path.exists(log_file):
            with open(log_file, 'r') as file:
                lines = file.readlines()
                if lines[-1].strip() == "completed":  # Check if the last line is "completed"
                    print("Process completed! Uploading to Hugging Face...")
                    with open(f"{folder_path}/processed.jsonl", 'r') as f:
                        data = [json.loads(line) for line in f]

                    # Create a Dataset from the data with additional fields
                    dataset = Dataset.from_dict({'prompt': [datum['prompt'] for datum in data],
                                                'is_pairwise': [datum['is_pairwise'] for datum in data],
                                                'captions': [datum['captions'] for datum in data], 
                                                'chosen_response': [datum['chosen_response'] for datum in data],
                                                'rejected_responses': [datum['rejected_responses'] for datum in data],
                                                'caption_media_scores': [datum['caption_media_scores'] for datum in data], 
                                                'medias': [datum['medias'] for datum in data],
                                                'media_coordinates': [datum['media_coordinates'] for datum in data],
                                                'media_types': [datum['media_types'] for datum in data],
                                                'metadata': [{'source': datum['metadata']['source'],
                                                            'params': datum['metadata']['params']} for datum in data],
                                                })
                    
                    features = Features({
                        'prompt': Value('string'),
                        'is_pairwise': Value('bool'),
                        'captions': Sequence(Value('string')),
                        'chosen_response': Value('string'),
                        'rejected_responses': Sequence(Value('string')),
                        'caption_media_scores': Sequence(Value('float')),
                        'medias': Sequence(Image()),  
                        'media_coordinates': Sequence(Sequence(Value('float'))),  
                        'media_types': Sequence(Value('string')),
                        'metadata': {
                            'source': Value('string'),  
                            'params': Value('string')
                        }
                    })

                    # Cast dataset to include the image and other variables as feature types
                    dataset = dataset.cast(features)

                    # Push the dataset to Hugging Face with the new fields
                    dataset.push_to_hub(repo_id, split=split) # ,private=True)

                    break
        time.sleep(240)  # Check every 240 seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor log file for completion status.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repo ID")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the directory to upload")
    parser.add_argument("--split", type=str, required=True, help="data split")

    args = parser.parse_args()
    # api = HfApi()
    # api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    check_completion(args.folder_path, args.repo_id, args.split)

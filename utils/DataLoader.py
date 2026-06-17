import random
from torch.utils.data import IterableDataset, Dataset
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path


"""
class VideoDatasetLoader(IterableDataset):
    def __init__(self, video_path_to_label_dict):
        # 输入是一个字典：{视频路径: 标签}
        self.video_path_to_label = video_path_to_label_dict
    
    def __iter__(self):
        # --- 打乱视频顺序 (模拟 shuffle) ---
        video_items = list(self.video_path_to_label.items())
        random.shuffle(video_items) # 在每个 Worker 开始时打乱
        
        for video_path, label in video_items:
            video_reader = decord.VideoReader(video_path)
            for frame_idx in range(len(video_reader)):
                frame = video_reader[frame_idx]
                
                yield {
                    'frame': frame.permute(2, 0, 1) / 255.0,
                    'video_id': video_path,
                    'label': label,          # 每一帧都带上整个视频的标签
                    'frame_idx': frame_idx,
                    'total_frames': len(video_reader)
                }


    # --- 使用 ---
    # 定义数据和标签
    data_with_labels = {
        'video1.mp4': 0,  # 0 代表 "打篮球"
        'video2.mp4': 1,  # 1 代表 "游泳"
        'video3.mp4': 0,
    }

    dataset = VideoFrameDataset(data_with_labels)
    dataloader = DataLoader(dataset, batch_size=1)
"""


class FrameDatasetLoader(Dataset):
    def __init__(self, frames_dir, json_path, device, set_type, features_dir=None):
        self.frames_dir = frames_dir
        self.device = device
        self.set_type = set_type
        self.features_dir = str(features_dir) if features_dir is not None else None

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.database = data['database']
        self.dataset_totle_frame = 0
        self.class_to_idx = {cls: idx for idx, cls in enumerate(data['classes'])}
        self.class_num = len(data['classes'])

        # 统计总帧数（用于 DataLoader 的长度，或者你可以直接用视频数量）
        # 这里我们用视频数量作为长度，每个 batch 处理一个视频
        self.video_list = []
        for video_id, info in self.database.items():
            if info['subset'] == self.set_type:
                self.video_list.append({
                    'video_id': video_id,
                    'fps': info['fps'],
                    'duration': info['duration'],
                    'total_frames': info['frame_num'],
                    'subset': info['subset'], # train/val
                    'annotations': info['annotations'],
                    'feature_path': str(Path(self.features_dir) / f"{video_id}.npy") if self.features_dir else None,
                })
                self.dataset_totle_frame += info['frame_num']

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_info = self.video_list[idx]
        video_id = video_info['video_id']
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        annotations = video_info['annotations']
        
        # 构建该视频帧文件夹路径
        video_folder = os.path.join(self.frames_dir, video_id) if self.frames_dir else None

        labels = []
        for ann in annotations:
            segment = torch.tensor(ann['segment'], dtype=torch.float32)
            cls_idx = torch.tensor(self.class_to_idx[ann['label']], dtype=torch.long)
            cls_one_hot = F.one_hot(cls_idx, num_classes=self.class_num).to(torch.float32)

            labels.append(torch.cat([segment, cls_one_hot], dim=0))
        labels = torch.stack(labels).to(self.device)  # [num_annotations, 2 + num_classes]


        # 返回视频的元信息和处理好的片段
        return {
            'video_id': video_id,
            'video_folder': video_folder,
            'feature_path': video_info.get('feature_path', None),
            'total_frames': total_frames,
            'labels': labels,
            'fps': fps
        }


def load_train_val_data(frames_dir, json_path, device, features_dir=None):
    return (
        FrameDatasetLoader(frames_dir, json_path, device, set_type='train', features_dir=features_dir),
        FrameDatasetLoader(frames_dir, json_path, device, set_type='val', features_dir=features_dir),
    )


if __name__ == "__main__":
    # --- 测试 FrameDatasetLoader ---
    dataset = FrameDatasetLoader(
        frames_dir='/home/liutianxiang/datasets/tennis_match_annalysis/frames',
        json_path='/home/liutianxiang/datasets/tennis_match_annalysis/MEM_TAD_format/tennisnet_annotations_high.json',
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"数据集中视频数量: {len(dataset)}")
    
    # 获取第一个视频的信息
    video_info = dataset[0]
    print(f"视频ID: {video_info['video_id']}")
    print(f"视频帧文件夹: {video_info['video_folder']}")
    print(f"总帧数: {video_info['total_frames']}")
    print(f"FPS: {video_info['fps']}")
    print("处理后的片段信息:")
    print(video_info['labels'])

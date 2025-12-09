# dataset_loader.py
import os
import torch
from torch.utils.data import Dataset
from preprocess import preprocess_image, preprocess_text, structured_features_from_text

class ProductDataset(Dataset):
    def __init__(self, dataframe, image_folder, is_train=True, image_transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.is_train = is_train
        self.transform = image_transform  # not used here because preprocess_image handles default

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = int(row['sample_id'])
        text = str(row.get('catalog_content', ""))

        image_path = os.path.join(self.image_folder, f"{sample_id}.jpg")
        image = preprocess_image(image_path)  # returns tensor
        text_inputs = preprocess_text(text)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]
        struct_feat = structured_features_from_text(text)

        if self.is_train:
            price = float(row['price'])
            # log scaling applied here in training script (we keep price raw here and let training do log1p)
            return image, input_ids, attention_mask, struct_feat, torch.tensor(price, dtype=torch.float32)
        else:
            return image, input_ids, attention_mask, struct_feat, sample_id

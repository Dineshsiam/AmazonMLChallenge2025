# predict.py
import torch
from torch.utils.data import DataLoader
import pandas as pd
from dataset_loader1 import ProductDataset
from model import FusionModel
from torch.amp import autocast  # updated import
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TEST_CSV = "/home/dinesh/AmazonMLChallange/dataset/test.csv"
IMAGE_FOLDER = "/home/dinesh/AmazonMLChallange/dataset/test_images"
MODEL_PATH = "best_fusion_model.pth"

# Load test dataset
df_test = pd.read_csv(TEST_CSV)
test_dataset = ProductDataset(df_test, IMAGE_FOLDER, is_train=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Load model
model = FusionModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

preds = []
sample_ids = []

with torch.no_grad():
    for images, input_ids, attention_mask, struct_feat, s_id in tqdm(test_loader):
        images = images.to(DEVICE, non_blocking=True)
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)
        struct_feat = struct_feat.to(DEVICE, non_blocking=True)

        # Updated autocast usage
        with autocast(device_type='cuda'):
            outputs = model(images, input_ids, attention_mask, struct_feat)

        # Inverse log1p to original scale
        batch_preds = torch.expm1(outputs.detach().cpu())
        preds.extend(batch_preds.tolist())
        sample_ids.extend([int(x) for x in s_id])

# Save predictions
df_out = pd.DataFrame({"sample_id": sample_ids, "price": preds})
df_out.to_csv("test_predictions.csv", index=False)
print(" Saved test_predictions.csv")

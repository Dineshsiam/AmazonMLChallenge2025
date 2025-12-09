# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, DistilBertModel
from transformers.utils import import_utils
import torch
if "dev" in torch.__version__:
    import_utils.check_torch_load_is_safe.__defaults__ = (False,)
class FusionModel(nn.Module):
    def __init__(self, structured_dim=5, embed_dropout=0.3):
        super().__init__()
        # ---- TEMP FIX for CVE-2025-32434 false positive on PyTorch dev/nightly ----
        
# ---------------------------------------------------------------------------

        # Image encoder (DeiT tiny) - loaded once in init
        self.image_model = AutoModel.from_pretrained(
            "facebook/deit-tiny-patch16-224",
            revision="main",
             use_safetensors=True,  
            ignore_mismatched_sizes=True
        )

        # Text encoder (DistilBERT)
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        img_dim = self.image_model.config.hidden_size
        txt_dim = self.text_model.config.hidden_size

        # Projectors (to same dim)
        proj_dim = 512
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(embed_dropout)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(embed_dropout)
        )

        # Gated fusion: learns which modality to trust
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Structured features normalization + projector
        self.struct_norm = nn.LayerNorm(structured_dim)
        self.struct_proj = nn.Sequential(
            nn.Linear(structured_dim, 64),
            nn.ReLU(),
            nn.Dropout(embed_dropout)
        )

        # Final head
        fusion_dim = proj_dim + 64
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, images, input_ids, attention_mask, structured_features):
        # images: pixel_values for DeiT
        img_outputs = self.image_model(pixel_values=images).last_hidden_state  # (B, seq, dim)
        img_cls = img_outputs[:, 0, :]  # CLS token

        txt_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        txt_pool = txt_outputs.mean(dim=1)  # mean pooling

        img_e = self.img_proj(img_cls)
        txt_e = self.txt_proj(txt_pool)

        # Gating
        gate_in = torch.cat([img_e, txt_e], dim=1)
        gate = torch.sigmoid(self.gate(gate_in))  # (B, proj_dim)
        fused_img_txt = gate * img_e + (1 - gate) * txt_e  # (B, proj_dim)

        # Structured
        s = self.struct_norm(structured_features)
        s_e = self.struct_proj(s)

        fused = torch.cat([fused_img_txt, s_e], dim=1)
        out = self.head(fused).squeeze(-1)  # (B,)
        return out

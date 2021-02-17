from typing import List, Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule
from torch.optim import Adam
from transformers import BertConfig
from vit_pytorch import ViT
from pytorch_lightning.metrics.functional import confusion_matrix

from configs import TrainingArguments
from models.modules.fast_transformers import LinearAttention


class ViTLinBertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

        self.scale = self.attention_head_size ** -0.5

        self.attention = LinearAttention(config, None)

        self.to_qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask=None):
        x_ = F.pad(x, [0, 0, 0, 1])
        qkv = self.to_qkv(x_).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.num_attention_heads), qkv)

        mask = torch.ones(1)
        if torch.cuda.is_available():
            mask = mask.cuda()
        out = self.attention(q, k, v, mask)

        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.to_out(out)[:, :-1, :]
        return out


class EfficientViT(ViT):
    def __init__(self, config: BertConfig):
        self.config = config
        super().__init__(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            num_classes=self.config.num_labels,
            channels=self.config.channels,
            dim=self.config.hidden_size,
            depth=self.config.num_hidden_layers,
            dim_head=self.config.hidden_size // self.config.num_attention_heads,
            heads=self.config.num_attention_heads,
            mlp_dim=self.config.intermediate_size,
            dropout=self.config.hidden_dropout_prob,
            emb_dropout=self.config.hidden_dropout_prob
        )

        for i, _ in enumerate(self.transformer.layers):
            if self.config.is_linear:
                self.transformer.layers[i][0].fn.fn = ViTLinBertSelfAttention(config)


class ViTModel(LightningModule):
    def __init__(self, config: BertConfig, training_args: TrainingArguments):
        super().__init__()
        self.config = config
        self.training_args = training_args
        self.save_hyperparameters()
        self.vit = EfficientViT(config=config)

    def configure_optimizers(self):
        return Adam(self.vit.parameters(), lr=self.training_args.learning_rate)

    def forward(self, img: torch.Tensor, mask: torch.Tensor = None):
        return self.vit(img, mask)

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, *_ = labels.shape
        output = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(output, labels, reduction='sum')
        loss = loss.sum() / batch_size
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Dict:  # type: ignore
        imgs, labels = batch
        logits = self(imgs)
        loss = self._calculate_loss(logits, labels)

        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}
        with torch.no_grad():
            conf_matrix = confusion_matrix(
                logits.argmax(-1),
                labels.squeeze(0),
                num_classes=self.config.num_labels
            )
            log["train/accuracy"] = conf_matrix.trace() / conf_matrix.sum()
        self.log_dict(log)

        return {"loss": loss, "confusion_matrix": conf_matrix}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:  # type: ignore
        imgs, labels = batch
        logits = self(imgs)
        loss = self._calculate_loss(logits, labels)
        with torch.no_grad():
            conf_matrix = confusion_matrix(
                logits.argmax(-1),
                labels.squeeze(0),
                num_classes=self.config.num_labels
            )

        return {"loss": loss, "confusion_matrix": conf_matrix}

    def test_step(self, batch: Any, batch_idx: int) -> Dict:  # type: ignore
        return self.validation_step(batch, batch_idx)

    # ========== On epoch end ==========

    def _shared_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": mean_loss}

            accumulated_conf_matrix = torch.zeros(
                self.config.num_labels, self.config.num_labels, requires_grad=False, device=self.device
            )
            for out in outputs:
                _conf_matrix = out["confusion_matrix"]
                max_class_index, _ = _conf_matrix.shape
                accumulated_conf_matrix[:max_class_index, :max_class_index] += _conf_matrix
            log[f"{group}/accuracy"] = (accumulated_conf_matrix.trace() / accumulated_conf_matrix.sum()).item()

            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test")

"""The classifier network: MiniLM encoder + mean pooling + dropout + linear head.

torch and transformers are imported lazily so the rest of the package stays importable
without the ``finetune`` extra.
"""

from pathlib import Path
from typing import Any

from edusync_blueprint_model.deps import require

_CLASS_CACHE: dict[str, Any] = {}


def _classifier_class() -> Any:  # pragma: no cover - needs torch
    """Define ``MiniLMClassifier`` once (it must subclass ``torch.nn.Module``)."""
    if "cls" in _CLASS_CACHE:
        return _CLASS_CACHE["cls"]
    torch = require("torch", "finetune")
    nn = torch.nn

    class MiniLMClassifier(nn.Module):
        def __init__(self, encoder: Any, num_labels: int, dropout: float) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(encoder.config.hidden_size, num_labels)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            tokens = hidden[0]
            # Mean pooling over real tokens only, as the sentence-transformer was trained.
            mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return self.head(self.dropout(pooled))

    _CLASS_CACHE["cls"] = MiniLMClassifier
    return MiniLMClassifier


def build_classifier(base_dir: Path, num_labels: int, dropout: float) -> Any:  # pragma: no cover
    """Load the encoder from ``base_dir`` and attach a fresh classification head."""
    transformers = require("transformers", "finetune")
    encoder = transformers.AutoModel.from_pretrained(str(base_dir))
    return _classifier_class()(encoder, num_labels, dropout)


def save_classifier(model: Any, finetuned_dir: Path, dropout: float) -> None:  # pragma: no cover
    """Write ``encoder/`` and ``head.pt`` (the tokenizer is saved by the caller)."""
    torch = require("torch", "finetune")
    finetuned_dir = Path(finetuned_dir)
    model.encoder.save_pretrained(str(finetuned_dir / "encoder"))
    meta = {"num_labels": model.head.out_features, "dropout": dropout}
    torch.save({"state_dict": model.head.state_dict(), **meta}, finetuned_dir / "head.pt")


def load_finetuned(finetuned_dir: Path) -> Any:  # pragma: no cover - needs torch
    """Rebuild the fine-tuned classifier saved by ``save_classifier``."""
    torch = require("torch", "finetune")
    finetuned_dir = Path(finetuned_dir)
    head_file = finetuned_dir / "head.pt"
    if not head_file.is_file():
        raise FileNotFoundError(f"No fine-tuned model in {finetuned_dir}; run the train step")
    saved = torch.load(head_file, map_location="cpu", weights_only=True)
    model = build_classifier(finetuned_dir / "encoder", saved["num_labels"], saved["dropout"])
    model.head.load_state_dict(saved["state_dict"])
    return model.eval()

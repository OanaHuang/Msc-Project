from pathlib import Path
import sys

import torch


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPIKEYOLO_ROOT = (
    PROJECT_ROOT
    / "External"
    / "SpikeYOLO"
)

WEIGHTS_PATH = (
    SPIKEYOLO_ROOT
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "SpikeYOLO"
    / "03_model_architecture"
)

OUTPUT_TXT = OUTPUT_DIR / "spikeyolo_architecture.txt"


# ============================================================
# 2. Import local SpikeYOLO
# ============================================================

sys.path.insert(0, str(SPIKEYOLO_ROOT))

from ultralytics import YOLO  # noqa: E402


# ============================================================
# 3. Helpers
# ============================================================

def count_parameters(module):
    total = sum(
        parameter.numel()
        for parameter in module.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ============================================================
# 4. Main
# ============================================================

def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found:\n{WEIGHTS_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS_PATH))
    torch_model = model.model

    total_params, trainable_params = count_parameters(
        torch_model
    )

    lines = []

    lines.append("=" * 100)
    lines.append("SpikeYOLO Architecture")
    lines.append("=" * 100)
    lines.append(f"Weights: {WEIGHTS_PATH}")
    lines.append(f"Model class: {torch_model.__class__.__name__}")
    lines.append(f"Total parameters: {total_params:,}")
    lines.append(
        f"Trainable parameters: {trainable_params:,}"
    )
    lines.append(
        f"Approximate size: {total_params / 1e6:.3f} M"
    )
    lines.append("")

    lines.append("=" * 100)
    lines.append("Top-level model")
    lines.append("=" * 100)
    lines.append(str(torch_model))
    lines.append("")

    lines.append("=" * 100)
    lines.append("Layer-by-layer structure")
    lines.append("=" * 100)

    network_layers = torch_model.model

    for index, layer in enumerate(network_layers):
        layer_params, trainable_layer_params = (
            count_parameters(layer)
        )

        from_index = getattr(layer, "f", None)
        layer_type = layer.__class__.__name__

        lines.append(
            f"[{index:02d}] "
            f"type={layer_type:<35} "
            f"from={str(from_index):<15} "
            f"params={layer_params:>12,} "
            f"trainable={trainable_layer_params:>12,}"
        )

        lines.append(f"      {layer}")
        lines.append("")

    lines.append("=" * 100)
    lines.append("All named modules containing SNN-related terms")
    lines.append("=" * 100)

    keywords = (
        "lif",
        "spike",
        "snn",
        "membrane",
        "neuron",
        "quant",
    )

    snn_module_count = 0

    for name, module in torch_model.named_modules():
        module_text = (
            f"{name} "
            f"{module.__class__.__name__}"
        ).lower()

        if any(
            keyword in module_text
            for keyword in keywords
        ):
            module_params, _ = count_parameters(module)

            lines.append(
                f"{name:<70} "
                f"{module.__class__.__name__:<35} "
                f"{module_params:>12,}"
            )

            snn_module_count += 1

    lines.append("")
    lines.append(
        f"Detected SNN-related modules: "
        f"{snn_module_count}"
    )

    output_text = "\n".join(lines)

    print(output_text)

    OUTPUT_TXT.write_text(
        output_text,
        encoding="utf-8",
    )

    print()
    print(f"Architecture saved to:\n{OUTPUT_TXT}")


if __name__ == "__main__":
    main()
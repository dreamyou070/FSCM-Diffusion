<h1 align="center">🌊 FSCM-Diffusion</h1>
<p align="center">
  <strong>Stepwise Control of Climate Phenomena using Fused Structural Condition Maps</strong><br>
  A diffusion-based image editing framework for realistic flood and snow intensity control in real-world scenes.
</p>

---

## 🌐 Overview

**FSCM-Diffusion** is a diffusion-based image editing framework that enables explicit, fine-grained, and physically plausible control over flooding and snowfall in real-world scenes.

Unlike prior methods that struggle with structural consistency or lack quantitative interpretability, we introduce:

- 🔢 **Leveling System**: Encodes user-specified instructions as discrete numeric values for controllable modulation.
- 🧩 **FSCM (Fused Structural Condition Map)**: Combines depth + normal maps for structural guidance.
- 🎨 **Style-LoRA Distillation**: Enables realistic texture synthesis from limited climate-specific data.

We train the model on a high-quality paired dataset combining synthetic Climate-NeRF scenes and temporally-aligned real YouTube footage. Our method generalizes well to in-the-wild images and outperforms previous methods across controllability, realism, and structure preservation.

---

## 🚀 Quick Demo

### 🔹 Input
- Prompt: `make the area more submerged`
- Level: `3`

<img src="assets/demo_input.jpg" width="300"/>

### 🔹 Run Inference
```bash
python inference.py \
  --input_image assets/demo_input.jpg \
  --depth_map assets/demo_depth.png \
  --normal_map assets/demo_normal.png \
  --level 3 \
  --output_dir outputs/demo/

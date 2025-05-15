<h1 align="center">FSCM-Diffusion: Stepwise Control of Climate Phenomena using Fused Structural Condition Maps</h1>


**FSCM-Diffusion** We present FSCM-Diffusion, a diffusion-based image editing framework that enables explicit, fine-grained, and physically plausible control over flooding in real-world scenes. Unlike prior methods that struggle with structural consistency or lack quantitative interpretability, we introduce a Leveling System that encodes user-specified water instructions as discrete numeric values, allowing precise and consistent modulation. To enhance realism and geometric coherence, we propose a Fused Structural Condition Map (FSCM) that integrates depth and surface normal maps as conditioning inputs. This fused representation guides the generation to better conform to underlying scene geometry and fluid dynamics, mitigating artifacts like surface misalignment or texture blur. We further incorporate a custom Style-LoRA, distilled from limited data, to inject localized rainfall and snowfall textures without requiring large-scale style datasets, enhancing stylistic fidelity and visual diversity. For training and evaluation, we construct a high-quality paired dataset combining synthetic Climate-NeRF renderings with temporally aligned real-world YouTube footage depicting progressive flood scenarios. Our method generalizes well to in-the-wild images, producing visually coherent edits across diverse urban and natural environments. Quantitative metrics and human evaluations confirm that FSCM-Diffusion outperforms existing approaches in generating structure-preserving, intensity-controlled climate transformations. This framework opens new possibilities for disaster simulation, climate-aware urban planning, and temporally consistent visual effects generation.


![설명 텍스트](assets/Figure1.png)

## 🌟 Features
- Fused Structural Condition Map (FSCM)
- Level-based flood/snow control
- Style-LoRA integration

## 📁 Model Pipeline
![설명 텍스트](assets/Figure2.png)

## 📁 Ablatino Stury
![설명 텍스트](assets/Figure6.png)
![설명 텍스트](assets/Figure7.png)


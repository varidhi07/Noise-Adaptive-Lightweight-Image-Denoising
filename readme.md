# Noise Adaptive-Lightweight Image Denoiser

A lightweight (`44.45K` parameters, `2.652` GMac at `256×256`) blind image denoising framework designed to handle **Gaussian**, **Impulse (salt-and-pepper)**, and **Mixed** noise within a single unified model, accompanied by a 3-seed IEEE evaluation and ablation suite.

## Overview

Single-noise denoisers degrade sharply when tested off-domain, while static mixed-trained backbones must compromise between incompatible noise statistics. `NoiseAdaptiveDenoiser` resolves this by estimating a continuous 2D noise descriptor $d = (\hat{d}_{\text{gauss}}, \hat{d}_{\text{imp}}) \in [0, 1]^2$ from a fixed high-pass residual and using zero-initialized **Post-Sigmoid FiLM** (`ResidualFiLM`) to modulate Efficient Channel Attention (`ECA`) weights across six depthwise-separable residual blocks.

## Architecture Summary

<img width="1327" height="937" alt="architecture" src="https://github.com/user-attachments/assets/aeea30e4-6085-4f71-9ab7-76221269f663" />


1. **`NoiseDescriptorHead`**:

   * Fixed $5\times 5$ binomial `HighPass` filter ($x - \text{blur}_{5\times 5}(x)$, `groups=3`) to strip image content and isolate noise.

   * Two $3\times 3$ `Conv2d` layers (`3 → 16 → 32`, `ReLU`) followed by resolution-invariant global `mean + std` pooling (`64`-D).

   * Two `Linear` layers (`64 → 32 → 2`) with `Sigmoid` output $d = (\hat{d}_{\text{gauss}}, \hat{d}_{\text{imp}})$, plus conditioning https://www.google.com/search?q=dropout ($p_{\text{https://www.google.com/search?q=drop}} = 0.25$ during training).

2. **Backbone (`Stem` + 6 × `NoiseConditionedRAB` + `Tail`)**:

   * `Stem`: $3\times 3$ `Conv2d` (`3 → 48`).

   * `NoiseConditionedRAB`: Two depthwise-separable blocks (`3×3` depthwise + `1×1` pointwise, `ReLU`), `ECA` (`GAP` + `Conv1d` $k=3$), and **Post-Sigmoid `ResidualFiLM`** ($\alpha = \text{clamp}(\sigma(z) \odot (1 + m\Delta\gamma) + m\beta,\, 0,\, 1)$) with local residual skip.

   * `Tail`: $3\times 3$ `Conv2d` (`48 → 3`) predicting the noise residual $\hat{n}$, subtracted from the input ($\hat{x} = x - \hat{n}$).

## Datasets

* **Training (`BSD400`, deduplicated)**: Downloads `BSDS500` train/val (`300` images) and removes all `68` `CBSD68` test image stems, leaving **`232` leak-free training images** (`repeats=4` random `64×64` patches/epoch = `928` samples/epoch, `1,450` gradient steps over `50` epochs).

* **Held-Out Test Benchmarks**:

  * `CBSD68` (`68` images, from `FFDNet`)

  * `Set12` (`12` images, from `KAIR`)

  * `Kodak24` (`24` images, lossless Kodak suite)

## Experimental Suite

All models are trained across **3 random seeds** (`[42, 123, 456]`) using `Adam` (`lr=1e-3`, `StepLR` at epoch 30, `gamma=0.5`) and unified best-train-PSNR checkpoint selection:

* **Literature-Pattern Baselines**: `HKCNN-style (44K, matched)`, `LWIFCNN-style`, `NAF-style`, `Dual-Attention (CBAM-style)`, `Cross-Channel (QMSANet-style)`, and `Noise-Adaptive (ours)`.

* **Ablation Variants**: `Full Model (Ours)` (Post-Sigmoid FiLM), `Pre-Sigmoid FiLM`, `w/o FiLM (Static)`, `w/o High-Pass`, `w/o Cond. Dropout`, `w/o Aux Supervision`, `Mean-Pool Only`, and inference-time `Full Model (cond_on=False)`.

* **Additional Evaluations**: Cross-domain single-noise specialists (`Gaussian-only`, `Impulse-only`), 9-level severity sweep on `CBSD68`, out-of-range severities ($\sigma \in \{10, 65\}$, $p=0.40$, mixed $35+0.20$), unseen noise types (**Speckle**, **Poisson**, **Gaussian+Speckle**), noise-head calibration (MAE, $R^2$), and 3-seed paired *t*-tests.

## How to Run

### Kaggle (Recommended)

1. Upload `imagedenoising_fixed_headv2.ipynb` to a Kaggle Notebook.

2. In **Session options**, set **Accelerator** to **GPU T4 x2** (or **GPU P100**) and toggle **Internet** to **On**.

3. Run all cells—or click **Save Version → Save & Run All (Commit)** to run unattended in the background.

4. Outputs are written to `/kaggle/working/denoising_project/ablation_ieee_results_v5/`.

### Google Colab

1. Upload `imagedenoising_fixed_headv2.ipynb` to Google Colab and select **Runtime → Change runtime type → T4 GPU**.

2. Run all cells; checkpoints and results persist to `/content/drive/MyDrive/denoising_project/ablation_ieee_results_v5/`.

## Exported IEEE Outputs

Inside `ablation_ieee_results_v5/`:

* **`ieee_ablation_and_comparison_results.xlsx`**: 10-sheet formatted workbook (`Literature_Comparison`, `Ablation_Summary`, `Cross_Domain_Eval`, `OutOfRange_UnseenNoise`, `Multi_Dataset_Eval`, `Multi_Severity_Eval`, `Literature_Rankings`, `Ablation_Rankings`, `Noise_Head_Accuracy`, `Statistical_Tests`).

* **`csv/`**: Individual CSV files for all tables and per-seed runs.

* **`ieee_comparison_table.tex` & `ieee_ablation_table.tex`**: Ready-to-paste `booktabs` LaTeX tables with `\textbf{}` and `\underline{}` highlighting.

* **`figures/`**: 300-DPI `.png` and vector `.pdf` IEEE figures (PSNR & rank heatmaps, cross-domain heatmap, out-of-range/unseen noise heatmap, Params/GMac Pareto frontier, and noise-head calibration).

## References

### Core Conditioning & Attention Mechanisms

* **FiLM**: E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual Reasoning with a General Conditioning Layer," *AAAI Conference on Artificial Intelligence (AAAI)*, 2018. ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11671) | [arXiv](https://arxiv.org/abs/1709.07871))

* **ECA-Net**: Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 11534–11542. ([CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html) | [arXiv](https://arxiv.org/abs/1910.03151))

### Literature-Pattern Baselines

*(Note: The baselines in the notebook are controlled 6-block architectural pattern re-implementations inspired by the works below, trained under an identical protocol.)*

* **HKCNN-style**: "Efficient image denoising with heterogeneous kernel-based CNN" (HKCNN). ([ResearchGate](https://www.researchgate.net/publication/380404203_Efficient_image_denoising_with_heterogeneous_kernel-based_CNN))

* **LWIFCNN-style**: "A lightweight CNN based information fusion for image denoising" (LWIFCNN). ([ResearchGate](https://www.researchgate.net/publication/373191992_A_lightweight_CNN_based_information_fusion_for_image_denoising))

* **NAF-style (NAFNet)**: L. Chen, X. Chu, X. Zhang, and J. Sun, "Simple Baselines for Image Restoration," *European Conference on Computer Vision (ECCV)*, 2022. ([ECCV PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf) | [arXiv](https://arxiv.org/abs/2204.04676))

* **Dual-Attention (CBAM-style)**: S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," *European Conference on Computer Vision (ECCV)*, 2018. ([CVF Open Access](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html) | [arXiv](https://arxiv.org/abs/1807.06521))

* **Cross-Channel (QMSANet-style)**: Y. Liu, Q. Xie, Y. Guo, G. Chen, B. Wu, D. Meng, J.-M. Morel, Q. Jin, and M. K. Ng, "A Quaternion Multi-Scale Attention Network for Color Image Denoising" (QMSANet). ([IPOL](https://www.ipol.im/pub/pre/607/))

### Evaluation Metrics & Datasets

* **SSIM**: Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, Apr. 2004. ([DBLP](https://dblp.org/rec/journals/tip/WangBSS04) | [PubMed](https://pubmed.ncbi.nlm.nih.gov/15376593/))

* **Datasets**: `BSDS500` ([GitHub](https://github.com/BIDS/BSDS500)), `KAIR` (`Set12`, [GitHub](https://github.com/cszn/KAIR)), `FFDNet` (`BSD68` / `CBSD68`, [GitHub](https://github.com/cszn/FFDNet)).

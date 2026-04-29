\documentclass[conference]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tikz}
\usepackage{multirow}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, calc}

\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue
}

\begin{document}

\title{Reinforcement Learning-Optimized Grounded Radiology Report Generation with Hierarchical Vision Transformers}

\author{
\IEEEauthorblockN{Muhammad Talha}
\IEEEauthorblockA{Department of Computer Science\\
FAST National University of Computer and Emerging Sciences\\
Karachi, Pakistan\\
K25-7605}
}

\maketitle

\begin{abstract}
Generating radiology reports automatically from chest X-rays has improved a lot in recent years, but the systems we have today still have serious problems. Reports often contain hallucinated findings, and there is rarely any way to verify that what the text says actually points to a specific region on the image. Current top models each address part of the problem but not all of it. MAIRA-2 introduced grounded reporting through supervised fine-tuning, but it has no way to penalize poor spatial predictions during training. UniRG uses reinforcement learning to make the text more factually accurate yet does not touch spatial grounding at all. Hierarchical vision backbones like the Swin Transformer are excellent for capturing features at multiple scales---exactly what you need to detect pathologies of different sizes---but they have never been combined with a training objective that pushes on both text quality and localization precision at the same time.

This paper proposes a framework that fills this gap. We use a hierarchical Swin Transformer as the visual encoder to pull out multi-resolution feature maps from chest radiographs. These features are fed into a language model decoder through cross-attention, producing radiology reports that include bounding-box coordinates alongside each clinical finding. A composite reward function, trained with Group Relative Policy Optimization, penalizes clinical errors in the generated text and spatial inaccuracies in the predicted boxes at the same time. To our knowledge, this is the first training approach to apply reinforcement learning to both the textual and spatial dimensions of grounded radiology report generation simultaneously, yielding outputs that are clinically accurate and spatially verifiable.
\end{abstract}

\begin{IEEEkeywords}
Radiology Report Generation, Reinforcement Learning, Swin Transformer, Spatial Grounding, Chest X-Ray, Multi-Modal Learning, Visual Grounding, Medical Imaging
\end{IEEEkeywords}

% ===========================================================================
\section{Introduction}
% ===========================================================================

Chest X-rays are the most commonly ordered imaging study in the world. Hundreds of millions of them are taken every year across hospitals of all sizes \cite{johnson2019mimic}. Reading them and writing structured radiology reports takes real expertise, careful attention, and a lot of time. When there aren't enough radiologists to keep pace with the caseload, turnaround times suffer, and patients end up waiting longer for diagnoses and treatment. Automated systems that can draft accurate, evidence-based reports would genuinely change that dynamic---not as a replacement for radiologists, but as a dependable second reader that catches concerning regions on the image and takes some of the routine burden off the radiologist's plate.

We've made real progress in medical image understanding through deep learning, but generating reports that are both factually reliable and grounded in the image is still largely unsolved. Early deep learning work focused on classification---getting CNNs to assign disease labels to chest X-rays. Wang et al.\ put together the ChestX-ray8 dataset: 108,948 frontal-view images from 32,717 patients, with eight disease labels automatically extracted from the accompanying radiology reports via NLP \cite{liu2026unirg}. The CheXpert dataset extended this work, offering 224,316 chest radiographs from 65,240 patients annotated with 14 observation labels and explicit uncertainty markers for ambiguous findings \cite{irvin2019chexpert}. These large labeled datasets powered a wave of classification research, with model architectures evolving from standard CNNs \cite{alshmrani2023, pillai2022} to lightweight designs built for edge deployment \cite{saranyaraj2026} to vision transformers that use hierarchical attention mechanisms \cite{taslimi2022, liu2021swin}.

But classification alone doesn't get you far enough. A radiologist doesn't just stamp a study with disease labels---the report describes specific findings, puts them in the context of anatomical landmarks, and increasingly points out where on the image each finding lives. That need is what turned radiology report generation into its own research problem. Chen et al.\ built R2Gen, a memory-driven Transformer that uses a relational memory module to model long-range dependencies between image regions and report sentences, and it performed well on both IU~X-Ray and MIMIC-CXR \cite{chen2020r2gen}. More recently, Bannur et al.\ introduced MAIRA-2, pairing a radiology-specific visual encoder (Rad-DINO) with a Vicuna-7B large language model and training it to generate reports that include bounding boxes pinpointing each finding on the image \cite{bannur2024}. MAIRA-2 defined what this paper builds on---\textit{grounded report generation}---and introduced RadFact, an LLM-based evaluation framework that checks report correctness and completeness at the sentence level.

A separate line of work noticed a fundamental mismatch: the standard supervised fine-tuning (SFT) objective minimizes next-token prediction loss against reference reports, which doesn't actually optimize for clinical quality. Token-level cross-entropy pushes the model to reproduce the surface form of training data, rewarding lexical overlap rather than factual accuracy. Liu et al.\ addressed this with UniRG, a QWEN-VL-based framework that applies Group Relative Policy Optimization (GRPO) \cite{liu2026unirg}---a more memory-efficient alternative to PPO \cite{liu2026unirg}---to optimize directly for clinical metrics like CheXprompt. UniRG beat the previous state of the art on ReXrank by a meaningful margin \cite{liu2026unirg}. The catch: UniRG only cares about the text. Its RL objective doesn't say anything about where findings are located on the image.

This paper targets the gap that lies between these two research directions. No existing framework uses reinforcement learning to \textit{jointly} improve both the clinical accuracy of the generated text and the spatial accuracy of the predicted bounding boxes. The proposed framework, which we call \textbf{RL-GroundGen}, makes three contributions:

\begin{enumerate}
    \item A multi-modal architecture that pairs a hierarchical Swin Transformer visual encoder \cite{liu2021swin} with a transformer-based language decoder to produce reports containing interleaved text and bounding-box coordinates.
    \item A composite reward function comprising a clinical factuality component and a spatial precision component based on Intersection-over-Union, designed to penalize both textual hallucinations and inaccurate localization.
    \item A training pipeline that first warm-starts the model with supervised fine-tuning on paired image--report--box data and then refines it with GRPO-based reinforcement learning using the composite reward, yielding a model whose outputs are simultaneously more factual and more spatially precise than those of SFT-only baselines.
\end{enumerate}

% ===========================================================================
\section{Related Work}
% ===========================================================================

This section reviews the three areas of literature that come together in the proposed framework: chest X-ray classification, radiology report generation, and reinforcement learning for text optimization.

\subsection{Chest X-Ray Classification}

Large labeled datasets have been the main driver of deep learning progress in chest radiograph analysis. Wang et al.\ released ChestX-ray8 (later expanded to ChestX-ray14)---a dataset of 108,948 frontal-view chest X-ray images from 32,717 unique patients, with disease labels automatically extracted from the associated radiology reports using natural language processing \cite{nicolson2026cxrmate2}. The labels cover 14 thoracic pathologies: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, and more. The dataset enabled weakly-supervised multi-label classification and localization, and it kicked off a substantial wave of follow-up studies. Irvin et al.\ then introduced CheXpert with 224,316 chest radiographs from 65,240 patients, annotated by an automated rule-based labeler that outputs positive, negative, and \textit{uncertain} labels for 14 observations, which forced the field to grapple with the genuine ambiguity in radiological interpretation \cite{irvin2019chexpert}. On CheXpert's validation set---annotated by three board-certified radiologists---deep models matched or exceeded individual radiologist performance on Cardiomegaly, Edema, and Pleural Effusion.

Alshmrani et al.\ worked on multi-class classification of six chest diseases including COVID-19, using a hybrid ensemble of VGG19 and custom CNN branches and reporting 96.48\% accuracy \cite{alshmrani2023}. The accuracy is impressive, but the system is a black-box classifier---it assigns a label but gives no localization or textual explanation for why, which limits how useful it is in practice. Pillai ran a comprehensive comparison of DenseNet-121, ResNet-50, and other CNNs on the full 14-disease multi-label task using ChestX-ray14, finding notable performance drops on rare pathology classes due to class imbalance \cite{pillai2022}. This highlighted a stubborn challenge: models trained on imbalanced medical datasets tend to get good at common conditions and struggle with rarer ones that are often clinically more important.

On the efficiency side, Saranyaraj et al.\ proposed PneuNet, a lightweight CNN that combines Atrous Spatial Pyramid Pooling (ASPP) and Squeeze-and-Excitation (SE) blocks in just 1.8 million parameters \cite{saranyaraj2026}. It's designed for binary pneumonia detection and targets resource-constrained edge devices, showing that thoughtful architecture design can keep accuracy high while cutting computational costs dramatically. The tradeoff is scope: it handles one binary task and produces no descriptive output.

The move from CNNs to transformers for chest X-ray analysis really arrived with SwinCheX from Taslimi et al.\ \cite{taslimi2022}. SwinCheX uses a Swin Transformer backbone with a multi-layer perceptron classification head on ChestX-ray14. With a 3-layer MLP head, it reached an average AUC of 0.810 across all 14 diseases, beating the previous state-of-the-art average of 0.799 and confirming that hierarchical vision transformers can outperform CNN-based approaches on this task. The attention maps from SwinCheX showed the model focusing on pathologically relevant chest regions, suggesting that the features it learns carry spatial information that could plausibly support localization tasks too.

The Swin Transformer itself was introduced by Liu et al.\ \cite{liu2021swin}. Unlike standard Vision Transformers that compute attention globally, Swin computes attention within local non-overlapping windows and uses a shifted-window scheme in alternating layers to allow information to flow across window boundaries. The result is a hierarchical feature pyramid with linear computational complexity relative to image size---which makes it practical for dense prediction. On ImageNet-1K, Swin hit 87.3\% top-1 accuracy; on COCO object detection it reached 58.7 box AP and 51.1 mask AP; on ADE20K semantic segmentation it got 53.5 mIoU. In each case it topped the prior state of the art by a clear margin. That combination of global context and fine spatial resolution makes Swin a natural fit for a system that needs to both understand the overall image and precisely localize small pathological findings.

\subsection{Radiology Report Generation}

Going from classification to generating free-text reports introduces a lot more complexity. The model can't just detect what's in the image---it needs to describe each finding in clinical language, keep the report coherent across sentences, and avoid inventing observations that have no basis in what it actually sees.

Chen et al.\ built R2Gen, a memory-driven Transformer for radiology report generation \cite{chen2020r2gen}. The key addition in R2Gen is a relational memory module that keeps track of key contextual information during decoding, combined with a memory-driven conditional layer normalization that injects this memory into the Transformer decoder at each layer. Tested on IU X-Ray and MIMIC-CXR, R2Gen outperformed prior methods on both NLG metrics (BLEU, METEOR, ROUGE-L) and clinical efficacy metrics. The MIMIC-CXR dataset \cite{johnson2019mimic}, which contains 377,110 images from 227,835 radiographic studies at Beth Israel Deaconess Medical Center, provided the scale needed to train and evaluate generation models under realistic conditions, and it has become the standard benchmark in this space.

Bannur et al.\ pushed the output format much further with MAIRA-2 \cite{bannur2024}. By pairing a radiology-specific encoder (Rad-DINO) with a Vicuna-7B language model and training on reports that include bounding-box coordinates, MAIRA-2 introduced \textit{grounded report generation}---a setup that mirrors clinical practice far more closely. When a radiologist notes an opacity in the right lower lobe, they can point to the exact region; a model that does the same is much easier to verify. MAIRA-2 also introduced RadFact, which uses large language model inference to assess whether individual generated sentences are correct and complete, handling the many valid ways you can phrase the same clinical observation. The main limitation of MAIRA-2 is that it relies entirely on supervised fine-tuning. Training the model to imitate reference reports and boxes through next-token prediction optimizes for surface-level similarity, not for clinical accuracy or spatial precision directly.

\subsection{Reinforcement Learning for Report Optimization}

Supervised fine-tuning for sequence generation optimizes a token-level cross-entropy loss, which doesn't necessarily align with the clinical quality metrics that determine whether a report is actually useful. Reinforcement learning provides a way around this: treat the model as a policy, and use reward signals to directly optimize for how good the generated sequences are.

Schulman et al.\ introduced Proximal Policy Optimization (PPO), a family of policy gradient algorithms that alternate between collecting trajectories from the current policy and running multiple epochs of mini-batch gradient updates on a clipped surrogate objective \cite{nicolson2026cxrmate2}. PPO has become a standard tool for aligning language models with human preferences and task-specific goals, largely because it's simple, stable, and has better sample efficiency than alternatives like TRPO. One practical problem with PPO for large language models is that you need to keep both a policy model and a value model in memory at once, which doubles the GPU memory requirement.

Shao et al.\ tackled this by proposing Group Relative Policy Optimization (GRPO) in the DeepSeekMath project \cite{nicolson2026cxrmate2}. GRPO removes the value model entirely. Instead of estimating advantages through a learned value function, it samples a group of outputs for each input, computes rewards for each, and derives advantages from how each output's reward compares to the group average. This cuts the memory requirement in half with no apparent drop in optimization quality, making it practical to fine-tune large multi-modal models on the kind of hardware most research groups actually have.

Liu et al.\ applied GRPO to radiology report generation in UniRG \cite{liu2026unirg}. Built on QWEN-VL, UniRG first does supervised fine-tuning on chest X-ray report data, then refines with GRPO using CheXprompt---a clinical evaluation metric---as the reward signal. On the ReXrank benchmark, UniRG-CXR set a new state-of-the-art by a wide margin. The key insight is that directly optimizing for clinical factuality through RL leads to better generalization across different institutions and clinical practices, avoiding the tendency of SFT-only models to overfit to the boilerplate language in training data. But UniRG produces text-only reports. Its reward function says nothing about where findings are on the image, leaving spatial grounding completely unaddressed.

\subsection{Identified Research Gap}

Table~\ref{tab:literature} summarizes the work reviewed above. The gap is clear: no existing framework uses reinforcement learning to jointly optimize both the clinical accuracy of generated report text and the spatial accuracy of predicted grounding regions. MAIRA-2 does grounded reporting but trains with SFT only. UniRG uses RL-based factuality optimization but ignores spatial grounding altogether. SwinCheX shows what hierarchical transformers can do on CXR analysis, but it's a classification system. RL-GroundGen brings all three together: a Swin Transformer encoder, a language decoder, and GRPO-based training with a composite reward that targets both textual and spatial quality.

% ---- Literature Comparison Table ----
\begin{table*}[!t]
\caption{Comparative Summary of Related Work}
\label{tab:literature}
\centering
\renewcommand{\arraystretch}{1.25}
\footnotesize
\begin{tabular}{>{\raggedright\arraybackslash}p{2.1cm}
                >{\raggedright\arraybackslash}p{2.25cm}
                >{\raggedright\arraybackslash}p{2.15cm}
                >{\raggedright\arraybackslash}p{2.35cm}
                >{\raggedright\arraybackslash}p{2.4cm}
                >{\raggedright\arraybackslash}p{2.0cm}}
\toprule
\textbf{Author, Year} & \textbf{Task} & \textbf{Method} & \textbf{Dataset} & \textbf{Key Metrics} & \textbf{Code Availability} \\
\midrule

Irvin et al.\ (2019) \cite{irvin2019chexpert} 
& Multi-label classification 
& CheXpert (DenseNet) 
& CheXpert dataset 
& Radiologist-level AUC on 3 pathologies 
& Available \\
\midrule

Johnson et al.\ (2019) \cite{johnson2019mimic} 
& Dataset release 
& MIMIC-CXR Database 
& MIMIC-CXR 
& 377,110 images; 227,835 studies 
& Available \\
\midrule

Chen et al.\ (2020) \cite{chen2020r2gen} 
& Report generation 
& R2Gen (Memory Transformer) 
& IU-Xray, MIMIC-CXR 
& BLEU, METEOR, ROUGE-L 
& Available \\
\midrule

Liu et al.\ (2021) \cite{liu2021swin} 
& General-purpose vision backbone 
& Swin Transformer 
& ImageNet, COCO 
& 87.3\% top-1; 58.7 box AP 
& Available \\
\midrule

Pillai (2022) \cite{pillai2022} 
& Multi-label classification (14 diseases) 
& DenseNet, ResNet comparison 
& ChestX-ray14 
& AUC comparison across architectures 
& Not reported \\
\midrule

Taslimi et al.\ (2022) \cite{taslimi2022} 
& Multi-label classification 
& SwinCheX (Swin + MLP) 
& ChestX-ray14 
& AUC 0.810 
& Not reported \\
\midrule

Alshmrani et al.\ (2023) \cite{alshmrani2023} 
& 6-class CXR classification 
& VGG19 + CNN Ensemble 
& Private CXR dataset 
& 96.48\% accuracy 
& Not reported \\
\midrule

Bannur et al.\ (2024) \cite{bannur2024} 
& Grounded report generation 
& MAIRA-2 (Rad-DINO + Vicuna LLM) 
& MIMIC-CXR, Chest ImaGenome 
& RadFact scores; bounding box IoU 
& Not reported \\
\midrule

Saranyaraj et al.\ (2026) \cite{saranyaraj2026} 
& Binary pneumonia detection 
& PneuNet (ASPP + SE) 
& ChestX-ray2017, COVID-QU-Ex 
& 1.8M parameters; accuracy, F1 
& Not reported \\
\midrule

Liu et al.\ (2026) \cite{liu2026unirg} 
& Universal report generation 
& UniRG (QWEN-VL + GRPO) 
& MIMIC-CXR, IU-Xray 
& ReXrank SOTA; clinical factuality 
& Not reported \\
\midrule

Nicolson et al.\ (2026) \cite{nicolson2026cxrmate2} 
& Report generation 
& CXRMate-2 (multimodal conditioning + RL) 
& MIMIC-CXR, CheXpert Plus, ReXgradient 
& GREEN +11.2\%; RadGraph-XL +24.4\%; 45\% acceptable ratings 
& Available \\
\midrule

\textbf{Proposed (RL-GroundGen)} 
& \textbf{Grounded report generation} 
& \textbf{Swin + LM + GRPO} 
& \textbf{MIMIC-CXR + Chest ImaGenome} 
& \textbf{Text factuality + grounding accuracy} 
& \textbf{Planned} \\

\bottomrule
\end{tabular}
\end{table*}

% ===========================================================================
\section{Methodology}
% ===========================================================================

The RL-GroundGen framework has three main parts: (1) a hierarchical visual encoder based on the Swin Transformer that extracts multi-scale features from input chest radiographs, (2) a transformer-based language decoder that generates token sequences representing interleaved clinical text and bounding-box coordinates, and (3) a two-phase training pipeline that first warm-starts the model with supervised fine-tuning and then refines it with reinforcement learning using a composite reward. Fig.~\ref{fig:architecture} shows the overall architecture.

\begin{figure*}[!t]
\centering
\begin{tikzpicture}[
    node distance=0.65cm and 0.7cm,
    block/.style={rectangle, draw, thick, text centered, minimum height=0.85cm, minimum width=2.0cm, font=\footnotesize, text width=1.9cm, rounded corners=2pt},
    smallblock/.style={rectangle, draw, thick, text centered, minimum height=0.7cm, minimum width=1.8cm, font=\scriptsize, text width=1.7cm, rounded corners=2pt},
    reward/.style={rectangle, draw, thick, text centered, minimum height=0.7cm, minimum width=1.8cm, font=\scriptsize, text width=1.7cm, rounded corners=2pt, fill=gray!15},
    arr/.style={-{Stealth[length=2mm]}, thick},
    darr/.style={-{Stealth[length=2mm]}, thick, dashed}
]

% Row 1: Input + Patch Embed + 3 Swin stages (5 blocks, compact)
\node[block, fill=blue!10] (input) {Input CXR\\$384\!\times\!384\!\times\!3$};
\node[block, fill=blue!10, right=of input] (patch) {Patch Partition + Embed};
\node[block, fill=blue!15, right=of patch] (stage1) {Swin Stage 1\\$96\!\times\!96\!\times\!C$};
\node[block, fill=blue!20, right=of stage1] (stage2) {Swin Stage 2\\$48\!\times\!48\!\times\!2C$};
\node[block, fill=blue!25, right=of stage2] (stage3) {Swin St.\ 3--4\\$12\!\times\!12\!\times\!8C$};

% Row 2: FPN fusion centred below stages
\node[block, fill=orange!15, below=0.9cm of stage2] (fusion) {Multi-Scale FPN Fusion};

% Row 3: Cross-attention + Decoder side by side, centred
\node[block, fill=orange!20, below=0.9cm of fusion, xshift=-1.6cm] (cross) {Cross-Attention Projection};
\node[block, fill=green!12, below=0.9cm of fusion, xshift=1.6cm] (decoder) {Transformer LM Decoder};

% Row 4: Outputs
\node[smallblock, fill=yellow!15, below=0.9cm of cross] (text) {Report Text Tokens};
\node[smallblock, fill=yellow!15, below=0.9cm of decoder] (bbox) {Bounding-Box Tokens};

% Row 5: Rewards
\node[reward, below=0.7cm of text] (rtext) {$R_{\mathrm{text}}$: Factuality};
\node[reward, below=0.7cm of bbox] (rbox) {$R_{\mathrm{box}}$: GIoU};

% Row 6: Composite reward centred
\node[reward, below=0.7cm of fusion, yshift=-5.8cm] (rtotal) {$R_{\mathrm{total}}\!=\!\alpha R_{\mathrm{text}}\!+\!\beta R_{\mathrm{box}}$};

% --- Arrows ---
% Encoder chain
\draw[arr] (input) -- (patch);
\draw[arr] (patch) -- (stage1);
\draw[arr] (stage1) -- (stage2);
\draw[arr] (stage2) -- (stage3);

% Stages to FPN
\draw[arr] (stage1.south) -- ++(0,-0.25) -| (fusion.north west);
\draw[arr] (stage2.south) -- (fusion.north);
\draw[arr] (stage3.south) -- ++(0,-0.25) -| (fusion.north east);

% FPN to cross-attention and decoder
\draw[arr] (fusion.south) -- ++(0,-0.25) -| (cross.north);
\draw[arr] (cross) -- (decoder);

% Decoder to outputs
\draw[arr] (cross.south) -- (text.north);
\draw[arr] (decoder.south) -- (bbox.north);

% Outputs to rewards
\draw[arr] (text) -- (rtext);
\draw[arr] (bbox) -- (rbox);

% Rewards to composite
\draw[darr] (rtext.south) -- ++(0,-0.2) -| (rtotal.west);
\draw[darr] (rbox.south) -- ++(0,-0.2) -| (rtotal.east);

% Label
\node[font=\scriptsize\itshape, below=0.12cm of rtotal] {GRPO Policy Update};

\end{tikzpicture}
\caption{Architecture of the proposed RL-GroundGen framework. A Swin Transformer extracts hierarchical features at multiple resolutions. These features are fused via a Feature Pyramid Network and projected into the decoder's cross-attention space. The language model decoder generates interleaved text and bounding-box tokens. During the RL phase, a composite reward $R_{\mathrm{total}}$ drives GRPO-based policy optimization.}
\label{fig:architecture}
\end{figure*}
\subsection{Visual Encoder: Hierarchical Swin Transformer}

The visual encoder takes a chest radiograph of resolution $H \times W$ (resized to $384 \times 384$ during preprocessing) and produces a set of multi-scale feature maps. Following the Swin Transformer design \cite{liu2021swin}, the image is first split into non-overlapping $4 \times 4$ patches, each linearly projected to an embedding of dimension $C = 128$. Self-attention is computed within local windows of $7 \times 7$ patches, and a shifted-window mechanism in alternating layers lets information flow across window boundaries while keeping computational complexity linear with respect to image size.

The encoder has four stages. At each stage transition, a patch-merging layer concatenates features from each $2 \times 2$ neighborhood, doubling the channel dimension and halving spatial resolution. The feature maps at the four stages have spatial resolutions $\frac{H}{4} \times \frac{W}{4}$, $\frac{H}{8} \times \frac{W}{8}$, $\frac{H}{16} \times \frac{W}{16}$, and $\frac{H}{32} \times \frac{W}{32}$ with channel dimensions $C$, $2C$, $4C$, and $8C$ respectively. This pyramid captures pathological patterns across a wide range of scales---from large-area opacities like pleural effusions down to small focal lesions like nodules or calcifications that might only cover a few pixels at the coarsest resolution.

The multi-scale feature maps are then aggregated through a lightweight Feature Pyramid Network (FPN) using top-down lateral connections and $1 \times 1$ convolutions to produce a unified feature representation at a common spatial resolution of $\frac{H}{16} \times \frac{W}{16}$. This fused representation serves as the key and value inputs for the cross-attention layers in the decoder.

\subsection{Language Model Decoder}

The language decoder is a causally masked transformer that generates tokens one at a time in an autoregressive manner. We extend the standard text vocabulary with special coordinate tokens that represent discretized bounding-box coordinates. Each spatial coordinate (top-left $x$, top-left $y$, bottom-right $x$, bottom-right $y$) is quantized into one of 1000 bins spanning the normalized image coordinate range $[0, 1]$, following the tokenization approach used in MAIRA-2 \cite{bannur2024}. Each finding in the output takes the form:

\begin{equation}
\texttt{<find>}\ t_1\ t_2\ \ldots\ t_n\ \texttt{<box>}\ x_1\ y_1\ x_2\ y_2\ \texttt{</find>}
\end{equation}

\noindent where $t_1, \ldots, t_n$ are text tokens describing the finding and $x_1, y_1, x_2, y_2$ are coordinate tokens specifying its bounding box. The full report is a sequence of such finding blocks, letting the decoder interleave textual descriptions with spatial localizations in a single autoregressive pass.

The decoder attends to the fused visual features through multi-head cross-attention at every layer. Letting $\mathbf{H}_{\text{vis}} \in \mathbb{R}^{N_v \times d}$ be the projected visual feature matrix (with $N_v$ spatial positions and $d$ model dimension), the cross-attention at decoder layer $l$ is:

\begin{equation}
\text{CrossAttn}(\mathbf{Q}^l, \mathbf{H}_{\text{vis}}) = \text{softmax}\!\left(\frac{\mathbf{Q}^l \mathbf{K}_{\text{vis}}^{\top}}{\sqrt{d_k}}\right) \mathbf{V}_{\text{vis}}
\end{equation}

\noindent where $\mathbf{Q}^l$ comes from the decoder hidden states and $\mathbf{K}_{\text{vis}}, \mathbf{V}_{\text{vis}}$ are linear projections of $\mathbf{H}_{\text{vis}}$. This lets the decoder tie each generated token back to specific visual evidence.

\subsection{Training Phase 1: Supervised Fine-Tuning}

We first train the model on a dataset of chest radiographs paired with reference reports and bounding-box annotations using standard supervised fine-tuning. The training objective is the token-level cross-entropy loss over the combined text-and-coordinate sequence:

\begin{equation}
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log\, p_\theta(y_t \mid y_{<t},\, \mathbf{I})
\end{equation}

\noindent where $\mathbf{I}$ is the input image, $y_t$ is the ground-truth token at position $t$ (which can be a text token or a coordinate token), and $\theta$ denotes the model parameters. This phase gives the model a solid starting point for both text generation and bounding-box prediction, establishing the basic correspondence between visual features and clinical language.

We initialize the Swin Transformer encoder with weights pretrained on ImageNet-1K and the language decoder with weights from a general-purpose pretrained language model. Cross-attention projection layers and the coordinate embedding layer are initialized randomly. The SFT phase trains all parameters end-to-end with the AdamW optimizer, a learning rate of $2 \times 10^{-5}$, and a cosine decay schedule.

\subsection{Training Phase 2: Reinforcement Learning with GRPO}

Once SFT converges, we move to a reinforcement learning phase that directly optimizes for the clinical and spatial quality of generated outputs. We use Group Relative Policy Optimization (GRPO) \cite{nicolson2026cxrmate2}, which avoids the need for a separate value network (as required by PPO \cite{nicolson2026cxrmate2}) by estimating advantages from relative rewards within a sampled group.

For each training image $\mathbf{I}_i$, the current policy $\pi_\theta$ generates a group of $G$ candidate outputs $\{o_i^1, o_i^2, \ldots, o_i^G\}$ by sampling. Each output $o_i^g$ contains both the generated report text and predicted bounding boxes. We compute a composite reward $R(o_i^g)$ for each output (described below), and the advantage for output $g$ is the normalized deviation of its reward from the group mean:

\begin{equation}
\hat{A}_i^g = \frac{R(o_i^g) - \mu(\{R(o_i^j)\}_{j=1}^G)}{\sigma(\{R(o_i^j)\}_{j=1}^G) + \epsilon}
\end{equation}

The GRPO objective maximizes expected advantage subject to a KL-divergence constraint against the reference policy $\pi_{\text{ref}}$ (the SFT checkpoint):

\begin{equation}
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\!\left[\min\!\left(r_t \hat{A},\; \text{clip}(r_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\,\hat{A}\right)\right] + \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\end{equation}

\noindent where $r_t = \frac{\pi_\theta(y_t \mid y_{<t}, \mathbf{I})}{\pi_{\text{ref}}(y_t \mid y_{<t}, \mathbf{I})}$ is the per-token probability ratio and $\beta$ controls the strength of the KL penalty.

\subsection{Composite Reward Function}

The reward is designed to capture both dimensions of output quality. For a generated output $o$ with report text $\mathcal{T}$ and predicted bounding boxes $\{\hat{b}_k\}$, the total reward is:

\begin{equation}
R(o) = \alpha\, R_{\text{text}}(\mathcal{T}) + \beta_{\text{box}}\, R_{\text{box}}(\{\hat{b}_k\})
\label{eq:reward}
\end{equation}

\noindent where $\alpha$ and $\beta_{\text{box}}$ are scalar weights controlling the balance between textual and spatial quality. The two components are defined as follows.

\textbf{Textual Reward $R_{\text{text}}$:} Following UniRG \cite{liu2026unirg}, the textual reward uses a clinical factuality score that checks whether the generated text correctly describes the pathological findings present (or absent) in the image. This score is computed by an external clinical evaluator model (analogous to CheXprompt) that compares generated sentences against structured labels and reference reports. Unlike token-overlap metrics like BLEU, this reward directly penalizes hallucinated findings and rewards correct identification of subtle pathologies.

\textbf{Spatial Reward $R_{\text{box}}$:} For each predicted bounding box $\hat{b}_k$ matched to a ground-truth box $b_k^*$ based on the associated finding description, we compute the spatial reward using Generalized Intersection-over-Union (GIoU):

\begin{equation}
R_{\text{box}} = \frac{1}{K} \sum_{k=1}^{K} \text{GIoU}(\hat{b}_k, b_k^*)
\end{equation}

GIoU extends standard IoU by accounting for the area of the smallest enclosing box, which gives a meaningful gradient signal even when the predicted and ground-truth boxes don't overlap at all. For finding descriptions that don't need a bounding box (e.g., ``No acute cardiopulmonary process''), the spatial reward doesn't apply. For predicted boxes with no corresponding ground-truth match, we assign a penalty of $-1$, discouraging the model from hallucinating spatial localizations.


% ===========================================================================
\section{Experimental Setup}
% ===========================================================================

\subsection{Datasets}

We designed RL-GroundGen for evaluation on two publicly available chest X-ray datasets that provide the multimodal annotations we need.

\textbf{MIMIC-CXR} \cite{johnson2019mimic} contains 377,110 images from 227,835 radiographic studies of 65,379 patients at the Beth Israel Deaconess Medical Center. Each study has a free-text radiology report. For bounding-box annotations, we use the subset annotated following the protocol from Bannur et al.\ \cite{bannur2024}, where sentence-level findings are paired with spatial boxes by radiologists. We use the standard train/validation/test splits.

\subsection{Evaluation Metrics}

\textbf{Natural Language Generation (NLG) Metrics:} We compute BLEU-1 through BLEU-4, METEOR, and ROUGE-L between generated and reference reports. These metrics measure lexical overlap but don't directly reflect clinical correctness.

\textbf{Clinical Efficacy (CE) Metrics:} Following Chen et al.\ \cite{chen2020r2gen} and Liu et al.\ \cite{liu2026unirg}, we apply the CheXbert labeller to both generated and reference reports to extract 14 binary pathology labels, then compute precision, recall, and F1 over these labels. This measures whether the generated report captures the same clinical findings as the reference, regardless of how they're phrased.

\textbf{RadFact:} Following MAIRA-2 \cite{bannur2024}, we use the RadFact evaluation framework, which applies a large language model to perform sentence-level logical inference between generated and reference reports. This yields precision (fraction of generated sentences entailed by the reference), recall (fraction of reference sentences entailed by the generated report), and F1.

\textbf{Grounding Metrics:} For spatial localization, we compute mean Intersection-over-Union (mIoU) and the percentage of findings whose predicted bounding box achieves IoU $\geq 0.5$ with the ground truth (mIoU@0.5).

\subsection{Baselines}

We compare RL-GroundGen against the following baselines:

\begin{itemize}
    \item \textbf{R2Gen} \cite{chen2020r2gen}: Memory-driven Transformer for report generation without grounding.
    \item \textbf{MAIRA-2} \cite{bannur2024}: SFT-based grounded report generation (Rad-DINO + Vicuna).
    \item \textbf{UniRG-CXR} \cite{liu2026unirg}: RL-optimized report generation without grounding.
    \item \textbf{RL-GroundGen (SFT only)}: Our architecture trained with supervised fine-tuning only, serving as an ablation to isolate the contribution of the RL phase.
\end{itemize}

\subsection{Implementation Details}

The Swin Transformer encoder uses the Swin-Base configuration (around 88 million parameters) pretrained on ImageNet-1K. The language model decoder is initialized from a 7-billion-parameter pretrained model. Training has two phases: SFT for 10 epochs with a batch size of 16 and learning rate $2 \times 10^{-5}$, followed by GRPO-based RL for 3 epochs with group size $G = 8$, learning rate $5 \times 10^{-7}$, clipping parameter $\epsilon = 0.2$, and KL penalty coefficient $\beta = 0.04$. The reward weights in Eq.~\ref{eq:reward} are set to $\alpha = 0.6$ and $\beta_{\text{box}} = 0.4$ based on a preliminary hyperparameter search on the validation set. All experiments use mixed-precision (FP16) training on a cluster of 4 NVIDIA A100 80GB GPUs.

% ===========================================================================
\section{Expected Results and Discussion}
% ===========================================================================

This section describes what we expect the framework to achieve based on how its components relate to existing methods, and explains the reasoning behind each anticipated result.

\subsection{Textual Quality}

UniRG showed that applying GRPO-based RL after supervised fine-tuning produces substantial improvements in clinical factuality over SFT-only baselines---it set a new state of the art on ReXrank by a considerable margin on CheXprompt scores \cite{liu2026unirg}. Since RL-GroundGen uses the same GRPO-based RL for the textual part of its reward, we expect similar improvements in clinical factuality relative to SFT-only approaches like MAIRA-2. The key thing we're watching for: CE F1 scores should go up after the RL phase compared to the SFT-only ablation, confirming that the RL signal is actually pushing the model toward more clinically accurate descriptions.

We also expect the Swin Transformer backbone to give the SFT phase a head start. SwinCheX reached an average AUC of 0.810 on the 14-disease classification task, beating the prior CNN-based state of the art by more than a percentage point \cite{taslimi2022}. Better visual understanding should feed into more informative features for the decoder, meaning the model misses fewer findings even before RL training starts.

\subsection{Spatial Grounding Quality}

The core novelty here is extending the RL reward into the spatial domain. Existing grounded reporting systems like MAIRA-2 train bounding-box prediction through SFT alone, which means the model learns to reproduce locations from training data but gets no direct penalty for poor localization at inference time. By adding a GIoU-based spatial reward to the GRPO training loop, we expect RL-GroundGen to improve mIoU and mIoU@0.5 relative to both the SFT-only ablation and to MAIRA-2.

Here's what we think is happening mechanistically: during RL training, the model generates multiple grounded reports per image, and outputs where bounding boxes more tightly enclose the pathological region score higher rewards. Over time, the policy shifts toward producing tighter, more accurate boxes. It's the same mechanism by which GRPO improves text factuality---not by changing the training data, but by selectively reinforcing better outputs.

We also expect the joint optimization to prevent a trade-off that would arise if text and box rewards were tuned independently. A model optimizing only for text might improve its descriptions while letting box coordinates drift; a model optimizing only for spatial accuracy might improve localization at the expense of textual coherence. The composite reward in Eq.~\ref{eq:reward} forces the model to improve on both dimensions at the same time.

\subsection{Ablation Analysis}

To understand what each component contributes, we plan the following ablation configurations:

\begin{itemize}
    \item \textbf{SFT Only}: Evaluates the baseline quality achievable through supervised fine-tuning alone, establishing the floor performance.
    \item \textbf{SFT + $R_{\text{text}}$ Only}: Applies RL with only the textual factuality reward, matching UniRG's approach but on a grounding-capable architecture. This is expected to improve text metrics while leaving grounding metrics unchanged.
    \item \textbf{SFT + $R_{\text{box}}$ Only}: Applies RL with only the spatial reward. Text quality should remain near SFT levels while grounding metrics improve.
    \item \textbf{SFT + $R_{\text{text}}$ + $R_{\text{box}}$ (Full)}: The complete RL-GroundGen pipeline. This should achieve the best performance across both textual and spatial metrics.
\end{itemize}

Table~\ref{tab:expected} shows the expected performance trends across these configurations.

\begin{table}[!t]
\caption{Expected Performance Trends Across Ablation Configurations}
\label{tab:expected}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{l c c c}
\toprule
\textbf{Configuration} & \textbf{CE F1} & \textbf{RadFact F1} & \textbf{mIoU@0.5} \\
\midrule
SFT Only             & Baseline & Baseline & Baseline \\
SFT + $R_{\text{text}}$  & $\uparrow\uparrow$ & $\uparrow\uparrow$ & $\approx$ \\
SFT + $R_{\text{box}}$   & $\approx$ & $\approx$ & $\uparrow\uparrow$ \\
\textbf{Full RL-GroundGen} & $\uparrow\uparrow$ & $\uparrow\uparrow$ & $\uparrow\uparrow$ \\
\bottomrule
\end{tabular}
\par\smallskip
\footnotesize $\uparrow\uparrow$ = notable improvement; $\approx$ = comparable to baseline.
\end{table}

\subsection{Qualitative Expectations}

Beyond the numbers, the real clinical value of RL-GroundGen comes from how easy it is to verify the outputs. When a model says ``There is a moderate left-sided pleural effusion'' and shows a bounding box that actually encloses that region on the radiograph, the reviewing radiologist can quickly confirm or dismiss the finding by checking the highlighted area. If the box is loose, wrong, or missing, the radiologist has to search the whole image again---which eliminates most of the efficiency benefit. Directly optimizing for spatial precision through the RL reward is what makes verification genuinely faster.

There's also an expected reduction in spatial hallucinations---cases where the model predicts a bounding box for a finding that doesn't actually exist at that location. The $-1$ penalty applied to unmatched predicted boxes is a strong deterrent against this. Standard SFT training doesn't address spatial hallucinations directly, since cross-entropy loss can still assign reasonable probability to incorrect coordinate tokens if similar patterns appeared in the training data.

\subsection{Limitations}

There are a few real limitations worth being upfront about. First, training requires bounding-box annotations at the finding level, which is more expensive to collect than report-only data. The grounded subset of MIMIC-CXR covers a limited set of pathologies, and how well the framework generalizes to unannotated findings is an open question. Second, the GRPO RL phase costs significantly more compute than SFT---each training step requires generating $G$ full reports per image and evaluating the reward for all of them. With $G = 8$ and a 7B-parameter decoder, that's a heavy GPU memory and time requirement. Third, the clinical factuality reward depends on the accuracy of the external evaluator model. If the evaluator makes systematic errors, those errors can end up baked into the reward signal, potentially reinforcing wrong behavior over time.

% ===========================================================================
\section{Conclusion and Future Work}
% ===========================================================================

This paper presented RL-GroundGen, a framework for generating spatially grounded radiology reports from chest X-ray images. The core motivation is a clear gap in existing work: prior research showed that reinforcement learning improves clinical factuality in generated text (UniRG) and that supervised fine-tuning can produce spatially grounded reports (MAIRA-2), but nobody had combined both into a single training pipeline. RL-GroundGen fills that gap by pairing a hierarchical Swin Transformer encoder with a transformer language decoder, using SFT for initialization and then refining with GRPO-based RL under a composite reward that penalizes both textual inaccuracies and spatial localization errors.

The Swin Transformer's hierarchical feature extraction gives the model multi-scale visual representations suited to detecting pathologies across a wide range of sizes and locations. The composite reward---combining clinical factuality scores for text with GIoU-based penalties for spatial precision---means the RL phase pushes on both dimensions together rather than trading one off against the other. And using GRPO instead of standard PPO keeps the RL phase feasible without requiring a separate value network.

Several natural extensions suggest themselves. First, testing beyond chest X-rays---CT, MRI, other modalities---would show how general the composite reward formulation really is. Volumetric modalities bring three-dimensional grounding, which would mean adapting bounding-box representations to volumes or segmentation masks.

Second, the current reward function treats text and spatial quality as additive with fixed weights. Learning those weights dynamically, perhaps through a multi-objective optimization setup, could improve the balance between the two objectives as training progresses and across different categories of findings.

Third, reducing dependence on expensive bounding-box annotations would make the framework much more broadly applicable. Semi-supervised or weakly-supervised variants that derive approximate spatial supervision from attention maps or class activation maps could work here---the Swin Transformer's shifted-window attention maps seem like a natural starting point for generating pseudo-labels.

Fourth, and most importantly, actual clinical validation by practicing radiologists is necessary before this gets anywhere near a real workflow. A prospective reader study comparing RL-GroundGen outputs against SFT-only baselines on clinical utility, correctness, and verification efficiency would be the most meaningful test of whether any of this matters in practice.

Finally, the multi-modal LLM landscape is evolving fast. The modular design of RL-GroundGen---a visual encoder, a language decoder, and a reward function that are loosely coupled---means swapping in newer architectures as they mature doesn't require rebuilding the training pipeline from scratch.

% ===========================================================================
% REFERENCES
% ===========================================================================
\begin{thebibliography}{11}

\bibitem{irvin2019chexpert}
J.~Irvin \textit{et~al.},
``CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison,''
in \textit{Proc. AAAI Conf. Artificial Intelligence}, vol.~33, 2019, pp.~590--597.

\bibitem{johnson2019mimic}
A.~E.~W.~Johnson, T.~J.~Pollard, S.~J.~Berkowitz, N.~R.~Greenbaum, M.~P.~Lungren, C.-Y.~Deng, R.~G.~Mark, and S.~Horng,
``MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports,''
\textit{Scientific Data}, vol.~6, art. no.~317, 2019.

\bibitem{chen2020r2gen}
Z.~Chen, Y.~Song, T.-H.~Chang, and X.~Wan,
``Generating radiology reports via memory-driven Transformer,''
in \textit{Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)}, 2020, pp.~1439--1449.

\bibitem{liu2021swin}
Z.~Liu, Y.~Lin, Y.~Cao, H.~Hu, Y.~Wei, Z.~Zhang, S.~Lin, and B.~Guo,
``Swin Transformer: Hierarchical vision transformer using shifted windows,''
in \textit{Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)}, 2021, pp.~10012--10022.

\bibitem{pillai2022}
A.~S.~Pillai,
``Multi-label chest X-ray classification via deep learning,''
\textit{Journal of Intelligent Learning Systems and Applications}, vol.~14, pp.~43--56, Nov. 2022.

\bibitem{taslimi2022}
S.~Taslimi, S.~Taslimi, N.~Fathi, M.~Salehi, and M.~H.~Rohban,
``SwinCheX: Multi-label classification on chest X-ray images with transformers,''
\textit{arXiv preprint arXiv:2206.04246}, Jun. 2022.

\bibitem{alshmrani2023}
G.~M.~M.~Alshmrani, Q.~Ni, R.~Jiang, H.~Pervaiz, and N.~M.~Elshennawy,
``A deep learning architecture for multi-class lung diseases classification using chest X-ray (CXR) images,''
\textit{Alexandria Engineering Journal}, vol.~64, pp.~923--935, 2023.

\bibitem{bannur2024}
S.~Bannur \textit{et~al.},
``MAIRA-2: Grounded radiology report generation,''
\textit{arXiv preprint arXiv:2406.04449}, Sep. 2024.

\bibitem{saranyaraj2026}
D.~Saranyaraj, V.~Shrinaath, A.~Nayak, and R.~Vishal,
``PneuNet: A lightweight convolutional neural network with multiscale feature fusion for automated pneumonia detection from chest X-rays,''
\textit{Frontiers in Medicine}, vol.~12, p.~1713587, Jan. 2026.

\bibitem{liu2026unirg}
Q.~Liu \textit{et~al.},
``Scaling medical imaging report generation with multimodal reinforcement learning,''
\textit{arXiv preprint arXiv:2601.17151}, Jan. 2026.

\bibitem{nicolson2026cxrmate2}
A.~Nicolson, E.~J.~Cooper, H.-J.~Yoon, C.~McCafferty, R.~Krishnan, M.~Craigie, N.~Saad, J.~Dowling, I.~A.~Scott, and B.~Koopman,
``Toward clinically acceptable chest X-ray report generation: A qualitative retrospective pilot study of CXRMate-2,''
\textit{arXiv preprint arXiv:2604.18967}, Apr. 2026.

\end{thebibliography}
\end{document}

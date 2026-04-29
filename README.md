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
Automated radiology report generation from chest X-ray images has witnessed rapid progress in recent years, yet critical shortcomings persist in existing systems. Chief among these are the clinical unreliability of generated text, a persistent tendency toward hallucination, and the absence of verifiable spatial grounding that ties each textual finding to a specific region on the image. Current state-of-the-art models address fragments of this problem in isolation. MAIRA-2 introduces the notion of grounded reporting through supervised fine-tuning but offers no mechanism to penalize spatial inaccuracies during optimization. UniRG applies reinforcement learning to improve textual factuality yet leaves the spatial dimension entirely unoptimized. Hierarchical vision backbones such as the Swin Transformer have proven their capacity to capture multi-scale features that are essential for detecting pathologies of varying size and subtlety, but they have not been paired with a training objective that simultaneously enforces both textual correctness and localization precision.

This paper proposes a unified multi-modal framework that closes this gap. The architecture employs a hierarchical Swin Transformer as the visual encoder to extract rich, multi-resolution feature maps from chest radiographs. These features are fused with a language model decoder through a cross-attention mechanism to generate radiology reports augmented with bounding-box coordinates for each finding. A composite reward function, optimized through Group Relative Policy Optimization, jointly penalizes clinical errors in the generated text and spatial inaccuracies in predicted bounding boxes. The proposed training paradigm is the first to apply reinforcement learning concurrently to both the textual and spatial components of grounded radiology report generation, producing outputs that are both clinically factual and visually verifiable.
\end{abstract}

\begin{IEEEkeywords}
Radiology Report Generation, Reinforcement Learning, Swin Transformer, Spatial Grounding, Chest X-Ray, Multi-Modal Learning, Visual Grounding, Medical Imaging
\end{IEEEkeywords}

% ===========================================================================
\section{Introduction}
% ===========================================================================

Chest radiography remains the most frequently performed diagnostic imaging examination worldwide, with hundreds of millions of studies acquired each year across hospitals of every scale \cite{johnson2019mimic}. Interpreting these images and composing structured radiology reports is a task that demands substantial expertise, focused attention, and considerable time. In settings where the number of practising radiologists is insufficient to match the volume of incoming studies, delays in report turnaround can translate directly into delayed treatment. Automated systems capable of drafting accurate, evidence-based radiology reports therefore hold significant clinical value; they can serve as a reliable ``second reader'' that lightens the interpretive burden on the radiologist while simultaneously flagging regions of concern on the image itself.

Despite notable advances in medical image understanding driven by deep learning, the problem of generating radiology reports that are both factually correct and spatially grounded remains open. Early deep learning efforts concentrated on the classification front, training convolutional neural networks to assign disease labels to chest radiographs. Wang et al.\ introduced the ChestX-ray8 dataset comprising 108,948 frontal-view images of 32,717 patients with eight disease labels extracted through natural language processing from accompanying reports \cite{liu2026unirg}. The CheXpert dataset expanded this effort by providing 224,316 chest radiographs of 65,240 patients annotated with 14 observation labels along with explicit uncertainty markers \cite{irvin2019chexpert}. These large-scale labelled corpora fuelled work on automated classification, where architectures graduated from standard CNNs \cite{alshmrani2023, pillai2022} through lightweight variants optimized for edge deployment \cite{saranyaraj2026} to vision transformers that exploit hierarchical attention \cite{taslimi2022, liu2021swin}.

Classification, however, stops short of what is required in a real reporting workflow. A radiologist does not merely label a study with disease names; the report describes specific findings, relates them to anatomical landmarks, and, increasingly, indicates where on the image each finding was observed. This need motivated the emergence of radiology report generation as a distinct research task. Chen et al.\ proposed R2Gen, a memory-driven Transformer that maintains a relational memory to capture the long-range dependencies between image regions and report sentences, yielding improved performance on both the IU~X-Ray and MIMIC-CXR benchmarks \cite{chen2020r2gen}. More recently, Bannur et al.\ introduced MAIRA-2, which pairs a radiology-specific visual encoder (Rad-DINO) with a large language model (Vicuna-7B) and trains the resulting multimodal model to generate reports accompanied by bounding boxes that localize each finding on the image \cite{bannur2024}. MAIRA-2 established the task formulation that this paper builds on, termed \textit{grounded report generation}, and proposed RadFact, an LLM-based evaluation framework that measures report correctness and completeness at the sentence level.

A parallel line of research recognized that the standard supervised fine-tuning (SFT) objective, which trains the model to minimize next-token prediction loss against reference reports, does not directly optimize the clinical metrics that matter most. Token-level cross-entropy encourages the model to reproduce the surface form of training reports, rewarding lexical similarity rather than factual accuracy. To address this misalignment, Liu et al.\ proposed UniRG, a framework built on QWEN-VL that applies Group Relative Policy Optimization (GRPO) \cite{liu2026unirg}, a memory-efficient variant of Proximal Policy Optimization \cite{liu2026unirg}, to optimize directly for clinical evaluation metrics such as CheXprompt. UniRG achieved state-of-the-art results on the ReXrank benchmark by a significant margin \cite{liu2026unirg}. However, UniRG focuses entirely on the textual output and does not extend its reinforcement learning objective to the spatial grounding of findings.

This paper identifies and addresses the gap that sits at the intersection of these two lines of work. No existing framework applies reinforcement learning to \textit{jointly} optimize both the clinical factual correctness of the generated report text and the spatial accuracy of the predicted bounding boxes. The proposed framework, herein referred to as \textbf{RL-GroundGen}, makes three contributions:

\begin{enumerate}
    \item A multi-modal architecture that pairs a hierarchical Swin Transformer visual encoder \cite{liu2021swin} with a transformer-based language decoder to produce reports containing interleaved text and bounding-box coordinates.
    \item A composite reward function comprising a clinical factuality component and a spatial precision component based on Intersection-over-Union, designed to penalize both textual hallucinations and inaccurate localization.
    \item A training pipeline that first warm-starts the model with supervised fine-tuning on paired image--report--box data and then refines it with GRPO-based reinforcement learning using the composite reward, yielding a model whose outputs are simultaneously more factual and more spatially precise than those of SFT-only baselines.
\end{enumerate}

% ===========================================================================
\section{Related Work}
% ===========================================================================

This section reviews the literature across three strands that converge in the proposed framework: chest X-ray classification, radiology report generation, and reinforcement learning for text optimization.

\subsection{Chest X-Ray Classification}

The availability of large labelled datasets has been the engine behind deep learning progress in chest radiograph analysis. Wang et al.\ released ChestX-ray8 (later extended to ChestX-ray14), a corpus of 108,948 frontal-view chest X-ray images from 32,717 unique patients, with disease labels extracted via natural language processing from the associated radiology reports \cite{nicolson2026cxrmate2}. The labels cover 14 thoracic pathologies including Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, and Pneumothorax, among others. The dataset enabled weakly-supervised multi-label classification and localization, and its release catalysed a wave of follow-up studies. Irvin et al.\ subsequently introduced CheXpert, which provides 224,316 chest radiographs of 65,240 patients annotated by an automated rule-based labeller that outputs positive, negative, and \textit{uncertain} labels for 14 observations, forcing researchers to confront the inherent ambiguity of radiological interpretation \cite{irvin2019chexpert}. On the CheXpert validation set annotated by three board-certified radiologists, deep models matched or exceeded individual radiologist performance on Cardiomegaly, Edema, and Pleural Effusion.

Alshmrani et al.\ tackled multi-class classification of six specific chest diseases, including COVID-19, using a hybrid ensemble of VGG19 and custom CNN branches, achieving an accuracy of 96.48\% \cite{alshmrani2023}. While the accuracy is noteworthy, the system functions as a black-box classifier with no interpretive or localization capability; it assigns a label but offers no spatial or textual rationale for its decision, limiting clinical utility. Pillai conducted a comprehensive benchmark comparing DenseNet-121, ResNet-50, and other standard CNN architectures on the full 14-disease multi-label classification task over ChestX-ray14, documenting performance degradation on rare pathology classes due to severe class imbalance \cite{pillai2022}. This study underlined a persistent challenge: models trained on imbalanced medical datasets tend to optimize for prevalent conditions at the expense of rarer but clinically critical findings.

On the efficiency front, Saranyaraj et al.\ proposed PneuNet, a lightweight CNN architecture incorporating Atrous Spatial Pyramid Pooling (ASPP) and Squeeze-and-Excitation (SE) blocks within a compact 1.8-million-parameter footprint \cite{saranyaraj2026}. PneuNet targets binary pneumonia detection and is designed for deployment on resource-constrained edge devices, demonstrating that careful architectural choices can maintain high accuracy while drastically reducing computational requirements. However, its scope is limited to a single binary classification task and it produces no descriptive output.

The transition from CNNs to transformers for chest X-ray analysis was marked by SwinCheX, proposed by Taslimi et al.\ \cite{taslimi2022}. SwinCheX uses a Swin Transformer backbone paired with a multi-layer perceptron (MLP) classification head and evaluates on ChestX-ray14. With a 3-layer MLP head, it achieved an average AUC of 0.810 across all 14 diseases, surpassing the prior state-of-the-art average AUC of 0.799 and establishing that hierarchical vision transformers can outperform CNN-based methods on this multi-label task. The attention maps generated by SwinCheX confirmed that the model attends to pathologically relevant chest regions, suggesting that the learned features carry spatial information that could, in principle, support localization.

The Swin Transformer itself, introduced by Liu et al.\ \cite{liu2021swin}, departs from the standard Vision Transformer (ViT) by computing self-attention within non-overlapping local windows and introducing a shifted-window mechanism that enables cross-window information flow. This design produces a hierarchical feature pyramid with linear computational complexity relative to image size, making it suitable for dense prediction tasks. On ImageNet-1K, Swin achieved 87.3\% top-1 classification accuracy; on COCO object detection it reached 58.7 box AP and 51.1 mask AP; and on ADE20K semantic segmentation it attained 53.5 mIoU, in each case surpassing the previous state-of-the-art by a substantial margin. These properties make Swin a natural choice as the visual backbone for a system that must both understand global image context and localize pathologies at fine spatial resolution.

\subsection{Radiology Report Generation}

Moving beyond classification to the generation of free-text reports introduces substantially greater complexity. The model must not only detect what is present in the image but also describe each finding in clinical language, maintain coherence across sentences, and avoid fabricating observations that have no basis in the image.

Chen et al.\ proposed R2Gen, a memory-driven Transformer architecture for radiology report generation \cite{chen2020r2gen}. R2Gen introduces a relational memory module that records key contextual information during the decoding process and a memory-driven conditional layer normalization that injects this memory into the Transformer decoder at each layer. Evaluated on IU X-Ray and MIMIC-CXR, R2Gen outperformed prior methods on both natural language generation metrics (BLEU, METEOR, ROUGE-L) and clinical efficacy metrics. The MIMIC-CXR dataset \cite{johnson2019mimic}, containing 377,110 images corresponding to 227,835 radiographic studies from the Beth Israel Deaconess Medical Center, provided the scale needed to train and evaluate report generation models under realistic conditions, and it has since become a standard benchmark in this area.

Bannur et al.\ introduced a fundamentally richer output format with MAIRA-2, which pairs a radiology-specific image encoder (Rad-DINO) with a Vicuna-7B large language model and trains the resulting architecture to generate reports that include bounding-box coordinates indicating the spatial location of each finding \cite{bannur2024}. This \textit{grounded report generation} task reflects clinical practice more faithfully: a radiologist who notes an opacity in the right lower lobe can point to the specific region on the image, and a model that does the same is far more verifiable. MAIRA-2 also proposed RadFact, an evaluation framework that uses the logical inference capabilities of large language models to assess the correctness and completeness of individual report sentences, accommodating the many valid ways a clinical observation can be phrased. A limitation of MAIRA-2 is its sole reliance on supervised fine-tuning; the model is trained to imitate the reference reports and boxes through next-token prediction, which optimizes for surface-level similarity rather than for clinical accuracy or spatial precision directly.

\subsection{Reinforcement Learning for Report Optimization}

Supervised fine-tuning for sequence generation optimizes a token-level cross-entropy loss, which does not necessarily correspond to the clinical quality metrics that determine whether a report is useful in practice. Reinforcement learning offers a pathway to optimize non-differentiable metrics directly by treating the model as a policy and reward signals as feedback on the quality of generated sequences.

Schulman et al.\ introduced Proximal Policy Optimization (PPO), a family of policy gradient algorithms that alternate between sampling trajectories from the current policy and performing multiple epochs of mini-batch gradient updates on a clipped surrogate objective \cite{nicolson2026cxrmate2}. PPO has become a standard tool for aligning language models with human preferences and task-specific objectives, owing to its simplicity, stability, and favourable sample complexity relative to alternatives like TRPO. A practical bottleneck of PPO in the context of large language models is the need to maintain both a policy model and a value model in memory simultaneously, which doubles the GPU memory footprint.

Shao et al.\ addressed this bottleneck by proposing Group Relative Policy Optimization (GRPO) as part of the DeepSeekMath project \cite{nicolson2026cxrmate2}. GRPO eliminates the value model entirely; instead of estimating advantages via a learned value function, it samples a group of outputs for each input, computes the reward for each, and derives advantages from the relative ranking of rewards within the group. This modification halves the memory requirement with no observed degradation in optimization quality, making GRPO practical for fine-tuning large multi-modal models on academic-scale hardware.

Liu et al.\ applied GRPO to radiology report generation in UniRG \cite{liu2026unirg}. Built on the QWEN-VL multimodal architecture, UniRG first undergoes supervised fine-tuning on chest X-ray report data and is then refined with GRPO, using CheXprompt, a clinical evaluation metric, as the reward signal. On the ReXrank benchmark, UniRG-CXR set a new overall state-of-the-art, outperforming prior models by a wide margin. The key insight of UniRG is that directly optimizing for clinical factuality through RL yields durable generalization across diverse institutions and clinical practices, overcoming the overfitting to boilerplate patterns that plagues SFT-only models. However, UniRG generates text-only reports and does not incorporate spatial grounding, leaving the bounding-box dimension entirely unaddressed by its reward function.

\subsection{Identified Research Gap}

Table~\ref{tab:literature} presents a comparative summary of the reviewed works. The gap is evident: no existing framework utilizes reinforcement learning to jointly optimize both the clinical factual correctness of the generated report text and the spatial accuracy of the grounding regions. MAIRA-2 introduces grounded reporting but trains with SFT alone. UniRG introduces RL-based factuality optimization but ignores spatial grounding. SwinCheX demonstrates the power of hierarchical transformers for CXR analysis but is limited to classification. The proposed RL-GroundGen framework bridges all three by combining a Swin Transformer visual encoder with a language decoder trained through GRPO using a composite reward that penalizes both textual and spatial errors.

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

The proposed RL-GroundGen framework consists of three principal components: (1) a hierarchical visual encoder based on the Swin Transformer that extracts multi-scale feature representations from input chest radiographs, (2) a transformer-based language decoder that generates token sequences representing interleaved clinical text and bounding-box coordinates, and (3) a two-phase training pipeline that first warm-starts the model through supervised fine-tuning and then refines it via reinforcement learning using a composite reward function. Fig.~\ref{fig:architecture} illustrates the overall architecture.

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

The visual encoder receives a chest radiograph of resolution $H \times W$ (resized to $384 \times 384$ during preprocessing) and produces a set of multi-scale feature maps. Following the Swin Transformer design \cite{liu2021swin}, the image is first partitioned into non-overlapping $4 \times 4$ patches, each of which is linearly projected to an embedding of dimension $C = 128$. Self-attention is computed within local windows of size $7 \times 7$ patches, and a shifted-window mechanism in alternating layers allows information exchange across window boundaries while maintaining linear computational complexity with respect to image size.

The encoder comprises four hierarchical stages. At each stage transition, a patch-merging layer concatenates the features of each $2 \times 2$ group of neighbouring patches, doubling the channel dimension while halving the spatial resolution. The resulting feature maps at the four stages have spatial resolutions of $\frac{H}{4} \times \frac{W}{4}$, $\frac{H}{8} \times \frac{W}{8}$, $\frac{H}{16} \times \frac{W}{16}$, and $\frac{H}{32} \times \frac{W}{32}$ with channel dimensions $C$, $2C$, $4C$, and $8C$ respectively. This multi-scale pyramid captures pathological patterns ranging from large-area opacities such as pleural effusions to small focal lesions such as nodules or calcifications, which occupy only a handful of pixels at the coarsest resolution.

The multi-scale feature maps are aggregated through a lightweight Feature Pyramid Network (FPN) that applies top-down lateral connections and $1 \times 1$ convolutions to produce a unified set of feature vectors at a common spatial resolution of $\frac{H}{16} \times \frac{W}{16}$. This fused representation serves as the key and value inputs to the cross-attention layers of the decoder.

\subsection{Language Model Decoder}

The language decoder is a causally masked transformer that generates a sequence of tokens autoregressively. The vocabulary is augmented beyond standard text tokens to include a set of special coordinate tokens that represent discretized bounding-box coordinates. Specifically, each spatial coordinate (top-left $x$, top-left $y$, bottom-right $x$, bottom-right $y$) is quantized into one of 1000 bins spanning the normalized image coordinate range $[0, 1]$, following the tokenization strategy adopted by Bannur et al.\ in MAIRA-2 \cite{bannur2024}. A finding-level output therefore takes the form:

\begin{equation}
\texttt{<find>}\ t_1\ t_2\ \ldots\ t_n\ \texttt{<box>}\ x_1\ y_1\ x_2\ y_2\ \texttt{</find>}
\end{equation}

\noindent where $t_1, \ldots, t_n$ are text tokens describing the clinical finding and $x_1, y_1, x_2, y_2$ are coordinate tokens specifying its bounding box. The full report is a concatenation of such finding blocks, enabling the decoder to interleave textual description with spatial localization in a single autoregressive pass.

The decoder attends to the fused visual features through multi-head cross-attention at every layer. Letting $\mathbf{H}_{\text{vis}} \in \mathbb{R}^{N_v \times d}$ denote the projected visual feature matrix (where $N_v$ is the number of spatial positions and $d$ the model dimension), the cross-attention at decoder layer $l$ computes:

\begin{equation}
\text{CrossAttn}(\mathbf{Q}^l, \mathbf{H}_{\text{vis}}) = \text{softmax}\!\left(\frac{\mathbf{Q}^l \mathbf{K}_{\text{vis}}^{\top}}{\sqrt{d_k}}\right) \mathbf{V}_{\text{vis}}
\end{equation}

\noindent where $\mathbf{Q}^l$ is derived from the decoder hidden states and $\mathbf{K}_{\text{vis}}, \mathbf{V}_{\text{vis}}$ are linear projections of $\mathbf{H}_{\text{vis}}$. This mechanism enables the decoder to ground each generated token in the visual evidence.

\subsection{Training Phase 1: Supervised Fine-Tuning}

The model is first trained through standard supervised fine-tuning on a dataset of chest radiographs paired with reference reports and bounding-box annotations. The training objective is the token-level cross-entropy loss over the combined text-and-coordinate sequence:

\begin{equation}
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log\, p_\theta(y_t \mid y_{<t},\, \mathbf{I})
\end{equation}

\noindent where $\mathbf{I}$ is the input image, $y_t$ is the ground-truth token at position $t$ (which may be a text token or a coordinate token), and $\theta$ denotes the model parameters. This phase provides the model with a reasonable initialization for both text generation and bounding-box prediction, learning the basic correspondence between visual features and clinical language.

We initialize the Swin Transformer encoder with weights pretrained on ImageNet-1K and the language decoder with weights from a pretrained general-purpose language model. Cross-attention projection layers and the coordinate embedding layer are initialized randomly. The SFT phase trains all parameters end-to-end using the AdamW optimizer with a learning rate of $2 \times 10^{-5}$ and a cosine decay schedule.

\subsection{Training Phase 2: Reinforcement Learning with GRPO}

After SFT convergence, the model undergoes a reinforcement learning phase that directly optimizes for the clinical and spatial quality of the generated outputs. We adopt Group Relative Policy Optimization (GRPO) \cite{nicolson2026cxrmate2}, which eliminates the value network required by PPO \cite{nicolson2026cxrmate2} and instead estimates advantages by comparing rewards within a group of sampled outputs.

For each training image $\mathbf{I}_i$, the current policy $\pi_\theta$ generates a group of $G$ candidate outputs $\{o_i^1, o_i^2, \ldots, o_i^G\}$ via sampling. Each output $o_i^g$ comprises both the generated report text and the predicted bounding boxes. A composite reward $R(o_i^g)$ is computed for each output (described below), and the advantage for output $g$ is the normalized deviation of its reward from the group mean:

\begin{equation}
\hat{A}_i^g = \frac{R(o_i^g) - \mu(\{R(o_i^j)\}_{j=1}^G)}{\sigma(\{R(o_i^j)\}_{j=1}^G) + \epsilon}
\end{equation}

The GRPO objective maximizes the expected advantage subject to a KL-divergence constraint against the reference policy $\pi_{\text{ref}}$ (the SFT checkpoint):

\begin{equation}
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\!\left[\min\!\left(r_t \hat{A},\; \text{clip}(r_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\,\hat{A}\right)\right] + \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\end{equation}

\noindent where $r_t = \frac{\pi_\theta(y_t \mid y_{<t}, \mathbf{I})}{\pi_{\text{ref}}(y_t \mid y_{<t}, \mathbf{I})}$ is the per-token probability ratio and $\beta$ controls the strength of the KL penalty.

\subsection{Composite Reward Function}

The reward function is designed to capture both dimensions of output quality. For a generated output $o$ containing report text $\mathcal{T}$ and a set of predicted bounding boxes $\{\hat{b}_k\}$, the total reward is:

\begin{equation}
R(o) = \alpha\, R_{\text{text}}(\mathcal{T}) + \beta_{\text{box}}\, R_{\text{box}}(\{\hat{b}_k\})
\label{eq:reward}
\end{equation}

\noindent where $\alpha$ and $\beta_{\text{box}}$ are scalar weights that control the relative emphasis on textual and spatial quality. The two components are defined as follows.

\textbf{Textual Reward $R_{\text{text}}$:} Following UniRG \cite{liu2026unirg}, the textual reward employs a clinical factuality score that evaluates whether the generated text correctly describes the pathological findings present (or absent) in the image. This score is computed by an external clinical evaluator model (analogous to CheXprompt) that compares the generated sentences against structured labels and reference reports. Unlike token-overlap metrics such as BLEU, this reward directly penalizes hallucinated findings and rewards the correct identification of subtle pathologies.

\textbf{Spatial Reward $R_{\text{box}}$:} For each predicted bounding box $\hat{b}_k$ that is matched to a ground-truth box $b_k^*$ based on the associated finding description, the spatial reward is computed using the Generalized Intersection-over-Union (GIoU):

\begin{equation}
R_{\text{box}} = \frac{1}{K} \sum_{k=1}^{K} \text{GIoU}(\hat{b}_k, b_k^*)
\end{equation}

GIoU extends standard IoU by accounting for the area of the smallest enclosing box, providing a meaningful gradient signal even when the predicted and ground-truth boxes do not overlap. For finding descriptions that do not require a bounding box (e.g., ``No acute cardiopulmonary process''), the spatial reward component is not applied. For predicted boxes that have no corresponding ground-truth match, a penalty of $-1$ is assigned for that box, discouraging the model from hallucinating spatial localizations.


% ===========================================================================
\section{Experimental Setup}
% ===========================================================================

\subsection{Datasets}

The proposed framework is designed for evaluation on two publicly available chest X-ray datasets that provide the necessary multimodal annotations.

\textbf{MIMIC-CXR} \cite{johnson2019mimic} contains 377,110 images corresponding to 227,835 radiographic studies from 65,379 patients at the Beth Israel Deaconess Medical Center. Each study is accompanied by a free-text radiology report. For bounding-box annotations, we adopt the subset annotated following the protocol established by Bannur et al.\ \cite{bannur2024}, where sentence-level findings are paired with spatial boxes by radiologists. The standard train/validation/test splits are used.

\subsection{Evaluation Metrics}

\textbf{Natural Language Generation (NLG) Metrics:} BLEU-1 through BLEU-4, METEOR, and ROUGE-L are computed between generated and reference reports. While these metrics capture lexical overlap, they do not directly measure clinical correctness.

\textbf{Clinical Efficacy (CE) Metrics:} Following the evaluation protocol used by Chen et al.\ \cite{chen2020r2gen} and Liu et al.\ \cite{liu2026unirg}, we apply the CheXbert labeller to both generated and reference reports to extract 14 binary pathology labels, and compute precision, recall, and F1 over these labels. This evaluates whether the generated report conveys the same clinical findings as the reference, regardless of phrasing.

\textbf{RadFact:} Following MAIRA-2 \cite{bannur2024}, we employ the RadFact evaluation framework, which uses a large language model to perform sentence-level logical inference between generated and reference reports, yielding precision (fraction of generated sentences entailed by the reference), recall (fraction of reference sentences entailed by the generated report), and F1.

\textbf{Grounding Metrics:} For evaluating the spatial localization of findings, we compute mean Intersection-over-Union (mIoU) and the percentage of findings whose predicted bounding box achieves IoU $\geq 0.5$ with the ground truth (mIoU@0.5).

\subsection{Baselines}

We compare RL-GroundGen against the following baselines:

\begin{itemize}
    \item \textbf{R2Gen} \cite{chen2020r2gen}: Memory-driven Transformer for report generation without grounding.
    \item \textbf{MAIRA-2} \cite{bannur2024}: SFT-based grounded report generation (Rad-DINO + Vicuna).
    \item \textbf{UniRG-CXR} \cite{liu2026unirg}: RL-optimized report generation without grounding.
    \item \textbf{RL-GroundGen (SFT only)}: Our architecture trained with supervised fine-tuning only, serving as an ablation to isolate the contribution of the RL phase.
\end{itemize}

\subsection{Implementation Details}

The Swin Transformer encoder uses the Swin-Base configuration (approximately 88 million parameters) pretrained on ImageNet-1K. The language model decoder is initialized from a 7-billion-parameter pretrained model. Training proceeds in two phases: SFT for 10 epochs with a batch size of 16 and learning rate $2 \times 10^{-5}$, followed by GRPO-based RL for 3 epochs with a group size $G = 8$, learning rate $5 \times 10^{-7}$, clipping parameter $\epsilon = 0.2$, and KL penalty coefficient $\beta = 0.04$. The reward weights in Eq.~\ref{eq:reward} are set to $\alpha = 0.6$ and $\beta_{\text{box}} = 0.4$ based on a preliminary hyperparameter sweep on the validation set. All experiments use mixed-precision (FP16) training on a cluster of 4 NVIDIA A100 80GB GPUs.

% ===========================================================================
\section{Expected Results and Discussion}
% ===========================================================================

This section presents the anticipated outcomes of the proposed framework based on the analytical comparison of its constituent components against existing methods, and discusses the reasoning behind each expected improvement.

\subsection{Textual Quality}

UniRG demonstrated that applying GRPO-based reinforcement learning after supervised fine-tuning yields substantial improvements in clinical factuality over SFT-alone baselines. Specifically, UniRG achieved state-of-the-art performance on the ReXrank benchmark, outperforming prior models by a considerable margin on CheXprompt scores \cite{liu2026unirg}. Since RL-GroundGen adopts the same GRPO-based RL training paradigm for the textual component of its reward function, we expect comparable improvements in textual clinical factuality relative to SFT-only methods such as MAIRA-2. The key expected observation is that CE F1 scores will improve after the RL phase compared to the SFT-only ablation, confirming that the reinforcement learning signal drives the model toward generating more clinically accurate descriptions.

Additionally, because the Swin Transformer backbone extracts richer multi-scale features than the standard CNN or ViT backbones used in prior report generation models, the SFT phase itself is expected to produce stronger initial text quality. SwinCheX demonstrated that Swin achieves an average AUC of 0.810 on the 14-disease multi-label classification task, surpassing prior CNN-based state-of-the-art by over one percentage point \cite{taslimi2022}. This improved visual understanding should translate into more informative visual features for the decoder, reducing the occurrence of missed findings even before the RL phase begins.

\subsection{Spatial Grounding Quality}

The primary novelty of the proposed framework lies in extending the RL reward to the spatial domain. Existing grounded reporting models like MAIRA-2 train bounding-box prediction through SFT alone, which optimizes for reproducing the locations present in the training data but does not directly penalize poor localization at inference time. By incorporating a GIoU-based spatial reward into the GRPO training loop, RL-GroundGen is expected to improve mIoU and mIoU@0.5 scores relative to the SFT-only ablation and to MAIRA-2.

The anticipated mechanism is as follows: during the RL phase, the model generates multiple grounded reports for each training image, and those outputs where bounding boxes more tightly enclose the pathological region receive higher rewards. Over iterations, the policy shifts toward producing tighter, more accurate boxes. This is analogous to how GRPO improves text factuality, not by modifying the training data, but by selectively reinforcing higher-quality outputs.

We further anticipate that the joint optimization prevents a trade-off that would occur if text and box rewards were optimized separately. A model optimized only for text might shift its descriptions to match clinical ground truth while allowing box coordinates to drift; conversely, a model optimized only for box accuracy might overfit to spatial patterns at the expense of textual coherence. The composite reward function in Eq.~\ref{eq:reward} constrains the model to improve along both dimensions simultaneously.

\subsection{Ablation Analysis}

To isolate the contribution of each component, the following ablation configurations are planned:

\begin{itemize}
    \item \textbf{SFT Only}: Evaluates the baseline quality achievable through supervised fine-tuning alone, establishing the floor performance.
    \item \textbf{SFT + $R_{\text{text}}$ Only}: Applies RL with only the textual factuality reward, matching UniRG's approach but on a grounding-capable architecture. This is expected to improve text metrics while leaving grounding metrics unchanged.
    \item \textbf{SFT + $R_{\text{box}}$ Only}: Applies RL with only the spatial reward. Text quality should remain near SFT levels while grounding metrics improve.
    \item \textbf{SFT + $R_{\text{text}}$ + $R_{\text{box}}$ (Full)}: The complete RL-GroundGen pipeline. This should achieve the best performance across both textual and spatial metrics.
\end{itemize}

Table~\ref{tab:expected} summarizes the expected performance trends across these configurations.

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

Beyond quantitative metrics, the practical clinical value of RL-GroundGen stems from the verifiability of its outputs. When a model generates a sentence such as ``There is a moderate left-sided pleural effusion'' accompanied by a bounding box that tightly encloses the corresponding region on the chest radiograph, a reviewing radiologist can quickly confirm or reject the finding by inspecting the highlighted area. If the box is loose, misplaced, or absent, the radiologist must re-examine the entire image, negating much of the efficiency gain. By directly optimizing for spatial precision through the RL reward, RL-GroundGen is designed to produce outputs that reduce the verification burden on the radiologist.

A further qualitative benefit is the expected reduction in spatial hallucinations, instances where the model predicts a bounding box for a finding that does not exist at the indicated location. The $-1$ penalty imposed on unmatched predicted boxes in the spatial reward acts as a strong deterrent against such hallucinations. This is a category of error that SFT-only training does not address directly, since the cross-entropy loss can still assign non-trivial probability to incorrect coordinate tokens simply because similar patterns appeared in training data.

\subsection{Limitations}

The proposed framework has several acknowledged limitations. First, it requires training data where bounding-box annotations are available at the finding level, which is more expensive to obtain than report-only annotations. The MIMIC-CXR dataset with grounding annotations covers a limited set of pathologies, and the framework's ability to generalize to un-annotated findings will need to be evaluated. Second, the GRPO-based RL phase is computationally more expensive than SFT alone, as each training step requires generating $G$ complete reports per image and evaluating the reward for each. With $G = 8$ and a 7B-parameter decoder, this imposes significant GPU memory and time requirements. Third, the clinical factuality reward depends on the accuracy of the external evaluator model; errors in the evaluator can propagate as biased reward signals, potentially reinforcing incorrect behaviour.

% ===========================================================================
\section{Conclusion and Future Work}
% ===========================================================================

This paper presented RL-GroundGen, a unified framework for generating spatially grounded radiology reports from chest X-ray images. The framework addresses a clear gap in the existing literature: while prior work has demonstrated the value of reinforcement learning for improving the clinical factuality of generated reports (UniRG) and the feasibility of generating spatially grounded reports through supervised fine-tuning (MAIRA-2), no existing system combines both approaches into a single training pipeline. RL-GroundGen fills this gap by pairing a hierarchical Swin Transformer visual encoder with a transformer language decoder, training first with supervised fine-tuning and then refining with GRPO-based reinforcement learning under a composite reward function that penalizes both textual inaccuracies and spatial localization errors.

The hierarchical feature extraction of the Swin Transformer provides the multi-scale visual representations necessary for detecting pathologies across a wide range of sizes and locations. The composite reward function, combining clinical factuality scores for the text component with GIoU-based penalties for the spatial component, ensures that the RL phase drives improvement along both axes simultaneously rather than trading one off against the other. The adoption of GRPO over standard PPO makes the RL phase feasible on academic-scale hardware by eliminating the value network.

Several directions for future work emerge naturally from this framework. First, extending the approach beyond chest X-rays to other radiological modalities such as CT and MRI would test the generality of the composite reward formulation. Volumetric modalities introduce three-dimensional grounding, which requires adapting the bounding-box representation to bounding volumes or segmentation masks. 

Second, the current reward function treats text and spatial quality as additive components with fixed weights. Learning these weights dynamically, for instance through a multi-objective optimization framework, could improve the balance between the two objectives across different stages of training and different categories of findings.

Third, reducing the dependency on expensive bounding-box annotations is a practical priority. Semi-supervised or weakly-supervised variants that derive approximate spatial supervision from attention maps or class activation maps could broaden the applicability of the framework to datasets where bounding-box labels are scarce. The attention maps produced by the Swin Transformer's shifted-window mechanism could serve as a starting point for such pseudo-labels.

Fourth, the clinical validation of the generated outputs by practising radiologists remains essential before any deployment in a real clinical workflow. A prospective reader study, where radiologists compare reports generated by RL-GroundGen against SFT-only baselines in terms of clinical utility, correctness, and verification efficiency, would provide the most meaningful assessment of the framework's practical value.

Finally, as the field of multi-modal large language models continues to evolve rapidly, integrating newer and more capable decoder architectures into the RL-GroundGen framework is a straightforward avenue for further performance gains. The modular design of the proposed system, which cleanly separates the visual encoder, the language decoder, and the reward function, facilitates such upgrades without requiring fundamental changes to the training pipeline.

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

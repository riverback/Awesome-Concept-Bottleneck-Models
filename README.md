# Awesome Concept Bottleneck Models
**Work in progress**: we have compiled and summarized relevant papers in this field and are continuing to improve the categorization and organization of the collection to help newcomers quickly understand the area. Feel free to suggest improvements or add new papers via a pull request.

## Introduction

The Concept Bottleneck Model (CBM) is an emerging self-explainable architecture that first maps inputs to a set of human-interpretable concepts before making predictions using an interpretable classifier, typically a single-layer linear model. Beyond inherent interpretability, CBMs provide an intervention interface through the concept bottleneck layer, allowing users to directly modify concept activations to refine model predictions, and this serves as the most significant difference between CBMs and other explainable models, such as the CapsulesNet and ProtoPNet.

![image_from_IntCEMs](https://github.com/riverback/Awesome-Concept-Bottleneck-Models/assets/IntCEMs.png)

(images from IntCEMs, highlighting the interpretability and intervention ability of CBM architectures)

## Papers sorted by research focus
TBD

## Papers sorted by publication year

#### 2025
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| AAAI 2025 | [Explanation Bottleneck Models](https://ojs.aaai.org/index.php/AAAI/article/view/35495) | [Code](https://github.com/yshinya6/xbm) |
| AAAI 2025 | [V2C-CBM: Building Concept Bottlenecks with Vision-to-Concept Tokenizer](https://ojs.aaai.org/index.php/AAAI/article/view/32352) | [Code](https://github.com/riverback/V2C-CBM) |
|   CVPR 2025   | [Interpretable Generative Models through Post-hoc Concept Bottlenecks](https://arxiv.org/pdf/2503.19377v1) | [Code](https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm) |
|  CVPR W 2025  | [PCBEAR: Pose Concept Bottleneck for Explainable Action Recognition](https://arxiv.org/abs/2504.13140) |                              -                               |
|   ICLR 2025   | [Counterfactual Concept Bottleneck Models](https://openreview.net/forum?id=w7pMjyjsKN) | [Code](https://github.com/gabriele-dominici/Counterfactual-CBM) |
|   ICLR 2025   | [Concept Bottleneck Large Language Models](https://openreview.net/forum?id=RC5FPYVQaH) |    [Code](https://github.com/Trustworthy-ML-Lab/CB-LLMs)     |
|   ICLR 2025   | [CONDA: Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts](https://iclr.cc/virtual/2025/poster/30736) |         [Code](https://github.com/jihyechoi77/CONDA)         |
|   ICLR 2025   | [Concept Bottleneck Language Models For Protein Design](https://iclr.cc/virtual/2025/poster/29243) |                              -                               |
|   ICML 2025   | [Editable Concept Bottleneck Models](https://dblp.uni-trier.de/rec/journals/corr/abs-2405-15476.html) |                              -                               |
|   ICML 2025   | [DCBM: Data-Efficient Visual Concept Bottleneck Models](https://arxiv.org/abs/2412.11576) |           [Code](https://github.com/KathPra/DCBM)            |
|   ICML 2025   | [Addressing Concept Mislabeling in Concept Bottleneck Models Through Preference]([arxiv.org/pdf/2504.18026](https://arxiv.org/pdf/2504.18026)) | [Code](https://github.com/Emilianopp/ConceptPreferenceOptimization) |
| IEEE TMM 2025 | [Leveraging Concise Concepts with Probabilistic Modeling for Interpretable Visual Recognition](https://ieeexplore.ieee.org/document/10948345) |                              -                               |
|  MICCAI 2025  | [Learning Concept-Driven Logical Rules for Interpretable and Generalizable Medical Image Classification](https://arxiv.org/abs/2505.14049) |            [Code](https://github.com/obiyoag/crl)            |
|  MICCAI 2025  | [Training-free Test-time Improvement for Explainable Medical Image Classification](https://arxiv.org/abs/2506.18070) |       [Code](https://github.com/riverback/TF-TTI-XMed)       |
|   TMLR 2025   | [Selective Concept Bottleneck Models Without Predefined Concepts](https://openreview.net/pdf?id=uuvujfQXZy) |      [Code](https://openreview.net/forum?id=PMO30TLI4l)      |
|  Arxiv 2025   | [ConceptCLIP: Towards Trustworthy Medical AI Via Concept-Enhanced Contrastive Langauge-Image Pre-training](https://arxiv.org/abs/2501.15579) |       [Code](https://github.com/JerrryNie/ConceptCLIP)       |

#### 2024
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| AAAI 2024 | [On the Concept Trustworthiness in Concept Bottleneck Models](https://doi.org/10.1609/aaai.v38i19.30109) | [Code](https://github.com/hqhQAQ/ProtoCBM) |
| AAAI 2024 | [Sparsity-guided holistic explanation for llms with interpretable inference-time intervention](https://ojs.aaai.org/index.php/AAAI/article/download/30160/32058) | [Code](https://github.com/zhen-tan-dmml/sparsecbm) |
| CVPR 2024 | [Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion](https://doi.org/10.1109/cvpr52733.2024.02538) | - |
| CVPR 2024 | [LVLM-Interpret: An Interpretability Tool for Large Vision-Language Models](https://arxiv.org/pdf/2404.03118) | [Code](https://github.com/IntelLabs/lvlm-interpret) |
| CVPR 2024 | [Incremental Residual Concept Bottleneck Models](https://doi.org/10.1109/cvpr52733.2024.01049) |  [Code](https://github.com/helloscm/res-cbm)|
| ECCV 2024 | [Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery](https://arxiv.org/pdf/2407.14499v2) | [Code](https://github.com/neuroexplicit-saar/discover-then-name) |
| ICLR 2024 | [Concept Bottleneck Generative Models](https://dblp.uni-trier.de/rec/conf/iclr/IsmailABRC24.html) | [Code](https://github.com/prescient-design/CBGM) |
| ICLR 2024 | [Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Probabilistic Interpretations](https://doi.org/10.48550/arxiv.2401.14142) | [Code](https://github.com/xmed-lab/ECBM) |
| ICLR 2024 | [Faithful Vision-Language Interpretation Via Concept Bottleneck Models](https://openreview.net/pdf?id=rp0EdI8X4e) | [Code](https://github.com/kaustpradalab/FVLC) |
| ICLR 2024 | [Concept Bottleneck Generative Models](https://dblp.uni-trier.de/rec/conf/iclr/IsmailABRC24.html) |  |
| ICML 2024 | [Post-hoc Part-prototype Networks](https://arxiv.org/abs/2406.03421) | - |
| ICML W 2024 | [XCoOp: Explainable Prompt Learning for Computer-Aided Diagnosis via Concept-guided Context Optimization](https://arxiv.org/pdf/2403.09410v1) | - |
| IEEE TPAMI | [The Decoupling Concept Bottleneck Model](https://ieeexplore.ieee.org/document/10740789) | [Code](https://github.com/deepopo/DCBM) |
| JBHI 2024 | [Guest Editorial: Trustworthy Machine Learning for Health Informatics](https://ieeexplore.ieee.org/iel8/6221020/10745910/10745914.pdf) | - |
| MICCAI 2024 | [Concept-Attention Whitening for Interpretable Skin Lesion Diagnosis](https://arxiv.org/pdf/2404.05997v2) |[Code](https://github.com/CAWframework/Concept-Attention-Whitening-for-Interpretable-Skin-Lesion-Diagnosis-Reproduction)|
| MICCAI 2024 | [Aligning human knowledge with visual concepts towards explainable medical image classification](https://arxiv.org/pdf/2406.05596) |[Code](https://github.com/yhygao/Explicd?tab=readme-ov-file)|
| MICCAI 2024 | [Evidential concept embedding models: Towards reliable concept explanations for skin disease diagnosis](https://arxiv.org/pdf/2406.19130) |[Code](https://github.com/obiyoag/evi-CEM)|
| MICCAI 2024 | [Learning a Clinically-Relevant Concept Bottleneck for Lesion Detection in Breast Ultrasound](https://arxiv.org/pdf/2407.00267v1) |[Code](https://github.com/hawaii-ai/bus-cbm)|
| MICCAI 2024 | [Mask-Free Neuron Concept Annotation for Interpreting Neural Networks in Medical Domain](https://arxiv.org/pdf/2407.11375v1) |  [Code](https://github.com/ailab-kyunghee/mammi)|
| MICCAI 2024 | [AdaCBM: an Adaptive Concept Bottleneck Model for Explainable and Accurate Diagnosis](https://doi.org/10.48550/arxiv.2105.02410) |  [Code](https://github.com/AIML-MED/AdaCBM)|
| MICCAI 2024 | [Explain Via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts](https://doi.org/10.1007/978-3-031-73016-0_8) |-|
| MICCAI 2024 | [Integrating Clinical Knowledge into Concept Bottleneck Models](https://doi.org/10.1007/978-3-031-72083-3_23) |[Code](https://github.com/pangwinnie0219/align_concept_cbm)|
| MedIA 2024 | [Interpretable and Intervenable Ultrasonography-Based Machine Learning Models for Pediatric Appendicitis](https://doi.org/10.1016/j.media.2023.103042) |[Code](https://github.com/i6092467/semi-supervised-multiview-cbm)|
| NeurIPS 2024 | [Stochastic Concept Bottleneck Models](https://arxiv.org/abs/2406.19272) | [Code](https://github.com/mvandenhi/scbm) |
| NeurIPS 2024 | [Coarse-to-Fine Concept Bottleneck Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bdeab378efe6eb289714e2a5abc6ed42-Abstract-Conference.html) | [Code](https://github.com/konpanousis/Coarse-To-Fine-CBMs) |
| NeurIPS 2024 | [VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance](https://arxiv.org/abs/2408.01432) | [Code](https://github.com/Trustworthy-ML-Lab/VLG-CBM) |
| NeurIPS 2024 | [A Theoretical Design of Concept Sets: Improving the Predictability of Concept Bottleneck Models](https://dblp.uni-trier.de/rec/conf/nips/LuytenS24.html) |-|
| NeurIPS 2024 | [Towards Multi-dimensional Explanation Alignment for Medical Classification](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea370419760b421ce12e3082eb2ae1a8-Abstract-Conference.html) |-|
| NeurIPS 2024 | [A Concept-Based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074) |  [Code](https://github.com/mshukor/xl-vlms)|
| NeurIPS 2024 | [Classifier Clustering and Feature Alignment for Federated Learning under Distributed Concept Drift](https://arxiv.org/abs/2410.18478) |  [Code](https://github.com/chen-junbao/fedccfa)|
| NeurIPS 2024 | [ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty](https://arxiv.org/abs/2408.14339) |[Code](https://github.com/princetonvisualai/ConceptMix)|
| NeurIPS 2024 | [Do LLMs Dream of Elephants (when Told Not To)? Latent Concept Association and Associative Memory in Transformers](https://arxiv.org/abs/2406.18400) |-|
| NeurIPS 2024 | [FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://arxiv.org/abs/2407.06567) |  [Code](https://github.com/MXGao-A/FAgent)|
| NeurIPS 2024 | [Free Lunch in Pathology Foundation Model: Task-specific Model Adaptation with Concept-Guided Feature Enhancement](https://arxiv.org/abs/2411.09894) |  [Code](https://github.com/HKU-MedAI/CATE)|
| NeurIPS 2024 | [From Causal to Concept-Based Representation Learning](https://dblp.uni-trier.de/rec/conf/nips/RajendranBASR24.html) |-|
| NeurIPS 2024 | [Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents](https://doi.org/10.48550/arxiv.2401.05821) |  [Code](https://github.com/k4ntz/scobots)|
| NeurIPS 2024 | [Interpretable Concept-Based Memory Reasoning](https://arxiv.org/abs/2407.15527) |  [Code](https://github.com/daviddebot/CMR)|
| NeurIPS 2024 | [Interpreting CLIP with Sparse Linear Concept Embeddings (Splice)](https://doi.org/10.48550/arxiv.2402.10376) |  [Code](https://github.com/ai4life-group/splice)|
| NeurIPS 2024 | [Learning Discrete Concepts in Latent Hierarchical Models](https://arxiv.org/abs/2406.00519) |-|
| NeurIPS 2024 | [LG-CAV: Train Any Concept Activation Vector with Language Guidance](https://arxiv.org/abs/2410.10308) |-|
| NeurIPS 2024 | [Neural Concept Binder](https://doi.org/10.48550/arxiv.2406.09949) |  [Code](https://github.com/ml-research/neuralconceptbinder)|
| NeurIPS 2024 | [No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance](https://doi.org/10.48550/arxiv.2404.04125) |  [Code](https://github.com/bethgelab/frequency_determines_performance)|
| NeurIPS 2024 | [PaCE: Parsimonious Concept Engineering for Large Language Models](https://arxiv.org/abs/2406.04331) |  [Code](https://github.com/peterljq/parsimonious-concept-engineering)|
| NeurIPS 2024 | [Relational Concept Bottleneck Models](https://arxiv.org/abs/2308.11991) |  [Code](https://github.com/diligmic/rcbm-neurips2024)|
| NeurIPS 2024 | [Uncovering Safety Risks of Large Language Models Through Concept Activation Vector](https://arxiv.org/abs/2404.12038) |  [Code](https://github.com/sproutnan/ai-safety_scav)|
| NeurIPS 2024 | [Towards Multi-dimensional Explanation Alignment for Medical Classification](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea370419760b421ce12e3082eb2ae1a8-Paper-Conference.pdf) | - |
| NeurIPS 2024 | [Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?](https://doi.org/10.48550/arxiv.2401.13544) | [Code](https://github.com/sonialagunac/beyond-cbm) |
| NeurIPS W 2024 | [Bayesian concept bottleneck models with llm priors](https://arxiv.org/abs/2410.15555) | [Code](https://github.com/jjfeng/bc-llm) |
| PAKDD 2024 | [Interpreting Pretrained Language Models Via Concept Bottlenecks](https://doi.org/10.1007/978-981-97-2259-4_5) | [Code](https://github.com/Zhen-Tan-dmml/CBM_NLP?tab=readme-ov-file) |
| Sci. Rep 2024 | [Pseudo-class Part Prototype Networks for Interpretable Breast Cancer Classification](https://doi.org/10.1038/s41598-024-60743-x) | [Code](https://github.com/MA-Choukali/PCPPN) |
| TMLR 2024 | [Reproducibility Study of "LICO: Explainable Models with Language-Image Consistency"](https://dblp.uni-trier.de/rec/journals/tmlr/FletcherKSVA24.html) | [Code](https://github.com/robertdvdk/lico-fact) |
| TMLR 2024 | [[Re].on the Reproducibility of Post-Hoc Concept Bottleneck Models](https://dblp.uni-trier.de/rec/journals/tmlr/MidavaineGCSC24.html) | [Code](https://github.com/dgcnz/FACT) |
| TMLR 2024 | [CLIP-QDA: an Explainable Concept Bottleneck Model](https://doi.org/10.48550/arxiv.2312.00110) |  |
| Arxiv 2024 | [Explainable and interpretable multimodal large language models: A comprehensive survey](https://arxiv.org/abs/2412.02104) | - |
| Arxiv 2024 | [Semi-supervised Concept Bottleneck Models](https://dblp.uni-trier.de/rec/journals/corr/abs-2406-18992.html) | [Code](https://github.com/Skyyyy0920/SSCBM) |
| Arxiv 2024 | [Self-eXplainable AI for Medical Image Analysis: A Survey and New Outlooks](https://arxiv.org/abs/2410.02331) | - |
| Arxiv 2024 | [Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis](https://arxiv.org/abs/2410.15446) | - |
| Arxiv 2024 | [Improving Concept Alignment in Vision-Language Concept Bottleneck Models](https://arxiv.org/pdf/2405.01825) | [Code](https://github.com/NMS05/Improving-Concept-Alignment-in-Vision-Language-Concept-Bottleneck-Models) |


#### 2023
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| AAAI 2023 | [Interactive Concept Bottleneck Models](https://ojs.aaai.org/index.php/AAAI/article/view/25736/25508) |-|
| CVPR 2023 | [Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Language_in_a_Bottle_Language_Model_Guided_Concept_Bottlenecks_for_CVPR_2023_paper.pdf) |[Code](https://github.com/yueyang1996/labo)|
| CVPR 2023 | [Learning bottleneck concepts in image classification](https://arxiv.org/pdf/2304.10131v1) |[Code](https://github.com/wbw520/botcl)|
| CVPR 2023 | [Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision](https://arxiv.org/pdf/2303.00885v1) |-|
|  EMNLP 2023  | [STAIR: Learning Sparse Text and Image Representation in Grounded Tokens](https://arxiv.org/pdf/2301.13081v2) |                              -                               |
|  EMNLP 2023  | [Cross-Modal Conceptualization in Bottleneck Models](https://doi.org/10.18653/v1/2023.emnlp-main.318) |         [Code](https://github.com/danisalukaev/xcbs)         |
| ICLR 2023 | [LABEL-FREE CONCEPT BOTTLENECK MODELS](https://arxiv.org/pdf/2304.06129) |  [Code](https://github.com/trustworthy-ml-lab/label-free-cbm)|
|  ICLR 2023   | [Post-hoc Concept Bottleneck Models](https://doi.org/10.48550/arxiv.2205.15480) |        [Code](https://github.com/mertyg/post-hoc-cbm)        |
|  ICML 2023   | [A Closer Look at the Intervention Procedure of Concept Bottleneck Models](https://doi.org/10.48550/arxiv.2302.14260) |  [Code](https://github.com/ssbin4/Closer-Intervention-CBM)   |
|  ICML 2023   | [Probabilistic Concept Bottleneck Models](https://doi.org/10.48550/arxiv.2306.01574) |         [Code](https://github.com/ejkim47/prob-cbm)          |
| ICML W 2023 | [A ChatGPT Aided Explainable Framework for Zero-Shot Medical Image Diagnosis](https://doi.org/10.48550/arxiv.2307.01981) |-|
| MICCAI 2023 | [Concept Bottleneck with Visual Concept Filtering for Explainable Medical Image Classification](https://doi.org/10.1007/978-3-031-47401-9_22) |-|
| NeurIPS 2023 | [Do Concept Bottleneck Models Respect Localities](https://arxiv.org/abs/2401.01259) |  [Code](https://github.com/naveenr414/Spurious-Concepts)|
| NeurIPS 2023 | [Learning to Receive Help: Intervention-Aware Concept Embedding Models](https://openreview.net/forum?id=4ImZxqmT1K) | [Code](https://github.com/mateoespinosa/cem) |
| Arxiv 2023 | [Robust and interpretable medical image classifiers via concept bottleneck models](https://doi.org/10.48550/arxiv.2310.03182) |-|

#### 2022
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| ICCV 2022 | [Explaining in Style: Training a GAN to Explain a Classifier in StyleSpace](https://doi.org/10.5281/zenodo.6574709) |-|
| ICLR 2023 | [CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks](https://arxiv.org/abs/2204.10965) |  [Code](https://github.com/trustworthy-ml-lab/clip-dissect)|
| NeurIPS 2022 | [Concept Embedding Models](https://arxiv.org/abs/2209.09056) |  [Code](https://github.com/mateoespinosa/cem)|
| NeurIPS 2022 | [Addressing Leakage in Concept Bottleneck Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf) | [Code](https://github.com/dtak/addressing-leakage) |

#### 2021
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| ICML 2021 | [Meaningfully Debugging Model Mistakes Using Conceptual Counterfactual Explanations](https://doi.org/10.48550/arxiv.2106.12723) |  [Code](https://github.com/mertyg/debug-mistakes-cce)|
| NMI 2021 | [A case-based interpretable deep learning model for classification of mass lesions in digital mammography](https://arxiv.org/pdf/2103.12308) |[Code](https://github.com/alinajadebarnett/iaiabl)|
| Arxiv 2021 | [Partially Interpretable Estimators (PIE): Black-Box-Refined Interpretable Machine Learning](https://doi.org/10.48550/arxiv.2105.02410) |-|

#### 2020
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| ICML 2020 | [Concept bottleneck models](http://proceedings.mlr.press/v119/koh20a/koh20a.pdf) |  [Code](https://github.com/yewsiang/ConceptBottleneck)|
| NML 2020 | [Concept whitening for interpretable image recognition](https://rdcu.be/cbOKj) |  [Code](https://github.com/zhiCHEN96/ConceptWhitening?tab=readme-ov-file)|



**Acknowledgement**

This project was originally inspired by https://github.com/kkzhang95/Awesome_Concept_Bottleneck_Models. We thank the authors for their contributions. Our main motivation is to provide an additional architecture organized by research focus, supplement it with more recent papers, and sort them by conference name for easier navigation.

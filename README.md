# Awesome Concept Bottleneck Models
**Work in progress**: we have compiled and summarized relevant papers in this field and are continuing to improve the categorization and organization of the collection to help newcomers quickly understand the area. Feel free to suggest improvements or add new papers via a pull request.

## Introduction

The Concept Bottleneck Model (CBM) is an emerging self-explainable architecture that first maps inputs to a set of human-interpretable concepts before making predictions using an interpretable classifier, typically a single-layer linear model. Beyond inherent interpretability, CBMs provide an intervention interface through the concept bottleneck layer, allowing users to directly modify concept activations to refine model predictions, and this serves as the most significant difference between CBMs and other explainable models, such as the CapsulesNet and ProtoPNet.

![image_from_IntCEMs](https://github.com/riverback/Awesome-Concept-Bottleneck-Models/blob/master/assets/IntCEMs.png)

(images from IntCEMs, highlighting the interpretability and intervention ability of CBM architectures)

## Papers Sorted by Research Focus

### Architecture Improvements

**Improving Concept Representations**



**Improving Intervention Ability**



### Finding Concepts (concept discovery, language-guided CBMs, etc.)

**Concept Discovery**



**Language-guided CBMs**



### CBMs for Non-visual Data

**Text**



**Table**



**Scientific Data**



### CBM Applications



### Datasets

**Concept Annotated Datasets**

| Name                                                         | Task                             | N. of concepts | N. of classes |
| ------------------------------------------------------------ | -------------------------------- | -------------- | ------------- |
| [CUB](https://worksheets.codalab.org/bundles/0x518829de2aa440c79cd9d75ef6669f27) | birds classification             | 312            | 200           |
| [AwA2](https://cvml.ista.ac.at/AwA2/)                        | animals classification           | 85             | 50            |
| [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)   | identities classification        | 40             | 10,177        |
| [OAI](https://nda.nih.gov/oai)                               | x-ray grading                    | 10             | 4             |
| [WBCAtt](https://github.com/apple2373/wbcatt)                | white blood cells classification | 31             | 5             |
| [Fitzpatrick 17k (subset)](https://skincon-dataset.github.io/index.html#dataset) | skin diseases classification     | 48             | 2             |
| [Diverse Dermatology Images (DDI)](https://skincon-dataset.github.io/index.html#dataset) | skin diseases classification     | 48             | 2             |
| [Skincon (Fitz sub + DDI annotated)](https://skincon-dataset.github.io/index.html#dataset) | skin diseases classification     | 48             | 2             |
| [DermaCon-IN](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W7OUZM) | skin diseases classification     | 47             | 8             |

## Papers Sorted by Publication Year

### 2025
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| AAAI | [Explanation Bottleneck Models](https://ojs.aaai.org/index.php/AAAI/article/view/35495) | [Code](https://github.com/yshinya6/xbm) |
| AAAI | [V2C-CBM: Building Concept Bottlenecks with Vision-to-Concept Tokenizer](https://ojs.aaai.org/index.php/AAAI/article/view/32352) | [Code](https://github.com/riverback/V2C-CBM) |
| ACL | [Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](10.48550/arXiv.2506.01334) | -` |
|   CVPR   | [Interpretable Generative Models through Post-hoc Concept Bottlenecks](https://arxiv.org/pdf/2503.19377v1) | [Code](https://github.com/Trustworthy-ML-Lab/posthoc-generative-cbm) |
| CVPR | [Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability](http://arxiv.org/abs/2503.20301) | [Code](https://github.com/tiggers23/ALBM) |
| CVPR | [Language Guided Concept Bottleneck Models for Interpretable Continual Learning](https://arxiv.org/abs/2503.23283) | [Code](https://github.com/FisherCats/CLG-CBM) |
| CVPR | [Discovering Fine-Grained Visual-Concept Relations by Disentangled Optimal Transport Concept Bottleneck Models](http://arxiv.org/abs/2505.07209) | - |
|  CVPR W  | [PCBEAR: Pose Concept Bottleneck for Explainable Action Recognition](https://arxiv.org/abs/2504.13140) |                              -                               |
| ICCV | [Intervening in Black Box: Concept Bottleneck Model for Enhancing Human Neural Network Mutual Understanding](http://arxiv.org/abs/2506.22803) | [Code](https://github.com/XiGuaBo/CBM-HNMU) |
|   ICLR   | [Counterfactual Concept Bottleneck Models](https://openreview.net/forum?id=w7pMjyjsKN) | [Code](https://github.com/gabriele-dominici/Counterfactual-CBM) |
|   ICLR   | [Concept Bottleneck Large Language Models](https://openreview.net/forum?id=RC5FPYVQaH) |    [Code](https://github.com/Trustworthy-ML-Lab/CB-LLMs)     |
|   ICLR   | [CONDA: Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts](https://iclr.cc/virtual/2025/poster/30736) |         [Code](https://github.com/jihyechoi77/CONDA)         |
|   ICLR   | [Concept Bottleneck Language Models For Protein Design](https://iclr.cc/virtual/2025/poster/29243) |                              -                               |
| ICLR W | [Causally Reliable Concept Bottleneck Models](https://arxiv.org/abs/2503.04363) | [Code](https://github.com/gdefe/causally-reliable-cbm) |
| ICLR W | [Adaptive Test-Time Intervention for Concept Bottleneck Models](https://openreview.net/forum?id=wBygggbUV8) | [Code](https://github.com/mattyshen/adaptiveTTI) |
|   ICML   | [Editable Concept Bottleneck Models](https://dblp.uni-trier.de/rec/journals/corr/abs-2405-15476.html) |                              -                               |
|   ICML   | [DCBM: Data-Efficient Visual Concept Bottleneck Models](https://arxiv.org/abs/2412.11576) |           [Code](https://github.com/KathPra/DCBM)            |
|   ICML   | [Addressing Concept Mislabeling in Concept Bottleneck Models Through Preference]([arxiv.org/pdf/2504.18026](https://arxiv.org/pdf/2504.18026)) | [Code](https://github.com/Emilianopp/ConceptPreferenceOptimization) |
| ICML | [Concept-Based Unsupervised Domain Adaptation](https://arxiv.org/pdf/2505.05195) | [Code](https://github.com/xmed-lab/CUDA) |
| IEEE TMI | [Concept-Based Lesion Aware Transformer for Interpretable Retinal Disease Diagnosis](https://ieeexplore.ieee.org/document/10599508) | [Code](https://github.com/Sorades/CLAT) |
| IEEE TMM | [Leveraging Concise Concepts with Probabilistic Modeling for Interpretable Visual Recognition](https://ieeexplore.ieee.org/document/10948345) |                              -                               |
| IJCAI | [MVP-CBM:Multi-layer Visual Preference-enhanced Concept Bottleneck Model for Explainable Medical Image Classification](http://arxiv.org/abs/2506.12568) | [Code](https://github.com/wcj6/MVP-CBM) |
|  MICCAI  | [Learning Concept-Driven Logical Rules for Interpretable and Generalizable Medical Image Classification](https://arxiv.org/abs/2505.14049) |            [Code](https://github.com/obiyoag/crl)            |
|  MICCAI  | [Training-free Test-time Improvement for Explainable Medical Image Classification](https://arxiv.org/abs/2506.18070) |       [Code](https://github.com/riverback/TF-TTI-XMed)       |
|   TMLR   | [Selective Concept Bottleneck Models Without Predefined Concepts](https://openreview.net/pdf?id=uuvujfQXZy) |      [Code](https://openreview.net/forum?id=PMO30TLI4l)      |
| xAI | [V-CEM: Bridging Performance and Intervenability in Concept-based Models](https://arxiv.org/abs/2504.03978) | [Code](https://github.com/francescoTheSantis/Variational-Concept-Embedding-Model) |
|  Arxiv   | [ConceptCLIP: Towards Trustworthy Medical AI Via Concept-Enhanced Contrastive Langauge-Image Pre-training](https://arxiv.org/abs/2501.15579) |       [Code](https://github.com/JerrryNie/ConceptCLIP)       |
| Arxiv | [Object Centric Concept Bottlenecks](http://arxiv.org/abs/2505.24492) | - |
| Arxiv | [Stable Vision Concept Transformers for Medical Diagnosis](http://arxiv.org/abs/2506.05286) | - |
| Arxiv | [Towards Reasonable Concept Bottleneck Models](http://arxiv.org/abs/2506.05014) | - |
| Arxiv | [Zero-shot Concept Bottleneck Models](https://arxiv.org/abs/2502.09018) | [Code](https://github.com/yshinya6/zcbm) |
| Arxiv | [CBVLM: Training-free Explainable Concept-based Large Vision Language Models for Medical Image Classification](https://arxiv.org/abs/2501.12266) | [Code](https://cristianopatricio.github.io/CBVLM/) |
| Arxiv | [Towards Achieving Concept Completeness for Textual Concept Bottleneck Models](https://arxiv.org/abs/2502.11100) | - |
| Arxiv | [Deferring Concept Bottleneck Models: Learning to Defer Interventions to Inaccurate Experts](https://arxiv.org/abs/2503.16199) | - |
| Arxiv | [If Concept Bottlenecks are the Question, are Foundation Models the Answer?](https://arxiv.org/abs/2504.19774v2) | [Code](https://github.com/debryu/CQA) |
| Arxiv | [DeCoDe: Defer-and-Complement Decision-Making via Decoupled Concept Bottleneck Models](https://doi.org/10.48550/arXiv.2505.19220) | - |

### 2024
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| AAAI | [On the Concept Trustworthiness in Concept Bottleneck Models](https://doi.org/10.1609/aaai.v38i19.30109) | [Code](https://github.com/hqhQAQ/ProtoCBM) |
| AAAI | [Sparsity-guided holistic explanation for llms with interpretable inference-time intervention](https://ojs.aaai.org/index.php/AAAI/article/download/30160/32058) | [Code](https://github.com/zhen-tan-dmml/sparsecbm) |
| CVPR | [Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion](https://doi.org/10.1109/cvpr52733.2024.02538) | - |
| CVPR | [LVLM-Interpret: An Interpretability Tool for Large Vision-Language Models](https://arxiv.org/pdf/2404.03118) | [Code](https://github.com/IntelLabs/lvlm-interpret) |
| CVPR | [Incremental Residual Concept Bottleneck Models](https://doi.org/10.1109/cvpr52733.2024.01049) |  [Code](https://github.com/helloscm/res-cbm)|
| ECCV | [Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery](https://arxiv.org/pdf/2407.14499v2) | [Code](https://github.com/neuroexplicit-saar/discover-then-name) |
| ECCV | [Explain Via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts](https://doi.org/10.1007/978-3-031-73016-0_8) | - |
| ICLR | [Concept Bottleneck Generative Models](https://dblp.uni-trier.de/rec/conf/iclr/IsmailABRC24.html) | [Code](https://github.com/prescient-design/CBGM) |
| ICLR | [Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Probabilistic Interpretations](https://doi.org/10.48550/arxiv.2401.14142) | [Code](https://github.com/xmed-lab/ECBM) |
| ICLR | [Faithful Vision-Language Interpretation Via Concept Bottleneck Models](https://openreview.net/pdf?id=rp0EdI8X4e) | [Code](https://github.com/kaustpradalab/FVLC) |
| ICLR | [Concept Bottleneck Generative Models](https://dblp.uni-trier.de/rec/conf/iclr/IsmailABRC24.html) |  |
| ICML | [Post-hoc Part-prototype Networks](https://arxiv.org/abs/2406.03421) | - |
| ICML W | [XCoOp: Explainable Prompt Learning for Computer-Aided Diagnosis via Concept-guided Context Optimization](https://arxiv.org/pdf/2403.09410v1) | - |
| IEEE TPAMI | [The Decoupling Concept Bottleneck Model](https://ieeexplore.ieee.org/document/10740789) | [Code](https://github.com/deepopo/DCBM) |
| JBHI | [Guest Editorial: Trustworthy Machine Learning for Health Informatics](https://ieeexplore.ieee.org/iel8/6221020/10745910/10745914.pdf) | - |
| MICCAI | [Concept-Attention Whitening for Interpretable Skin Lesion Diagnosis](https://arxiv.org/pdf/2404.05997v2) |[Code](https://github.com/CAWframework/Concept-Attention-Whitening-for-Interpretable-Skin-Lesion-Diagnosis-Reproduction)|
| MICCAI | [Aligning human knowledge with visual concepts towards explainable medical image classification](https://arxiv.org/pdf/2406.05596) |[Code](https://github.com/yhygao/Explicd?tab=readme-ov-file)|
| MICCAI | [Evidential concept embedding models: Towards reliable concept explanations for skin disease diagnosis](https://arxiv.org/pdf/2406.19130) |[Code](https://github.com/obiyoag/evi-CEM)|
| MICCAI | [Learning a Clinically-Relevant Concept Bottleneck for Lesion Detection in Breast Ultrasound](https://arxiv.org/pdf/2407.00267v1) |[Code](https://github.com/hawaii-ai/bus-cbm)|
| MICCAI | [Mask-Free Neuron Concept Annotation for Interpreting Neural Networks in Medical Domain](https://arxiv.org/pdf/2407.11375v1) |  [Code](https://github.com/ailab-kyunghee/mammi)|
| MICCAI | [AdaCBM: an Adaptive Concept Bottleneck Model for Explainable and Accurate Diagnosis](https://doi.org/10.48550/arxiv.2105.02410) |  [Code](https://github.com/AIML-MED/AdaCBM)|
| MICCAI | [Integrating Clinical Knowledge into Concept Bottleneck Models](https://doi.org/10.1007/978-3-031-72083-3_23) |[Code](https://github.com/pangwinnie0219/align_concept_cbm)|
| MedIA | [Interpretable and Intervenable Ultrasonography-Based Machine Learning Models for Pediatric Appendicitis](https://doi.org/10.1016/j.media.2023.103042) |[Code](https://github.com/i6092467/semi-supervised-multiview-cbm)|
| NeurIPS | [Stochastic Concept Bottleneck Models](https://arxiv.org/abs/2406.19272) | [Code](https://github.com/mvandenhi/scbm) |
| NeurIPS | [Coarse-to-Fine Concept Bottleneck Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bdeab378efe6eb289714e2a5abc6ed42-Abstract-Conference.html) | [Code](https://github.com/konpanousis/Coarse-To-Fine-CBMs) |
| NeurIPS | [VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance](https://arxiv.org/abs/2408.01432) | [Code](https://github.com/Trustworthy-ML-Lab/VLG-CBM) |
| NeurIPS | [A Theoretical Design of Concept Sets: Improving the Predictability of Concept Bottleneck Models](https://dblp.uni-trier.de/rec/conf/nips/LuytenS24.html) |-|
| NeurIPS | [Towards Multi-dimensional Explanation Alignment for Medical Classification](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea370419760b421ce12e3082eb2ae1a8-Abstract-Conference.html) |-|
| NeurIPS | [A Concept-Based Explainability Framework for Large Multimodal Models](https://arxiv.org/abs/2406.08074) |  [Code](https://github.com/mshukor/xl-vlms)|
| NeurIPS | [Classifier Clustering and Feature Alignment for Federated Learning under Distributed Concept Drift](https://arxiv.org/abs/2410.18478) |  [Code](https://github.com/chen-junbao/fedccfa)|
| NeurIPS | [ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty](https://arxiv.org/abs/2408.14339) |[Code](https://github.com/princetonvisualai/ConceptMix)|
| NeurIPS | [Do LLMs Dream of Elephants (when Told Not To)? Latent Concept Association and Associative Memory in Transformers](https://arxiv.org/abs/2406.18400) |-|
| NeurIPS | [FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://arxiv.org/abs/2407.06567) |  [Code](https://github.com/MXGao-A/FAgent)|
| NeurIPS | [Free Lunch in Pathology Foundation Model: Task-specific Model Adaptation with Concept-Guided Feature Enhancement](https://arxiv.org/abs/2411.09894) |  [Code](https://github.com/HKU-MedAI/CATE)|
| NeurIPS | [From Causal to Concept-Based Representation Learning](https://dblp.uni-trier.de/rec/conf/nips/RajendranBASR24.html) |-|
| NeurIPS | [Interpretable Concept Bottlenecks to Align Reinforcement Learning Agents](https://doi.org/10.48550/arxiv.2401.05821) |  [Code](https://github.com/k4ntz/scobots)|
| NeurIPS | [Interpretable Concept-Based Memory Reasoning](https://arxiv.org/abs/2407.15527) |  [Code](https://github.com/daviddebot/CMR)|
| NeurIPS | [Interpreting CLIP with Sparse Linear Concept Embeddings (Splice)](https://doi.org/10.48550/arxiv.2402.10376) |  [Code](https://github.com/ai4life-group/splice)|
| NeurIPS | [Learning Discrete Concepts in Latent Hierarchical Models](https://arxiv.org/abs/2406.00519) |-|
| NeurIPS | [LG-CAV: Train Any Concept Activation Vector with Language Guidance](https://arxiv.org/abs/2410.10308) |-|
| NeurIPS | [Neural Concept Binder](https://doi.org/10.48550/arxiv.2406.09949) |  [Code](https://github.com/ml-research/neuralconceptbinder)|
| NeurIPS | [No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance](https://doi.org/10.48550/arxiv.2404.04125) |  [Code](https://github.com/bethgelab/frequency_determines_performance)|
| NeurIPS | [PaCE: Parsimonious Concept Engineering for Large Language Models](https://arxiv.org/abs/2406.04331) |  [Code](https://github.com/peterljq/parsimonious-concept-engineering)|
| NeurIPS | [Relational Concept Bottleneck Models](https://arxiv.org/abs/2308.11991) |  [Code](https://github.com/diligmic/rcbm-neurips2024)|
| NeurIPS | [Uncovering Safety Risks of Large Language Models Through Concept Activation Vector](https://arxiv.org/abs/2404.12038) |  [Code](https://github.com/sproutnan/ai-safety_scav)|
| NeurIPS | [Towards Multi-dimensional Explanation Alignment for Medical Classification](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea370419760b421ce12e3082eb2ae1a8-Paper-Conference.pdf) | - |
| NeurIPS | [Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?](https://doi.org/10.48550/arxiv.2401.13544) | [Code](https://github.com/sonialagunac/beyond-cbm) |
| NeurIPS W | [Bayesian concept bottleneck models with llm priors](https://arxiv.org/abs/2410.15555) | [Code](https://github.com/jjfeng/bc-llm) |
| PAKDD | [Interpreting Pretrained Language Models Via Concept Bottlenecks](https://doi.org/10.1007/978-981-97-2259-4_5) | [Code](https://github.com/Zhen-Tan-dmml/CBM_NLP?tab=readme-ov-file) |
| Sci. Rep | [Pseudo-class Part Prototype Networks for Interpretable Breast Cancer Classification](https://doi.org/10.1038/s41598-024-60743-x) | [Code](https://github.com/MA-Choukali/PCPPN) |
| TMLR | [Reproducibility Study of "LICO: Explainable Models with Language-Image Consistency"](https://dblp.uni-trier.de/rec/journals/tmlr/FletcherKSVA24.html) | [Code](https://github.com/robertdvdk/lico-fact) |
| TMLR | [[Re].on the Reproducibility of Post-Hoc Concept Bottleneck Models](https://dblp.uni-trier.de/rec/journals/tmlr/MidavaineGCSC24.html) | [Code](https://github.com/dgcnz/FACT) |
| TMLR | [CLIP-QDA: an Explainable Concept Bottleneck Model](https://doi.org/10.48550/arxiv.2312.00110) |  |
| Arxiv | [Explainable and interpretable multimodal large language models: A comprehensive survey](https://arxiv.org/abs/2412.02104) | - |
| Arxiv | [Semi-supervised Concept Bottleneck Models](https://dblp.uni-trier.de/rec/journals/corr/abs-2406-18992.html) | [Code](https://github.com/Skyyyy0920/SSCBM) |
| Arxiv | [Self-eXplainable AI for Medical Image Analysis: A Survey and New Outlooks](https://arxiv.org/abs/2410.02331) | - |
| Arxiv | [Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis](https://arxiv.org/abs/2410.15446) | - |
| Arxiv | [Improving Concept Alignment in Vision-Language Concept Bottleneck Models](https://arxiv.org/pdf/2405.01825) | [Code](https://github.com/NMS05/Improving-Concept-Alignment-in-Vision-Language-Concept-Bottleneck-Models) |
| Arxiv | [CAT: Concept-level backdoor ATtacks for Concept Bottleneck Models](https://arxiv.org/abs/2410.04823) | [Code](https://github.com/xll0328/CAT_CBM-Backdoor) |
| Arxiv | [Tree-Based Leakage Inspection and Control in Concept Bottleneck Models](https://arxiv.org/abs/2410.06352) | [Code](https://github.com/ai4ai-lab/mixed-cbm-with-trees) |


### 2023
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| AAAI | [Interactive Concept Bottleneck Models](https://ojs.aaai.org/index.php/AAAI/article/view/25736/25508) |-|
| CVPR | [Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Language_in_a_Bottle_Language_Model_Guided_Concept_Bottlenecks_for_CVPR_2023_paper.pdf) |[Code](https://github.com/yueyang1996/labo)|
| CVPR | [Learning bottleneck concepts in image classification](https://arxiv.org/pdf/2304.10131v1) |[Code](https://github.com/wbw520/botcl)|
| CVPR | [Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision](https://arxiv.org/pdf/2303.00885v1) |-|
|  EMNLP  | [STAIR: Learning Sparse Text and Image Representation in Grounded Tokens](https://arxiv.org/pdf/2301.13081v2) |                              -                               |
|  EMNLP  | [Cross-Modal Conceptualization in Bottleneck Models](https://doi.org/10.18653/v1/2023.emnlp-main.318) |         [Code](https://github.com/danisalukaev/xcbs)         |
| ICLR | [Label-free Concept Bottleneck Models](https://arxiv.org/pdf/2304.06129) |  [Code](https://github.com/trustworthy-ml-lab/label-free-cbm)|
|  ICLR   | [Post-hoc Concept Bottleneck Models](https://doi.org/10.48550/arxiv.2205.15480) |        [Code](https://github.com/mertyg/post-hoc-cbm)        |
|  ICML   | [A Closer Look at the Intervention Procedure of Concept Bottleneck Models](https://doi.org/10.48550/arxiv.2302.14260) |  [Code](https://github.com/ssbin4/Closer-Intervention-CBM)   |
|  ICML   | [Probabilistic Concept Bottleneck Models](https://doi.org/10.48550/arxiv.2306.01574) |         [Code](https://github.com/ejkim47/prob-cbm)          |
| ICML W | [A ChatGPT Aided Explainable Framework for Zero-Shot Medical Image Diagnosis](https://doi.org/10.48550/arxiv.2307.01981) |-|
| MICCAI | [Concept Bottleneck with Visual Concept Filtering for Explainable Medical Image Classification](https://doi.org/10.1007/978-3-031-47401-9_22) |-|
| NeurIPS | [Do Concept Bottleneck Models Respect Localities](https://arxiv.org/abs/2401.01259) |  [Code](https://github.com/naveenr414/Spurious-Concepts)|
| NeurIPS | [Learning to Receive Help: Intervention-Aware Concept Embedding Models](https://openreview.net/forum?id=4ImZxqmT1K) | [Code](https://github.com/mateoespinosa/cem) |
| Arxiv | [Robust and interpretable medical image classifiers via concept bottleneck models](https://doi.org/10.48550/arxiv.2310.03182) |-|

### 2022
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| ICCV | [Explaining in Style: Training a GAN to Explain a Classifier in StyleSpace](https://doi.org/10.5281/zenodo.6574709) |-|
| ICLR | [CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks](https://arxiv.org/abs/2204.10965) |  [Code](https://github.com/trustworthy-ml-lab/clip-dissect)|
| NeurIPS | [Concept Embedding Models](https://arxiv.org/abs/2209.09056) |  [Code](https://github.com/mateoespinosa/cem)|
| NeurIPS | [Addressing Leakage in Concept Bottleneck Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/944ecf65a46feb578a43abfd5cddd960-Paper-Conference.pdf) | [Code](https://github.com/dtak/addressing-leakage) |

### 2021
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| ICML | [Meaningfully Debugging Model Mistakes Using Conceptual Counterfactual Explanations](https://doi.org/10.48550/arxiv.2106.12723) |  [Code](https://github.com/mertyg/debug-mistakes-cce)|
| NMI | [A case-based interpretable deep learning model for classification of mass lesions in digital mammography](https://arxiv.org/pdf/2103.12308) |[Code](https://github.com/alinajadebarnett/iaiabl)|
| Arxiv | [Partially Interpretable Estimators (PIE): Black-Box-Refined Interpretable Machine Learning](https://doi.org/10.48550/arxiv.2105.02410) |-|

### 2020
| Publication |    Paper Title     |   Code/Project                                                 |
|:----:|-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------:|
| ICML | [Concept bottleneck models](http://proceedings.mlr.press/v119/koh20a/koh20a.pdf) |  [Code](https://github.com/yewsiang/ConceptBottleneck)|
| NMI | [Concept whitening for interpretable image recognition](https://rdcu.be/cbOKj) |  [Code](https://github.com/zhiCHEN96/ConceptWhitening?tab=readme-ov-file)|



**Acknowledgement**

This project was originally inspired by https://github.com/kkzhang95/Awesome_Concept_Bottleneck_Models. We thank the authors for their contributions. Our main motivation is to provide an additional architecture organized by research focus, supplement it with more recent papers, and sort them by conference name for easier navigation.

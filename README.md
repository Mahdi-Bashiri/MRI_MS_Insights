[updated_readme.md](https://github.com/user-attachments/files/24014885/updated_readme.md)
# Population-Specific Lesion Patterns in Multiple Sclerosis: Automated Deep Learning Analysis of a Large Iranian Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11+](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)
[![Multiple Sclerosis](https://img.shields.io/badge/application-Multiple%20Sclerosis-red.svg)](https://github.com/topics/multiple-sclerosis)

## 🧠 Overview

This repository implements a large-scale neuroanatomical profiling study of Multiple Sclerosis using deep learning-based automated segmentation. Our comprehensive analysis of **1,381 subjects** from Northwest Iran provides detailed statistical characterization of brain structural changes in an underrepresented Middle Eastern population.

### 🎯 Key Contributions

- **🔬 Large-Scale MS Neuroimaging Study**: 1,381 participants (381 MS patients, 1,000 healthy controls)
- **🌍 Population-Specific Research**: Addresses gap in Middle Eastern MS neuroimaging research
- **🤖 Comparative Architecture Evaluation**: Four deep learning models evaluated (U-Net, Attention U-Net, Trans-U-Net, DeepLabV3Plus)
- **⚡ Optimal Model Selection**: Attention U-Net achieved superior performance (DSC=0.858, HD95=12.0mm)
- **📊 Comprehensive Statistics**: Multi-dimensional analysis across age, gender, and anatomical regions
- **🎯 Clinical Translation**: Population-specific normative values for MS biomarkers

### 📈 Key Clinical Findings

| Metric | MS Patients | Healthy Controls | Statistical Significance |
|--------|-------------|------------------|-------------------------|
| **Lesion Burden Disparity** | 5.5x higher | Baseline | p<0.001, r=0.82 (large effect) |
| **Periventricular Involvement** | 58.02±28.35% | Minimal | Consistent predominance |
| **Age-Related Progression** | 0.13% → 0.71% | 0.02% (stable) | Progressive accumulation |
| **Gender Differences (PEWMH)** | ♂: 61.85% vs ♀: 56.45% | - | p=0.018, r=0.12 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA support (RTX 3060 or equivalent)
- 64GB+ RAM (recommended for large-scale processing)
- FLAIR MRI sequences in NIfTI format

### Installation

```bash
# Clone the repository
git clone https://github.com/Mahdi-Bashiri/MRI_MS_Insights.git
cd MRI_MS_Insights

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🗂️ Repository Structure

The repository follows a **5-phase pipeline architecture**:

```
├── 📁 Article_Figures/              # 7 main figures from published article
│   ├── Figure_1.tif                 # Study flowchart
│   ├── Figure_2.tif                 # Study population demographics
│   ├── Figure_3.tif                 # Sample FLAIR annotations
│   ├── Figure_4.tif                 # Sample FLAIR slices with performances of models
│   ├── Figure_5.tif                 # Ventricular burden analysis
│   ├── Figure_6.tif                 # Lesion burden analysis
│   ├── Figure_7.tif                 # Anatomical lesion distribution
│   └── Figure_8.tif                 # Statistical correlation matrices
├── 📁 Article_Tables/               # 3 comprehensive tables
│   ├── Table_1.docx                # Demographic characteristics
│   ├── Table_2.docx                # MRI imaging protocols
│   ├── Table_3.docx                # Deep learning models performances
│   ├── Table_4.docx                # Detailed attention U-Net model performance
│   └── Table_5.docx                # Lesion subtype statistical comparison
├── 📁 Phase1_data_preprocessing/    # FLAIR image preprocessing
│   ├── 📁 raw_data/                # Sample data (10 patients)
│   └── pre_processing_flair.py     # 4-step preprocessing pipeline
├── 📁 Phase2_data_preparation_for_model_training/
│   ├── 📁 Original_FLAIRs_prep/    # Preprocessed FLAIR images
│   ├── 📁 abWMH_manual_segmentations/ # Manual abnormal WMH masks
│   ├── 📁 vent_manual_segmentations/  # Manual ventricle masks
│   └── 📁 manual_3l_masks/         # Generated 3-class masks
├── 📁 Phase3_model_training_and_inferencing_and_evaluation/
│   ├── 📁 dataset_3l_man/          # Training/testing datasets
│   ├── 📁 model_performance/       # Model performance for all 4 architectures
│   ├── 📁 trained_models/          # Pre-trained models (all 4 architectures)
│   │   ├── trans_unet_model/
│   │   └── Pretrained_Models.md    # Link to the Hugging Face repository for accessing the pre-trained models
│   ├── wmh_vent_segmentation_Fa.py    # Model training
│   ├── unet_model.py                  # U-Net Model
│   ├── attn_unet_model.py             # Attention U-Net Model
│   ├── trans_unet_model.py            # TransU-Net Model
│   ├── dlv3_unet_model.py             # DeepLabV3+ U-Net Model
│   ├── postprocessing.py              # Postprocessing
│   ├── metrics_functions.py           # Metrics
│   └── loss_functions.py              # Loss Functions
├── 📁 Phase4_data_processing/         # Quantitative analysis
│   ├── brain_mri_analysis_results_PROCESSED_updated.xlsx
│   ├── excel_extractor.py             # Data extraction
│   └── excel_filler_brain_TIA.py      # Brain area normalization
├── 📁 Phase5_statistical_analysis/   # Comprehensive statistics
│   └── 📁 csv_analysis_outputs/      # Statistical results
└── 📄 README.md                    # This file
```

---

## 🔬 Methodology

### Study Population

- **Total Participants**: 1,381 subjects from Northwest Iran (2021-2024)
  - **MS Patients**: 381 subjects (71.7% female, 28.3% male)
  - **Healthy Controls**: 1,000 subjects (67.4% female, 32.6% male)
- **Age Range**: 18-74 years (HC), 18-68 years (MS)
- **Mean Age**: HC: 34.7±10.2 years, MS: 37.4±10.2 years
- **Location**: Golghasht Medical Imaging Center, Tabriz, Iran
- **Ethics Approval**: Tabriz University of Medical Sciences Research Ethics Committee (IR.TBZMED.REC.1402.902)

### Deep Learning Architecture Evaluation

**Comparative Analysis of 4 Architectures:**

| Model | Ventricle DSC | WMH DSC | Overall DSC | Overall HD95 (mm) | Selection |
|-------|--------------|---------|-------------|-----------|-----------|
| **U-Net** | 0.912 | 0.801 | 0.856 | 12.2 | Baseline |
| **Attention U-Net** | 0.917 | 0.799 | **0.858** | **12.0** | ⭐ **Optimal** |
| **Trans-U-Net** | 0.904 | 0.742 | 0.823 | 16.8 | Good |
| **DeepLabV3Plus** | 0.862 | 0.726 | 0.794 | 17.8 | Good |

**DeepLabV3Plus Implementation:**
- **Architecture**: Atrous Spatial Pyramid Pooling (ASPP) with encoder-decoder
- **Backbone**: ResNet-50 for robust feature extraction
- **Input**: Single-modality FLAIR sequences (256×256 pixels)
- **Output**: 3-class segmentation (background, ventricles, WMH)
- **Processing Speed**: 32ms per image (inference)
- **Training Details**: 100 epochs with hybrid loss function strategy (WCE + Focal)

### Preprocessing Pipeline

1. **Noise Reduction**: 3×3 median filter + Gaussian smoothing (σ=1.0)
2. **Intensity Normalization**: Slice-wise z-score standardization
3. **Dimension Standardization**: Isotropic resampling to 1×1 mm² pixels
4. **Matrix Standardization**: Resizing to 256×256 pixels

### Neuroanatomical Lesion Classification

**Distance-Based Classification Criteria:**
- **PEWMH** (Periventricular): ≤3mm from ventricular surface
- **JCWMH** (Juxtacortical): ≤3mm from gray-white matter junction, area ≤20mm²
- **PAWMH** (Paraventricular): Deep white matter lesions not meeting above criteria

---

## 📊 Key Clinical Findings

### Model Performance

**Attention U-Net Achieved Optimal Performance:**
- **Dice Similarity Coefficient**: 0.858 (overall)
- **Hausdorff Distance (95th percentile)**: 12.0mm
- **Ventricle Segmentation**: DSC=0.917, Precision=0.912, Recall=0.923
- **WMH Segmentation**: DSC=0.799, Precision=0.834, Recall=0.767

### Lesion Burden Analysis

- **MS vs HC Comparison**: 5.5-fold higher lesion burden in MS patients
- **Age-Related Progression**: 
  - MS: 0.13% (18-29 years) → 0.71% (60+ years)
  - HC: 0.02% (stable across age groups)
- **Statistical Significance**: Mann-Whitney U test, p<0.001, effect size r=0.82 (large)

### Anatomical Distribution

- **Periventricular Predominance**: 58.02±28.35% of total lesion burden
  - Age-related increase: 44.97% (young) → 71.43% (older cohorts)
- **Gender-Specific Patterns**:
  - Males: 61.85±25.84% periventricular (higher proportion)
  - Females: 56.45±29.12% periventricular
  - Statistical comparison: U=12,847, p=0.018, r=0.12
- **Paraventricular**: 20.76±21.45%
- **Juxtacortical**: 17.73±19.82%

### Ventricular Burden

- **No significant difference between MS and HC**: t=1.42, p=0.156
- **Age-related correlations**:
  - HC: r=0.241 (weak positive)
  - MS: r=0.319 (moderate positive)
- **Pattern**: Progressive enlargement with age in both groups

### Population-Specific Context

- **Regional Prevalence**: Iran shows 100 per 100,000 vs global 35.9 per 100,000
- **Study Contribution**: First large-scale neuroanatomical characterization in Middle Eastern population
- **Clinical Utility**: Population-specific normative values for therapeutic monitoring

---

## 🛠️ Technical Specifications

### Hardware Requirements

- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or equivalent
- **CPU**: Intel Core i7-7700K (8 cores) or equivalent
- **RAM**: 64GB DDR4 (recommended for large-scale processing)
- **Storage**: 100GB+ free space for datasets and processing

### Software Stack

```python
# Core Dependencies
tensorflow==2.11.0
cuda==11.8
nibabel>=3.2.0
scikit-learn>=1.0.0
scipy>=1.7.0
scikit-image>=0.19.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### MRI Acquisition Specifications

- **Scanner**: 1.5-Tesla TOSHIBA Vantage (Canon Medical Systems)
- **T2-FLAIR Sequence Parameters**:
  - TR = 10,000 ms
  - TE = 100 ms
  - TI = 2,500 ms
  - Flip angle = 90°
  - Field of view = 230 × 230 mm²
  - Slice thickness = 6 mm
  - Voxel size = 0.9 × 0.9 mm²
  - Acquisition matrix = [0, 256, 192, 0]

---

**Dataset Composition:**
- 100 MS patients from local dataset
- Split: 90% training, 10% testing
- Total: 1,775 training images, 199 testing

**Training Configuration:**
- Optimizer: Adam (lr=1×10⁻⁴, β₁=0.9, β₂=0.999)
- Batch size: 8
- Epochs: 100 (early stopping patience=10)
- Loss: Hybrid strategy (weighted categorical cross-entropy → unified focal loss)
- Learning rate schedule: ReduceLROnPlateau

**Extracted Metrics:**
- Total ventricular area and ratio
- Total WMH area and ratio
- PEWMH, PAWMH, JCWMH areas and proportions
- Age and gender stratification
- Normalized values (% of total intracranial area)

**Statistical Methods:**
- Age stratification: 5 groups (18-29, 30-39, 40-49, 50-59, 60+ years)
- Normality testing: Shapiro-Wilk
- Group comparisons: Independent t-tests / Mann-Whitney U tests
- Effect sizes: Cohen's d / Pearson's r
- Correlation analysis: Spearman correlation matrices
- Multiple comparison correction: Bonferroni adjustment

---

## 📈 Performance Benchmarks

### Computational Efficiency

| Operation | Time | Hardware |
|-----------|------|----------|
| **Training (per epoch)** | 40-45 seconds | RTX 3060 |
| **Inference (per image)** | 32 milliseconds | RTX 3060 |
| **Full cohort processing** | ~2 hours | 1,381 subjects |
| **Statistical analysis** | ~10 minutes | i7-7700K |

**Clinical Acceptability:**
- ✅ Suitable for population-level analysis
- ✅ Consistent performance across age groups
- ✅ Robust to clinical imaging variability

---

## 🌍 Clinical Impact

### Research Contributions

- **Population-Specific Data**: First comprehensive MS neuroimaging study in Middle Eastern population
- **Normative Values**: Baseline references for Iranian/Middle Eastern MS patients
- **Methodological Framework**: Scalable approach for large-scale neuroimaging studies
- **Open-Source Tools**: Reproducible pipeline for global MS research community

### Clinical Applications

- **Diagnostic Support**: Automated lesion quantification and classification
- **Disease Monitoring**: Longitudinal tracking of lesion burden
- **Treatment Assessment**: Quantitative biomarkers for therapeutic response
- **Risk Stratification**: Age and gender-specific lesion patterns

### Future Research Directions

1. **Longitudinal Studies**: Track individual patient trajectories
2. **Multi-Modal Integration**: Combine FLAIR with DTI, T1-weighted sequences
3. **Clinical Correlation**: Integrate with EDSS scores and disease duration
4. **Genetic Association**: Link imaging biomarkers with genetic profiles
5. **Multi-Center Expansion**: Validate across diverse Iranian regions

---

## 📚 Documentation

### Available Resources
- **[Article Manuscript](p4_Manuscript_BAWIL2025_Fa.pdf)**: Complete research article
- **[Figure Collection](Article_Figures/)**: All 8 publication-quality figures
- **[Statistical Tables](Article_Tables/)**: Comprehensive demographic and performance tables

### Key Figures
- **Figure 1**: Study flowchart
- **Figure 2**: Study population demographics
- **Figure 3**: Expert manual annotation examples
- **Figure 4**: Samples of FLAIR slices and model performances
- **Figure 5**: Age-stratified ventricular burden
- **Figure 6**: WMH burden comparison
- **Figure 7**: Anatomical lesion distribution
- **Figure 8**: Correlation matrices (HC vs MS)

---

## 🤝 Contributing

We welcome contributions to advance MS neuroimaging research! Areas for contribution:

- **Algorithm Improvements**: Enhanced architectures or training strategies
- **Population Expansion**: Extension to other geographic regions
- **Clinical Validation**: Real-world deployment studies
- **Multi-Modal Analysis**: Integration with advanced imaging sequences

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{bashiri2026population,
    title={Population-Specific Lesion Patterns in Multiple Sclerosis: Automated Deep Learning Analysis of a Large Iranian Dataset},
    author={Bashiri Bawil, Mahdi and Shamsi, Mousa and Shakeri Bavil, Abolhassan},
    journal={Under Review - Journal of Soft Computing and Information Technology},
    year={2026},
    note={Ethics Approval: IR.TBZMED.REC.1402.902},
    url={https://github.com/Mahdi-Bashiri/MRI_MS_Insights}
}
```

---

## 🏥 Study Information

### Clinical Site
**Golghasht Medical Imaging Center**  
Tabriz, Iran  
- Data collection: 2021-2024
- Expert neuroradiologist validation (20+ years experience)
- Standardized clinical MRI protocols

### Ethics & Compliance
- **IRB Approval**: Tabriz University of Medical Sciences Research Ethics Committee
- **Approval Number**: IR.TBZMED.REC.1402.902
- **Compliance**: 1964 Helsinki Declaration and amendments
- **Patient Consent**: Written informed consent obtained
- **Data Protection**: Comprehensive anonymization protocols

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Golghasht Medical Imaging Center** for clinical dataset and imaging resources
- **Expert neuroradiologists** for manual annotations and validation
- **Study participants** (381 MS patients, 1,000 healthy volunteers)
- **Tabriz University of Medical Sciences** for institutional support and ethics oversight
- **Open-source community** for foundational deep learning tools

---

## 📞 Contact & Support

### Repository
- **GitHub**: [https://github.com/Mahdi-Bashiri/MRI_MS_Insights](https://github.com/Mahdi-Bashiri/MRI_MS_Insights)
- **Issues**: [GitHub Issues](https://github.com/Mahdi-Bashiri/MRI_MS_Insights/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/MRI_MS_Insights/discussions)

### Research Inquiries
For research collaborations, clinical implementations, or dataset access, please use GitHub communication channels or contact through institutional email.

---

## 🌟 Impact Statement

This research represents a contribution to understanding Multiple Sclerosis through neuroimaging in an underrepresented population. By providing:

- **Open-source tools** for automated MS lesion analysis
- **Population-specific normative data** for Middle Eastern populations
- **Methodological framework** for large-scale neuroimaging studies
- **Clinical translation pathway** from research to practice

We aim to support improved patient care and advance global MS research initiatives.

---

[![Star History Chart](https://api.star-history.com/svg?repos=Mahdi-Bashiri/MRI_MS_Insights&type=Date)](https://star-history.com/#Mahdi-Bashiri/MRI_MS_Insights&Date)

---

*Version: 1.0 | Last Updated: January 2026 | Status: Manuscript Under Review*

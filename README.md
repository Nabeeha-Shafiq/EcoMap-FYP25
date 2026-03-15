# EcoMap Modular Pipeline 🗺️

**Unified, dataset-agnostic spatial transcriptomics ML pipeline for ecotype classification**

## Quick Start

### 1. Install Dependencies
```bash
cd modular_pipeline
pip install -r requirements.txt
```

### 2. Prepare Your Data
Create a YAML config (see `config/geo_exam ple.yaml` or `config/zenodo_example.yaml`):
```yaml
dataset:
  name: "MY_DATASET"
  n_samples: 25000
  n_classes: 5

embeddings:
  UNI:
    name: "UNI"
    file_path: "/path/to/uni_embeddings.npy"
    file_format: "npy"
    n_dims: 150

# ... (see config examples for complete format)
```

### 3. Run Conversion
```bash
python scripts/1_convert_embeddings.py --config config/my_dataset.yaml
```

**Output:** Single HDF5 file with all embeddings + metadata ✅

## Project Structure

```
modular_pipeline/
├── src/                          # Core library modules
│   ├── __init__.py
│   ├── config_loader.py          # YAML → Pydantic (validation)
│   ├── embeddings.py             # NumPy/CSV → HDF5 conversion
│   ├── validation.py             # Data quality checks
│   ├── pca_reduction.py          # (Week 2)
│   ├── fusion.py                 # (Week 2)
│   ├── train.py                  # (Week 3)
│   ├── visualize.py              # (Week 3)
│   └── utils.py                  # Colors, logging, metrics
│
├── scripts/                       # Executable entry points
│   ├── 1_convert_embeddings.py   # Main: Embeddings → HDF5
│   ├── 2_validate_embeddings.py  # (Week 2)
│   ├── 3_pca_reduction.py        # (Week 2)
│   ├── 4_train.py                # (Week 3)
│   ├── 5_visualize.py            # (Week 3)
│   ├── run_full_pipeline.py      # (Week 3)
│   ├── prepare_geo_data.py       # Test data prep
│   └── prepare_zenodo_data.py    # Test data prep
│
├── config/                        # Configuration templates
│   ├── default.yaml              # Blank template
│   ├── geo_example.yaml          # GEO dataset (23.3K samples, 303D)
│   └── zenodo_example.yaml       # Zenodo dataset (16.6K samples, 197D)
│
├── tests/                         # (Week 4)
│   ├── test_config_loader.py
│   ├── test_embeddings.py
│   └── test_end_to_end.py
│
├── notebooks/                     # (Week 4)
│   └── tutorial_basic_usage.ipynb
│
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── .gitignore
```

## Features

### ✅ Phase 1: Complete (Week 1)
- [x] Configuration via YAML + Pydantic validation
- [x] Auto-detect embedding files by modality name
- [x] Support both NPY and CSV formats
- [x] Efficient HDF5 storage with metadata
- [x] Comprehensive logging and error handling
- [x] Works on ANY spatial transcriptomics dataset

### 🔄 Phase 2: In Progress (Week 2)
- [ ] Data quality validation module
- [ ] Per-modality PCA reduction
- [ ] Modality fusion strategies
- [ ] Spatial visualization framework

### ⏳ Phase 3: Scheduled (Week 3)
- [ ] MLP training with 5-fold CV
- [ ] Class weight balancing
- [ ] Early stopping + model selection
- [ ] Spatial heatmaps + Moran's I
- [ ] Per-class accuracy and confusion matrices

### ⏳ Phase 4: Scheduled (Week 4)
- [ ] Full unit test suite
- [ ] Integration tests
- [ ] Jupyter tutorials
- [ ] Final documentation

## Data Flow

```
Raw Embeddings (NPY/CSV)
    ↓
[1_convert_embeddings.py] ← YAML config
    ↓ (validate, align barcodes)
Single HDF5 File ← Meta data stored as attributes
    ↓
[2_validate_embeddings.py] (Week 2)
    ↓
[3_pca_reduction.py] (Week 2)
    ↓
[4_train.py] (Week 3) → 5-fold CV results
    ↓
[5_visualize.py] (Week 3) → Spatial maps + metrics
```

## Configuration

### Minimal Example
```yaml
dataset:
  name: "MY_DATA"
  n_samples: 20000

embeddings:
  UNI:
    name: "UNI"
    file_path: "/data/uni.npy"
    file_format: "npy"
    n_dims: 150

labels_metadata:
  labels_path: "/data/labels.npy"
  barcode_path: "/data/barcodes.csv"

output_dir: ./results
```

### Full Example
See [config/geo_example.yaml](config/geo_example.yaml)

## Usage Examples

### Example 1: Convert GEO dataset
```bash
python scripts/1_convert_embeddings.py --config config/geo_example.yaml
# → creates results_geo/GEO_23342.h5
```

### Example 2: Convert Zenodo dataset
```bash
python scripts/1_convert_embeddings.py --config config/zenodo_example.yaml
# → creates results_zenodo/ZENODO_16639.h5
```

### Example 3: Custom dataset
```bash
# 1. Create config/my_dataset.yaml (copy from geo_example.yaml)
# 2. Update paths to your embeddings, labels, barcodes
python scripts/1_convert_embeddings.py --config config/my_dataset.yaml --verbose
```

## Testing

### Test on GEO Dataset (23.3K samples, 303D)
- UNI-150D + scVI-128D + RCTD-25D
- Validates cross-modal barcode alignment
- Tests compression: 40 MB → 26 MB HDF5

### Test on Zenodo Dataset (16.6K samples, 197D)
- UNI-150D + scVI-32D + RCTD-15D
- Different dimensions per modality
- Tests flexibility: 12.14 MB HDF5

## Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| NPY | ✅ | ❌ | Fast, binary |
| CSV | ✅ | ❌ | Human-readable |
| HDF5 | ✅ | ✅ | Efficient, hierarchical |

## Colors Used

5 vivid, high-contrast colors for ecotypes:
- Fibrotic: **Red** (#FF0000)
- Immunosuppressive: **Blue** (#0047AB)
- Invasive_Border: **Green** (#00AA00)
- Metabolic: **Purple** (#AA00AA)
- Normal_Adjacent: **Orange** (#FF6600)

##  Dependencies

```
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
pydantic==2.4.2
pyyaml==6.0.1
h5py==3.9.0
```

## Architecture Decisions

**Why HDF5?**
- Single file per dataset (no scattered files)
- Hierarchical structure (embeddings/ group for modalities)
- Metadata embedded as attributes
- Compression included (gzip)
- Standard in scientific computing

**Why YAML + Pydantic?**
- Human-readable configuration
- Type validation at startup
- Clear error messages
- Industry standard

**Why 5-Fold CV (not LOPO)?**
- Dataset-agnostic (works on any split)
- Faster than leave-one-patient-out
- Appropriate for 5-patient datasets

## Timeline

- **Week 1** ✅: Core I/O infrastructure (HDF5 creation, config system)
- **Week 2** 🔄: Validation + PCA modules
- **Week 3** ⏳: Training + visualization
- **Week 4** ⏳: Testing + documentation

## Authors

FYP Team - Spatial Transcriptomics ML Pipeline

## License

TBD - Nabeeha-Shafiq/EcoMap-FYP25

---

**Questions?** See the design documents in the parent directory:
- `MODULAR_PIPELINE_DESIGN_REPORT.md`
- `DETAILED_IMPLEMENTATION_SPEC.md`
- `ARCHITECTURE_DIAGRAM.md`

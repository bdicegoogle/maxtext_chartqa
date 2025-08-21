# Gemma 3-12B Fine-tuning on ChartQA with MaxText

End-to-end pipeline for fine-tuning Google's Gemma 3-12B model on the ChartQA dataset using MaxText and Google Cloud TPUs.

## 🚀 Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env with your configuration

# 2. Run the complete pipeline
./run_experiment.sh
```

## 📋 Prerequisites

- **Google Cloud Project** with billing enabled and TPU quota
- **gcloud CLI** installed and authenticated
- **Docker** installed and configured
- **HuggingFace Token** with access to Gemma models
- **Python 3.12+** with `uv` package manager

## 🔧 Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure Google Cloud:**
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your project settings and HF token
   ```

## 🏗️ Pipeline Overview

The `run_experiment.sh` script performs these steps:

1. **🛠️ Infrastructure Setup** - Enable APIs, configure permissions, create buckets
2. **🔄 Model Conversion** - Convert HuggingFace Gemma to MaxText format
3. **✅ Model Validation** - Test converted model with sample inference
4. **🎯 Fine-tuning** - SFT on ChartQA dataset using TPUs
5. **🔍 Model Testing** - Validate fine-tuned model performance
6. **🚀 Serving Deploy** - Deploy with JetStream for high-performance inference
7. **📦 Export** - Convert back to HuggingFace format

## 📊 Key Features

- **TPU Acceleration** - Optimized for Google Cloud TPU v5p/v6e
- **Multimodal Support** - Vision-language capabilities for chart understanding
- **Production Ready** - JetStream serving for high-throughput inference
- **Interactive Monitoring** - TensorBoard integration and step-by-step logging

## 🎛️ Configuration

Key settings in `.env`:

| Variable | Description | Example |
|----------|-------------|---------|
| `TPU_TYPE` | TPU hardware type | `v5p-8`, `v6e-8` |
| `SFT_STEPS` | Fine-tuning steps | `250`, `1000` |
| `HF_TOKEN` | HuggingFace access token | `hf_xxx...` |
| `CLUSTER_NAME` | GKE cluster name | `gemma-experiment` |

## 📈 Monitoring

- **Training Progress:** `tensorboard --logdir=gs://your-bucket/path/tensorboard/`
- **Pod Status:** `kubectl get pods | grep gemma`
- **Logs:** `kubectl logs <pod-name> -f`

## 💰 Cost Considerations

- **TPU Usage:** pricing is available at : https://cloud.google.com/tpu/pricing?hl=en 
- **Storage:** Minimal costs for model checkpoints
- **Duration:** 2-4 hours for complete pipeline

> ⚠️ **Important:** Monitor costs in Google Cloud Console and delete resources when done.

## 🔍 Expected Performance

- **Fine-tuning Speed:** ~250 steps in 30-60 minutes on v5p-8
- **Serving Throughput:** ~20-25 QPS with JetStream
- **Model Quality:** Enhanced chart reasoning capabilities

## 🛟 Troubleshooting

| Issue | Solution |
|-------|----------|
| TPU quota exceeded | Request quota increase or change region |
| HF authentication | Verify token has Gemma model access |
| Docker push fails | Run `gcloud auth configure-docker` |
| Pod pending | Check node availability and resource requests |

## 📁 Project Structure

```
maxtext_chartqa/
├── run_experiment.sh    # Main pipeline script
├── .env.example        # Configuration template
├── .env               # Your configuration (create from .env.example)
├── inference/         # JetStream serving container
├── pyproject.toml     # Python dependencies
└── README.md          # This file
```

## 🔗 Resources

- [MaxText Documentation](https://github.com/AI-Hypercomputer/maxtext)
- [ChartQA Dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
- [XPK User Guide](https://github.com/google/xpk)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)

## 📄 License

Apache 2.0 - See pipeline scripts for full license text.

---

*Built with ❤️ for high-performance multimodal AI on Google Cloud TPUs*

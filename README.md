# Distributed Scientific Retrieval System

Multimodal retrieval infrastructure for scientific documents built with 
PyTorch.

## Overview

This project implements a scalable semantic retrieval system capable of:

- ingesting collections of scientific PDFs
- extracting and processing document content
- training retrieval models with PyTorch
- performing hybrid semantic + keyword search
- serving low-latency retrieval APIs
- supporting distributed training and inference workflows

## Current Goals

- [x] Build PDF ingestion pipeline
- [x] Create document chunking and preprocessing scheme
- [x] Implement embedding generation
- [ ] Train dual-encoder retrieval models
- [ ] Add reranking pipeline
- [ ] Deploy scalable API
- [ ] Optimize inference latency
- [ ] Add monitoring

## Architecture

```text
PDF Documents
Ingestion Pipeline
OCR + Metadata Extraction
Chunking + Cleaning
Embedding Generation
Vector Indexing
Hybrid Retrieval
Reranking
Inference API
```

---

## Repo Structure

```text
src/
├── data/
├── models/
├── training/
├── inference/
├── retrieval/
└── utils/

tests/
configs/
scripts/
docker/
docs/
```

---

## Tech Stack

### ML
- PyTorch
- Transformers
- FAISS

### Infrastructure
- Docker
- Kubernetes
- FastAPI

### Experimentation
- Weights & Biases
- MLflow

### Monitoring
- Prometheus
- Grafana

---

## Setup

### Environment

```bash
uv venv
source .venv/bin/activate
```

### Install Dependencies

```bash
uv pip install -e .
```

### Run Tests

```bash
pytest
```

### Run Linting

```bash
ruff check .
```

---

## Motivation

Most public ML projects focus on model training. This project focuses on engineering challenges involved in building production-grade ML retrieval systems:
- data ingestion
- distributed training
- inference optimization
- deployment infrastructure
- observability
- evaluation pipelines
- scalability

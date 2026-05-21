# Distributed Scientific Retrieval System

Production-oriented multimodal retrieval infrastructure for scientific documents built with 
PyTorch.

## Overview

This project implements a scalable semantic retrieval system capable of:

- ingesting large collections of scientific PDFs
- extracting and processing document content
- training dense retrieval models with PyTorch
- performing hybrid semantic + keyword search
- serving low-latency retrieval APIs
- supporting distributed training and inference workflows

This is designed to emulate real-word ML engineering and retrieval infrastructure used in modern AI 
organizations.

## Current Goals

- [x] Build PDF ingestion pipeline
- [ ] Create document chunking and preprocessing scheme
- [ ] Implement embedding generation
- [ ] Train dual-encoder retrieval models
- [ ] Add reranking pipeline
- [ ] Deploy scalable API
- [ ] Optimize inference latency
- [ ] Add monitoring

## Planned Architecture

```text
PDF Documents
    ↓
Ingestion Pipeline
    ↓
OCR + Metadata Extraction
    ↓
Chunking + Cleaning
    ↓
Embedding Generation
    ↓
Vector Indexing
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
Inference API
```

---

## Repository Structure

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

### ML / Training
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

## Development Setup

### Create Environment

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

Most public ML projects focus exclusively on model training.

This project focuses on the broader engineering challenges involved in building production-grade ML retrieval 
systems:

- data ingestion
- distributed training
- inference optimization
- deployment infrastructure
- observability
- evaluation pipelines
- scalability

---

## Status

Early development.

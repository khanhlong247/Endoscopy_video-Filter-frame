# Endoscopy Video Frame Deduplication Pipeline

A high-performance video frame deduplication system for intestinal endoscopy. The project utilizes a self-monitoring model **DINOv2**, a parallel task queue **Celery + Redis**, and a vector database **ChromaDB** combined with a **Vectorized Global Filtering** algorithm running directly in RAM to eliminate duplicate frames before feeding them into an AI Object Detection model.

---

## 📌 Project Overview

In video-based intestinal endoscopy, the camera typically moves very slowly, rotates its head, or remains stationary while the doctor injects water/aspirates fluid or closely observes the mucosa. This results in an extremely large amount of duplicate frames (often accounting for 50% to 80% of the video duration).

If all these frames are directly fed into an Object Detection model (such as YOLO, DETR, etc.) which only processes static images:
1. **Server burden:** Extremely high GPU resource consumption for inference on identical images.
2. **Inconsistent results:** Because static models lack temporal memory, the same block in two consecutive frames can produce two different prediction results (causing flickering).

**Project's solution:** Build a preprocessing pipeline using a hierarchical architecture. The system extracts features (embedding) from each frame using the **DINOv2 ViT-S** model, stores the vector and metadata in **ChromaDB**, and then applies parallel matrix multiplication in RAM to filter out duplicate frames throughout the video (including loop closures repeated at distant time points). Only frames containing new information are sent to the AI ​​Object Detection model for prediction.

---

## 🛠️ Technologies Used

The project is built on advanced backend and AI technologies:

* **Virtual Environment & Package Management:** `uv` (Astral's ultra-fast package manager for Python 3.10, completely solving dependency resolution issues).

* **Message Broker & Task Queue:** Celery collaborates with Redis (Dockerized) to distribute vector embedding extraction tasks in parallel across GPUs/CPUs.

* **Image Embedding Model:** DINOv2 ViT-S/14 (HuggingFace Transformers & PyTorch) - A self-monitoring model from Meta AI, extremely sensitive to the surface texture and geometry of the intestinal mucosa. Designed as a Singleton Pattern to optimize RAM/VRAM usage on workers.

* **Vector Database:** ChromaDB (Persistent mode) - Stores vector embeddings and accompanying metadata, optimized for Cosine Similarity.

* **Image Processing & Mathematics:** OpenCV (frame extraction) and NumPy (optimized filtering algorithm using vectorized matrix multiplication).

---

## 🔄 Detailed Operational Pipeline

The video processing process goes through the following four main stages:

```
[ Video Input (.mp4) ]

│

▼

1. Frame Extraction ──► Assign Timestamp to each Frame (CPU)

│

▼

2. Celery Worker (Group) ──► Run DINOv2 Embedding (GPU/CPU) ──► Save to ChromaDB

│

▼

3. Matrix (RAM) Duplicate Filtering ──► [O(N * K) Vectorized Global Filtering] ──► Remove duplicates

│

▼

4. Export JSON Metadata File ──► Ready for Object Model Detection
```

### 1. Frame Extraction & Timestamping
Video is split into static frames with a defined target frame rate (e.g., TARGET_FPS = 5 fps). Each frame is assigned a unique timestamp (or frame_index) to ensure temporal sequence and the ability to reverse-map the original video after processing.

### 2. Embedding & Parallel Storage (Celery + DINOv2 + ChromaDB)
* To avoid bottlenecking the main processing thread, the metadata list of frames is sent as a parallel task (group) to the Celery Workers.

* Each worker uses the DINOv2Singleton class to pre-load the `facebook/dinov2-small` model into memory (loaded only once).

* The image features are extracted using the feature vector $d = 384$ via DINOv2, then **L2 Normalized** and stored directly in **ChromaDB** along with metadata (path, timestamp, run_id).

### 3. Vectorized Global Filtering Algorithm

To overcome the weakness of local filters (which only compare adjacent frames), the algorithm performs global filtering in RAM using the NumPy library with an extremely optimized time complexity of $O(N \times K)$:

- **Initialization:** The vector of the first Frame (Frame 0) is taken as the first row of the matrix $E_{keep}\in\mathbb{R}^{1\times d}$ and added to the `keep_Frames` list.

- **Sequential Traversal:** For each subsequent Frame $i$ with normalized vector $v_i\in\mathbb{R}^{d}$:

  - Calculate the cosine similarity of Frame $i$ with **all previously stored frames** through matrix multiplication:

    $$
    S = E_{keep}\cdot v_i
    $$

  - Find the maximum similarity value:

    $$
    S_{\max}=\max(S)
    $$

  - **Decision**

    - If $S_{\max}\ge\theta$ (e.g., $\theta=0.96$), this frame is too similar to a location the camera has visited before.
      $\rightarrow$ **Remove**.

    - If $S_{\max}<\theta$, this frame contains new information.
      $\rightarrow$ Add $v_i^T$ to the last row of $E_{keep}$ and add Frame $i$ to the `Keep_Frames` list.

### 4. Export Results (Metadata Export)
The list of unique surviving frames is exported as a JSON file containing complete metadata, allowing the Object Detection model to run inference extremely smoothly and accurately.

---

## 📁 Project Directory Structure

The system is organized according to the Hierarchical Architecture standard:

```text
endoscopy_pipeline/
├── data/
│ ├── video/                            # Directory containing input video (e.g., sample.mp4)
│ ├── frame/                            # Directory containing extracted frames (automatically generated according to runtime encoding)
│ └── processed_frame/                  # Directory storing the JSON file of the results after duplicate filtering
├── database/
│ ├── __init__.py
│ └── chroma_db.py                      # Manages initialization and CRUD operations with ChromaDB
├── service/
│ ├── __init__.py
│ ├── model_service.py                  # Initialize DINOv2 model using Singleton Pattern
│ ├── embedding_service.py              # Define Celery Tasks to extract embedding vectors
│ └── filter_service.py                 # Implement Vectorized Global Filtering algorithm
├── utils/
│ ├── __init__.py
│ ├── celery_app.py                     # Initialize and configure Celery application
│ ├── video_utils.py                    # Function to extract frames and assign timestamps from video
│ └── filter_utils.py                   # Function to save results to a file JSON
├── .env                                # Contains important system configuration variables
├── config.py                           # Reads and maps environment variables from the .env file
├── docker-compose.yml                  # Deploys Redis and Celery Worker via Docker
├── Dockerfile.celery                   # Dockerfile specific to Celery Worker
├── main.py                             # Script to run the main Pipeline processing thread
└── README.md                           # Project documentation

```

---

## 🚀 System Installation & Run Guide

### 1. Prepare the Server Environment (Local)

Ensure your computer has **Docker Desktop** and the **uv** manager installed.

```bash
uv sync
```
### 2. System Configuration (`.env`)

Create a `.env` file in the root directory and edit the parameters if necessary:
```env
CELERY_BROKER_URL=redis://localhost:6379/0
CHROMA_DB_PATH=./database/chroma_data
COLLECTION_NAME=endoscopy_frames
TARGET_FPS=5
COSINE_THRESHOLD=0.96
```

### 3. Launch Services using Docker

Use Docker Compose to launch Redis and Celery Worker in the background:
```bash
docker compose up --build -d
```

To check the operational status of the containers:
```bash
docker compose ps
```

### 4. Prepare Data & Launch Pipeline

1. 1. Place your endoscopy video in the `data/video/` folder, for example, `sample.mp4`.

2. Launch the main process of the pipeline:
```bash
python main.py
```

After running, the duplicate filtering results will be saved at `data/processed_frame/sample_<time_code>_filtered.json`.

---

## 🎯 Conclusion

The project successfully solved the problem of frame duplication filtering for intestinal endoscopy videos through a harmonious combination of backend software techniques and artificial intelligence:
* Using **Celery + Redis** maximizes hardware power by processing I/O-bound and GPU-bound in parallel.

* Applying native **DINOv2** (Zero-shot) provides superior image feature rendering without the need for expensive fine-tuning.

* The **Vectorized Global Filtering** algorithm on RAM completely overcomes the "filtering loop" error of traditional filtering methods, reducing the number of duplicate frames fed into the Object Detection model by **50% - 80%**, directly reducing server infrastructure load and improving the quality of clinical diagnostic display.
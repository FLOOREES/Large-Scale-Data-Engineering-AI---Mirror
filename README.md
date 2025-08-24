# End-to-End Data Platform for Catalan Socio-Economic Analysis

> **Note:** This repository is a public mirror of a private project. It has been made available for portfolio and CV purposes to demonstrate the project's architecture and capabilities.

## 1. Project Overview

This project implements a comprehensive, end-to-end data pipeline that ingests, processes, and analyzes socio-economic and geographical data from Catalonia. The pipeline follows a multi-layered Medallion architecture (Landing, Formatted, Trusted, Exploitation) built with PySpark and Delta Lake.

The final exploitation layer produces two key assets:
1.  A **Knowledge Graph (KG)** that models municipalities, socio-economic indicators, and their relationships.
2.  A consolidated, feature-rich **tabular dataset** for predictive modeling.

These assets are then used in an advanced analysis layer that showcases a variety of powerful techniques, including SPARQL querying, Large Language Model (LLM)-based Retrieval-Augmented Generation (RAG), Knowledge Graph Embeddings, and predictive machine learning.

## 2. Architecture & Pipeline Stages

The project is orchestrated by a central pipeline controller (`src/pipeline.py`) that manages the execution of each stage in sequence.

 #### **Stage 1: Landing Zone**
* **Purpose:** Ingests raw data from external public APIs.
* **Process:** Scripts in `src/landing/` connect to APIs for regional indicators (Idescat), rental market data (Lloguer), and income statistics (RFDBC). Data is downloaded and saved in its raw format (Parquet/JSON) to `data/landing/`.
* **Key Tech:** `requests`, `pandas`, `pyarrow`.

#### **Stage 2: Formatted Zone**
* **Purpose:** Enforces a consistent schema and data types.
* **Process:** PySpark jobs in `src/formatted/` read the raw landing data, cast columns to their correct types (e.g., string, double, timestamp), standardize identifiers (e.g., normalizing municipality codes), and save the results as versioned Delta Lake tables in `data/formatted/`.
* **Key Tech:** `pyspark`, `delta-lake`.

#### **Stage 3: Trusted Zone**
* **Purpose:** Applies data quality rules and validation to create a clean, reliable dataset.
* **Process:** Data is loaded from the Formatted Zone, and a series of validation rules (Denial Constraints) are applied in `src/trusted/`. This includes filtering null keys, checking value ranges (e.g., realistic population counts, valid geographical coordinates), and removing duplicates. The final, trusted data is saved to `data/trusted/`.
* **Key Tech:** `pyspark`, `delta-lake`.

#### **Stage 4: Exploitation Zone**
* **Purpose:** Prepares the final, analysis-ready data assets.
* **Process:** The `src/exploitation/exploitation.py` script consumes data from the Trusted Zone to generate two primary outputs in `data/exploitation/`:
    1.  **Knowledge Graph (`knowledge_graph.ttl`):** A comprehensive RDF graph built with `rdflib`. It integrates demographic, economic, and geospatial data (including neighborhood relationships derived from shapefiles using `geopandas`) into a semantic model.
    2.  **ML Table (`municipal_annual`):** A consolidated Delta table optimized for machine learning, containing annualized data per municipality with features from all sources.
* **Key Tech:** `pyspark`, `rdflib`, `geopandas`, `delta-lake`.

## 3. Core Features: The Analysis Zone

The Analysis Zone (`src/analysis/`) demonstrates the power of the generated Knowledge Graph and ML table through four distinct modules.

### 3.1. SPARQL Querying
The Knowledge Graph can be directly queried using SPARQL, the standard query language for RDF. This allows for powerful, structured retrieval of complex relational data that would be difficult with traditional SQL.

* **Functionality (`kg_query.py`):** Provides a simple pipeline to execute SPARQL queries against the graph and return results as a Pandas DataFrame.
* **Example Use Case:** "Find all municipalities in the Girona province with an average 2021 rent below 700€ that are neighbors of a town with over 20,000 inhabitants."

### 3.2. LLM-Powered GraphRAG (Retrieval-Augmented Generation)
To make the Knowledge Graph accessible to non-technical users, this module leverages a Large Language Model (LLM) to translate natural language questions into SPARQL queries.

* **Functionality (`kg_graphrag.py`):**
    1.  A user asks a question in plain English (e.g., "What is the population of Girona?").
    2.  A robust prompt, containing the authoritative graph schema, guides an LLM (GPT-4o) to generate a precise SPARQL query.
    3.  The generated query is executed against the KG.
    4.  The query results are passed back to the LLM, which synthesizes a natural language answer.
* **Key Tech:** `langchain`, `langchain-openai`, `rdflib`.

### 3.3. Knowledge Graph Embeddings
This module explores the latent relationships within the KG by training and evaluating various Knowledge Graph Embedding (KGE) models. These models learn dense vector representations (embeddings) for each entity (e.g., municipality), capturing their semantic and relational properties.

* **Functionality (`kg_embeddings.py`):**
    * Uses the **PyKEEN** library to systematically train and evaluate a suite of KGE models (TransE, TransH, RotatE, ComplEx, RGCN, etc.) across different embedding dimensions (64, 128, 256).
    * Implements early stopping for efficient training.
    * Automatically generates comparison reports, including performance heatmaps and faceted bar plots of key metrics like Mean Reciprocal Rank (MRR) and Hits@k, to identify the best-performing model.
* **Key Tech:** `pykeen`, `pytorch`, `matplotlib`, `seaborn`.

### 3.4. Predictive Modeling with KG Features
This module demonstrates the practical value of KGEs by using them as features to enhance a traditional machine learning task: predicting the average monthly rent of a municipality.

* **Functionality (`kg_prediction.py`):**
    1.  Loads the consolidated ML table and the pre-trained embeddings of the best KGE model (identified in the previous step).
    2.  Enriches the tabular data by merging the embeddings as new features for each municipality.
    3.  Performs time-series feature engineering (e.g., creating lagged variables).
    4.  Trains a **LightGBM** regression model to predict `avg_monthly_rent` using a time-based train-validation-test split.
    5.  Evaluates the model and visualizes results, including feature importance, which highlights the predictive power of both the tabular and the KG embedding features.
* **Key Tech:** `lightgbm`, `scikit-learn`, `pyspark`.

## 4. Tech Stack

* **Data Processing & Storage:** PySpark, Delta Lake, Pandas, GeoPandas
* **Knowledge Graph:** RDFLib, PyKEEN, OWL-RL
* **Machine Learning:** PyTorch, Scikit-learn, LightGBM
* **LLM & RAG:** LangChain, OpenAI
* **APIs & Web:** Requests, Python-dotenv
* **Visualization:** Matplotlib, Seaborn

## 5. Setup and Execution

### Prerequisites
* Git
* Python 3.9+
* For Windows users, it is **highly recommended** to use **WSL (Windows Subsystem for Linux)**, as some dependencies may have compatibility issues with native Windows.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FLOOREES/Large-Scale-Data-Engineering-AI---Mirror.git
    cd Large-Scale-Data-Engineering-AI---Mirror
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Linux / macOS / WSL
    python3 -m venv venv
    source venv/bin/activate

    # For Windows (PowerShell)
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install the required dependencies:**
    The project uses a `requirements.txt` file which includes a special index URL for PyTorch with CUDA support.
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
The GraphRAG component (`kg_graphrag.py`) requires an OpenAI API key.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API key to this file:
    ```
    OPENAI_API_KEY='your-sk-api-key-here'
    ```

### Running the Full Pipeline
The entire pipeline can be executed from a single entry point. The script is configured to run all stages sequentially, from data ingestion to final analysis.

```bash
python src/run.py
```

You can customize which stages or analysis parts to run by modifying the configuration variables at the bottom of the `src/run.py` file.

## 6. Dependencies

This project relies on the following major Python libraries, as listed in `requirements.txt`:

* **Core & Data Manipulation:** `numpy`, `pandas`, `pyarrow`
* **Big Data:** `pyspark`
* **Machine Learning:** `torch`, `scikit-learn`, `lightgbm`
* **Geospatial:** `geopandas`
* **Knowledge Graph & Semantic Web:** `rdflib`, `owlrl`, `pykeen`
* **LLM Integration:** `langchain-openai`
* **Visualization:** `matplotlib`, `seaborn`
* **Utilities:** `requests`, `tqdm`, `python-dotenv`
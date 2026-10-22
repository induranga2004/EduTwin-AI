# Complete Research Flow & Model/Dataset Architecture: EduTwin C1

This document provides a complete visual diagram and reference map of the research methodology for **Component C1**, detailing every input, model, dataset, and output across both the **Offline Research/Evaluation Phase** and the **Online Runtime Phase**.

---

## 1. Master Research Architecture Diagram

The diagram below maps all datasets, pre-trained models, and trained classifiers across the entire pipeline:

```mermaid
flowchart TD
    subgraph DataInputs["DATA INPUTS & DATASETS"]
        D1["Lecture Document<br/>(PDF or PPTX slides)"]
        D2["Live Lecturer Audio<br/>(Microphone stream)"]
        D3[("Synthetic / Annotated Dataset<br/>coverage_examples.csv")]
        D4[("Ground-Truth DAG Benchmarks<br/>curriculum_graphs.json")]
    end

    subgraph Phase1["PHASE 1: PRE-LECTURE (Planned Curriculum Graph)"]
        D1 --> M1["Text & Layout Extractor<br/>(pypdf / python-pptx)"]
        M1 --> M2["Concept Keyphrase Extractor<br/>(Rule / Keyphrase Heuristic)"]
        M2 --> M3["Prerequisite & Slide Flow Builder<br/>(NetworkX / Directed DAG)"]
        M3 --> G_PLAN[("Graph 1: G_planned<br/>Nodes: Concepts + Slide Numbers<br/>Edges: Prerequisite Dependencies")]
    end

    subgraph Phase2["PHASE 2: LIVE RUNTIME (Speech to Delivered Graph)"]
        D2 --> M4["Model A: ASR Speech-to-Text<br/>(OpenAI Whisper / faster-whisper)"]
        M4 --> UTT["Timestamped Utterances<br/>[t_start, t_end, text]"]
        UTT --> M5["Model B: Semantic Embedder<br/>(all-MiniLM-L6-v2)"]
        G_PLAN -.-> M5
        M5 --> SIM["Cosine Similarities vs Concepts<br/>Matrix: [1 x N_concepts]"]
        SIM --> M6["Model C: Coverage Confidence Model<br/>(Logistic Regression / Bayesian)"]
        M6 --> G_DELIV[("Graph 2: G_delivered(t)<br/>Nodes: Concepts with Graded Confidence [0-1]<br/>Edges: Spoken Discourse Transitions")]
    end

    subgraph Phase3["PHASE 3: REAL-TIME INTELLIGENCE & INTERVENTIONS"]
        G_PLAN --> COMP["Graph Comparison Engine<br/>Delta = G_planned \ G_delivered(t)"]
        G_DELIV --> COMP
        COMP --> CHECK1{"Prerequisite Skipped<br/>or Inadequately Covered?"}
        CHECK1 -->|Yes| ALERT["Lecturer In-Session Notification<br/>'Skipped Base Case on Slide 4!'"]
        CHECK1 -->|No| PROGRESS["Update Live Coverage Dashboard"]

        G_DELIV --> CHECK2{"Student Joins Late?<br/>(timestamp = t_join)"}
        CHECK2 -->|Yes| TOPO["Topological Sorting Engine<br/>(Sort missed subgraph 0 -> t_join)"]
        TOPO --> M7["Summary & Quiz Generator<br/>(Micro-Lesson Synthesizer)"]
        M7 --> CATCHUP["5-Minute Student Catch-Up Pack<br/>- Prerequisite takeaways<br/>- 30s audio replay anchors<br/>- 2-3 question interactive quiz"]
    end

    subgraph Phase4["PHASE 4: RESEARCH EVALUATION & TRAINING (Academic Validity)"]
        D3 --> TRAIN["Model C Training Split (75%)<br/>(scikit-learn Pipeline)"]
        TRAIN --> M6
        D3 --> TEST["Model C Held-Out Test Split (25%)<br/>(Never seen by model during training)"]
        D4 --> TEST
        TEST --> EVAL["Evaluation Metrics Report:<br/>1. Coverage F1 Score<br/>2. Skip Precision & Recall<br/>3. False Alert Rate (FAR / hr)<br/>4. Matching Latency (ms)"]
    end
```

---

## 2. Models Used in Component C1

Here is the exact inventory of every machine learning model in your component:

| Model ID | Model Name & Architecture | Type | Input | Output | Need Training? |
|---|---|---|---|---|---|
| **Model A** | **`faster-whisper`** (OpenAI Whisper `base` or `small`) | Pre-trained Deep ASR (Encoder-Decoder Transformer) | Raw 16kHz audio waveform | Timestamped utterance text: `[start, end, text]` | **No** (Pre-trained on 680,000 hrs of audio) |
| **Model B** | **`all-MiniLM-L6-v2`** (Sentence-Transformers) | Pre-trained Dense Bi-Encoder (MiniLM BERT) | Spoken utterance + Concept definitions | 384-dimensional dense vectors & Cosine similarity scores | **No** (Pre-trained on 1B+ sentence pairs) |
| **Model C** | **`Coverage Confidence Classifier`** (TF-IDF + Features + Logistic Regression) | Supervised Feature Classifier / Bayesian Accumulator | `[similarity, duration, prereq_ratio, mention_count, text]` | Graded coverage label (`not_yet_observed`, `introduced`, `adequately_covered`) & confidence score $[0.0, 1.0]$ | **YES** (Trained locally in seconds on your dataset) |
| **Algorithm D** | **`Topological Graph Sorter`** | Graph Algorithm (Kahn's / DFS on DAG) | Missed subgraph from $G_{\text{delivered}}(t_{\text{join}})$ | Prerequisite-ordered linear study sequence | **No** (Deterministic discrete graph algorithm) |

---

## 3. Datasets Used in Component C1

There are four distinct datasets/data artifacts across the lifecycle:

```mermaid
flowchart LR
    subgraph D_Inputs["1. Raw Real-World Inputs"]
        PDF["Course Lecture Slides<br/>(PDF / PPTX)"]
        AUDIO["Audio Streams<br/>(Live / Recorded WAV/MP3)"]
    end

    subgraph D_Synthetic["2. Synthetic Training & Evaluation Dataset"]
        CSV["coverage_examples.csv<br/>- lecture_id<br/>- segment_id<br/>- timestamp_seconds<br/>- concept<br/>- transcript_text<br/>- semantic_similarity<br/>- prerequisite_coverage<br/>- previous_evidence_count<br/>- ground_truth_label"]
        DAG_GT["curriculum_dag_ground_truth.json<br/>- Verified concept nodes<br/>- Verified prerequisite edges (A -> B)"]
    end

    subgraph D_Usage["3. Where They Are Used"]
        CSV -->|Trains| M_C["Model C (Coverage Classifier)"]
        CSV & DAG_GT -->|Evaluates| BENCH["Research Benchmark Report (F1, Skip P/R, Latency)"]
        PDF & AUDIO -->|Drives| RUNTIME["Live BigBlueButton Meeting Execution"]
    end
```

### Dataset Specifications:

#### 1. Input Lecture Documents (PDF / PPTX)
- **Role:** Uploaded before class by the lecturer.
- **Contents:** Slide titles, bullet points, code snippets, visual explanations.
- **Used by:** Phase 1 to construct $G_{\text{planned}}$.

#### 2. Live Audio Stream (Speech Utterances)
- **Role:** Captured live during the BigBlueButton meeting.
- **Contents:** Spoken audio stream segmented by Voice Activity Detection (VAD).
- **Used by:** Model A (Whisper) to produce speech segments.

#### 3. `coverage_examples.csv` (Annotated Utterance Dataset)
- **Role:** The primary ML research dataset.
- **Contents:** Thousands of utterance segments annotated with human (or synthetic) ground truth labels:
  - Features: `semantic_similarity`, `prerequisite_coverage`, `previous_evidence_count`, `timestamp_seconds`.
  - Target Label: `not_yet_observed`, `introduced`, `adequately_covered`, `insufficiently_covered`.
- **Used by:** Model C training split and test evaluation split.

#### 4. `curriculum_dag_ground_truth.json` (Curriculum Benchmark)
- **Role:** Gold-standard prerequisite graphs for 3–5 sample university courses (e.g. Data Structures, Calculus, Physics).
- **Contents:** Exact list of concepts, their slide numbers, and verified prerequisite edges ($A \to B$).
- **Used by:** Graph Comparison Engine and Skip-Detection Precision/Recall evaluation.

---

## 5. End-to-End User Experience Summary

1. **Before Meeting:**
   The lecturer uploads `Lecture04_Trees.pdf`.
   The system instantly builds **Graph 1 ($G_{\text{planned}}$)**.
2. **During Meeting (Lecturer View):**
   The lecturer teaches. The system transcribes speech with **Whisper**, maps meaning with **MiniLM**, and updates **Graph 2 ($G_{\text{delivered}}$)** with confidence scores.
   If the lecturer jumps to slide 10 without teaching the concept on slide 4, the **Graph Comparator** triggers a notification: *"Warning: You skipped 'Binary Tree Traversals' before introducing 'BST Balancing'."*
3. **During Meeting (Late-Joining Student View):**
   A student joins at minute 25.
   The **Topological Sorter** extracts what was taught in minutes 0–25, formats a **5-minute micro-lesson**, links to the exact **30-second audio replay timestamps**, and gives a **2-question quiz**. Once completed, the student is seamlessly synced into the live class!

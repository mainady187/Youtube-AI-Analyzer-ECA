# 🇪🇬 YouTube Data Retriever & NLP Pipeline for Egyptian Dialect

An end-to-end Natural Language Processing (NLP) pipeline designed to extract, transcribe, and analyze YouTube videos spoken explicitly in **Egyptian Colloquial Arabic (ECA)**. Traditional Arabic NLP models often struggle with regional dialects; this project bridges that gap by implementing custom preprocessing, specialized lexicons, and machine learning/deep learning benchmarks tailored for Egyptian slang and daily expressions.

 **Primary Data Source:** The core dataset for this pipeline was acquired and curated from **"El-Daheh" (الدحيح)** YouTube episodes. This source was strategically chosen as it represents one of the most challenging benchmarks for Egyptian dialect NLP—featuring rapid-fire colloquial speech, mixed scientific jargon, and culturally specific idioms.
---

## System Architecture & Interface
*The dashboard features a high-contrast, cyberpunk-themed UI engineered with Streamlit.*
---

## Tech Stack & Architecture
* **Data Extraction:** `yt-dlp` (Automated extraction of YouTube video streams, audio, and metadata).
* **Speech-to-Text:** `SpeechRecognition` / Whisper (Converts colloquial Egyptian audio into structured text).
* **Text Preprocessing:** `Pandas` & `Re` (Advanced Regular Expressions designed for dialect text normalization and linguistic noise filtering).
* **Feature Extraction:** `Scikit-Learn` (`TfidfVectorizer`) & Custom `Word2Vec` word embeddings.
* **UI Dashboard:** `Streamlit` (Custom dark-mode / neon aesthetic).

---

## Specialized Dialect Preprocessing Pipeline

To handle the fluid nature of **Egyptian Colloquial Arabic (ECA)**, traditional root-based stemming (like `ISRIStemmer`) was intentionally avoided. Stemming often distorts crucial dialectal sentiment indicators (e.g., words like "تحفة", "جامد", "خربانة"). Instead, a rigorous **Normalization and Tokenization** pipeline was developed:

1. **HTML & Tag Stripping:** Cleans any underlying metadata noise.
2. **Character Normalization:** Standardizes variations of Arabic characters (e.g., mapping `إ, أ, آ` to `ا`, `ة` to `ه`, and `ى` to `ي`) to reduce data sparsity.
3. **Regex Alphabet Filtering:** Strips non-Arabic scripts, emojis, and punctuation while retaining pure Arabic dialect words.
4. **Colloquial Stopword Elimination:** Filters out both standard Arabic stopwords and common Egyptian filler words (e.g., "ده", "بقى", "اللي", "عشان", "يعني").

## Model Evaluation & Benchmarking

| Model Architecture | Vectorizer / Embedding | Accuracy | Technical Insights |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **TF-IDF** | **83.33%** | **Top Performer:** Highly efficient at mapping distinct dialectal emotional keywords weights. |
| **BiLSTM (Deep Learning)** | **Word2Vec (Custom)** | **83.33%** | **Strong Sequential Logic:** Excellent at capturing context and negation flips in sentences. |
| **Random Forest** | **Word2Vec (Custom)** | **50.00%** | **Underperformed:** Averaging word vectors (`np.mean`) flattens sequential patterns needed for tree splits. |

### Preprocessing & Embedding Implementation:
```python
# Tokenization strategy
tokenized_data = [str(text).split() for text in df['clean_version'].fillna('')]

# Custom localized Word2Vec configuration
w2v_model = Word2Vec(sentences=tokenized_data, vector_size=20, window=5, min_count=1, workers=4)

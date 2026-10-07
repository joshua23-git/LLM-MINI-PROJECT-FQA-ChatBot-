# NLP & Transformer Projects

A collection of Natural Language Processing (NLP) projects built while learning and working with **Transformers, BERT, XLNet, Hugging Face, PyTorch, and LLM-based applications**.

This repository currently contains two major projects:

1. **FAQ Chatbot using BERT**
2. **Emotion Classification using XLNet**

The projects demonstrate how transformer models can be used for different real-world NLP tasks such as question answering, FAQ retrieval, and emotion classification.

---

# Projects

## 1. FAQ Chatbot using BERT

A question-answering chatbot designed to answer frequently asked questions using a **BERT-based NLP model**.

The chatbot takes a user's question as input and identifies the most relevant FAQ answer from the available knowledge base.

### Example

**User:**

```text
What are your working hours?
```

**Chatbot:**

```text
Our working hours are 9:00 AM to 5:00 PM, Monday to Friday.
```

### Main Features

* FAQ-based question answering
* BERT transformer model
* Natural language understanding
* Question matching
* Answer retrieval
* Confidence-based responses
* Handles different ways of asking the same question

### Technologies

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* NLP
* Pandas
* Scikit-learn

### Basic Workflow

```text
User Question
      ↓
Text Preprocessing
      ↓
BERT Tokenization
      ↓
BERT Model
      ↓
Question Representation
      ↓
FAQ Matching
      ↓
Best Matching Answer
      ↓
Chatbot Response
```

### Example Questions

The chatbot can be trained or configured to handle questions such as:

```text
What are your working hours?

When do you open?

When does the company close?

How can I contact support?

Where are you located?

How can I reset my password?
```

Different questions can have the same intended meaning, allowing the chatbot to provide the appropriate FAQ answer.

---

# 2. XLNet Emotion Classification

An NLP text-classification project that uses **XLNet** to classify text into four emotions.

### Supported Emotions

| Label | Emotion |
| ----- | ------- |
| 0     | Anger   |
| 1     | Fear    |
| 2     | Joy     |
| 3     | Sadness |

### Example

**Input:**

```text
I am extremely happy today!
```

**Expected prediction:**

```text
Joy
```

---

# Technologies Used

The repository uses several modern NLP and machine-learning technologies:

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* BERT
* XLNet
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* Conda

---

# Project Architecture

The repository contains two different NLP applications built around transformer models.

```text
                    NLP Projects
                         |
             +-----------+-----------+
             |                       |
        FAQ Chatbot            Emotion Classifier
             |                       |
            BERT                    XLNet
             |                       |
      Question Matching        Text Classification
             |                       |
       FAQ Answer              Emotion Prediction
```

---

# FAQ Chatbot

## What is BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** is a transformer-based language model developed for understanding the context of words in text.

BERT processes text bidirectionally, allowing it to use information from both the left and right context of a word.

For example:

```text
I went to the bank to deposit money.
```

and:

```text
I sat near the river bank.
```

The meaning of "bank" changes based on the surrounding context.

BERT uses contextual information to understand this difference.

## FAQ Chatbot Workflow

The FAQ chatbot follows these general steps:

### 1. User Input

The user enters a question.

```text
How do I reset my password?
```

### 2. Tokenization

The question is converted into tokens using a BERT tokenizer.

### 3. BERT Processing

The tokenized input is passed through the BERT model to obtain contextual representations.

### 4. Question Matching

The user's question is compared with the available FAQ questions.

### 5. Best Match

The system identifies the most relevant FAQ.

### 6. Response

The corresponding FAQ answer is returned to the user.

---

# XLNet Emotion Classification

## Dataset

The emotion classifier uses three datasets:

```text
emotions_data/
├── emotion-labels-train.csv
├── emotion-labels-test.csv
└── emotion-labels-val.csv
```

The datasets contain text samples and corresponding emotion labels.

The datasets are combined and processed before training.

## Data Preprocessing

The preprocessing pipeline includes:

* Removing emojis
* Removing user mentions
* Cleaning text
* Balancing emotion classes
* Converting labels into numerical values
* Splitting the dataset

Example:

```python
data['text_clean'] = data['text'].apply(
    lambda x: clean(x, no_emoji=True)
)

data['text_clean'] = data['text_clean'].apply(
    lambda x: re.sub('@[^\s]+', '', x)
)
```

## Dataset Split

The processed data is divided into:

| Dataset    | Samples |
| ---------- | ------: |
| Training   |    4414 |
| Validation |     491 |
| Testing    |    1227 |

## XLNet Tokenization

The project uses:

```python
XLNetTokenizer.from_pretrained("xlnet-base-cased")
```

The text is tokenized with a maximum sequence length of 128 tokens.

```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=128,
        truncation=True
    )
```

## Model

The classification model is:

```python
XLNetForSequenceClassification
```

configured for four classes.

```python
model = XLNetForSequenceClassification.from_pretrained(
    "xlnet-base-cased",
    num_labels=4,
    id2label={
        0: "anger",
        1: "fear",
        2: "joy",
        3: "sadness"
    }
)
```

## Training

The model is fine-tuned using the Hugging Face `Trainer` API.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

## Evaluation

Accuracy is used as the initial evaluation metric.

```python
metric = evaluate.load("accuracy")
```

The model's predicted class is obtained using:

```python
np.argmax(logits, axis=-1)
```

---

# Repository Structure

A recommended structure for the repository is:

```text
NLP-Transformer-Projects/
│
├── FAQ_Chatbot/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   └── README.md
│
├── XLNet_Emotion_Classification/
│   ├── emotions_data/
│   │   ├── emotion-labels-train.csv
│   │   ├── emotion-labels-test.csv
│   │   └── emotion-labels-val.csv
│   │
│   ├── XLNet_Emotion_Classification.ipynb
│   └── README.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Environment Setup

Create the Conda environment:

```bash
conda create -n llms_course_env python=3.11
```

Activate it:

```bash
conda activate llms_course_env
```

Install the required libraries:

```bash
pip install transformers datasets evaluate pandas numpy scikit-learn clean-text
```

Install PyTorch according to your system requirements.

---

# Running the Projects

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Enter the project directory:

```bash
cd NLP-Transformer-Projects
```

Activate the environment:

```bash
conda activate llms_course_env
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the corresponding notebook for the project you want to run.

---

# Learning Objectives

Through these projects, the repository demonstrates practical experience with:

* Natural Language Processing
* Transformer architectures
* BERT
* XLNet
* Transfer learning
* Fine-tuning
* Text preprocessing
* Tokenization
* Question answering
* FAQ chatbots
* Text classification
* Emotion detection
* Hugging Face Transformers
* Hugging Face Datasets
* PyTorch
* Model evaluation

---

# Future Improvements

## FAQ Chatbot

* Add more FAQ categories
* Improve semantic question matching
* Add conversation history
* Add fallback responses
* Add confidence thresholds
* Build a web interface
* Create a REST API
* Add an LLM-based response generation layer
* Deploy the chatbot

## Emotion Classifier

* Train using the complete dataset
* Improve model accuracy
* Add precision, recall and F1-score
* Generate a confusion matrix
* Perform hyperparameter tuning
* Compare BERT, RoBERTa, DistilBERT and XLNet
* Save and load the fine-tuned model
* Build a real-time prediction application

---

# Key Concepts Learned

### BERT

Used for understanding contextual relationships between words and building the FAQ chatbot.

### XLNet

Used for transformer-based emotion classification.

### Tokenization

Converts natural-language text into tokens that transformer models can process.

### Fine-Tuning

Adapts a pretrained transformer model to a specific NLP task.

### Transfer Learning

Uses knowledge learned from large-scale pretraining and adapts it to a smaller task-specific dataset.

---

# Author

**Joshua**

B.E. Computer Science & Engineering

Interested in:

* Artificial Intelligence
* Machine Learning
* Natural Language Processing
* Large Language Models
* Generative AI
* AI Engineering

---

# License

This repository is intended primarily for educational and learning purposes.

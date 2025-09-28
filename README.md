### **GitHub Repository Description**

Sentiment analysis on Amazon Kindle reviews comparing three NLP feature extraction methods: Bag of Words, TF-IDF, and Word2Vec using Scikit-learn and Gensim.

-----

### **README.md File**

# Kindle Review Sentiment Analysis with Scikit-learn & Gensim

This repository contains a Jupyter notebook that performs sentiment analysis on Amazon Kindle reviews. The primary goal is to classify reviews as positive or negative and to compare the performance of three different NLP feature extraction techniques: **Bag of Words (BoW)**, **TF-IDF**, and **Word2Vec**.

The entire pipeline—from data cleaning and preprocessing to model training and evaluation—is implemented to provide a clear comparison of these methods.

-----

## Core Concepts Demonstrated

  * **Data Cleaning & Preprocessing**:

      * Handling missing values and exploring data distribution.
      * Converting multi-class ratings (1-5 stars) into binary sentiment labels (0 for negative, 1 for positive).
      * Text normalization: lowercasing, removing special characters, URLs, and HTML tags.
      * Stopword removal using `NLTK`.
      * **Lemmatization** with `NLTK`'s `WordNetLemmatizer` to reduce words to their base form.

  * **Feature Extraction Techniques**:

      * **Bag of Words**: Using `sklearn.feature_extraction.text.CountVectorizer`.
      * **TF-IDF**: Using `sklearn.feature_extraction.text.TfidfVectorizer`.
      * **Word2Vec**: Training a custom word embedding model with `gensim` and creating document vectors by averaging word vectors.

  * **Machine Learning Models**:

      * **Multinomial Naive Bayes**: Applied to the BoW and TF-IDF features.
      * **Random Forest Classifier**: Applied to the Word2Vec embedding features.

  * **Model Evaluation**:

      * Splitting the data into training and testing sets.
      * Evaluating models using **Accuracy**, **Classification Report** (Precision, Recall, F1-Score), and **Confusion Matrix**.

-----

## Getting Started

Follow these instructions to set up your environment and run the project notebook.

### Prerequisites

  * Python 3.x
  * Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install the required libraries:**

    ```bash
    pip install pandas nltk scikit-learn gensim beautifulsoup4 lxml
    ```

3.  **Download NLTK Data:**
    Run the following in a Python interpreter or Jupyter cell to download the necessary NLTK models for stopword removal and lemmatization.

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

-----

## Project Workflow & Methodology

The notebook follows a structured approach to compare the different feature extraction techniques.

### 1\. Data Loading and Preprocessing

The `all_kindle_review.csv` dataset is loaded, and only the `reviewText` and `rating` columns are selected. The 5-star ratings are converted into a binary sentiment:

  * **Negative (0)**: Ratings 1 and 2.
  * **Positive (1)**: Ratings 3, 4, and 5.

The review text then undergoes a rigorous cleaning pipeline to prepare it for feature extraction.

### 2\. Feature Extraction & Modeling

After cleaning, the data is split into training and testing sets. Three different models are then trained and evaluated in parallel:

#### Method 1: Bag of Words (BoW) + Multinomial Naive Bayes

  * **Feature Extraction**: The preprocessed text is converted into a matrix of token counts using `CountVectorizer`. This method represents each review by the frequency of the words it contains.
  * **Model**: A `MultinomialNB` classifier is trained on the resulting BoW feature matrix.

#### Method 2: TF-IDF + Multinomial Naive Bayes

  * **Feature Extraction**: The text is converted into a matrix using `TfidfVectorizer`. TF-IDF gives higher weight to words that are frequent in a document but rare across the entire corpus, capturing importance better than simple counts.
  * **Model**: A `MultinomialNB` classifier is trained on the TF-IDF features.

#### Method 3: Average Word2Vec + Random Forest

  * **Feature Extraction**: A `Word2Vec` model is trained from scratch on the review corpus to learn semantic embeddings for words. Each review is then converted into a single numerical vector by **averaging the vectors** of all the words within it.
  * **Model**: A `RandomForestClassifier` is trained on the averaged Word2Vec feature vectors, as these dense vectors work well with tree-based models.

-----

## Results and Conclusion

The performance of each model was evaluated on the same test set. The results are summarized below, comparing both accuracy and F1-score.

| Method | Accuracy | F1-Score (Positive Class) |
| :--- | :---: | :---: |
| **Bag of Words** | **83.25%** | **0.87** |
| Average Word2Vec | 76.67% | 0.84 |
| TF-IDF | 67.08% | 0.80 |

### Conclusion

For this specific dataset and sentiment analysis task, the **Bag of Words (BoW) model performed the best**, achieving the highest accuracy and F1-score. This suggests that for this binary classification problem, simple word frequency was a very effective feature.

While the Word2Vec approach did not achieve the highest accuracy, it demonstrates a powerful technique for capturing semantic meaning, which can be superior in more complex NLP tasks where context and word relationships are critical. The TF-IDF model, surprisingly, performed the worst, indicating that its term-weighting scheme was less effective for this dataset than the other two methods.

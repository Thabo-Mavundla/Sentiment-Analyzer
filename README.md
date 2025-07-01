Here’s a complete and well-structured `README.md` for your Sentiment Analysis Web App:

---

# 🧠 AI Sentiment Analysis Web App

This project is a **Gradio-based web application** that performs **AI-powered sentiment analysis** using the OpenRouter API. The app provides:

* Manual sentiment analysis for individual text inputs.
* Batch sentiment analysis from CSV files with accuracy evaluation.
* Comparative sentiment analysis between two texts.
* Downloadable results in CSV, Excel, and TXT formats.
* Visual performance metrics including a sentiment distribution chart and confusion matrix.

---

## 🚀 Features

* **Manual Sentiment Analysis:** Enter single texts and get real-time sentiment predictions with explanations.
* **Batch File Processing:** Upload CSV files containing texts and their ground truth labels to evaluate model performance.
* **Accuracy Metrics:** Calculates accuracy, precision, recall, F1 score, and generates a confusion matrix and classification report.
* **Comparative Analysis:** Compare the sentiment of two texts side by side with a visualization.
* **Export Options:** Download the sentiment analysis results in CSV, Excel, or TXT formats.
* **Concurrent Requests:** Utilizes multithreading for faster batch processing.
* **Visual Reports:** Interactive charts for sentiment distribution and confusion matrices using Plotly.

---

## 📂 File Format Requirement

For batch processing, the uploaded CSV file **must include** the following columns:

* `Text` (The text to analyze)
* `GroundTruth` (The actual sentiment label: Positive, Neutral, or Negative)

Example:

```csv
Text,GroundTruth
"I love this product!",Positive
"This is okay.",Neutral
"I dislike this experience.",Negative
```

---

## 🔧 Setup Instructions

### Prerequisites

* Python 3.8+
* Required Python libraries:

```bash
pip install gradio pandas plotly scikit-learn requests
```

### Running the App

```bash
python app.py
```

The app will automatically launch in your browser.

---

## 🔑 API Configuration

The app uses the **OpenRouter API** for sentiment analysis. You can set your API key as an environment variable:

```bash
export API_KEY=your_openrouter_api_key
```

Or directly update the `API_KEY` variable in the script:

```python
API_KEY = "your_openrouter_api_key"
```

---

## 🖥️ Application Structure

* **Manual Input Tab:** Analyze individual text entries.
* **File Upload Tab:** Upload and process CSV files, view accuracy, charts, and download results.
* **Comparative Analysis Tab:** Compare sentiments of two texts side by side.

---

## 📈 Performance Settings

* `MAX_WORKERS = 2` (Number of concurrent API calls)
* `REQUEST_DELAY = 10` seconds (Delay after every batch of `MAX_WORKERS` requests to prevent rate limiting)

You can fine-tune these values based on your API rate limits.

---

## 📤 Export Options

The following file formats are supported for exporting results:

* CSV
* Excel
* TXT

---

## 💡 Notes

* The app currently supports the model: `mistralai/mistral-small-3.2-24b-instruct:free` via OpenRouter.
* Only **CSV files** are supported for batch processing.
* Ensure your API key is valid and you have an active internet connection.

---

## 📜 License

This project is for educational and non-commercial use. Please review the terms of the OpenRouter API for any commercial restrictions.

---

If you would like, I can help you customize this further for GitHub with badges, example screenshots, or deployment instructions for Hugging Face Spaces. Let me know!

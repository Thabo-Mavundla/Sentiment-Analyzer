import gradio as gr
import pandas as pd
import plotly.express as px
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import plotly.figure_factory as ff

# 🔑 API Key
API_KEY = os.getenv("API_KEY") or "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_WORKERS = 2
REQUEST_DELAY = 10
result_df_global = pd.DataFrame()

# 📊 Sentiment Analysis Function
def analyze_sentiment(text, retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost:7860",
        "X-Title": "SentimentAnalyserApp"
    }

    prompt = (
        "Please classify the sentiment of the following text strictly as 'Positive', 'Neutral', or 'Negative'. "
        "Provide your answer exactly in this format:\n"
        "Sentiment: [Positive/Neutral/Negative]\nExplanation: [Your brief explanation]\n\n"
        f"Text: {text}"
    )

    payload = {
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 429:
                time.sleep(10)
                continue
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']

            if "Sentiment:" in content:
                sentiment = content.split("Sentiment:")[1].split("\n")[0].strip()
            else:
                sentiment = "Unknown"

            if "Explanation:" in content:
                explanation = content.split("Explanation:")[1].strip()
            else:
                explanation = "No explanation provided."

            return sentiment, explanation

        except Exception as e:
            if attempt == retries - 1:
                return "Error", f"API error: {str(e)}"
            time.sleep(5)

    return "Error", "Maximum retries exceeded."

# 🔍 Manual Text Processing
def process_single_text(text):
    sentiment, explanation = analyze_sentiment(text)
    return f"Sentiment: {sentiment}", f"Explanation: {explanation}"

# 📂 File Processing with Accuracy Calculation
def process_file(file):
    global result_df_global

    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            return pd.DataFrame({'Error': ['Only CSV files are supported for accuracy analysis']}), None, None, None

        if 'Text' not in df.columns or 'GroundTruth' not in df.columns:
            return pd.DataFrame({'Error': ['CSV must contain "Text" and "GroundTruth" columns']}), None, None, None

        texts = df['Text'].tolist()
        ground_truths = df['GroundTruth'].tolist()

        results = []
        predictions = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_text = {executor.submit(analyze_sentiment, text): text for text in texts}

            for i, future in enumerate(as_completed(future_to_text)):
                sentiment, explanation = future.result()
                predictions.append(sentiment)
                results.append({
                    'Text': future_to_text[future],
                    'AI_Prediction': sentiment,
                    'Explanation': explanation
                })

                if (i + 1) % MAX_WORKERS == 0:
                    time.sleep(REQUEST_DELAY)

        result_df = pd.DataFrame(results)
        result_df['GroundTruth'] = ground_truths

        global result_df_global
        result_df_global = result_df.copy()

        # Calculate performance metrics
        accuracy = accuracy_score(ground_truths, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(ground_truths, predictions, labels=['Positive', 'Neutral', 'Negative'])
        report = classification_report(ground_truths, predictions)

        sentiment_counts = result_df['AI_Prediction'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Distribution')

        # Confusion Matrix Plot
        conf_fig = ff.create_annotated_heatmap(z=conf_matrix, x=['Positive', 'Neutral', 'Negative'], y=['Positive', 'Neutral', 'Negative'],
                                               colorscale='Blues', showscale=True)
        conf_fig.update_layout(title_text='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')

        # Build summary
        summary = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}\n\nClassification Report:\n{report}"

        return result_df, fig, conf_fig, summary

    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]}), None, None, None

# 📥 Export Functions
def export_csv():
    global result_df_global
    file_path = "sentiment_results.csv"
    result_df_global.to_csv(file_path, index=False)
    return file_path

def export_excel():
    global result_df_global
    file_path = "sentiment_results.xlsx"
    result_df_global.to_excel(file_path, index=False)
    return file_path

def export_txt():
    global result_df_global
    file_path = "sentiment_results.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        for index, row in result_df_global.iterrows():
            f.write(f"Text: {row['Text']}\nPredicted: {row['AI_Prediction']}\nGround Truth: {row['GroundTruth']}\nExplanation: {row['Explanation']}\n\n")
    return file_path

# 🔄 Comparative Analysis
def compare_texts(text1, text2):
    sentiment1, explanation1 = analyze_sentiment(text1)
    sentiment2, explanation2 = analyze_sentiment(text2)

    comparison_data = pd.DataFrame({
        'Text': ['Text 1', 'Text 2'],
        'Sentiment': [sentiment1, sentiment2]
    })

    fig = px.bar(
        comparison_data,
        x='Text',
        color='Sentiment',
        title='Comparative Sentiment Analysis',
        barmode='group',
        text='Sentiment',
        color_discrete_map={
            'Positive': 'green',
            'Neutral': 'gray',
            'Negative': 'red'
        }
    )

    return f"Text 1 Sentiment: {sentiment1}\nExplanation: {explanation1}", \
           f"Text 2 Sentiment: {sentiment2}\nExplanation: {explanation2}", fig

# 🌐 Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI Sentiment Analysis Web App with Accuracy Evaluation")

    with gr.Tab("🔹 Manual Input"):
        text_input = gr.Textbox(label="Enter Text", placeholder="Type or paste text here...")
        analyze_button = gr.Button("Analyze Sentiment")
        sentiment_output = gr.Textbox(label="Sentiment")
        explanation_output = gr.Textbox(label="Explanation")

        analyze_button.click(process_single_text, inputs=text_input, outputs=[sentiment_output, explanation_output])

    with gr.Tab("📂 File Upload & Accuracy Report"):
        file_input = gr.File(label="Upload CSV file with 'Text' and 'GroundTruth' columns")
        process_button = gr.Button("Process File")
        file_output = gr.Dataframe(label="Sentiment Results")
        chart_output = gr.Plot(label="Sentiment Distribution Chart")
        conf_matrix_output = gr.Plot(label="Confusion Matrix")
        summary_output = gr.Textbox(label="Performance Summary", lines=10)

        process_button.click(process_file, inputs=file_input, outputs=[file_output, chart_output, conf_matrix_output, summary_output])

        gr.Markdown("### 📥 Export Results")
        export_csv_btn = gr.Button("Download as CSV")
        export_excel_btn = gr.Button("Download as Excel")
        export_txt_btn = gr.Button("Download as TXT")

        csv_file = gr.File(label="CSV File")
        excel_file = gr.File(label="Excel File")
        txt_file = gr.File(label="Text File")

        export_csv_btn.click(export_csv, outputs=csv_file)
        export_excel_btn.click(export_excel, outputs=excel_file)
        export_txt_btn.click(export_txt, outputs=txt_file)

    with gr.Tab("⚖️ Comparative Analysis"):
        text1_input = gr.Textbox(label="Enter First Text")
        text2_input = gr.Textbox(label="Enter Second Text")
        compare_button = gr.Button("Compare Sentiments")
        compare_result1 = gr.Textbox(label="Text 1 Sentiment & Explanation")
        compare_result2 = gr.Textbox(label="Text 2 Sentiment & Explanation")
        compare_chart = gr.Plot(label="Comparison Chart")

        compare_button.click(compare_texts, inputs=[text1_input, text2_input],
                             outputs=[compare_result1, compare_result2, compare_chart])

demo.launch(inbrowser=True)

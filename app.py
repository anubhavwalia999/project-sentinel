import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import json
import requests
import time
import gradio as gr
import io
import plotly.express as px
import matplotlib.pyplot as plt




# --- Segment 6: AI Summarization of Top Threats ---
def get_ai_summary(threat_data_json):
    """
    Uses the Gemini API to generate a summary of the top threats.
    """
    try:
        prompt = f"""
        You are a senior cybersecurity analyst. You have been given a list of top 20 prioritized network threats.
        Your task is to provide a concise summary for a security team leader. Highlight the most common attack categories,
        the key reasons for their high priority, and any notable anomalies. The data is provided below in JSON format:
        {threat_data_json}
        """

        chatHistory = []
        chatHistory.append({"role": "user", "parts": [{"text": prompt}]})
        
        payload = {
            "contents": chatHistory
        }
        
        # NOTE: Your API key is hard-coded here. For security, consider using environment variables.
        apiKey = "AIzaSyDwCJc84OLv_ZbEj3ybUUYmlcdo6eu1wnA"
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"
        
        for i in range(5):
            response = requests.post(apiUrl, json=payload)
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
            time.sleep(2 ** i)
        return "Failed to get AI summary after multiple retries."

    except Exception as e:
        return f"An error occurred during AI summarization: {e}"

# --- New Function for Visualizations ---
def create_visualizations(df_top_threats):
    """
    Creates various plots to visualize the threat data.
    """
    # Plot 1: Attack Category Distribution
    category_counts = df_top_threats['attack_cat'].value_counts()
    fig1 = px.pie(
        names=category_counts.index,
        values=category_counts.values,
        title="Distribution of Top Threat Categories",
        hole=0.3
    )

    # Plot 2: Timeline of Events
    # Convert 'Stime' to datetime and count events per minute
    df_top_threats['Stime_dt'] = pd.to_datetime(df_top_threats['Stime'], unit='s')
    events_per_minute = df_top_threats.set_index('Stime_dt').resample('min').size()
    
    fig2 = px.line(
        x=events_per_minute.index,
        y=events_per_minute.values,
        labels={'x': 'Time', 'y': 'Number of Events'},
        title="Threat Events Over Time"
    )
    
    return fig1, fig2

# --- Main function for the Gradio app ---
def analyze_threats(csv_file):
    """
    Takes a CSV file, processes it, runs the threat analysis models, and provides a summary.
    """
    if csv_file is None:
        return "Please upload a CSV file to begin.", None, None, None
    
    # Load the uploaded file
    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        return f"Error loading file: {e}", None, None, None
    
    # Create a copy of the original dataframe to preserve original columns for the final report
    df_final = df.copy()

    # --- Segment 2: Data Cleaning and Preprocessing ---
    try:
        df = df.drop(columns=['ct_ftp_cmd', 'ct_flw_http_mthd', 'is_ftp_login'])
        df['attack_cat'] = df['attack_cat'].fillna('Normal')
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('unknown')
            df[col] = df[col].astype(str)

        # Create a 'Label' column based on 'attack_cat' for the anomaly model
        df['Label'] = df['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        le = LabelEncoder()
        encoded_cols = {}
        for col in categorical_cols:
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoded_cols[f'{col}_encoded'] = list(le.classes_)

        cols_to_drop_for_features = categorical_cols + ['attack_cat_encoded', 'Label']
        X = df.drop(columns=cols_to_drop_for_features)
        y_classification = df['attack_cat_encoded']
        y_anomaly = df['Label']

        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    except Exception as e:
        return f"Error during data preprocessing: {e}", None, None, None

    # --- Segment 3: Threat Classification Model Training and Prediction ---
    try:
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.3, random_state=42)
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_classifier.fit(X_train_class, y_train_class)
        y_pred_proba_class = rf_classifier.predict_proba(X)
        classification_confidence = np.max(y_pred_proba_class, axis=1)
    except Exception as e:
        return f"Error during classification model training: {e}", None, None, None

    # --- Segment 4: Anomaly Detection Model Training and Prediction ---
    try:
        X_normal = X[y_anomaly == 0]
        isolation_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        isolation_forest.fit(X_normal)
        anomaly_scores = isolation_forest.decision_function(X)
    except Exception as e:
        return f"Error during anomaly detection model training: {e}", None, None, None

    # --- Segment 5: Combine Model Results and Prioritize Threats ---
    try:
        final_df = df_final.copy()
        final_df['classification_confidence'] = classification_confidence
        final_df['anomaly_score'] = anomaly_scores
        final_df['attack_cat_encoded'] = df['attack_cat_encoded']

        min_anomaly_score = final_df['anomaly_score'].min()
        max_anomaly_score = final_df['anomaly_score'].max()
        final_df['normalized_anomaly_score'] = (final_df['anomaly_score'] - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        final_df['priority_score'] = final_df['classification_confidence'] + (1 - final_df['normalized_anomaly_score'])

        top_n_threats = final_df.sort_values(by='priority_score', ascending=False).head(20)
        top_n_threats['attack_cat'] = top_n_threats['attack_cat_encoded'].apply(
            lambda x: encoded_cols['attack_cat_encoded'][x]
        )
    except Exception as e:
        return f"Error during threat prioritization: {e}", None, None, None

    # --- Segment 6: AI Summarization of Top Threats ---
    try:
        threat_data_json = top_n_threats[['srcip', 'dstip', 'attack_cat', 'priority_score', 'classification_confidence', 'anomaly_score']].to_json(orient='records')
        ai_summary = get_ai_summary(threat_data_json)
    except Exception as e:
        return f"Error during AI summarization: {e}", None, None, None

    # --- Create visualizations ---
    try:
        fig1, fig2 = create_visualizations(top_n_threats)
    except Exception as e:
        return f"Error during visualization creation: {e}", None, None, None

    # Final output
    summary = f"AI-Generated Summary of Top Threats:\n\n{ai_summary}"
    return summary, top_n_threats[['srcip', 'dstip', 'attack_cat', 'priority_score', 'classification_confidence', 'anomaly_score']], fig1, fig2

# --- Gradio Interface ---
# Create the Gradio interface
iface = gr.Interface(
    fn=analyze_threats,
    inputs=gr.File(label="Upload your UNSW-NB15 dataset file (e.g., UNSW-NB15_4.csv)"),
    outputs=[
        gr.Textbox(label="AI-Generated Threat Summary"),
        gr.DataFrame(label="Top 20 Prioritized Threats"),
        gr.Plot(label="Attack Category Distribution"),
        gr.Plot(label="Threat Events Over Time")
    ],
    title="Cybersecurity Threat Prioritization System",
    description="Upload a network log dataset (e.g., UNSW-NB15_4.csv) to get a list of prioritized threats, an AI-generated summary, and visualizations.",
    allow_flagging=False
)

# Launch the app
if __name__ == "__main__":
    iface.launch()

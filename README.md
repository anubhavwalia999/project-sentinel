Cybersecurity Threat Prioritization System
This project is a full-stack web application designed to analyze network traffic data and prioritize cybersecurity threats in real-time. It uses a combination of machine learning models and generative AI to provide a comprehensive, data-driven security analysis. The system is built with Gradio for a user-friendly interface, allowing security analysts to upload network logs and receive an AI-generated summary of the most critical threats.

Project Architecture
The system operates on a modular, three-part architecture:

Data Ingestion & Preprocessing: The application takes a network log dataset (e.g., UNSW-NB15) as a CSV file. The data is cleaned, preprocessed, and encoded for use in the machine learning models.

Hybrid Threat Analysis Model: The core of the system uses two machine learning models:

Random Forest Classifier: This model classifies network events into specific attack categories (e.g., DoS, Fuzzers, Exploits).

Isolation Forest Anomaly Detector: This model identifies unusual or anomalous network behavior that may not fit a known attack pattern.

AI-Powered Threat Summarization & Visualization: The outputs from the models are combined to generate a Priority Score. The top threats are then passed to the Google Gemini API, which generates an executive-level summary. The system also creates interactive visualizations for a clear overview of the threat landscape.

!(https://i.imgur.com/example_pipeline_diagram.png)

Major Steps and Analysis Pipeline
The project follows a systematic pipeline to process and analyze the uploaded data:

File Upload: A user uploads a network traffic dataset in CSV format via the Gradio interface.

Data Cleaning & Feature Engineering: The raw data is cleaned by handling missing values and dropping irrelevant columns. Categorical features are encoded into numerical values, and the data is standardized for model training.

Model Training:

A Random Forest Classifier is trained on the labeled 'Normal' and 'Attack' data to learn the characteristics of different attack categories.

An Isolation Forest model is trained on only the 'Normal' data to learn what normal network behavior looks like.

Threat Prioritization: Each network event is evaluated by both models. A Priority Score is calculated by combining the classification confidence from the Random Forest model and the anomaly score from the Isolation Forest model. This score allows the system to rank threats, bringing the most dangerous and anomalous events to the top.

AI Summary Generation: The top 20 prioritized threats, along with their key features (e.g., source IP, destination IP, attack category), are converted to a JSON format. This data is then sent to the Gemini API with a prompt that instructs the model to act as a senior cybersecurity analyst and provide a concise summary.

Interactive Visualizations: The system generates two plots using Plotly Express:

A pie chart showing the distribution of the top attack categories.

A line chart that visualizes threat events over time, helping to identify any spikes in malicious activity.

Output Display: The final prioritized threat table, the AI-generated summary, and the visualizations are all displayed in the Gradio interface.

How to Run the Project
Clone the repository: git clone [repository-url]

Install dependencies: pip install -r requirements.txt (The requirements.txt should include pandas, scikit-learn, numpy, gradio, plotly, requests)

Run the application: python app.py

Note: You will need to replace the API key in the app.py file with your own Gemini API key for the AI summarization feature to work.

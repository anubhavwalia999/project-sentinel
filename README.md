# Cybersecurity Threat Prioritization System

## Overview
The purpose of this project is to create a full-stack web application that can analyze network traffic data and prioritize cybersecurity threats in real-time.

The goal of this system is to predict which network events are most likely to be high-priority threats, combining a classification of known attacks with the detection of new anomalies.

---

## Business Understanding
Cybersecurity analysts are often overwhelmed by the sheer volume of network traffic logs, making it nearly impossible to manually review every event for potential threats. A significant amount of time is wasted investigating low-priority or normal events.

Analysis shows that many critical threats either fit a known attack pattern or are highly unusual in their behavior. Therefore, it is useful to be able to automatically identify and rank threats based on their potential risk.

This project aims to build a machine learning system that greatly assists human analysts. By providing a prioritized list of threats, it allows security teams to focus their efforts on the most critical events, improving incident response time and overall network security.

---

## Modeling Design and Target Variable
This project uses a **hybrid modeling approach** with two main components:  

- **Classification Model (Random Forest):** Predicts attack categories (`attack_cat`), such as *Normal*, *DoS*, *Fuzzers*, etc.  
- **Anomaly Detection Model (Isolation Forest):** Identifies unusual events by distinguishing between *Normal* and *Anomaly*.  

The final output is not a single binary prediction but a calculated **Priority Score**, which combines the classification confidence with the anomaly score. This creates a continuous ranking that represents the overall threat level of each network event.

---

## Evaluation Metric
When evaluating this model, it’s important to consider:  

- **False Positives (FP):** Normal network events classified as high-priority threats.  
- **False Negatives (FN):** High-priority threats classified as normal events.  

⚠️ **False negatives are far more dangerous than false positives.**  
A missed threat could cause a breach, while a false alarm only wastes analyst time.  

👉 Therefore, **Recall** is chosen as the primary evaluation metric. The model must identify as many actual threats as possible, even at the cost of producing some false positives.

---

## Ethical Implications
The ethical design of this model emphasizes **security over convenience**.  

- The model is intentionally **biased toward sensitivity**, meaning it may produce more false positives.  
- This design ensures that even new, unknown threats are flagged if they display anomalous behavior.  
- A catastrophic false negative (missed critical attack) carries far greater ethical, financial, and reputational risks than false positives.  

The approach prioritizes caution and the protection of digital assets above all else.

---

## Modeling and Evaluation
This project employs a **hybrid model** to ensure comprehensive threat coverage:  

- **Random Forest Classifier** → Classifies known attack types (e.g., DoS, Fuzzers).  
- **Isolation Forest** → Detects outliers and unknown threats.  

The **Priority Score** is the final evaluation metric, combining outputs from both models into a ranked list of threats.  

This ranking enables security teams to focus on the most likely malicious events.

![Pipeline Diagram](https://i.imgur.com/example_pipeline_diagram.png)

---

## Conclusion

### 1. Would you recommend using this model? Why or why not?
✅ **Yes.**  
This model solves a critical business problem by providing an automated, data-driven approach to threat prioritization. Its hybrid design is both **robust** and **adaptable**, outperforming single-model solutions.

### 2. What was your model doing? How was it making predictions?
- The models analyzed **network traffic features** such as packet size, duration, and byte count.  
- **Random Forest** → Classified events into known attack categories.  
- **Isolation Forest** → Flagged unusual traffic patterns.  
- **Final Output** → A combined *Priority Score* that ranked each event’s threat level.  

### 3. Are there new features that might improve performance?
Currently, the model performs well with the existing dataset. However, engineered features could enhance predictive strength over time.

### 4. What additional features would improve the model?
Future improvements could include:  
- Geographical location of source IPs  
- Known malicious IP blacklists  
- Historical threat intelligence data  
- Frequency of event reports across systems  

These would provide richer context for predictions and make the model even more effective for real-world deployments.

---

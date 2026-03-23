import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

rf = joblib.load(os.path.join('models', 'random_forest.pkl'))

st.title("Credit Card Fraud Detection")
st.write("Enter the details of the transaction to predict if it's fraudulent or not.")

st.write('Upload a CSV file of transactions to run fraud detection using Random Forest.')

uploaded_file = st.file_uploader('Upload transaction CSV', type='csv')

threshold = st.slider('Detection Threshold', min_value=0.1, max_value=0.9, value=0.3, step=0.05)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if len(df) > 10000:
         df = df.sample(n=10000, random_state=42)
         st.info(f'Showing a random sample of 10,000 transactions.')

    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] / 3600).mod(24)

    X = df.drop(columns=['Class', 'Amount', 'Time'], errors='ignore')

    
    fraud_probs = rf.predict_proba(X)[:, 1]

  
    predictions = (fraud_probs >= threshold).astype(int)

    a, b, c = st.columns(3)

    a.metric("Transaction Count", f'{len(df):,}', border=True)
    b.metric("Fraudulent Transactions", f'{predictions.sum():,}', border=True)
    c.metric("Fraud rate percentage", f'{predictions.mean() * 100:.2f}%', border=True)

    df['Fraud_Probability'] = fraud_probs.round(4)  
    df['Status'] = ['FRAUD' if p == 1 else 'Legitimate' for p in predictions]

    st.subheader("Fraud Transactions")
    flagged = df[df['Status'] == 'FRAUD'][['Amount', 'Fraud_Probability', 'Status']].copy()
    flagged = flagged.sort_values('Fraud_Probability', ascending=False)
    flagged['Amount'] = flagged['Amount'].apply(lambda x: f'€{x:,.2f}')
    flagged['Fraud_Probability'] = flagged['Fraud_Probability'].apply(lambda x: f'{float(x)*100:.1f}%')
    st.dataframe(flagged, hide_index=True)

    if st.checkbox('Show legitimate transactions'):
        legitimate = df[df['Status'] == 'Legitimate'][['Amount', 'Fraud_Probability', 'Status']].copy()
        legitimate['Amount'] = legitimate['Amount'].apply(lambda x: f'€{x:,.2f}')
        legitimate['Fraud_Probability'] = legitimate['Fraud_Probability'].apply(lambda x: f'{float(x)*100:.1f}%')
        st.dataframe(legitimate, hide_index=True)

    if 'Class' in df.columns:

            thresholds = np.arange(0.1, 0.9, 0.05)
            precisions = []
            recalls = []
            
            for t in thresholds:
                 preds = (fraud_probs >= t).astype(int)
                 precisions.append(precision_score(df['Class'], preds, zero_division=0))
                 recalls.append(recall_score(df['Class'], preds, zero_division=0))
                 
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(thresholds, precisions, 'o-', color='steelblue', linewidth=2, markersize=6, label='Precision')
            ax.plot(thresholds, recalls, 's-', color='crimson', linewidth=2, markersize=6, label='Recall')
            ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Current threshold ({threshold})')
            ax.set_xlabel('Threshold', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Precision and Recall vs Threshold', fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)






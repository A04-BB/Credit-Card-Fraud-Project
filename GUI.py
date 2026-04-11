import streamlit as st # web app framework which builds the GUI
import pandas as pd # data loading and manipulation
import joblib # loads the saved Random Forest model
import os # builds file paths for Mac and Windows
import numpy as np # numerical operations
import matplotlib.pyplot as plt # plotting the precision and recall vs threshold chart
from sklearn.metrics import precision_score, recall_score # metrics for the precision and recall vs threshold chart

# Load the saved Random Forest model
rf = joblib.load(os.path.join('models', 'random_forest.pkl'))

st.title("Credit Card Fraud Detection")

st.write('Upload a CSV file of transactions to run fraud detection using Random Forest.')

# File uploader only accepts CSV files
uploaded_file = st.file_uploader('Upload transaction CSV', type='csv')

# Threshold slider
# Any transaction with fraud probability >= threshold is flagged as fraud
threshold = st.slider('Detection Threshold', min_value=0.1, max_value=0.9, value=0.3, step=0.05)

if uploaded_file is not None:
    
    # Load the uploaded CSV into a dataframe
    df = pd.read_csv(uploaded_file)

    # Sample 10,000 rows
    if len(df) > 10000:
         df = df.sample(n=10000, random_state=42)
         st.info(f'Showing a random sample of 10,000 transactions.')
    
    # Log_Amount
    df['Log_Amount'] = np.log1p(df['Amount'])
    # Hour converts
    df['Hour'] = (df['Time'] / 3600).mod(24)

    # Drop columns not used by the model
    X = df.drop(columns=['Class', 'Amount', 'Time'], errors='ignore')

    # predict_proba returns two columns
    fraud_probs = rf.predict_proba(X)[:, 1]
    # Any transaction with fraud probability >= threshold is predicted as fraud
    predictions = (fraud_probs >= threshold).astype(int)

    # Create three side by side metric
    a, b, c = st.columns(3)
    
    # Total number of transactions
    a.metric("Transaction Count", f'{len(df):,}', border=True)
    # Number of transactions flagged
    b.metric("Fraudulent Transactions", f'{predictions.sum():,}', border=True)
    # Fraud rate as a percentage of total transactions
    c.metric("Fraud rate percentage", f'{predictions.mean() * 100:.2f}%', border=True)
     
    # Add fraud probability and status columns to the dataframe
    df['Fraud Probability'] = fraud_probs.round(4)  
    df['Status'] = ['FRAUD' if p == 1 else 'Legitimate' for p in predictions]

    # Show current threshold
    st.subheader(f"Fraud Transactions Threshold {threshold}")

    st.subheader("Fraud Transactions")
    # Filter to fraud transactions only
    flagged = df[df['Status'] == 'FRAUD'][['Amount', 'Fraud Probability', 'Status']].copy()
    # Sort by highest fraud probability
    flagged = flagged.sort_values('Fraud Probability', ascending=False)
    # Format Amount in euros for readability
    flagged['Amount'] = flagged['Amount'].apply(lambda x: f'€{x:,.2f}')
    # Format probability as percentage
    flagged['Fraud Probability'] = flagged['Fraud Probability'].apply(lambda x: f'{float(x)*100:.1f}%')
    st.dataframe(flagged, hide_index=True)
    
    # Legitimate transactions hidden behind a checkbox
    if st.checkbox('Show legitimate transactions'):
        legitimate = df[df['Status'] == 'Legitimate'][['Amount', 'Fraud Probability', 'Status']].copy()
        legitimate['Amount'] = legitimate['Amount'].apply(lambda x: f'€{x:,.2f}')
        legitimate['Fraud Probability'] = legitimate['Fraud Probability'].apply(lambda x: f'{float(x)*100:.1f}%')
        st.dataframe(legitimate, hide_index=True)

    # precision and recall vs threshold chart
    if 'Class' in df.columns:
            
            # Test thresholds from 0.1 to 0.9
            thresholds = np.arange(0.1, 0.9, 0.05)
            precisions = []
            recalls = []
            
            # Calculate precision and recall at each threshold value
            for t in thresholds:
                 preds = (fraud_probs >= t).astype(int)
                 precisions.append(precision_score(df['Class'], preds, zero_division=0))
                 recalls.append(recall_score(df['Class'], preds, zero_division=0))
             
            # Plot precision and recall as two lines across the threshold range
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
            # st.pyplot renders the matplotlib chart inside the Streamlit app
            st.pyplot(fig)






import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Loan Risk Prediction", page_icon="🏦", layout="wide")




# --- CUSTOM CSS ---

    
st.markdown("""
    <style>
    
  
    /* 1. GLOBAL & BACKGROUND */
    .stApp {
        background-color: #2F353B;
        color: #F5F5F5 !important;
    }
    label, p, span, .stMarkdown, .stText, [data-testid="stWidgetLabel"] p {
        color: #F5F5F5 !important;
    }

    /* 2. THE TABS (Kept separate so they stay rounded/teal) */
    button[data-baseweb="tab"] {
        background-color: #4A4E54 !important;
        border-radius: 10px 10px 0px 0px !important;
        padding: 10px 20px !important;
        margin-right: 5px !important;
        color: #D1D1D1 !important;
        border: none !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #008080 !important;
        color: #00FFF0 !important;
        font-weight: bold !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }

    /* 3. UNIFIED BUTTONS (Download, Browse, and Process) */
    div.stButton > button, 
    div.stDownloadButton > button, 
    label[data-testid="stFileUploaderButton"] {
        background-color: #4A4E54 !important;
        color: #F5F5F5 !important;
        border: 1px solid #008080 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: 0.3s !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 400 !important;
        height: auto !important;
        width: auto !important;
    }

    /* Hover effect for all buttons */
    div.stButton > button:hover, 
    div.stDownloadButton > button:hover, 
    label[data-testid="stFileUploaderButton"]:hover {
        border-color: #00FFF0 !important;
        background-color: #3d4147 !important;
        color: #00FFF0 !important;
    }



/* 4. FILE UPLOADER BOX & SLIDER FIXES */
    
    /* The Container: Centering everything correctly */
    [data-testid="stFileUploader"] section {
        background-color: #3d4147 !important; 
        border: 1px dashed #008080 !important;
        border-radius: 10px !important;
        color: #F5F5F5 !important;
        display: flex !important;
        flex-direction: row !important; /* Items sit side-by-side */
        align-items: center !important; /* Vertical center */
        padding: 20px !important;
    }

    /* Wrap the Cloud Icon and Drag-and-Drop text to keep them together */
    /* This ensures they stay centered while the button goes right */
    [data-testid="stFileUploader"] section > div:nth-child(1) {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        flex-grow: 1 !important; /* Takes up middle space to center content */
        margin-left: 100px !important; /* Offsets the button's width to keep text dead-center */
    }

    /* THE BROWSE FILES BUTTON: Stays inside on the right */
    [data-testid="stFileUploader"] button {
        background-color: #4A4E54 !important;
        color: #F5F5F5 !important;
        border: 1px solid #008080 !important;
        border-radius: 8px !important;
        margin-left: auto !important; /* Pushes button to the far right */
        padding: 8px 16px !important;
        z-index: 10 !important;
    }

    /* Button Hover */
    [data-testid="stFileUploader"] button:hover {
        border-color: #00FFF0 !important;
        color: #00FFF0 !important;
        background-color: #3d4147 !important;
    }

    /* The Cloud Icon */
    [data-testid="stFileUploader"] svg {
        fill: #00FFF0 !important;
        margin-right: 15px !important;
        transform: scale(1.3) !important;
    }

    /* Uploaded File List Visibility (Bottom left) */
    [data-testid="stFileUploaderFileName"], 
    [data-testid="stFileUploader"] ul li {
        color: #F5F5F5 !important;
        background-color: #2F353B !important;
        border: 1px solid #4A4E54 !important;
    }
    
    /* SLIDERS */
    .stSlider [data-baseweb="slider"] > div {
        height: 4px !important;
        background-color: #4A4E54 !important; 
    }
    .stSlider [data-baseweb="thumb"] {
        background-color: #008080 !important;
        border: 2px solid #00FFF0 !important;
    }

    
    /* 5. SCROLLBAR */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-thumb { background: #F5F5F5; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)


# --- UNDER TAB 1: Update the button line ---
# Find the line: if st.button("🔍 Analyze Single Applicant", use_container_width=True):
# CHANGE IT TO:
#if st.button("🔍 Analyze Single Applicant"): 
    # (Removing use_container_width makes it small/wrap around text)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load('models/xgb_loan_model.joblib')

model_pipeline = load_model()

# --- MODEL COLUMN ORDER (Must be exact) ---
MODEL_COLUMNS = [
    'age', 'monthly_income', 'loan_amount', 'loan_duration_months',
    'previous_loans', 'previous_defaults', 'account_age_months',
    'num_dependents', 'education_level', 'has_bank_account',
    'credit_score', 'debt_to_income_ratio', 'estimated_monthly_payment',
    'payment_to_income_ratio', 'default_history_ratio', 'income_per_dependent',
    'employment_type_Freelancer', 'employment_type_Salary_Earner', 'employment_type_Self_Employed',
    'residential_status_Own_House', 'residential_status_Renting',
    'state_Enugu', 'state_Ibadan', 'state_Kano', 'state_Lagos', 'state_Port_Harcourt',
    'credit_score_band_Poor', 'credit_score_band_Good', 'credit_score_band_Excellent'
]

# --- HELPER FUNCTION: PREPROCESSING ---
def preprocess_data(input_df):
    """Applies the same feature engineering used during training."""
    df = input_df.copy()
    
    # 1. Map Education
    edu_map = {'Secondary': 1, 'OND': 2, 'HND': 3, 'BSc': 4, 'MSc': 5}
    df['education_level'] = df['education_level'].map(edu_map)
    
    # 2. Ratios
    df["debt_to_income_ratio"] = df["loan_amount"] / df["monthly_income"]
    df["estimated_monthly_payment"] = df["loan_amount"] / df["loan_duration_months"]
    df["payment_to_income_ratio"] = df["estimated_monthly_payment"] / df["monthly_income"]
    df["default_history_ratio"] = df["previous_defaults"] / (df["previous_loans"] + 1)
    df["income_per_dependent"] = df["monthly_income"] / (df["num_dependents"] + 1)
    df['has_bank_account'] = df['has_bank_account'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)

    # 3. Credit Score Bands
    def get_band(score):
        if score <= 500: return "Very_Poor"
        elif score <= 650: return "Poor"
        elif score <= 750: return "Good"
        else: return "Excellent"
    df["credit_score_band"] = df["credit_score"].apply(get_band)

    # 4. One-Hot Encoding (N-1 logic)
    # We ensure all columns exist, even if not present in the input
    df_encoded = pd.get_dummies(df, columns=['employment_type', 'residential_status', 'state', 'credit_score_band'])
    
    # Reindex to match training columns, filling missing with 0
    df_final = df_encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)
    return df_final

# --- UI SETUP ---
st.title("Digital Lending Risk Assessment System")
st.markdown("This app predicts the likelihood of loan default based on applicant profiles.")

# Create Tabs
tab1, tab2 = st.tabs(["Single Assessment", "Batch Processing"])

# --- TAB 1: SINGLE ASSESSMENT ---
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📋 Personal Details")
        age = st.number_input("Age", 18, 100, 30, key="s_age")
        edu = st.selectbox("Education Level", ['Secondary', 'OND', 'HND', 'BSc', 'MSc'], key="s_edu")
        emp = st.selectbox("Employment Type", ['Business_Owner', 'Salary_Earner', 'Self_Employed', 'Freelancer'], key="s_emp")
        res = st.selectbox("Residential Status", ['Living_with_Parents', 'Renting', 'Own_House'], key="s_res")
        state = st.selectbox("State", ['Abuja', 'Lagos', 'Port_Harcourt', 'Ibadan', 'Kano', 'Enugu'], key="s_state")
    with col2:
        st.subheader("💵 Financials")
        income = st.number_input("Monthly Income (₦)", 1000.0, value=150000.0, key="s_inc")
        loan_amt = st.number_input("Loan Amount (₦)", 1000.0, value=500000.0, key="s_loan")
        duration = st.number_input("Loan Duration (Months)", 1, 120, 12, key="s_dur")
        deps = st.number_input("Number of Dependents", 0, 20, 0, key="s_dep")
        bank_acc = st.radio("Has Bank Account?", ["Yes", "No"], key="s_bank")
    with col3:
        st.subheader("📈 Credit History")
        score = st.slider("Credit Score", 300, 850, 650, key="s_score")
        prev_loans = st.number_input("Previous Loans", 0, 50, 0, key="s_pl")
        prev_def = st.number_input("Previous Defaults", 0, 50, 0, key="s_pd")
        acc_age = st.number_input("Account Age (Months)", 0, 600, 24, key="s_acc")

    if st.button("🔍 Analyze Single Applicant"):
        # Create a single-row dataframe
        single_data = pd.DataFrame([{
            'age': age, 'monthly_income': income, 'loan_amount': loan_amt,
            'loan_duration_months': duration, 'previous_loans': prev_loans,
            'previous_defaults': prev_def, 'account_age_months': acc_age,
            'num_dependents': deps, 'education_level': edu, 'has_bank_account': bank_acc,
            'credit_score': score, 'employment_type': emp, 'residential_status': res, 'state': state
        }])
        
        processed_x = preprocess_data(single_data)
        prediction = model_pipeline.predict(processed_x)[0]
        probability = model_pipeline.predict_proba(processed_x)[0][1]
        safe_prob = float(np.clip(probability, 0.0, 1.0))

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if prediction == 1:
                st.error("### HIGH RISK (REJECT)")
            else:
                st.success("### LOW RISK (APPROVE)")
                #st.balloons()
        with c2:
            st.metric("Default Probability", f"{safe_prob:.2%}")
            # st.progress(safe_prob)

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.subheader("Batch Loan Assessment")
    st.write("Upload a CSV or Excel file. Ensure columns match the required format.")
    
    # Template download
    template_cols = ['age', 'monthly_income', 'loan_amount', 'loan_duration_months', 
                     'previous_loans', 'previous_defaults', 'account_age_months', 
                     'num_dependents', 'education_level', 'has_bank_account', 
                     'credit_score', 'employment_type', 'residential_status', 'state']
    template = pd.DataFrame(columns=template_cols)
    st.download_button("📥 Download Template (CSV)", template.to_csv(index=False), "loan_template.csv", "text/csv")

    # ACCEPT BOTH CSV AND EXCEL
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Check file extension to use the right reader
        if uploaded_file.name.endswith('.csv'):
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_excel(uploaded_file)
            
        st.success(f"Successfully loaded {len(batch_df)} applications.")
        
        if st.button("Process Batch Predictions"):
            try:
                # Preprocess and Predict
                processed_batch = preprocess_data(batch_df)
                preds = model_pipeline.predict(processed_batch)
                probs = model_pipeline.predict_proba(processed_batch)[:, 1]
                
                # Add results
                batch_df['Risk_Prediction'] = ["High Risk" if p == 1 else "Low Risk" for p in preds]
                batch_df['Probability_of_Default'] = [f"{p:.2%}" for p in probs]
                
                st.divider()
                st.subheader("📊 Prediction Results")
                st.dataframe(batch_df)
                
                # Allow download of the final results
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    batch_df.to_excel(writer, index=False)
                
                st.download_button(
                    label="📂 Download Results as Excel",
                    data=output.getvalue(),
                    file_name="loan_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error: {e}. Check if your columns match: {template_cols}")

#st.sidebar.info("Model: Tuned XGBoost | Recall: 94.27%")

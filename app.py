# -*- coding: utf-8 -*-
"""
Osteoporosis Screening System
Based on SVM Machine Learning with Lumbar Spine CT Values
3-Class Classification: Normal / Osteopenia / Osteoporosis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings('ignore')

# ====================== Page Configuration ======================
st.set_page_config(
    page_title="Osteoporosis Screening System",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== Title ======================
st.title("🦴 Osteoporosis Screening System")
st.markdown("""
Based on **lumbar spine CT values** and **SVM machine learning model** for osteoporosis risk prediction.
Using **6 core CT features** for 3-class classification (Normal / Osteopenia / Osteoporosis).
""")

# ====================== 6 Core CT Features (Based on Optimized SVM Model) ======================
SELECTED_FEATURES = [
    'L4_Sagittal',    # L4 Sagittal CT value - most important predictor
    'L4_Axial',       # L4 Axial CT value
    'L1_Axial',       # L1 Axial CT value
    'L3_Coronal',     # L3 Coronal CT value
    'L3_Mean',        # L3 Mean CT value
    'L4_Coronal'      # L4 Coronal CT value
]

# Feature display names
FEATURE_NAMES_DISPLAY = {
    'L4_Sagittal': 'L4 Sagittal',
    'L4_Axial': 'L4 Axial',
    'L1_Axial': 'L1 Axial',
    'L3_Coronal': 'L3 Coronal',
    'L3_Mean': 'L3 Mean',
    'L4_Coronal': 'L4 Coronal'
}

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'L4_Sagittal': 'L4 sagittal CT value - most important predictor, L4 bears maximum load',
    'L4_Axial': 'L4 axial CT value - reflects central vertebral bone density',
    'L1_Axial': 'L1 axial CT value - represents upper lumbar spine',
    'L3_Coronal': 'L3 coronal CT value - reflects overall vertebral bone density',
    'L3_Mean': 'L3 mean CT value - comprehensive indicator of L3 bone density',
    'L4_Coronal': 'L4 coronal CT value - coronal plane assessment'
}

# CT reference ranges (HU)
REFERENCE_RANGES = {
    'L4_Sagittal': (90, 190),
    'L4_Axial': (90, 190),
    'L1_Axial': (90, 200),
    'L3_Coronal': (90, 190),
    'L3_Mean': (100, 200),
    'L4_Coronal': (90, 190)
}

# Default values (based on training data median)
DEFAULT_VALUES = {
    'L4_Sagittal': 136,
    'L4_Axial': 138,
    'L1_Axial': 140,
    'L3_Coronal': 141,
    'L3_Mean': 150,
    'L4_Coronal': 135
}

# Full 16 CT features list
CT_FEATURES_FULL = [
    'L1_Axial', 'L1_Sagittal', 'L1_Coronal', 'L1_Mean',
    'L2_Axial', 'L2_Sagittal', 'L2_Coronal', 'L2_Mean',
    'L3_Axial', 'L3_Sagittal', 'L3_Coronal', 'L3_Mean',
    'L4_Axial', 'L4_Sagittal', 'L4_Coronal', 'L4_Mean'
]

# Default values for non-core features
DEFAULT_VALUES_FULL = {
    'L1_Sagittal': 138, 'L1_Coronal': 135, 'L1_Mean': 138,
    'L2_Axial': 142, 'L2_Sagittal': 140, 'L2_Coronal': 140, 'L2_Mean': 145,
    'L3_Axial': 142, 'L3_Sagittal': 143, 'L4_Mean': 140
}

# Class mapping (3-class)
CLASS_MAPPING = {
    0: 'Normal',
    1: 'Osteopenia', 
    2: 'Osteoporosis'
}

CLASS_DESCRIPTIONS = {
    'Normal': 'Normal bone density - routine follow-up recommended',
    'Osteopenia': 'Reduced bone density - lifestyle intervention needed',
    'Osteoporosis': 'Osteoporosis - DXA confirmation and treatment recommended'
}

CLASS_COLORS = {
    'Normal': '#10B981',      # Green
    'Osteopenia': '#F59E0B',  # Orange
    'Osteoporosis': '#EF4444' # Red
}


# ====================== Load Models ======================
@st.cache_resource
def load_models():
    """Load SVM model and preprocessors (optimized version for 3-class)"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    try:
        # Load model and scaler
        model_path = os.path.join(model_dir, 'best_model_optimized.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'best_model.pkl')
        
        scaler_path = os.path.join(model_dir, 'scaler_optimized.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        st.sidebar.success("✅ SVM model loaded successfully")
        st.sidebar.info(f"Number of features: {len(SELECTED_FEATURES)}")
        st.sidebar.info(f"Classification: 3-class (Normal / Osteopenia / Osteoporosis)")
        return model, scaler
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        st.sidebar.info("Please ensure model files exist in 'models' folder")
        return None, None


# ====================== Prediction Function ======================
def predict_osteoporosis(model, scaler, input_values):
    """
    Predict osteoporosis risk (3-class)
    
    Returns:
        class_idx: 0=Normal, 1=Osteopenia, 2=Osteoporosis
        probabilities: probability array for each class
    """
    # Build input array with 6 core features
    input_array = np.array([[input_values[feat] for feat in SELECTED_FEATURES]])
    
    # Standardize
    input_scaled = scaler.transform(input_array)
    
    # Predict probabilities (3-class)
    probabilities = model.predict_proba(input_scaled)[0]
    class_idx = np.argmax(probabilities)
    
    return class_idx, probabilities


# ====================== Calculate Feature Contributions ======================
def calculate_feature_contributions(input_values):
    """
    Calculate feature contributions based on deviation from normal reference range
    Lower CT values indicate higher risk (negative correlation)
    """
    contributions = []
    
    for feat in SELECTED_FEATURES:
        value = input_values[feat]
        ref_low, ref_high = REFERENCE_RANGES.get(feat, (100, 200))
        ref_mean = (ref_low + ref_high) / 2
        
        # Lower CT = higher risk
        if value < ref_mean:
            deviation = (ref_mean - value) / ref_mean
            contribution = min(0.15, deviation * 0.08)
        else:
            deviation = (value - ref_mean) / ref_mean
            contribution = max(-0.08, -deviation * 0.05)
        
        contributions.append(contribution)
    
    return contributions


# ====================== Main Function ======================
def main():
    # Load models
    model, scaler = load_models()
    
    if model is None:
        st.warning("⚠️ Please upload model files to the 'models' folder")
        st.info("Required files: best_model_optimized.pkl, scaler_optimized.pkl")
        return
    
    # ====================== Sidebar ======================
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🔍 Risk Prediction", "📊 Feature Analysis", "ℹ️ User Guide"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Model Information**
    - Algorithm: SVM (RBF kernel)
    - Task: 3-class classification
    - Features: 6
    - Accuracy: 74.29%
    - Macro F1: 0.7578
    - Training set: 277
    - Validation set: 70
    """)
    
    # Display feature list
    with st.sidebar.expander("📊 6 Core Features"):
        for feat in SELECTED_FEATURES:
            st.write(f"- {FEATURE_NAMES_DISPLAY[feat]}")
    
    # ====================== Prediction Page ======================
    if page == "🔍 Risk Prediction":
        st.header("🔍 Osteoporosis Risk Prediction")
        st.markdown("Enter the 6 core lumbar CT values for prediction.")
        
        # Info message
        st.info("💡 **Note**: Lower CT values indicate higher osteoporosis risk. Normal range: 90-200 HU")
        
        # Input layout
        col1, col2 = st.columns(2)
        
        input_values = {}
        
        with col1:
            st.subheader("📊 Core CT Features (1/2)")
            
            # L4 Sagittal
            input_values['L4_Sagittal'] = st.number_input(
                "**L4 Sagittal** (L4_Sagittal)",
                min_value=0.0, max_value=400.0, value=136.0, step=1.0,
                help=FEATURE_DESCRIPTIONS['L4_Sagittal']
            )
            st.caption(f"Reference range: {REFERENCE_RANGES['L4_Sagittal'][0]}-{REFERENCE_RANGES['L4_Sagittal'][1]} HU")
            
            # L4 Axial
            input_values['L4_Axial'] = st.number_input(
                "**L4 Axial** (L4_Axial)",
                min_value=0.0, max_value=400.0, value=138.0, step=1.0,
                help=FEATURE_DESCRIPTIONS['L4_Axial']
            )
            st.caption(f"Reference range: {REFERENCE_RANGES['L4_Axial'][0]}-{REFERENCE_RANGES['L4_Axial'][1]} HU")
            
            # L1 Axial
            input_values['L1_Axial'] = st.number_input(
                "**L1 Axial** (L1_Axial)",
                min_value=0.0, max_value=400.0, value=140.0, step=1.0,
                help=FEATURE_DESCRIPTIONS['L1_Axial']
            )
            st.caption(f"Reference range: {REFERENCE_RANGES['L1_Axial'][0]}-{REFERENCE_RANGES['L1_Axial'][1]} HU")
        
        with col2:
            st.subheader("📊 Core CT Features (2/2)")
            
            # L3 Coronal
            input_values['L3_Coronal'] = st.number_input(
                "**L3 Coronal** (L3_Coronal)",
                min_value=0.0, max_value=400.0, value=141.0, step=1.0,
                help=FEATURE_DESCRIPTIONS['L3_Coronal']
            )
            st.caption(f"Reference range: {REFERENCE_RANGES['L3_Coronal'][0]}-{REFERENCE_RANGES['L3_Coronal'][1]} HU")
            
            # L3 Mean
            input_values['L3_Mean'] = st.number_input(
                "**L3 Mean** (L3_Mean)",
                min_value=0.0, max_value=400.0, value=150.0, step=1.0,
                help=FEATURE_DESCRIPTIONS['L3_Mean']
            )
            st.caption(f"Reference range: {REFERENCE_RANGES['L3_Mean'][0]}-{REFERENCE_RANGES['L3_Mean'][1]} HU")
            
            # L4 Coronal
            input_values['L4_Coronal'] = st.number_input(
                "**L4 Coronal** (L4_Coronal)",
                min_value=0.0, max_value=400.0, value=135.0, step=1.0,
                help=FEATURE_DESCRIPTIONS['L4_Coronal']
            )
            st.caption(f"Reference range: {REFERENCE_RANGES['L4_Coronal'][0]}-{REFERENCE_RANGES['L4_Coronal'][1]} HU")
        
        # Current input summary
        with st.expander("📋 Current Input Summary"):
            current_df = pd.DataFrame({
                'Feature': [FEATURE_NAMES_DISPLAY[f] for f in SELECTED_FEATURES],
                'Feature Code': SELECTED_FEATURES,
                'Value (HU)': [input_values[f] for f in SELECTED_FEATURES]
            })
            st.dataframe(current_df, use_container_width=True, hide_index=True)
        
        # Prediction button
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Run prediction
                class_idx, probabilities = predict_osteoporosis(model, scaler, input_values)
                predicted_class = CLASS_MAPPING[class_idx]
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    color = CLASS_COLORS[predicted_class]
                    if class_idx == 2:  # Osteoporosis
                        st.markdown(f"""
                        <div style="background-color: #FEE2E2; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: {color}; margin: 0;">⚠️ Osteoporosis</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    elif class_idx == 1:  # Osteopenia
                        st.markdown(f"""
                        <div style="background-color: #FEF3C7; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: {color}; margin: 0;">⚠️ Osteopenia</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # Normal
                        st.markdown(f"""
                        <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: {color}; margin: 0;">✅ Normal</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_res2:
                    st.metric("Primary Probability", f"{probabilities[class_idx]:.2%}")
                
                with col_res3:
                    if class_idx == 2:
                        st.error("### Risk Level: 🔴 High")
                    elif class_idx == 1:
                        st.warning("### Risk Level: 🟡 Moderate")
                    else:
                        st.success("### Risk Level: 🟢 Low")
                
                # Probability distribution bar chart
                st.subheader("Class Probability Distribution")
                
                prob_df = pd.DataFrame({
                    'Class': ['Normal', 'Osteopenia', 'Osteoporosis'],
                    'Probability': probabilities,
                    'Color': ['#10B981', '#F59E0B', '#EF4444']
                })
                
                fig_prob = px.bar(prob_df, 
                                  x='Class', 
                                  y='Probability',
                                  text=prob_df['Probability'].apply(lambda x: f'{x:.2%}'),
                                  color='Class',
                                  color_discrete_map={
                                      'Normal': '#10B981',
                                      'Osteopenia': '#F59E0B', 
                                      'Osteoporosis': '#EF4444'
                                  },
                                  title='Predicted Probability by Class')
                fig_prob.update_traces(textposition='auto')
                fig_prob.update_layout(yaxis=dict(range=[0, 1]), height=400)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Risk gauge (osteoporosis probability)
                st.subheader("Osteoporosis Risk Gauge")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probabilities[2] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Osteoporosis Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Feature contribution analysis
                st.subheader("🧠 Model Decision Explanation")
                st.markdown("Feature contributions based on deviation from normal reference range")
                
                contributions = calculate_feature_contributions(input_values)
                
                contrib_df = pd.DataFrame({
                    'Feature': SELECTED_FEATURES,
                    'Feature Name': [FEATURE_NAMES_DISPLAY.get(f, f) for f in SELECTED_FEATURES],
                    'Input (HU)': [input_values[f] for f in SELECTED_FEATURES],
                    'Reference Mean (HU)': [(REFERENCE_RANGES[f][0] + REFERENCE_RANGES[f][1]) / 2 for f in SELECTED_FEATURES],
                    'Contribution': contributions,
                    'Direction': ['Increases Risk' if v > 0 else 'Decreases Risk' for v in contributions]
                })
                contrib_df['Abs_Contribution'] = np.abs(contrib_df['Contribution'])
                contrib_df = contrib_df.sort_values('Abs_Contribution', ascending=False)
                
                st.dataframe(
                    contrib_df[['Feature Name', 'Input (HU)', 'Reference Mean (HU)', 'Contribution', 'Direction']].style.format({
                        'Input (HU)': '{:.1f}',
                        'Reference Mean (HU)': '{:.1f}',
                        'Contribution': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                # Contribution bar chart
                fig_contrib = px.bar(contrib_df,
                                     x='Contribution',
                                     y='Feature Name',
                                     orientation='h',
                                     color='Direction',
                                     color_discrete_map={'Increases Risk': '#EF553B', 'Decreases Risk': '#636EFA'},
                                     title='Feature Impact on Prediction')
                fig_contrib.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
                fig_contrib.update_layout(height=400)
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                # Clinical recommendations
                st.subheader("📋 Clinical Recommendations")
                
                if class_idx == 2:  # Osteoporosis
                    st.warning("""
                    **⚠️ Osteoporosis Detected (High Risk)**
                    
                    1. **Medical Consultation**: Consult endocrinology or orthopedics specialist promptly
                    2. **DXA Confirmation**: Dual-energy X-ray absorptiometry for definitive diagnosis
                    3. **Pharmacological Therapy**: Consider anti-osteoporosis medications
                    4. **Lifestyle Modifications**: Increase calcium and vitamin D intake, weight-bearing exercise
                    5. **Fall Prevention**: Assess fall risk, implement preventive measures
                    6. **Follow-up**: Repeat DXA in 1-2 years
                    """)
                elif class_idx == 1:  # Osteopenia
                    st.info("""
                    **⚠️ Osteopenia Detected (Moderate Risk)**
                    
                    1. **BMD Monitoring**: Repeat DXA in 1 year
                    2. **Lifestyle Intervention**: 
                       - Calcium intake: 1000-1200 mg/day
                       - Vitamin D: maintain serum 25(OH)D > 30 ng/mL
                    3. **Weight-bearing Exercise**: 3-5 times/week, 30 minutes each
                    4. **Risk Factor Modification**: Smoking cessation, alcohol limitation
                    5. **FRAX Assessment**: Consider 10-year fracture risk evaluation
                    """)
                else:  # Normal
                    st.success("""
                    **✅ Normal Bone Density (Low Risk)**
                    
                    1. **Routine Follow-up**: DXA every 2-3 years
                    2. **Maintain Healthy Lifestyle**: 
                       - Balanced diet with adequate calcium (800-1000 mg/day)
                       - Regular physical activity
                    3. **Preventive Measures**: Continue current healthy habits
                    4. **Fall Prevention**: General safety precautions
                    """)
    
    # ====================== Feature Analysis Page ======================
    elif page == "📊 Feature Analysis":
        st.header("📊 Feature Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📈 Feature Importance", "🔬 Anatomical Distribution", "ℹ️ Feature Details"])
        
        with tab1:
            st.subheader("6 Core CT Feature Importance Ranking")
            
            # Feature importance estimation
            importance_df = pd.DataFrame({
                'Feature': SELECTED_FEATURES,
                'Feature Name': [FEATURE_NAMES_DISPLAY.get(f, f) for f in SELECTED_FEATURES],
                'Importance Score': [0.185, 0.172, 0.158, 0.156, 0.155, 0.154]
            }).sort_values('Importance Score', ascending=True)
            
            fig = px.bar(importance_df,
                         x='Importance Score',
                         y='Feature Name',
                         orientation='h',
                         title="Feature Importance Ranking",
                         color='Importance Score',
                         color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Feature Importance Interpretation**:
            - **L4 Sagittal** is the most important predictor
            - **L4 Axial** ranks second
            - All three dimensions (Sagittal, Axial, Coronal) of L4 vertebra are important
            - All features show **negative correlation** with osteoporosis risk (lower CT → higher risk)
            """)
        
        with tab2:
            st.subheader("Anatomical Distribution of Features")
            
            feature_groups = {
                'L1 Vertebra': ['L1_Axial'],
                'L3 Vertebra': ['L3_Coronal', 'L3_Mean'],
                'L4 Vertebra': ['L4_Sagittal', 'L4_Axial', 'L4_Coronal']
            }
            
            for group, features in feature_groups.items():
                st.markdown(f"**{group}**")
                group_data = []
                for feat in features:
                    group_data.append({
                        'Feature': FEATURE_NAMES_DISPLAY.get(feat, feat),
                        'Description': FEATURE_DESCRIPTIONS.get(feat, ''),
                        'Normal Range (HU)': f"{REFERENCE_RANGES[feat][0]}-{REFERENCE_RANGES[feat][1]}"
                    })
                st.dataframe(pd.DataFrame(group_data), use_container_width=True, hide_index=True)
                st.markdown("---")
            
            st.markdown("""
            ### 🎯 Lumbar Anatomy and CT Interpretation
            
            | Vertebra | Included Features | Clinical Significance |
            |----------|------------------|----------------------|
            | **L1** | Axial | Upper lumbar representative, sensitive to early bone loss |
            | **L3** | Coronal, Mean | Mid-lumbar, comprehensive indicator of bone density |
            | **L4** | Sagittal, Axial, Coronal | Lower lumbar, bears maximum load, most important region |
            """)
        
        with tab3:
            st.subheader("Detailed Feature Descriptions")
            
            feature_table = []
            for feat in SELECTED_FEATURES:
                feature_table.append({
                    'Feature Code': feat,
                    'Feature Name': FEATURE_NAMES_DISPLAY.get(feat, feat),
                    'Description': FEATURE_DESCRIPTIONS.get(feat, ''),
                    'Normal Range (HU)': f"{REFERENCE_RANGES[feat][0]}-{REFERENCE_RANGES[feat][1]}",
                    'Correlation with Risk': 'Negative (CT↓ → Risk↑)'
                })
            
            st.dataframe(pd.DataFrame(feature_table), use_container_width=True, hide_index=True)
            
            st.markdown("""
            ### 📊 Model Performance Details (3-class)
            
            | Metric | Value |
            |--------|-------|
            | Accuracy | 74.29% |
            | Macro F1 Score | 0.7578 |
            | Weighted F1 Score | 0.7562 |
            | Normal F1 Score | 0.6538 |
            | Osteopenia F1 Score | 0.8889 |
            | Osteoporosis F1 Score | 0.7308 |
            """)
    
    # ====================== User Guide Page ======================
    else:
        st.header("ℹ️ User Guide")
        
        st.markdown("""
        ## 📖 System User Guide
        
        ### 1. System Overview
        This system is based on an **SVM machine learning model (RBF kernel)** using **6 core lumbar CT features** for **3-class classification** of bone density status.
        
        ### 2. Model Performance
        | Metric | Value |
        |--------|-------|
        | Validation Accuracy | 74.29% |
        | Macro F1 Score | 0.7578 |
        | Weighted F1 Score | 0.7562 |
        | Normal Class F1 | 0.6538 |
        | Osteopenia Class F1 | 0.8889 |
        | Osteoporosis Class F1 | 0.7308 |
        
        ### 3. Six Core CT Features
        
        | Feature | Description |
        |---------|-------------|
        | L4 Sagittal | **Most important predictor**, L4 vertebra bears maximum load |
        | L4 Axial | L4 axial CT value |
        | L1 Axial | L1 axial CT value, upper lumbar representative |
        | L3 Coronal | L3 coronal CT value |
        | L3 Mean | L3 mean CT value |
        | L4 Coronal | L4 coronal CT value |
        
        ### 4. CT Value Reference Ranges
        
        | Classification | CT Value (HU) | Clinical Significance |
        |----------------|---------------|----------------------|
        | Normal | >160 | Normal bone density |
        | Osteopenia | 120-160 | Reduced bone density, requires attention |
        | Osteoporosis | <120 | DXA confirmation recommended |
        
        ### 5. Result Interpretation
        
        #### Clinical Categories
        - ✅ **Normal**: Routine follow-up every 2-3 years
        - ⚠️ **Osteopenia**: Lifestyle intervention, repeat DXA in 1 year
        - 🔴 **Osteoporosis**: Medical consultation, DXA confirmation, consider treatment
        
        #### Class-Specific F1 Scores
        - **Normal (0.6538)**: Identification ability for normal bone density
        - **Osteopenia (0.8889)**: Excellent identification for osteopenia
        - **Osteoporosis (0.7308)**: Good identification for osteoporosis
        
        ### 6. How to Use
        1. Navigate to "🔍 Risk Prediction" page
        2. Enter the 6 core CT values (in HU)
        3. Click "Run Prediction" button
        4. Review prediction results and clinical recommendations
        
        ### 7. Important Disclaimer
        ⚠️ **This system is an opportunistic screening tool and cannot replace DXA gold standard diagnosis**
        Prediction results are for reference only. Final diagnosis should be made by qualified physicians.
        
        ### 8. Citation
        If you use this system in your research, please cite:

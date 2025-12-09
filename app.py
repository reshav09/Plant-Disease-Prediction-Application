"""
Plant Disease Detection Web Application
Streamlit-based interface for crop disease prediction and management.
"""

import streamlit as st
import os
import json
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    load_and_preprocess_image, 
    predict_disease, 
    load_model_and_classes,
    get_confidence_color,
    format_percentage,
    parse_class_name,
    check_image_quality
)
from disease_info import get_disease_info, get_all_crops, get_diseases_by_crop

# Import advanced features
try:
    from gradcam import GradCAM, visualize_gradcam_grid
    GRADCAM_AVAILABLE = True
except:
    GRADCAM_AVAILABLE = False

try:
    from iot_integration import (
        IoTSensorData, 
        DiseaseRiskAssessment,
        integrate_iot_with_prediction,
        generate_simulated_data
    )
    IOT_AVAILABLE = True
except:
    IOT_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection - AI-Powered",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language support
LANGUAGES = {
    'English': {
        'title': '🌾 Plant Disease Detection System',
        'subtitle': 'AI-Powered Crop Disease Identification and Management',
        'upload': 'Upload Plant Image',
        'analyze': 'Analyze Image',
        'results': 'Analysis Results',
        'disease_info': 'Detailed Disease Information',
        'confidence': 'Confidence',
        'crop': 'Crop',
        'prevention': 'Prevention',
        'treatment': 'Treatment',
        'symptoms': 'Symptoms'
    },
    'Hindi': {
        'title': '🌾 पौधों की बीमारी का पता लगाना',
        'subtitle': 'AI-संचालित फसल रोग पहचान और प्रबंधन',
        'upload': 'पौधे की छवि अपलोड करें',
        'analyze': 'छवि का विश्लेषण करें',
        'results': 'विश्लेषण परिणाम',
        'disease_info': 'विस्तृत रोग जानकारी',
        'confidence': 'विश्वास',
        'crop': 'फसल',
        'prevention': 'रोकथाम',
        'treatment': 'उपचार',
        'symptoms': 'लक्षण'
    },
    'Spanish': {
        'title': '🌾 Sistema de Detección de Enfermedades de Plantas',
        'subtitle': 'Identificación y Gestión de Enfermedades de Cultivos con IA',
        'upload': 'Subir Imagen de Planta',
        'analyze': 'Analizar Imagen',
        'results': 'Resultados del Análisis',
        'disease_info': 'Información Detallada de la Enfermedad',
        'confidence': 'Confianza',
        'crop': 'Cultivo',
        'prevention': 'Prevención',
        'treatment': 'Tratamiento',
        'symptoms': 'Síntomas'
    }
}

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .healthy-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
    }
    .diseased-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
    }
    .info-section {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and class names."""
    model_path = 'models/efficientnetb0_best.h5'
    classes_path = 'models/efficientnetb0_classes.json'
    
    # Check if model files exist
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please train the model first using: `python train.py`")
        return None, None
    
    if not os.path.exists(classes_path):
        st.error(f"❌ Classes file not found: {classes_path}")
        return None, None
    
    try:
        model, class_names = load_model_and_classes(model_path, classes_path)
        return model, class_names
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


def create_confidence_chart(top_predictions):
    """Create a bar chart for top predictions."""
    classes = [pred['class'].split('___')[1].replace('_', ' ') if '___' in pred['class'] 
               else pred['class'] for pred in top_predictions]
    confidences = [pred['confidence'] * 100 for pred in top_predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=classes,
            orientation='h',
            marker=dict(
                color=confidences,
                colorscale='RdYlGn',
                showscale=False
            ),
            text=[f"{c:.2f}%" for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top 3 Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Disease Class",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def display_disease_info(disease_class):
    """Display detailed disease information."""
    info = get_disease_info(disease_class)
    
    # Determine if healthy or diseased
    is_healthy = 'healthy' in disease_class.lower()
    
    if is_healthy:
        st.markdown(f"""
        <div class="healthy-box">
            <h3>✅ {info['crop']} - {info['disease']}</h3>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("🎉 Great! Your plant appears to be healthy!")
        
        with st.expander("📋 Maintenance Tips"):
            st.write("**Keep your plant healthy by:**")
            for tip in info['prevention']:
                st.write(f"• {tip}")
    
    else:
        st.markdown(f"""
        <div class="diseased-box">
            <h3>⚠️ {info['crop']} - {info['disease']}</h3>
            <p><strong>Severity:</strong> {info['severity']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning(f"Disease detected: {info['disease']}")
        
        # Description
        st.markdown("### 📖 Description")
        st.write(info['description'])
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["🔴 Symptoms", "🛡️ Prevention", "💊 Treatment"])
        
        with tab1:
            st.write("**Look for these symptoms:**")
            for symptom in info['symptoms']:
                st.write(f"• {symptom}")
        
        with tab2:
            st.write("**Preventive measures:**")
            for prevention in info['prevention']:
                st.write(f"• {prevention}")
        
        with tab3:
            st.write("**Treatment recommendations:**")
            for treatment in info['treatment']:
                st.write(f"• {treatment}")


def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Plant Disease Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Crop Disease Identification and Management</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/plant-under-sun.png", width=80)
        st.title("About")
        st.info("""
        This application uses deep learning to detect and identify plant diseases from leaf images.
        
        **Features:**
        - 🔍 Disease detection
        - 📊 Confidence scores
        - 💡 Treatment recommendations
        - 🛡️ Prevention strategies
        """)
        
        st.markdown("---")
        st.subheader("Supported Crops")
        crops = get_all_crops()
        for crop in crops[:5]:  # Show first 5
            st.write(f"🌱 {crop}")
        if len(crops) > 5:
            with st.expander("Show more..."):
                for crop in crops[5:]:
                    st.write(f"🌱 {crop}")
        
        st.markdown("---")
        st.subheader("Model Info")
        st.write("**Architecture:** EfficientNetB0")
        st.write("**Dataset:** PlantVillage")
        st.write("**Classes:** 38 diseases")
    
    # Load model
    model, class_names = load_model()
    
    if model is None or class_names is None:
        st.error("⚠️ Model not loaded. Please check the error messages above.")
        st.stop()
    
    # Main content
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Plant Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image of a plant leaf",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the plant leaf"
        )
        
        # Sample images option
        st.markdown("**Or try a sample image:**")
        sample_option = st.selectbox(
            "Select a sample",
            ["None", "Sample 1 (Healthy)", "Sample 2 (Diseased)", "Sample 3 (Diseased)"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Check image quality
            is_valid, message = check_image_quality(image)
            if not is_valid:
                st.error(message)
                st.stop()
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... Please wait."):
                    # Preprocess image
                    processed_image, _ = load_and_preprocess_image(uploaded_file)
                    
                    # Make prediction
                    prediction_result = predict_disease(model, processed_image, class_names)
                    
                    # Store results in session state
                    st.session_state['prediction'] = prediction_result
                    st.session_state['image'] = image
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            predicted_class = prediction['predicted_class']
            confidence = prediction['confidence']
            
            # Display prediction
            parsed = parse_class_name(predicted_class)
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #2E7D32;">
                    {parsed['disease']}
                </h2>
                <h3 style="text-align: center; color: #666;">
                    Crop: {parsed['crop']}
                </h3>
                <h3 style="text-align: center; color: {get_confidence_color(confidence)};">
                    Confidence: {format_percentage(confidence)}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            if confidence >= 0.9:
                st.success("✅ High confidence prediction")
            elif confidence >= 0.7:
                st.warning("⚠️ Moderate confidence - consider consulting an expert")
            else:
                st.error("❌ Low confidence - please upload a clearer image")
            
            # Display top 3 predictions chart
            st.plotly_chart(
                create_confidence_chart(prediction['top_3_predictions']),
                use_container_width=True
            )
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Disease information section (full width)
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.subheader("📚 Detailed Disease Information")
        display_disease_info(st.session_state['prediction']['predicted_class'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🌾 Plant Disease Detection System | Powered by Deep Learning & Streamlit</p>
            <p>⚠️ Disclaimer: This tool is for educational purposes. 
            Always consult agricultural experts for critical decisions.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

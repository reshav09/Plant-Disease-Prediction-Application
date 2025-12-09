"""
DEMO/PROTOTYPE VERSION - Plant Disease Detection System
Works without trained model - Shows system capabilities for demonstration
"""

import streamlit as st
import os
from pathlib import Path
from PIL import Image
import random

# Import disease info database
from disease_info import get_disease_info, get_all_crops, DISEASE_INFO

# Demo mode configuration
DEMO_MODE = True

# Simulated predictions based on filename patterns
DEMO_PREDICTIONS = {
    'AppleCedarRust': 'Apple___Cedar_apple_rust',
    'AppleScab': 'Apple___Apple_scab',
    'CornCommonRust': 'Corn_(maize)___Common_rust_',
    'PotatoEarlyBlight': 'Potato___Early_blight',
    'PotatoHealthy': 'Potato___healthy',
    'TomatoEarlyBlight': 'Tomato___Early_blight',
    'TomatoHealthy': 'Tomato___healthy',
    'TomatoYellowCurlVirus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    # Fallback patterns
    'Apple': 'Apple___Apple_scab',
    'Tomato': 'Tomato___Late_blight',
    'Potato': 'Potato___Late_blight',
    'Corn': 'Corn_(maize)___Common_rust_',
    'Pepper': 'Pepper,_bell___Bacterial_spot',
    'Grape': 'Grape___Black_rot',
}


def simulate_prediction(image_name, folder_name=None):
    """
    Simulate a prediction based on folder name (disease class) or image filename
    Returns prediction result similar to real model output
    
    Args:
        image_name: Name of the image file
        folder_name: Name of the folder (disease class) containing the image
    
    Returns:
        dict: Prediction results with class, confidence, and top-3 predictions
    """
    # If folder name is provided (new structure), use it as the disease class
    if folder_name and folder_name in DISEASE_INFO:
        disease_class = folder_name
        confidence = random.uniform(0.88, 0.98)  # High confidence since we know the actual class
    else:
        # Fallback: Try to match filename to known patterns
        disease_class = None
        for pattern, class_name in DEMO_PREDICTIONS.items():
            if pattern.lower() in image_name.lower():
                disease_class = class_name
                confidence = random.uniform(0.85, 0.95)
                break
        
        # If still no match, random prediction
        if disease_class is None:
            disease_class = random.choice(list(DISEASE_INFO.keys()))
            confidence = random.uniform(0.75, 0.90)
    
    # Generate top 3 predictions
    all_classes = list(DISEASE_INFO.keys())
    if disease_class in all_classes:
        all_classes.remove(disease_class)
    random.shuffle(all_classes)
    other_classes = all_classes[:2]
    
    # Calculate remaining confidence for other predictions
    remaining_confidence = 1.0 - confidence
    second_conf = random.uniform(0.4, 0.7) * remaining_confidence
    third_conf = remaining_confidence - second_conf
    
    top_3 = [
        {'class': disease_class, 'confidence': confidence},
        {'class': other_classes[0], 'confidence': second_conf},
        {'class': other_classes[1], 'confidence': third_conf}
    ]
    
    return {
        'predicted_class': disease_class,
        'confidence': confidence,
        'top_3_predictions': top_3
    }


def parse_class_name(class_name):
    """Parse disease class name into crop and disease"""
    if '___' in class_name:
        parts = class_name.split('___')
        crop = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ')
        return {'crop': crop, 'disease': disease}
    return {'crop': 'Unknown', 'disease': class_name.replace('_', ' ')}


def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.9:
        return '#28a745'  # Green
    elif confidence >= 0.7:
        return '#ffc107'  # Yellow
    else:
        return '#dc3545'  # Red


def format_percentage(value):
    """Format confidence as percentage"""
    return f"{value * 100:.2f}%"


# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection - DEMO",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .demo-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        border: 3px solid #ff4444;
        padding: 25px;
        margin: 20px 0;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
        color: white;
    }
    .warning-box strong {
        font-size: 1.3rem;
        display: block;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .warning-box code {
        background: rgba(0,0,0,0.3);
        padding: 8px 12px;
        border-radius: 5px;
        display: block;
        margin-top: 10px;
        font-size: 1rem;
        color: #fff;
        border: 1px solid rgba(255,255,255,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Plant Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><span class="demo-badge">🚀 PROTOTYPE/DEMO MODE</span></div>', unsafe_allow_html=True)
    
    # # Demo mode warning
    # st.markdown("""
    #     <div class="warning-box">
    #         <strong>⚠️ DEMO MODE ACTIVE - PROTOTYPE DEMONSTRATION</strong>
    #         This is a prototype to showcase system capabilities. Predictions are <b>simulated</b> for demonstration purposes only.
    #         <br><br>
    #         <b>📌 To get REAL AI predictions, train a model:</b>
    #         <code>python train.py --model efficientnetb0 --epochs 50</code>
    #         <br>
    #         <small>💡 Training takes 2-4 hours and uses your dataset in data/train/</small>
    #     </div>
    # """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/plant-under-sun.png", width=80)
        st.title("About Demo")
        st.info("""
        **Prototype Features:**
        
        ✨ **Working:**
        - Image upload & display
        - Simulated predictions
        - Disease information
        - Treatment recommendations
        - Interactive interface
        
        📊 **Demo Data:**
        - 38 disease classes
        - 1 sample per class
        - Organized by disease
        - Full disease database
        
        🎯 **Next Steps:**
        1. Choose input method
        2. Browse disease classes OR
        3. Upload your own image
        4. See disease information
        5. Train real model later
        """)
        
        st.markdown("---")
        st.subheader("Test Dataset Info")
        test_dir = Path('data/test')
        if test_dir.exists():
            # Count disease folders and images
            disease_folders = [d for d in test_dir.iterdir() if d.is_dir()]
            total_images = sum(len(list(folder.glob('*.JPG')) + list(folder.glob('*.jpg')) + list(folder.glob('*.png'))) 
                             for folder in disease_folders)
            st.success(f"✅ **{len(disease_folders)} disease classes**")
            st.success(f"✅ **{total_images} test images**")
            st.info(f"📊 ~{total_images // len(disease_folders) if disease_folders else 0} images per class")
        else:
            st.warning("⚠️ No test images found")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Plant Image")
        
        # Selection mode
        upload_mode = st.radio(
            "Choose input method:",
            ["📤 Upload My Own Image", "📁 Use Sample Test Image"],
            horizontal=True
        )
        
        uploaded_file = None
        selected_folder = None
        
        if upload_mode == "📁 Use Sample Test Image":
            test_dir = Path('data/test')
            if test_dir.exists():
                # Get all disease folders
                disease_folders = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
                
                if disease_folders:
                    # Select disease class folder
                    selected_folder = st.selectbox(
                        "Select disease class:",
                        disease_folders,
                        format_func=lambda x: x.replace('___', ' - ').replace('_', ' ')
                    )
                    
                    if selected_folder:
                        folder_path = test_dir / selected_folder
                        # Get only first image from each folder as sample
                        all_images = list(folder_path.glob('*.JPG')) + list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
                        
                        if all_images:
                            # Use only the first image as sample
                            uploaded_file = all_images[0]
                            image = Image.open(uploaded_file)
                            st.image(image, caption=f"Sample: {selected_folder.replace('___', ' - ').replace('_', ' ')}", use_container_width=True)
                            st.info(f"📷 Using sample image from this class")
                        else:
                            st.warning(f"No images found in {selected_folder}")
                else:
                    st.warning("No disease folders found in data/test")
        else:
            # Manual upload
            uploaded_file = st.file_uploader(
                "Choose an image of a plant leaf",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of the plant leaf"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    # Simulate prediction
                    image_name = uploaded_file.name if hasattr(uploaded_file, 'name') else str(uploaded_file.name)
                    
                    # Pass folder name if available (for organized test images)
                    is_sample = (upload_mode == "📁 Use Sample Test Image")
                    prediction_result = simulate_prediction(image_name, selected_folder if is_sample else None)
                    
                    # Store in session state
                    st.session_state['prediction'] = prediction_result
                    st.session_state['image_name'] = image_name
                    st.rerun()
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            predicted_class = prediction['predicted_class']
            confidence = prediction['confidence']
            
            parsed = parse_class_name(predicted_class)
            
            # Display results
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="text-align: center; color: #2E7D32; margin: 0;">
                    {parsed['disease']}
                </h2>
                <h3 style="text-align: center; color: #666; margin: 10px 0;">
                    Crop: {parsed['crop']}
                </h3>
                <h3 style="text-align: center; color: {get_confidence_color(confidence)};">
                    Confidence: {format_percentage(confidence)}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            if confidence >= 0.9:
                st.success("✅ High confidence prediction")
            elif confidence >= 0.7:
                st.warning("⚠️ Moderate confidence")
            else:
                st.error("❌ Low confidence - Consider expert review")
            
            # Top 3 predictions
            st.subheader("Top 3 Predictions")
            for i, pred in enumerate(prediction['top_3_predictions'], 1):
                parsed_pred = parse_class_name(pred['class'])
                conf_pct = pred['confidence'] * 100
                st.write(f"{i}. **{parsed_pred['disease']}** ({parsed_pred['crop']}) - {conf_pct:.2f}%")
                st.progress(pred['confidence'])
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Disease Information
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.subheader("📚 Disease Information & Recommendations")
        
        predicted_class = st.session_state['prediction']['predicted_class']
        confidence = st.session_state['prediction']['confidence']
        
        info = get_disease_info(predicted_class)
        
        # Tabs for different information
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Overview",
            "🛡️ Prevention",
            "💊 Treatment",
            "🌡️ Symptoms"
        ])
        
        with tab1:
            st.write("**Description:**")
            st.write(info['description'])
            st.write(f"\n**Severity Level:** {info.get('severity', 'Unknown')}")
            
            if confidence >= 0.9:
                st.success("✅ **Recommendation:** High confidence - proceed with suggested treatments")
            elif confidence >= 0.7:
                st.warning("⚠️ **Recommendation:** Moderate confidence - consider expert verification")
            else:
                st.error("❌ **Recommendation:** Low confidence - expert inspection strongly recommended")
        
        with tab2:
            st.write("**Prevention Measures:**")
            for i, prevention in enumerate(info['prevention'], 1):
                st.write(f"{i}. {prevention}")
        
        with tab3:
            st.write("**Treatment Options:**")
            for i, treatment in enumerate(info['treatment'], 1):
                st.write(f"{i}. {treatment}")
        
        with tab4:
            st.write("**Symptoms to Look For:**")
            for i, symptom in enumerate(info['symptoms'], 1):
                st.write(f"{i}. {symptom}")
    
    # Disease Database Browser
    st.markdown("---")
    st.subheader("📚 Browse Disease Database")
    
    crops = get_all_crops()
    selected_crop = st.selectbox("Select a crop to view diseases:", [''] + crops)
    
    if selected_crop:
        crop_diseases = [
            {'class': k, 'disease': v['disease'], 'severity': v.get('severity', 'Unknown')}
            for k, v in DISEASE_INFO.items()
            if v['crop'] == selected_crop
        ]
        
        st.write(f"**Diseases affecting {selected_crop}:** ({len(crop_diseases)} found)")
        
        for disease in crop_diseases:
            with st.expander(f"{disease['disease']} (Severity: {disease['severity']})"):
                info = get_disease_info(disease['class'])
                st.write(f"**Description:** {info['description']}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Key Symptoms:**")
                    for symptom in info['symptoms'][:3]:
                        st.write(f"• {symptom}")
                
                with col_b:
                    st.write("**Quick Actions:**")
                    for action in info['treatment'][:3]:
                        st.write(f"• {action}")
    
    # Footer
    st.markdown("---")
    
    # # Interactive Footer with multiple sections
    # footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    # with footer_col1:
    #     st.markdown("### 📊 Quick Stats")
    #     stats_data = {
    #         "Disease Classes": 38,
    #         "Crops Supported": 14,
    #         "Model Accuracy": "90-95%",
    #         "Prediction Time": "< 2s"
    #     }
    #     for key, value in stats_data.items():
    #         st.metric(label=key, value=value)
    
    # with footer_col2:
    #     st.markdown("### 🚀 Get Started")
    #     st.markdown("""
    #     **Train Your Model:**
    #     ```bash
    #     python train.py --model efficientnetb0 --epochs 50
    #     ```
        
    #     **Run Full App:**
    #     ```bash
    #     streamlit run app_advanced.py
    #     ```
    #     """)
        
    #     if st.button("📖 View Documentation", use_container_width=True):
    #         st.info("Check **START_HERE.md** and **README_SIMPLE.md** in the project folder for detailed guides!")
    
    # with footer_col3:
    #     st.markdown("### ℹ️ About System")
    #     st.markdown("""
    #     **Tech Stack:**
    #     - 🧠 TensorFlow/Keras
    #     - 🌐 Streamlit
    #     - 🐍 Python 3.8-3.11
        
    #     **Features:**
    #     - ✅ Real-time detection
    #     - ✅ Treatment recommendations
    #     - ✅ Disease database
    #     - ✅ Mobile-friendly UI
    #     """)
    
    # # Bottom bar with important info
    # st.markdown("---")
    # st.markdown("""
    #     <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    #                 padding: 20px; 
    #                 border-radius: 10px; 
    #                 text-align: center;
    #                 color: white;
    #                 margin-top: 20px;">
    #         <h3 style="margin: 0; color: white;">🌾 Plant Disease Detection System</h3>
    #         <p style="margin: 10px 0 0 0; opacity: 0.9;">
    #             💡 <strong>DEMO MODE</strong> - Helping farmers detect crop diseases early and save yields
    #         </p>
    #         <p style="margin: 10px 0 0 0; font-size: 0.9em; opacity: 0.8;">
    #             🎓 Educational Prototype | ⚠️ Train model for production | 🌍 Making agriculture smarter
    #         </p>
    #     </div>
    # """, unsafe_allow_html=True)
    
    # Expandable sections for additional info
    # st.markdown("<br>", unsafe_allow_html=True)
    
    # with st.expander("💻 System Requirements"):
    #     st.markdown("""
    #     **Minimum Requirements:**
    #     - Python 3.8 or higher
    #     - 8GB RAM
    #     - 2GB free disk space
        
    #     **Recommended:**
    #     - Python 3.11
    #     - 16GB RAM
    #     - GPU (NVIDIA CUDA) for faster training
    #     - 10GB free disk space
    #     """)
    
    with st.expander("📚 Supported Crops & Diseases"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Crops:**
            - 🍎 Apple
            - 🌽 Corn (Maize)
            - 🍅 Tomato
            - 🥔 Potato
            - 🍇 Grape
            - 🌶️ Pepper
            - 🍑 Peach
            """)
        with col_b:
            st.markdown("""
            **More Crops:**
            - 🍒 Cherry
            - 🍓 Strawberry
            - 🫐 Blueberry
            - 🍊 Orange
            - 🌿 Soybean
            - 🎃 Squash
            - 🍇 Raspberry
            """)
    
    # with st.expander("🎯 Training Tips"):
    #     st.markdown("""
    #     **For Best Results:**
        
    #     1. **Data Preparation:**
    #        - Use high-quality images (224x224 or higher)
    #        - Ensure balanced dataset (similar images per class)
    #        - Remove corrupted or low-quality images
        
    #     2. **Model Selection:**
    #        - `efficientnetb0` - Best balance (Recommended)
    #        - `mobilenetv2` - Fastest, mobile-friendly
    #        - `resnet50` - High accuracy, larger size
        
    #     3. **Training Parameters:**
    #        - Start with 30-50 epochs
    #        - Use early stopping (patience: 10)
    #        - Monitor validation accuracy
    #        - Save best model only
        
    #     4. **Performance Optimization:**
    #        - Use GPU if available (10x faster)
    #        - Batch size: 32 (adjust for your RAM)
    #        - Enable data augmentation
    #        - Use learning rate scheduling
    #     """)
    
    # with st.expander("❓ FAQ"):
    #     st.markdown("""
    #     **Q: Why are predictions simulated?**  
    #     A: This is a demo version. Train a model with `python train.py` for real predictions.
        
    #     **Q: How accurate is the real model?**  
    #     A: With proper training, expect 90-95% validation accuracy on PlantVillage dataset.
        
    #     **Q: Can I use my own dataset?**  
    #     A: Yes! Place images in `data/train/[disease_class]/` folders and run training.
        
    #     **Q: How long does training take?**  
    #     A: 2-4 hours with GPU, 8-12 hours with CPU (depends on dataset size).
        
    #     **Q: Is this production-ready?**  
    #     A: After training, yes! The advanced app includes Grad-CAM visualization and batch processing.
        
    #     **Q: Can I deploy this online?**  
    #     A: Yes! Deploy on Streamlit Cloud, Heroku, or AWS. Check deployment guides.
    #     """)

    st.markdown("""
            <div style="text-align: center; color: #666; padding: 20px;">
                <p>🌾 Plant Disease Detection System - Prototype Demo</p>
                <p>🎓 For demonstration and testing purposes</p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

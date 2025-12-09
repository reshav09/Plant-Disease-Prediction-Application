"""
Disease Information Database
Contains detailed information about plant diseases, symptoms, and management strategies.
"""

DISEASE_INFO = {
    # Apple Diseases
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "description": "A fungal disease caused by Venturia inaequalis that affects apple trees, causing dark, scabby lesions on leaves, fruits, and sometimes young twigs.",
        "symptoms": [
            "Olive-green to dark brown spots on leaves",
            "Scabby lesions on fruits",
            "Premature leaf drop",
            "Deformed or cracked fruits"
        ],
        "prevention": [
            "Plant resistant apple varieties",
            "Remove fallen leaves and infected plant debris",
            "Ensure good air circulation by proper pruning",
            "Apply preventive fungicides in early spring"
        ],
        "treatment": [
            "Apply fungicides (captan, myclobutanil) during growing season",
            "Remove and destroy infected leaves and fruits",
            "Use sulfur-based fungicides for organic control"
        ],
        "severity": "Moderate to High"
    },
    
    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Black Rot",
        "description": "A fungal disease caused by Botryosphaeria obtusa that causes leaf spot, fruit rot, and limb cankers.",
        "symptoms": [
            "Purple spots on leaves that turn brown",
            "Concentric rings on infected fruits",
            "Mummified fruits on tree",
            "Cankers on branches"
        ],
        "prevention": [
            "Prune dead and diseased branches",
            "Remove mummified fruits",
            "Maintain tree health with proper fertilization",
            "Avoid tree injuries"
        ],
        "treatment": [
            "Apply fungicides during bloom and fruit development",
            "Remove infected fruits and branches",
            "Improve orchard sanitation"
        ],
        "severity": "High"
    },
    
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "description": "A fungal disease requiring both apple and cedar trees to complete its life cycle, causing orange spots on leaves.",
        "symptoms": [
            "Bright orange spots on upper leaf surface",
            "Yellow-orange lesions on fruits",
            "Premature leaf drop",
            "Reduced fruit quality"
        ],
        "prevention": [
            "Remove nearby cedar trees if possible",
            "Plant resistant apple varieties",
            "Apply preventive fungicides in spring"
        ],
        "treatment": [
            "Apply fungicides (myclobutanil, propiconazole)",
            "Remove infected plant parts",
            "Control alternate hosts"
        ],
        "severity": "Moderate"
    },
    
    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "description": "The plant shows no signs of disease and appears healthy.",
        "symptoms": ["No visible symptoms"],
        "prevention": [
            "Maintain regular monitoring",
            "Ensure proper nutrition and watering",
            "Practice good sanitation"
        ],
        "treatment": ["No treatment needed"],
        "severity": "None"
    },
    
    # Corn Diseases
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "disease": "Gray Leaf Spot",
        "description": "A fungal disease caused by Cercospora zeae-maydis that causes rectangular gray lesions on corn leaves.",
        "symptoms": [
            "Rectangular gray to brown lesions on leaves",
            "Lesions parallel to leaf veins",
            "Severe defoliation in advanced stages",
            "Reduced grain yield"
        ],
        "prevention": [
            "Plant resistant hybrids",
            "Practice crop rotation",
            "Till crop residue after harvest",
            "Ensure proper plant spacing"
        ],
        "treatment": [
            "Apply fungicides at early disease stages",
            "Use azoxystrobin or propiconazole",
            "Remove severely infected plants"
        ],
        "severity": "High"
    },
    
    "Corn_(maize)___Common_rust": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "description": "A fungal disease caused by Puccinia sorghi that produces rusty brown pustules on corn leaves.",
        "symptoms": [
            "Circular to elongated brown pustules on leaves",
            "Pustules release reddish-brown spores",
            "Yellowing and drying of leaves",
            "Reduced photosynthesis"
        ],
        "prevention": [
            "Plant resistant corn varieties",
            "Proper plant spacing for air circulation",
            "Early planting to avoid peak infection period"
        ],
        "treatment": [
            "Apply fungicides if disease is severe",
            "Use triazole fungicides",
            "Remove infected plant debris"
        ],
        "severity": "Moderate"
    },
    
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "description": "A fungal disease caused by Exserohilum turcicum that produces long, cigar-shaped lesions on corn leaves.",
        "symptoms": [
            "Long, elliptical gray-green lesions",
            "Lesions turn tan with age",
            "Severe defoliation possible",
            "Reduced grain quality and yield"
        ],
        "prevention": [
            "Use resistant hybrids",
            "Practice crop rotation (2-3 years)",
            "Bury crop residue by tillage",
            "Balanced fertilization"
        ],
        "treatment": [
            "Apply fungicides at early infection",
            "Use strobilurin or triazole fungicides",
            "Multiple applications may be needed"
        ],
        "severity": "High"
    },
    
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "disease": "Healthy",
        "description": "The plant shows no signs of disease and appears healthy.",
        "symptoms": ["No visible symptoms"],
        "prevention": [
            "Maintain regular monitoring",
            "Ensure proper nutrition",
            "Practice crop rotation"
        ],
        "treatment": ["No treatment needed"],
        "severity": "None"
    },
    
    # Grape Diseases
    "Grape___Black_rot": {
        "crop": "Grape",
        "disease": "Black Rot",
        "description": "A serious fungal disease caused by Guignardia bidwellii affecting all green parts of the grape vine.",
        "symptoms": [
            "Circular tan spots on leaves with dark borders",
            "Black, mummified fruits",
            "Brown lesions on shoots and tendrils",
            "Severe fruit loss"
        ],
        "prevention": [
            "Remove mummified fruits and infected canes",
            "Ensure good air circulation",
            "Apply preventive fungicides",
            "Practice proper vineyard sanitation"
        ],
        "treatment": [
            "Apply fungicides from bud break to harvest",
            "Use mancozeb or captan",
            "Remove infected plant materials"
        ],
        "severity": "High"
    },
    
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "disease": "Esca (Black Measles)",
        "description": "A complex disease involving multiple fungi that affects the wood and leaves of grapevines.",
        "symptoms": [
            "Tiger-stripe pattern on leaves (yellowing between veins)",
            "Dark streaks in wood",
            "Shriveled, darkened berries",
            "Sudden vine collapse possible"
        ],
        "prevention": [
            "Use clean pruning tools",
            "Avoid large pruning wounds",
            "Maintain vine health and vigor",
            "Remove infected vines"
        ],
        "treatment": [
            "No effective chemical treatment available",
            "Prune out infected wood",
            "Support vine health through proper management",
            "Consider trunk renewal in severe cases"
        ],
        "severity": "Very High"
    },
    
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "disease": "Leaf Blight",
        "description": "A fungal disease causing leaf spots and defoliation in grapes.",
        "symptoms": [
            "Dark brown to black spots on leaves",
            "Spots with yellow halos",
            "Premature leaf drop",
            "Reduced photosynthesis"
        ],
        "prevention": [
            "Improve air circulation through pruning",
            "Remove infected leaves",
            "Apply preventive fungicides",
            "Avoid overhead irrigation"
        ],
        "treatment": [
            "Apply copper-based fungicides",
            "Use systemic fungicides in severe cases",
            "Remove and destroy infected leaves"
        ],
        "severity": "Moderate"
    },
    
    "Grape___healthy": {
        "crop": "Grape",
        "disease": "Healthy",
        "description": "The plant shows no signs of disease and appears healthy.",
        "symptoms": ["No visible symptoms"],
        "prevention": [
            "Maintain proper pruning and training",
            "Ensure adequate nutrition",
            "Monitor regularly for pests and diseases"
        ],
        "treatment": ["No treatment needed"],
        "severity": "None"
    },
    
    # Tomato Diseases
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease caused by Xanthomonas species that affects leaves, stems, and fruits of tomato plants.",
        "symptoms": [
            "Small dark brown spots on leaves",
            "Raised spots on fruits",
            "Yellow halos around leaf spots",
            "Severe defoliation possible"
        ],
        "prevention": [
            "Use disease-free seeds and transplants",
            "Practice crop rotation (3 years)",
            "Avoid overhead irrigation",
            "Use drip irrigation"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Remove infected plants",
            "Use resistant varieties",
            "Improve sanitation practices"
        ],
        "severity": "High"
    },
    
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani that affects older leaves first, causing concentric ring patterns.",
        "symptoms": [
            "Dark brown spots with concentric rings (target pattern)",
            "Yellow halo around spots",
            "Lower leaves affected first",
            "Stem lesions near soil line"
        ],
        "prevention": [
            "Practice crop rotation",
            "Remove infected plant debris",
            "Mulch around plants to prevent soil splash",
            "Ensure proper plant spacing"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb)",
            "Remove infected lower leaves",
            "Use organic options like copper fungicides"
        ],
        "severity": "Moderate to High"
    },
    
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "description": "A devastating disease caused by Phytophthora infestans, the same organism that caused the Irish potato famine.",
        "symptoms": [
            "Large, greasy-looking dark brown spots on leaves",
            "White fungal growth under leaves in humid conditions",
            "Rapid plant collapse",
            "Brown, firm lesions on fruits"
        ],
        "prevention": [
            "Use resistant varieties",
            "Avoid overhead watering",
            "Ensure good air circulation",
            "Remove volunteer tomato and potato plants"
        ],
        "treatment": [
            "Apply fungicides immediately (chlorothalonil, mancozeb)",
            "Remove and destroy infected plants",
            "Do not compost infected material"
        ],
        "severity": "Very High"
    },
    
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "description": "A fungal disease caused by Passalora fulva, common in greenhouse tomatoes with high humidity.",
        "symptoms": [
            "Pale green to yellow spots on upper leaf surface",
            "Olive-green to brown fuzzy growth on lower surface",
            "Older leaves affected first",
            "Reduced yield but fruits not usually infected"
        ],
        "prevention": [
            "Reduce humidity in greenhouses",
            "Ensure good ventilation",
            "Space plants properly",
            "Use resistant varieties"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb)",
            "Remove infected leaves",
            "Improve air circulation"
        ],
        "severity": "Moderate"
    },
    
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "description": "A fungal disease caused by Septoria lycopersici that affects tomato foliage.",
        "symptoms": [
            "Small circular spots with dark borders",
            "Gray center with tiny black dots (spore structures)",
            "Lower leaves affected first",
            "Progressive defoliation from bottom up"
        ],
        "prevention": [
            "Rotate crops",
            "Remove infected plant debris",
            "Mulch to prevent soil splash",
            "Stake or cage plants"
        ],
        "treatment": [
            "Apply fungicides regularly",
            "Remove infected lower leaves",
            "Ensure plants are properly staked"
        ],
        "severity": "Moderate"
    },
    
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Spider Mites",
        "description": "Not a disease but a pest infestation. Tiny arachnids that suck plant juices, causing stippling and webbing.",
        "symptoms": [
            "Fine stippling or speckling on leaves",
            "Yellowing and bronzing of leaves",
            "Fine webbing on undersides of leaves",
            "Leaves may dry and drop"
        ],
        "prevention": [
            "Maintain adequate soil moisture",
            "Avoid water stress",
            "Encourage beneficial insects",
            "Regular monitoring"
        ],
        "treatment": [
            "Use insecticidal soap or neem oil",
            "Spray water to dislodge mites",
            "Use miticides in severe infestations",
            "Introduce predatory mites"
        ],
        "severity": "Moderate"
    },
    
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "description": "A fungal disease caused by Corynespora cassiicola producing concentric ring patterns on leaves and fruits.",
        "symptoms": [
            "Brown spots with concentric rings on leaves",
            "Dark lesions on stems and fruits",
            "Spots may have yellow halos",
            "Severe defoliation in warm, humid conditions"
        ],
        "prevention": [
            "Use disease-free transplants",
            "Practice crop rotation",
            "Ensure good air circulation",
            "Avoid overhead irrigation"
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, azoxystrobin)",
            "Remove infected plant parts",
            "Maintain proper plant spacing"
        ],
        "severity": "Moderate to High"
    },
    
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "description": "A viral disease transmitted by whiteflies causing severe stunting and yield loss.",
        "symptoms": [
            "Upward curling of leaf margins",
            "Yellowing of leaf edges",
            "Severe stunting of plants",
            "Reduced fruit size and yield"
        ],
        "prevention": [
            "Control whitefly populations",
            "Use insect-proof screens in greenhouses",
            "Remove infected plants immediately",
            "Plant resistant varieties"
        ],
        "treatment": [
            "No cure available for viral infections",
            "Control whitefly vectors with insecticides",
            "Remove and destroy infected plants",
            "Use reflective mulches to repel whiteflies"
        ],
        "severity": "Very High"
    },
    
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "description": "A viral disease causing mottled patterns on leaves and reduced plant vigor.",
        "symptoms": [
            "Light and dark green mottled pattern on leaves",
            "Distorted, fern-like leaves",
            "Stunted plant growth",
            "Reduced fruit quality"
        ],
        "prevention": [
            "Use virus-free seeds and transplants",
            "Disinfect tools and hands",
            "Control aphids",
            "Remove infected plants"
        ],
        "treatment": [
            "No chemical treatment available",
            "Remove infected plants immediately",
            "Control insect vectors",
            "Practice good sanitation"
        ],
        "severity": "High"
    },
    
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "description": "The plant shows no signs of disease and appears healthy.",
        "symptoms": ["No visible symptoms"],
        "prevention": [
            "Maintain proper watering and fertilization",
            "Monitor regularly for pests and diseases",
            "Ensure good air circulation"
        ],
        "treatment": ["No treatment needed"],
        "severity": "None"
    },
    
    # Potato Diseases
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "description": "A fungal disease caused by Alternaria solani affecting potato foliage and tubers.",
        "symptoms": [
            "Dark brown spots with concentric rings on leaves",
            "Yellow halo around spots",
            "Lower leaves affected first",
            "Dark, sunken lesions on tubers"
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Practice 3-4 year crop rotation",
            "Hill soil around plants",
            "Remove volunteer potato plants"
        ],
        "treatment": [
            "Apply fungicides preventively",
            "Use chlorothalonil or mancozeb",
            "Remove infected foliage"
        ],
        "severity": "Moderate to High"
    },
    
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "description": "A devastating disease caused by Phytophthora infestans that can destroy entire potato crops.",
        "symptoms": [
            "Water-soaked lesions on leaves",
            "White fungal growth on undersides",
            "Rapid blackening and death of foliage",
            "Brown rot in tubers"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Use certified seed potatoes",
            "Avoid overhead irrigation",
            "Hill plants properly to protect tubers"
        ],
        "treatment": [
            "Apply fungicides immediately upon detection",
            "Destroy infected plants",
            "Do not harvest affected tubers"
        ],
        "severity": "Very High"
    },
    
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "description": "The plant shows no signs of disease and appears healthy.",
        "symptoms": ["No visible symptoms"],
        "prevention": [
            "Use certified seed potatoes",
            "Practice crop rotation",
            "Monitor regularly"
        ],
        "treatment": ["No treatment needed"],
        "severity": "None"
    },
    
    # Pepper Diseases
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Pepper (Bell)",
        "disease": "Bacterial Spot",
        "description": "A bacterial disease affecting pepper plants, causing leaf spots and fruit lesions.",
        "symptoms": [
            "Small, dark brown spots on leaves",
            "Raised, scabby lesions on fruits",
            "Defoliation in severe cases",
            "Reduced yield and fruit quality"
        ],
        "prevention": [
            "Use disease-free seeds and transplants",
            "Practice crop rotation",
            "Avoid overhead irrigation",
            "Remove plant debris"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Remove severely infected plants",
            "Use resistant varieties"
        ],
        "severity": "High"
    },
    
    "Pepper,_bell___healthy": {
        "crop": "Pepper (Bell)",
        "disease": "Healthy",
        "description": "The plant shows no signs of disease and appears healthy.",
        "symptoms": ["No visible symptoms"],
        "prevention": [
            "Maintain proper care and nutrition",
            "Monitor for pests and diseases",
            "Ensure adequate watering"
        ],
        "treatment": ["No treatment needed"],
        "severity": "None"
    }
}


def get_disease_info(disease_class):
    """
    Get detailed information about a disease.
    
    Args:
        disease_class: Disease class name
        
    Returns:
        Dictionary with disease information
    """
    return DISEASE_INFO.get(disease_class, {
        "crop": "Unknown",
        "disease": "Unknown Disease",
        "description": "No information available for this disease.",
        "symptoms": ["Information not available"],
        "prevention": ["Consult local agricultural extension service"],
        "treatment": ["Consult local agricultural extension service"],
        "severity": "Unknown"
    })


def get_all_crops():
    """Get list of all crops in the database."""
    crops = set()
    for info in DISEASE_INFO.values():
        crops.add(info["crop"])
    return sorted(list(crops))


def get_diseases_by_crop(crop_name):
    """Get all diseases for a specific crop."""
    diseases = []
    for class_name, info in DISEASE_INFO.items():
        if info["crop"] == crop_name:
            diseases.append({
                "class": class_name,
                "disease": info["disease"],
                "severity": info["severity"]
            })
    return diseases


if __name__ == "__main__":
    # Test the database
    print("🌾 Plant Disease Information Database")
    print("=" * 50)
    print(f"\nTotal diseases in database: {len(DISEASE_INFO)}")
    print(f"\nCrops covered: {', '.join(get_all_crops())}")
    
    # Example usage
    test_disease = "Tomato___Early_blight"
    info = get_disease_info(test_disease)
    print(f"\n\nExample - {info['disease']} ({info['crop']}):")
    print(f"Severity: {info['severity']}")
    print(f"Description: {info['description']}")

import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import gdown
import time
# ✅ Step 5: Streamlit UI
st.set_page_config(page_title="Waste Classification", layout="wide")

st.title("♻️ Waste Sorting Deep Learning  Model")
# Constants
MODEL_PATH = "mymodel.h5"
FILE_ID = "1UMKoI_y3mZhOyFi-RurNNuyOsqT6BcuX"  # Replace with your Google Drive file_id
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Function to download and load the model
@st.cache_resource
def load_model(path):
    """Loads a TensorFlow/Keras model from local path"""
    return tf.keras.models.load_model(path, compile=False)

# Check if model exists
if os.path.exists(MODEL_PATH):
    st.success("Model is already downloaded ✅")
    model = load_model(MODEL_PATH)
    st.write("Model loaded successfully!")
else:
    st.warning("Model not downloaded yet ❌")
    
    if st.button("Download Model"):
        st.info("📥 Downloading model... This may take a while.")
        progress_bar = st.progress(0)
        
        # Download model from Google Drive
        gdown.download(URL, MODEL_PATH, quiet=False)
        
        # Fake progress for visual feedback
        for i in range(101):
            time.sleep(0.01)
            progress_bar.progress(i)
        
        st.success("✅ Download complete!")
        
        # Load model after download
        model = load_model(MODEL_PATH)
        st.write("Model loaded successfully!")

st.write("Upload an image of waste and the model will sort it.")
# ✅ Step 3: Define waste classes
# List of waste classes
output_class = ["Battery", "Electronic Waste", "Glass", "Metal", "Organic Waste", "Plastic"]

# Waste class details
class_info = {
"Battery": {
        "description": "**Battery** includes materials that are toxic, flammable, or corrosive, posing risks to human health and the environment.",
        "example_items": [
            "Batteries (lithium-ion, lead-acid)",
            "Pesticides and fertilizers",
            "Paints, solvents, and industrial chemicals",
            "Medical waste (syringes, expired medications)"
        ],
        "disposal_tips": [
            "⚠️ **Do Not Mix with Regular Trash:** Hazardous materials require special handling.",
            "⚠️ **Use Certified Disposal Centers:** Many cities offer hazardous waste collection points.",
            "⚠️ **Switch to Eco-Friendly Alternatives:** Opt for natural cleaning agents and organic pesticides."
        ],
        "youtube_link": "https://youtu.be/f9zkRa4tG-g?si=3Lqm6scdX1QZ5bMZ",
        "blog_link": "https://www.epa.gov/hw/household-hazardous-waste-hhw"
    },

    "Electronic Waste": {
        "description": "**Electronic waste (E-Waste)** includes discarded electronic devices containing harmful substances like lead and mercury.",
        "example_items": [
            "Mobile phones, chargers, and tablets",
            "Laptops, desktops, and computer peripherals",
            "Televisions and monitors",
            "Household appliances (microwaves, toasters, refrigerators)"
        ],
        "disposal_tips": [
            "✅ **E-Waste Recycling Centers:** Many manufacturers offer take-back programs.",
            "✅ **Donation:** Working electronics can be donated to schools or NGOs.",
            "✅ **Recycling Precious Metals:** Specialized facilities extract valuable metals like gold and copper."
        ],
        "youtube_link": "https://youtu.be/U3KUJTDPsSE?si=saZHlWv1j4MuZBea",
        "blog_link": "https://earth911.com/recycling-guide/how-to-recycle-electronics/"
    },

    "Glass": {
        "description": "**Glass waste** includes used glass products that can be melted and reshaped multiple times without losing quality.",
        "example_items": [
            "Glass bottles (wine bottles, soda bottles)",
            "Broken glassware (jars, mirrors, windows)",
            "Light bulbs and fluorescent tubes"
        ],
        "disposal_tips": [
            "✅ **Recycling:** Glass can be processed and reused infinitely without degrading in quality.",
            "✅ **Reuse:** Bottles and jars can be cleaned and repurposed.",
            "✅ **Avoid Landfills:** Glass takes thousands of years to decompose, so recycling is crucial."
        ],
        "youtube_link": "https://youtu.be/xj5Fgg-tuzo?si=_lWb2eDSBLShlBKf",
        "blog_link": "https://www.nrdc.org/stories/glass-recycling-explained"
    },

    "Metal": {
        "description": "**Metal waste** consists of discarded metal products, many of which can be recycled indefinitely.",
        "example_items": [
            "Aluminum cans (soda cans, beer cans)",
            "Scrap iron and steel (car parts, tools)",
            "Copper wires and plumbing materials",
            "Metal packaging"
        ],
        "disposal_tips": [
            "✅ **Recycling:** Scrap metal can be melted and reused for manufacturing.",
            "✅ **Selling to Scrap Dealers:** Metal waste can be sold to scrap yards for repurposing.",
            "✅ **Avoid Dumping:** Improper disposal can lead to heavy metal pollution."
        ],
        "youtube_link": "https://youtu.be/_ErocQ2S080?si=0b2pkqz382jGYAF4",
        "blog_link": "https://www.thebalancesmb.com/metal-recycling-facts-2877923"
    },

"Organic Waste": {
        "description": "**Organic Waste** consists of organic materials that decompose naturally.",
        "example_items": [
            "Food waste (fruit peels, vegetable scraps, leftover meals)",
            "Garden waste (grass clippings, leaves, small branches)",
            "Paper-based products (newspapers, tissue paper, coffee filters)",
            "Animal waste and manure"
        ],
        "disposal_tips": [
            "✅ **Composting:** Converts biodegradable waste into nutrient-rich soil amendments.",
            "✅ **Biogas Production:** Organic waste can be used in biogas plants to generate renewable energy.",
            "✅ **Landfilling (Controlled):** If not composted, manage in controlled landfill sites to minimize methane emissions."
        ],
        "youtube_link": "https://youtu.be/2I8Tjb4Fy-Q?si=cUbgM6950fHEB3sr",
        "blog_link": "https://www.epa.gov/recycle/composting-home"
    },

    "Plastic": {
        "description": "**Plastic waste** consists of discarded plastic materials, many of which do not decompose for hundreds of years.",
        "example_items": [
            "Plastic bottles and food containers",
            "Shopping bags and packaging",
            "Straws, utensils, and disposable cutlery"
        ],
        "disposal_tips": [
            "✅ **Recycle Plastics:** Sort and recycle according to plastic type.",
            "✅ **Use Reusable Alternatives:** Replace single-use plastics with glass or stainless steel.",
            "✅ **Avoid Dumping:** Plastics harm marine life and ecosystems."
        ],
        "youtube_link": "https://youtu.be/M5Ml6Po4d9Q?si=uZPO_ieRo7wMsj2U",
        "blog_link": "https://plasticpollutioncoalition.org/"
    }}

# ✅ Step 4: Waste classification function
def waste_prediction(uploaded_file):
    # Load and preprocess the image
    test_image = Image.open(uploaded_file)
    plt.axis("off")
    plt.imshow(test_image)
    plt.show()

    test_image = test_image.resize((224, 224))  # Resize for model input
    test_image = np.array(test_image) / 255.0   # Normalize
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

    # Make prediction
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)

    return predicted_value, predicted_accuracy

# ✅ Step 6: Upload image
# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open and resize image for display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)  # Adjust width as needed    
    # Predict the waste class
    class_name, confidence = waste_prediction(uploaded_file)

    # Display prediction result
    st.success(f"Predicted Waste Class: **{class_name}** with {confidence}% accuracy")


     # Sidebar for waste class details
    with st.sidebar:
        st.header("🗑️ Waste Information")
        
        if class_name in class_info:
            info = class_info[class_name]
            st.subheader(class_name)
            st.write(info["description"])

            st.subheader("📋 Example Items")
            for item in info["example_items"]:
                st.write(f"- {item}")

            st.subheader("♻️ Proper Disposal Tips")
            for tip in info["disposal_tips"]:
                st.write(tip)

            st.subheader("📚 Learn More")
            st.markdown(f"[📺 Watch on YouTube]({info['youtube_link']})", unsafe_allow_html=True)
            st.markdown(f"[📖 Read More]({info['blog_link']})", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        padding: 10px;
        text-align: center;
        font-size: 14px;
        color: black;
        box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
    }
    .footer img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
    }
    .tech-icons img {
        width: 35px;
        height: 35px;
        margin: 5px;
    }
    </style>
    
    <div class="footer">
        <img src="C:/Users/DELL/Videos/Model Deployment/images/Mirason.jpg" alt="Profile Image">
        <span>Developed by <b>Nwankwo Nazarene Chisom</b></span>
        <br>
        <div class="tech-icons">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python">
            <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="TensorFlow">
            <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Pillow_logo.svg" alt="PIL">
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/3f/Numpy_logo_2020.svg" alt="NumPy">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

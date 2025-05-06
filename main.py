import streamlit as st
import tensorflow as tf
import numpy as np
import base64

def set_home_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background:url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    
def set_sidebar_text_color(color="#32a834"):
    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] * {{
        color: {color} !important;
    }}

    /* Dropdown menu selected item */
    section[data-testid="stSidebar"] .css-1wa3eu0-placeholder {{
        color: {color} !important;
    }}

    /* Dropdown options (when open) */
    div[data-baseweb="select"] > div {{
        color: {color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

def set_main_text_color(color="#32a834"):
    st.markdown(f"""
    <style>
    .stApp {{
        color: {color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


#tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr) 
    result_index=np.argmax(prediction)
    return result_index

#sidebar
set_sidebar_text_color("#32a834")
st.sidebar.title("Explore")
app_mode = st.sidebar.selectbox("Select Page ▼", ["Home","About","Disease Prediction"])

#Home Page
if(app_mode=="Home"):
    set_home_background("image.jpeg")  # Path to your transparent image
    st.header("PLANT DISEASE PREDICTOR")
    st.markdown("""
    ### 🔍 Diagnose. Discover. Defend.

    🖼️ Upload a leaf image  
    🧠 Let the AI analyze it  
    📋 Get the diagnosis instantly
    
    Nature speaks through patterns. Our AI listens.

    Each leaf tells a story—of growth, of stress, of disease. With the power of machine learning, we translate that story into action. Upload a leaf image, and let our system decode what your plant is trying to tell you.

    Because healthy plants make a healthy planet.


    ### Start Your Diagnosis Journey
    📌 **Click on "Disease Recognition"** in the sidebar on the left.
    🚀 It’s that simple — no registration, no delay!

    ### About Us
    Learn more about the project and our goals on the **About** page.           
""")
    
#About Page
elif(app_mode=="About"):
    set_main_text_color("#32a834")
    st.header("ABOUT")
    st.markdown("""            
    Our mission is simple: to help you protect your plants from diseases. Whether you're a farmer, gardener, or plant lover, this system can help you quickly identify any potential diseases affecting your plants. Just upload a picture of the plant, and our system will analyze it and give you the diagnosis.

    #### Key Features:
    - **Speedy Diagnosis**: Get your results almost instantly!
    - **High Accuracy**: Our model has been trained on a huge variety of plant images. It has an accuracy of 92%.
    - **Easy-to-Use**: Simply upload an image, and you're good to go.
""")
    image_path = "vgraph.png"
    st.image(image_path,use_container_width=True)    
    st.markdown("""
#### The Dataset:
The model was trained with over 70,000 plant leaf images from 38 disease categories. The dataset includes images of both healthy and diseased plants, ensuring that the model can differentiate between them.

#### What's Next?
We plan to expand the system with the following features:

- **Mobile App Integration**: Make predictions on-the-go.
- **Real-Time Camera Support**: Live detection using smartphone or webcam.
- **Remedy Suggestions**: After predicting the disease, suggest appropriate treatments or pesticides.
- **Localization**: Multilingual support for accessibility in rural regions.


##### Made By:
Masoom Muskan | B.Tech Student and Nature Enthusiast

""")

#Prediction Page
elif(app_mode=="Disease Prediction"):
    set_main_text_color("#32a834")
    st.header("Disease Prediction")
    test_image = st.file_uploader("Select an image then click on the predict button to get the diagnosis:")
    if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)
        
    #Predict Button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Define class
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success("Model is predicting, it's a {}".format(class_name[result_index]))
        
    
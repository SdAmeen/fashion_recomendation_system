import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st

background_image_path = "assets/background.jpg"

# Add custom CSS for background image and header styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url('/assets/background.jpg') no-repeat center center fixed;
        background-size: cover;
    }}
    .header {{
        color: #ffffff; /* Change this color to contrast with your background image */
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.5); /* Optional: semi-transparent background for header */
        border-radius: 10px; /* Optional: rounded corners for header */
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header">Fashion Recommendation System</div>', unsafe_allow_html=True)

Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.models.Sequential([model,
                                    GlobalMaxPool2D()
                                    ])
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# File uploader
upload_file = st.file_uploader("Upload Image")

if upload_file is not None:
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())

    # Extract features from uploaded image
    input_img_features = extract_features_from_images(upload_file, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    # Create a list of options with corresponding image paths
    image_options = [filenames[indices[0][i]] for i in range(1, 6)]

    # Initialize selected image as None
    selected_image = None

    # Display recommended images with selectable option
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("view", key="btn1"):
            selected_image = image_options[0]
        st.image(image_options[0])

    with col2:
        if st.button("view", key="btn2"):
            selected_image = image_options[1]
        st.image(image_options[1])

    with col3:
        if st.button("view", key="btn3"):
            selected_image = image_options[2]
        st.image(image_options[2])

    with col4:
        if st.button("view", key="btn4"):
            selected_image = image_options[3]
        st.image(image_options[3])

    with col5:
        if st.button("view", key="btn5"):
            selected_image = image_options[4]
        st.image(image_options[4])

    # Display the uploaded image and the selected image side by side
    st.subheader('Uploaded Image ----------------------> Selected  Image')
    col6, col7 = st.columns(2)

    with col6:
        st.image(upload_file, caption="Uploaded Image", use_column_width=True)

    with col7:
        if selected_image:
            st.image(selected_image, width=200, caption="Selected Main Image")


  



































# import numpy as np
# import pickle as pkl
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPool2D

# from sklearn.neighbors import NearestNeighbors
# import os
# from numpy.linalg import norm
# import streamlit as st 


# background_image_path = "assets/background.jpg"

# # Add custom CSS for background image and header styling
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: url('/assets/background.jpg') no-repeat center center fixed;
#         background-size: cover;
#     }}
#     .header {{
#         color: #ffffff; /* Change this color to contrast with your background image */
#         font-size: 36px;
#         font-weight: bold;
#         text-align: center;
#         padding: 20px;
#         background-color: rgba(0, 0, 0, 0.5); /* Optional: semi-transparent background for header */
#         border-radius: 10px; /* Optional: rounded corners for header */
#     }}
#     .stButton>button {{
#         background-color: #4CAF50;
#         color: white;
#         padding: 10px;
#         border-radius: 5px;
#         border: none;
#         cursor: pointer;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown('<div class="header">Fashion Recommendation System</div>', unsafe_allow_html=True)




# Image_features = pkl.load(open('Images_features.pkl','rb'))
# filenames = pkl.load(open('filenames.pkl','rb'))

# def extract_features_from_images(image_path, model):
#     img = image.load_img(image_path, target_size=(224,224))
#     img_array = image.img_to_array(img)
#     img_expand_dim = np.expand_dims(img_array, axis=0)
#     img_preprocess = preprocess_input(img_expand_dim)
#     result = model.predict(img_preprocess).flatten()
#     norm_result = result/norm(result)
#     return norm_result
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# model.trainable = False

# model = tf.keras.models.Sequential([model,
#                                    GlobalMaxPool2D()
#                                    ])
# neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
# neighbors.fit(Image_features)
# upload_file = st.file_uploader("Upload Image")
# if upload_file is not None:
#     with open(os.path.join('upload', upload_file.name), 'wb') as f:
#         f.write(upload_file.getbuffer())
#     st.subheader('Uploaded Image')
#     st.image(upload_file)
#     input_img_features = extract_features_from_images(upload_file, model)
#     distance,indices = neighbors.kneighbors([input_img_features])
#     st.subheader('Recommended Images')
#     col1,col2,col3,col4,col5 = st.columns(5)
#     with col1:
#         st.image(filenames[indices[0][1]])
#     with col2:
#         st.image(filenames[indices[0][2]])
#     with col3:
#         st.image(filenames[indices[0][3]])
#     with col4:
#         st.image(filenames[indices[0][4]])
#     with col5:
#         st.image(filenames[indices[0][5]])
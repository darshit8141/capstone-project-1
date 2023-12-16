import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/darshit/Desktop/Ct_Scan_Covid.h5')
def preprocess_image(img):
    # Resize the image to the target size (224, 224)
    img = cv2.resize(img, (180, 180))
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_image(img):
    

    input_image_scaled = img/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,180,180,3])
    
    input_prediction = model.predict(input_image_reshaped)
    if input_prediction>0.5:
        st.markdown('''<h2 class="title">Prediction: COVID Negetive happy to say you don't have covid</h2>''',unsafe_allow_html=True)
    else:
        st.markdown('''<h2 class="title">Prediction: COVID Positive sorry to say you have covid Please Stay in Home</h2>''',unsafe_allow_html=True)
   
def set_bg_hack_url():
     '''
     A function to unpack an image from url and set as bg.
     Returns
     -------
     The background.
     '''
         
     st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://prakashhospitals.in/wp-content/uploads/2021/04/covid-19-blog.jpeg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 16px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def main():
    st.markdown('<h1 class="title">Covid-19 prediction using CT-SCAN images</h1><br>', unsafe_allow_html=True)
    st.markdown('''<h3 class="title1">Upload an image (jpg or png), and we'll predict whether it's COVID-19 or non-COVID.</h3>''',unsafe_allow_html=True)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        # Preprocess and make predictions
        img = preprocess_image(img)
        predictions = predict_image(img)


if __name__ == "__main__":
    main()

    import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/nitesh yadav/Desktop/Ct_Scan_Covid.h5')
def preprocess_image(img):
    # Resize the image to the target size (224, 224)
    img = cv2.resize(img, (180, 180))
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_image(img):
    

    input_image_scaled = img/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,180,180,3])
    
    input_prediction = model.predict(input_image_reshaped)
    if input_prediction>0.5:
        st.markdown('''<h2 class="title">Prediction: COVID Negetive happy to say you don't have covid</h2>''',unsafe_allow_html=True)
    else:
        st.markdown('''<h2 class="title">Prediction: COVID Positive sorry to say you have covid Please Stay in Home</h2>''',unsafe_allow_html=True)
   
def set_bg_hack_url():
     '''
     A function to unpack an image from url and set as bg.
     Returns
     -------
     The background.
     '''
         
     st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://prakashhospitals.in/wp-content/uploads/2021/04/covid-19-blog.jpeg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 16px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def main():
    st.markdown('<h1 class="title">Covid-19 prediction using CT-SCAN images</h1><br>', unsafe_allow_html=True)
    st.markdown('''<h3 class="title1">Upload an image (jpg or png), and we'll predict whether it's COVID-19 or non-COVID.</h3>''',unsafe_allow_html=True)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        # Preprocess and make predictions
        img = preprocess_image(img)
        predictions = predict_image(img)


if __name__ == "__main__":
    main()

    vvvimport streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/nitesh yadav/Desktop/Ct_Scan_Covid.h5')
def preprocess_image(img):
    # Resize the image to the target size (224, 224)
    img = cv2.resize(img, (180, 180))
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_image(img):
    

    input_image_scaled = img/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,180,180,3])
    
    input_prediction = model.predict(input_image_reshaped)
    if input_prediction>0.5:
        st.markdown('''<h2 class="title">Prediction: COVID Negetive happy to say you don't have covid</h2>''',unsafe_allow_html=True)
    else:
        st.markdown('''<h2 class="title">Prediction: COVID Positive sorry to say you have covid Please Stay in Home</h2>''',unsafe_allow_html=True)
   
def set_bg_hack_url():
     '''
     A function to unpack an image from url and set as bg.
     Returns
     -------
     The background.
     '''
         
     st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://prakashhospitals.in/wp-content/uploads/2021/04/covid-19-blog.jpeg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 16px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def main():
    st.markdown('<h1 class="title">Covid-19 prediction using CT-SCAN images</h1><br>', unsafe_allow_html=True)
    st.markdown('''<h3 class="title1">Upload an image (jpg or png), and we'll predict whether it's COVID-19 or non-COVID.</h3>''',unsafe_allow_html=True)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        # Preprocess and make predictions
        img = preprocess_image(img)
        predictions = predict_image(img)


if __name__ == "__main__":
    main()

    import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/nitesh yadav/Desktop/Ct_Scan_Covid.h5')
def preprocess_image(img):
    # Resize the image to the target size (224, 224)
    img = cv2.resize(img, (180, 180))
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_image(img):
    

    input_image_scaled = img/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,180,180,3])
    
    input_prediction = model.predict(input_image_reshaped)
    if input_prediction>0.5:
        st.markdown('''<h2 class="title">Prediction: COVID Negetive happy to say you don't have covid</h2>''',unsafe_allow_html=True)
    else:
        st.markdown('''<h2 class="title">Prediction: COVID Positive sorry to say you have covid Please Stay in Home</h2>''',unsafe_allow_html=True)
   
def set_bg_hack_url():
     '''
     A function to unpack an image from url and set as bg.
     Returns
     -------
     The background.
     '''
         
     st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://prakashhospitals.in/wp-content/uploads/2021/04/covid-19-blog.jpeg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 16px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def main():
    st.markdown('<h1 class="title">Covid-19 prediction using CT-SCAN images</h1><br>', unsafe_allow_html=True)
    st.markdown('''<h3 class="title1">Upload an image (jpg or png), and we'll predict whether it's COVID-19 or non-COVID.</h3>''',unsafe_allow_html=True)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        # Preprocess and make predictions
        img = preprocess_image(img)
        predictions = predict_image(img)


if __name__ == "__main__":
    main()

    import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/nitesh yadav/Desktop/Ct_Scan_Covid.h5')
def preprocess_image(img):
    # Resize the image to the target size (224, 224)
    img = cv2.resize(img, (180, 180))
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_image(img):
    

    input_image_scaled = img/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,180,180,3])
    
    input_prediction = model.predict(input_image_reshaped)
    if input_prediction>0.5:
        st.markdown('''<h2 class="title">Prediction: COVID Negetive happy to say you don't have covid</h2>''',unsafe_allow_html=True)
    else:
        st.markdown('''<h2 class="title">Prediction: COVID Positive sorry to say you have covid Please Stay in Home</h2>''',unsafe_allow_html=True)
   
def set_bg_hack_url():
     '''
     A function to unpack an image from url and set as bg.
     Returns
     -------
     The background.
     '''
         
     st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://prakashhospitals.in/wp-content/uploads/2021/04/covid-19-blog.jpeg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 16px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def main():
    st.markdown('<h1 class="title">Covid-19 prediction using CT-SCAN images</h1><br>', unsafe_allow_html=True)
    st.markdown('''<h3 class="title1">Upload an image (jpg or png), and we'll predict whether it's COVID-19 or non-COVID.</h3>''',unsafe_allow_html=True)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        # Preprocess and make predictions
        img = preprocess_image(img)
        predictions = predict_image(img)


if __name__ == "__main__":
    main()

    import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('C:/Users/nitesh yadav/Desktop/Ct_Scan_Covid.h5')
def preprocess_image(img):
    # Resize the image to the target size (224, 224)
    img = cv2.resize(img, (180, 180))
    # Convert BGR to RGB (OpenCV loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_image(img):
    

    input_image_scaled = img/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,180,180,3])
    
    input_prediction = model.predict(input_image_reshaped)
    if input_prediction>0.5:
        st.markdown('''<h2 class="title">Prediction: COVID Negetive happy to say you don't have covid</h2>''',unsafe_allow_html=True)
    else:
        st.markdown('''<h2 class="title">Prediction: COVID Positive sorry to say you have covid Please Stay in Home</h2>''',unsafe_allow_html=True)
   
def set_bg_hack_url():
     '''
     A function to unpack an image from url and set as bg.
     Returns
     -------
     The background.
     '''
         
     st.markdown(
          f"""
          <style>
          .stApp {{
              background: url("https://prakashhospitals.in/wp-content/uploads/2021/04/covid-19-blog.jpeg");
              background-size: cover
          }}
          </style>
          """,
          unsafe_allow_html=True
      )
set_bg_hack_url()

st.markdown(
    """
    <style>
    /* CSS for title */
    .title {
        font-size: 36px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* CSS for title */
    .title1 {
        font-size: 16px;
        color: white; /* Black font color */
        text-align: center;
        background-color: black; /* White background color */
        padding: 10px; /* Add padding for spacing */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def main():
    st.markdown('<h1 class="title">Covid-19 prediction using CT-SCAN images</h1><br>', unsafe_allow_html=True)
    st.markdown('''<h3 class="title1">Upload an image (jpg or png), and we'll predict whether it's COVID-19 or non-COVID.</h3>''',unsafe_allow_html=True)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        # Preprocess and make predictions
        img = preprocess_image(img)
        predictions = predict_image(img)


if __name__ == "__main__":
    main()

    
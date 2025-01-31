import streamlit as st
import joblib
import numpy as np
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

#from streamlit_drawable_canvas import st_canvas

file_path = r"C:\Users\yad\Documents\NBI Handelsakademin\Delkurs 2\Kunskapskontroll\best_model_RandomForestClassifier.joblib"
model = joblib.load(file_path)

# Funktion för att förbehandla bilden
def preprocess_image(image):
    image = image.convert("L")  # Konvertera till gråskala
    #image = image.resize((28, 28))  # Ändra storlek till 28x28 (anpassa efter modellens input)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(image)  # Konvertera till numpy-array
    image_array = cv2.adaptiveThreshold(image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    st.write(f"Bildstorlek efter omformning: {image_array.shape}")
    st.image(image, caption="Förbehandlad bild", width=150)
    image_array = image_array / 255.0  # Normalisera
    image_array = image_array.flatten().reshape(1, -1)  # Platta ut och forma för modellen
    image_array = 255 - image_array

    return image_array

# Funktion för att förbehandla ritad bild
def preprocess_canvas_image(canvas_image):
    
    # Konvertera till gråskala
    image = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2GRAY)
    # Invertera färger (om bakgrund är svart och siffran är vit)
    #image = cv2.bitwise_not(image)

    # Tillämpa binär tröskel (för att förstärka siffran)
    _, thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

    # Hitta konturer och beskära onödiga kanter
    #coords = cv2.findNonZero(thresh)  # Hitta icke-svarta pixlar
    #x, y, w, h = cv2.boundingRect(coords)  # Skapa en rektangel runt siffran
    #cropped = thresh[y:y+h, x:x+w]  # Beskär bilden så att bara siffran återstår

    st.write(f"Bildstorlek före omformning: {thresh.shape}")
    
    # Ändra storlek till 28x28 pixlar
    resized = cv2.resize(thresh, (28, 28))

    st.write(f"Bildstorlek efter omformning: {resized.shape}")
    st.image(resized, caption="Förbehandlad bild", width=150)

    
    # Normalisera pixelvärdena
    normalized = resized / 255.0
    # Platta ut och omforma för modellen
    processed_image = normalized.flatten().reshape(1, -1)
    return processed_image

  
nav = st.sidebar.radio("Sifferigenkänning",["Ladda upp en bild", "Rita på ritytan", "Använd datorns kamera"])

# Streamlit UI
if nav == "Ladda upp en bild":
    st.title("Sifferklassificering")
    st.write("Ladda upp en bild av en siffra så kommer modellen att försöka identifiera den.")

    uploaded_file = st.file_uploader("Ladda upp en bild...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uppladdad bild", use_column_width=True)
        
        # Förbehandla bilden
        processed_image = preprocess_image(image)

        # Gör en prediktion
        prediction = model.predict(processed_image)
        
        # Visa resultatet
        st.write(f"**Predikterad siffra:** {prediction[0]}")

if nav == "Rita på ritytan":
    point_display_radius = st.sidebar.slider("Penseltjocklek: ", 1, 30, 10)
    st.write("Rita en siffra i rutan nedan så försöker modellen identifiera den.")
    
    # Skapa ritsektionen
    canvas_result = st_canvas(
        fill_color="black",  # Färg på bakgrund (svart)
        stroke_width=point_display_radius,  # Penselstorlek
        stroke_color="white",  # Färg på ritningen (vit)
        background_color="black",  # Bakgrundsfärg
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None:
        # Konvertera canvas till numpy-array
        image_data = np.array(canvas_result.image_data)
    
        # Kontrollera om användaren har ritat något
        if np.any(image_data[:, :, :-1] != 0):  # Kollar om bilden innehåller något annat än bakgrunden
            # Förbehandla bilden
            processed_image = preprocess_canvas_image(image_data)
    
            # Gör en prediktion
            prediction = model.predict(processed_image)
    
            # Visa resultatet
            st.write(f"**Predikterad siffra:** {prediction[0]}")
        else:
            st.write("Rita en siffra")

if nav == "Använd datorns kamera":
    st.title("Ta ett foto med kameran")

    # Skapa kamerainslag
    camera_image = st.camera_input("Tryck på knappen nedan för att ta ett foto")
    
    if camera_image is not None:
        # Konvertera bild till PIL-format
        image = Image.open(camera_image)
        
        # Visa bilden
        st.image(image, caption="Ditt foto", use_column_width=True)

        # Förbehandla bilden
        processed_image = preprocess_image(image)

        # Gör en prediktion
        prediction = model.predict(processed_image)
        st.write(f"🧠 Modellens prediktion: **{prediction[0]}**")
        

        
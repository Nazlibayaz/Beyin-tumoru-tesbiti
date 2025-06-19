import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Modeli yükle
model = load_model("beyin_tumor_modeli.h5")
etiketler = ['notumor', 'glioma', 'meningioma', 'pituitary']

st.title("🧠 Beyin Tümörü Tahmin Arayüzü")
st.write("Lütfen bir MR görüntüsü yükleyin:")

# Dosya yükleme
yuklenen_dosya = st.file_uploader("Resim Seç", type=["jpg", "png", "jpeg"])

if yuklenen_dosya is not None:
    # Görseli ekranda göster
    #image = Image.open(yuklenen_dosya)
    image = Image.open(yuklenen_dosya).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli hazırlayıp tahmin yap
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0) 

    # Modelden tahmin al
    tahmin = model.predict(img)
    sonuc = etiketler[np.argmax(tahmin)]

    st.success(f"Tahmin edilen sınıf: *{sonuc.upper()}*")
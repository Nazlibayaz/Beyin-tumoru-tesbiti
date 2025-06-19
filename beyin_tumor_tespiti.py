import os
import cv2
import numpy as np
import seaborn as sns # karmaşıklık matrisi için
import tensorflow as tf
import matplotlib.pyplot as plt  # doğruluk grafiği için

from sklearn.metrics import confusion_matrix  # karmaşıklık matrisi için
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Sınıfları tanımla
klasorler = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3
}

# Görselleri yükleyen fonksiyon
def veri_yukle(base_path, klasorler):
    veriler = []
    for klasor, etiket in klasorler.items():
        klasor_yolu = os.path.join(base_path, klasor)
        for dosya in os.listdir(klasor_yolu):
            try:
                img_path = os.path.join(klasor_yolu, dosya)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))
                veriler.append((img, etiket))
            except Exception as e:
                print(f"Hata: {img_path} — {e}")
    return veriler

egitim_verileri = veri_yukle("Training", klasorler)
test_verileri = veri_yukle("Testing", klasorler)

print("Eğitim verisi:", len(egitim_verileri))
print("Test verisi:", len(test_verileri))

for klasor, etiket in klasorler.items():
    klasor_yolu = os.path.join("Training", klasor)
    print(f"Klasör kontrol: {klasor_yolu}")
    if os.path.exists(klasor_yolu):
        print("Klasör bulundu.")
        print("İçindekiler:", os.listdir(klasor_yolu)[:5])
    else:
        print("Klasör bulunamadı!")

# Veriyi ayır ve normalleştir
X = np.array([x for x, _ in egitim_verileri]) / 255.0
y = np.array([y for _, y in egitim_verileri])
y = to_categorical(y, num_classes=4)

X_test = np.array([x for x, _ in test_verileri]) / 255.0
y_test = np.array([y for _, y in test_verileri])
y_test = to_categorical(y_test, num_classes=4)

# Eğitim verisini içten ayır (opsiyonel)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model tanımı
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 sınıf için
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Eğitim/Doğrulama doğruluğunu çiz
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Eğitim vs Doğrulama Doğruluğu')
plt.legend()
plt.grid(True)
plt.show()

# Test verisi üzerinden tahmin yap
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # tahmin edilen sınıflar
y_true = np.argmax(y_test, axis=1)          # gerçek sınıflar

# Confusion matrix oluştur
cm = confusion_matrix(y_true, y_pred_classes)

# Etiket isimleri
etiketler = ['notumor', 'glioma', 'meningioma', 'pituitary']

# Matris görselleştirme
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=etiketler, yticklabels=etiketler)
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("📊 Confusion Matrix - Test Verisi")
plt.show()


# Test verisi doğruluğunu yazdır
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {test_acc:.4f}")

# Modeli kaydet
model.save(r"C:\Users\hp\OneDrive\Masaüstü\archive (3)\beyin_tumor_modeli.h5")
print("Model başarıyla kaydedildi!")
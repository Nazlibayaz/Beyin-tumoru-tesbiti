# Beyin-tumoru-tesbiti
Bu proje, beyin MR görüntüleriyle beyin tümörü tespiti yapmayı amaçlamaktadır.

##  Dosya Açıklamaları

- `beyin_tumor_tespiti.py` → Ana model eğitim kodları bu dosyanın altındadır.
- `TahminEkrani.py` → Eğitilen modelle tahmin yapılmasını sağlayan arayüz bu dosya altındadır.
- `manueltestdosyasi` → Manuel test için oluşturduğumuz sayfa bu dosya altındadır.

##  Veri Seti

Veri setimiz Kaggle üzerinden alınmış olup linki aşağıdaki şekildedir.
🔗 [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
Veri setini indirdikten sonra, `Training/` ve `Testing/` klasörleri olarak uygun şekilde projeye yerleştiriniz.

##  Kullanılan Kütüphaneler

-tensorflow
-opencv-python
-numpy
-matplotlib
-seaborn
-scikit-learn
-streamlit
-pillow
şeklindedir.

Bu kütüphaneleri Visual Studio Code`da kullanmak için terminalde parantez içindeki eklentiyi çalıştırmanız yeterlidir.
(  pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn streamlit pillow  )

## Dikkat Edilmesi Gereken Yerler 

İlk olarak Python ve Visual Studio Codun kurulumunu yapınız ama bu kurulumu yaparken lütfen tenserflow kütüphanesinin Pythonun 3.11.6 sürümünden güncel olan sürümlerinde çalışmadığını unutmayınız.
Bu proje bu sebepten ötürü Python 3.11.6 sürümü ile yapılmıştır.
Ayrıca `TahminEkrani.py` Arayüzünü başlatmak için terminale aşağıda bparantez içinde bulunan komutu yazmanız yeterlidir: 
( streamlit run TahminEkrani.py ) 

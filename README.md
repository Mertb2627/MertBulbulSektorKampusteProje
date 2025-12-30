# 🧠 Yapay Zeka Destekli Görüntü Sınıflandırıcı  
## Sektör Kampüste Proje – Streamlit Animals-10 Uygulaması

Bu proje, **PyTorch** kullanılarak eğitilmiş bir **görüntü sınıflandırma modeli** ile
kullanıcının yüklediği bir görselin hangi sınıfa ait olduğunu tahmin eden
**yapay zeka destekli bir web uygulamasıdır**.

Uygulama, **Animals-10 veri seti** üzerinde **ResNet18 (Transfer Learning)**
yaklaşımıyla eğitilmiş bir modeli kullanır ve sonuçları
**Streamlit tabanlı kullanıcı dostu bir arayüz** üzerinden sunar.

---

## 🎯 Proje Amacı

Bu projenin amacı;

- Makine öğrenimi tekniklerini kullanarak görsel verileri sınıflandıran bir model geliştirmek  
- Eğitilmiş modeli basit ve erişilebilir bir web arayüzü ile entegre etmek  
- Kullanıcıdan alınan görüntüyü ön işleme tabi tutarak doğru sınıf tahmini üretmek  
- Yapay zeka tabanlı, işlevsel ve anlaşılır bir uygulama ortaya koymaktır  

---

## ⚙️ Fonksiyonel Özellikler

- Kullanıcı bilgisayarından bir **görüntü yükleyebilir**
- Yüklenen görüntü **otomatik olarak ön işleme** alınır
  - Yeniden boyutlandırma
  - Normalize etme
- Model, görüntüyü sınıflandırır
- Tahmin sonucu:
  - Sınıf adı
  - Güven oranı (%)
  - Top-K olasılık listesi
- Sonuçlar **kullanıcı dostu web arayüzünde** gösterilir

---

## 🧠 Kullanılan Model ve Teknik Detaylar

- **Model:** ResNet18  
- **Yöntem:** Transfer Learning  
- **Framework:** PyTorch  
- **Veri Seti:** Animals-10 (10 farklı hayvan sınıfı)  
- **Ön İşleme:**
  - Görüntü yeniden boyutlandırma
  - Normalize etme
- **Performans Metrikleri:**
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-score (macro)

Elde edilen metrikler uygulama arayüzünde de gösterilmektedir.

---

## 🖥️ Kullanılan Arayüz

- **Streamlit** kullanılarak geliştirilmiştir
- Sade, erişilebilir ve kullanıcı dostu tasarım
- Görsel yükleme alanı
- Önizleme ekranı
- Tahmin et butonu
- Sonuç ve güven oranı gösterimi

---

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Repoyu klonla
```bash
git clone https://github.com/Mertb2627/SektorKampusteProje.git
cd SektorKampusteProje


2️⃣ Sanal ortam oluştur (önerilir)
python -m venv venv

Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

Gerekli paketleri yükle
pip install -r requirements.txt

4️⃣ Uygulamayı çalıştır
streamlit run app.py

SektorKampusteProje/
│
├── app.py                  # Streamlit arayüzü
├── predict.py              # Tahmin fonksiyonları
├── train_animals10.py      # Model eğitim scripti
├── prepare_animals10.py    # Veri ön işleme scripti
├── requirements.txt        # Gerekli kütüphaneler
├── models/
│   └── animals10_resnet18.pth
└── README.md


Kullanılan Veri Seti

Animals-10 Dataset

10 farklı hayvan sınıfı

Dengeli ve etiketli görseller

Kaynak: https://www.kaggle.com/datasets/alessiocorrado99/animals10


Geliştirici:

Mert Bülbül
Öğrenci No: 220502006
Sektör Kampüste Projesi

<img width="2560" height="1347" alt="image" src="https://github.com/user-attachments/assets/3d111f89-53df-432e-9d93-6288fe2992c2" />



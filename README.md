# 🏨 Hotel Reservation Intelligence (HRI)

Otel rezervasyonlarının **iptal edilme olasılığını** tahmin eden bir makine öğrenmesi uygulaması. Resepsiyon ve yönetim ekiplerine, bir rezervasyonun riskini check-in öncesinde göstererek proaktif aksiyon almalarını sağlar.

> **Yazar:** Aslı Torun · Data Scientist — Hotel Analytics
> Data Science Portfolio · 2026 · v2.0

---

## ✨ Öne Çıkanlar

- **Random Forest** modeli ile **%90.34 doğruluk** ile iptal tahmini
- **İki dilli arayüz** (EN / TR) — sağ üstten anlık geçiş
- **Resepsiyon paneli:** tek rezervasyon için iptal olasılığı, misafir tipi, bağlılık skoru, oda ve hizmet önerisi
- **Yönetici paneli:** günlük özet, feature importance, segment risk profili, misafir segment matrisi, model karşılaştırması
- **Youden's J** ile optimize edilmiş karar eşiği (0.375)
- Modelin yokluğunda **kural tabanlı yedek mod**

---

## 📂 Proje Yapısı

```
hotel_project/
├── Hotel Reservations.csv     # Ham veri seti (kaynak)
├── hotel_optimizer.py         # 1. Veri temizleme → hotel_cleaned.csv
├── hotel_cleaned.csv          # Temizlenmiş & encode edilmiş veri
├── model.py                   # 2. Model eğitimi (LR + Random Forest)
├── save_threshold.py          # 3. Optimal karar eşiği hesaplama
├── app.py                     # 4. Streamlit arayüzü
├── rf_model.pkl               # Eğitilmiş Random Forest modeli
├── scaler.pkl                 # StandardScaler nesnesi
├── threshold.pkl              # Optimal karar eşiği (0.375)
├── demo.mp4                   # Projenin nasıl yapıldığını anlatan video
├── requirements.txt           # Python bağımlılıkları
└── README.md
```

---

## 🔄 Çalışma Akışı (Pipeline)

```
Hotel Reservations.csv
        │  hotel_optimizer.py  (Booking_ID at, fiyatı 0 olanları temizle,
        │                       hedefi sayısallaştır, one-hot encode)
        ▼
hotel_cleaned.csv
        │  model.py  (train/test ayır, ölçekle, LR + RF eğit)
        ▼
rf_model.pkl + scaler.pkl
        │  save_threshold.py  (Youden's J ile optimal eşik)
        ▼
threshold.pkl
        │  app.py  (Streamlit arayüzü)
        ▼
   Canlı uygulama
```

---

## 🚀 Kurulum & Çalıştırma

### 1. Sanal ortamı etkinleştir

```powershell
# Windows / PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Git Bash
source .venv/Scripts/activate
```

### 2. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 3. (Opsiyonel) Pipeline'ı sıfırdan çalıştır

`rf_model.pkl`, `scaler.pkl` ve `threshold.pkl` zaten depoda mevcut. Modeli baştan eğitmek istersen:

```bash
python hotel_optimizer.py   # ham veriyi temizle
python model.py             # modelleri eğit ve kaydet
python save_threshold.py    # optimal eşiği hesapla
```

### 4. Uygulamayı başlat

```bash
streamlit run app.py
```

Uygulama tarayıcıda `http://localhost:8501` adresinde açılır.

---

## 🧠 Model Detayları

| Model | Accuracy | Precision (iptal) | Recall (iptal) | F1 (iptal) |
|-------|----------|-------------------|----------------|------------|
| Logistic Regression (baseline) | %78.1 | %64 | %77 | %70 |
| **Random Forest ✓ (seçilen)** | **%90.3** | **%88** | **%82** | **%85** |

- **Eğitim verisi:** 35.730 rezervasyon kaydı
- **Genel iptal oranı:** %33.2
- **En kritik faktör:** Lead time (önceden rezervasyon süresi) — %30.9 önem derecesi
- **Karar eşiği:** Youden's J ile 0.375 (TPR=0.869, FPR=0.088)

### Öne Çıkan Özellikler (Feature Importance)

| Özellik | Önem |
|---------|------|
| Lead time | %30.9 |
| Oda fiyatı | %14.6 |
| Özel istek sayısı | %11.5 |
| Varış tarihi | %9.2 |
| Varış ayı | %8.6 |

---

## 🖥️ Arayüz

### Resepsiyon Sekmesi
Rezervasyon ve misafir bilgilerini girerek anlık analiz alınır:
- İptal olasılığı (% + risk seviyesi)
- Misafir tipi (Sessiz / Orta / Kalabalık aile)
- Bağlılık skoru ve segment (VIP / Kritik / Potansiyel / Özel İlgi)
- Misafir profiline uygun **oda önerisi**
- Özel istek bazlı **hizmet önerisi** ve risk altındaki tahmini gelir

### Yönetici Sekmesi
- Günlük özet metrikleri
- Feature importance grafiği
- Segment bazlı risk profili
- Misafir segment matrisi (bağlılık × risk) ve hizmet yaklaşımları
- Model karşılaştırma tablosu

---

## 🎬 Tanıtım Videosu

Projenin nasıl yapıldığını adım adım anlatan video proje klasöründe yer alır:

**[`demo.mp4`](demo.mp4)** — veri temizleme, model eğitimi ve Streamlit arayüzünün kurulumu anlatılır.

> Video dosyası ~48 MB'tır. GitHub'a yüklerken dosya boyutu sınırlarına dikkat edin; gerekirse `.gitignore`'a ekleyip videoyu ayrı bir bağlantı (YouTube, Drive vb.) üzerinden paylaşabilirsiniz.

---

## 🛠️ Teknoloji Yığını

- **Python** · pandas · numpy · scipy
- **scikit-learn** — RandomForestClassifier, LogisticRegression
- **Streamlit** — web arayüzü
- **Plotly** — interaktif grafikler
- **SHAP** — model açıklanabilirliği

---

## 📊 Veri Seti

`Hotel Reservations.csv` — 36.000+ otel rezervasyon kaydı. Her satırda misafir profili (yetişkin/çocuk sayısı, geceler), rezervasyon bilgileri (lead time, fiyat, oda tipi, yemek planı, market segmenti), geçmiş davranış (tekrar müşteri, önceki iptaller) ve hedef değişken `booking_status` (iptal edildi / edilmedi) yer alır.

**Temizleme adımları:** `Booking_ID` sütunu atıldı, fiyatı 0 olan hatalı kayıtlar silindi, hedef değişken sayısallaştırıldı, kategorik sütunlar one-hot encode edildi.

---

## 📝 Notlar

- Veri sızıntısını önlemek için `StandardScaler` yalnızca eğitim setine `fit` edilip test setine `transform` uygulanmıştır.
- `rf_model.pkl` dosyası ~64 MB'tır; depoya dahildir.
- `app.py` model dosyalarını bulamazsa otomatik olarak kural tabanlı moda geçer.

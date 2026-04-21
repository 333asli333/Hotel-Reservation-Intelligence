"""
Hotel Reservation Intelligence (HRI)
Model Eğitim Scripti
Author: Aslı Torun
"""

import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. VERİ YÜKLEME
# =============================================================================

df = pd.read_csv("hotel_cleaned.csv")

print("=" * 60)
print("VERİ GENEL BAKIŞ")
print("=" * 60)
print(f"Toplam kayıt: {len(df)}")
print("\nHedef değişken dağılımı:")
print(df["booking_status"].value_counts())
print("\nOran:")
print(df["booking_status"].value_counts(normalize=True).round(3))


# =============================================================================
# 2. TRAIN-TEST AYIRIMI (Tek seferlik — tüm modeller bu split'i kullanır)
# =============================================================================

X = df.drop("booking_status", axis=1)
y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Veri sızıntısını önlemek için test setine sadece transform uygulandı
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nEğitim seti: {X_train.shape[0]} kayıt")
print(f"Test seti  : {X_test.shape[0]} kayıt")


# =============================================================================
# 3. LOJİSTİK REGRESYON
# =============================================================================

print("\n" + "=" * 60)
print("LOJİSTİK REGRESYON")
print("=" * 60)

model_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
model_lr.fit(X_train_scaled, y_train)

y_pred_lr = model_lr.predict(X_test_scaled)
print(f"Doğruluk: %{accuracy_score(y_test, y_pred_lr) * 100:.2f}")
print(classification_report(y_test, y_pred_lr))

# Katsayı analizi
katsayilar = pd.DataFrame({
    "Değişken": X.columns,
    "Ağırlık": model_lr.coef_[0]
}).sort_values("Ağırlık", ascending=False)

print("\n--- İPTALİ EN ÇOK TETİKLEYEN 5 FAKTÖR (Pozitif Ağırlıklar) ---")
print(katsayilar.head(5).to_string(index=False))

print("\n--- İPTALİ EN ÇOK ENGELLEYEN 5 FAKTÖR (Negatif Ağırlıklar) ---")
print(katsayilar.tail(5).to_string(index=False))

# Sadakat skorlaması için önemli değişkenlerin katsayıları
loyalty_vars = [
    "no_of_special_requests",
    "repeated_guest",
    "required_car_parking_space",
    "no_of_previous_bookings_not_canceled"
]
print("\n--- SADAKAT SKORU İÇİN KRİTİK DEĞİŞKENLER ---")
print(katsayilar[katsayilar["Değişken"].isin(loyalty_vars)].to_string(index=False))


# =============================================================================
# 4. RANDOM FOREST
# =============================================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

model_rf = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42
)
model_rf.fit(X_train_scaled, y_train)

y_pred_rf = model_rf.predict(X_test_scaled)
print(f"Doğruluk: %{accuracy_score(y_test, y_pred_rf) * 100:.2f}")
print(classification_report(y_test, y_pred_rf))

# Değişken önem dereceleri
onem = pd.DataFrame({
    "Değişken": X.columns,
    "Önem Derecesi (%)": model_rf.feature_importances_ * 100
}).sort_values("Önem Derecesi (%)", ascending=False)

print("\n--- İPTAL KARARINDAKİ EN KRİTİK 5 FAKTÖR ---")
print(onem.head(5).to_string(index=False))


# =============================================================================
# 5. OPTİMAL EŞİK — YOUDEN'S J (Random Forest üzerinde)
# =============================================================================

print("\n" + "=" * 60)
print("OPTİMAL EŞİK ANALİZİ (Youden's J)")
print("=" * 60)

probs_rf = model_rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs_rf)

youden_j = tpr - fpr
optimal_idx = youden_j.argmax()
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Eşik : {optimal_threshold:.3f}")
print(f"TPR (Recall) : {tpr[optimal_idx]:.3f}")
print(f"FPR          : {fpr[optimal_idx]:.3f}")


# =============================================================================
# 6. KALİBRASYON ANALİZİ
# =============================================================================

print("\n" + "=" * 60)
print("KALİBRASYON ANALİZİ")
print("=" * 60)

prob_true, prob_pred = calibration_curve(y_test, probs_rf, n_bins=10)
calibration_df = pd.DataFrame({
    "Tahmin Edilen Olasılık": prob_pred.round(3),
    "Gerçek Oran": prob_true.round(3)
})
print(calibration_df.to_string(index=False))


# =============================================================================
# 7. KEŞİFSEL ANALİZLER
# =============================================================================

print("\n" + "=" * 60)
print("KEŞİFSEL ANALİZLER")
print("=" * 60)

print(f"\nArrival month medyan : {df['arrival_month'].median()}")
print(f"Arrival date medyan  : {df['arrival_date'].median()}")

print("\n--- Özel İstek Sayısına Göre İptal Oranı ---")
print(df.groupby("no_of_special_requests")["booking_status"].mean().round(3))

print("\n--- Çocuk Sayısına Göre İptal Oranı ---")
print(df.groupby("no_of_children")["booking_status"].agg(["count", "mean"]).round(3))

print("\n--- Yetişkin Sayısına Göre İptal Oranı ---")
print(df.groupby("no_of_adults")["booking_status"].agg(["count", "mean"]).round(3))

print("\n--- Önceki İptal Etmeme Sayısı Dağılımı ---")
print(df["no_of_previous_bookings_not_canceled"].describe().round(3))


# =============================================================================
# 8. WILSON GÜVENİ ARALIĞI ANALİZİ (Misafir Gürültü Profili)
# =============================================================================

print("\n" + "=" * 60)
print("WILSON GÜVEN ARALIĞI — ÇOCUK SAYISINA GÖRE İPTAL RİSKİ")
print("=" * 60)

z = 1.96
for name, group in df.groupby("no_of_children")["booking_status"]:
    n = len(group)
    p = group.mean()
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    print(f"Çocuk={name}: n={n:5d} | İptal Oranı={p:.3f} | "
          f"CI=[{center - margin:.3f}, {center + margin:.3f}]")


# =============================================================================
# 9. MODELLERİ KAYDET
# =============================================================================

print("\n" + "=" * 60)
print("MODEL KAYIT")
print("=" * 60)

with open("rf_model.pkl", "wb") as f:
    pickle.dump(model_rf, f)

with open("lr_model.pkl", "wb") as f:
    pickle.dump(model_lr, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("optimal_threshold.pkl", "wb") as f:
    pickle.dump(optimal_threshold, f)

print("✓ rf_model.pkl kaydedildi")
print("✓ lr_model.pkl kaydedildi")
print("✓ scaler.pkl kaydedildi")
print(f"✓ optimal_threshold.pkl kaydedildi ({optimal_threshold:.3f})")
print("\nModel eğitimi tamamlandı.")
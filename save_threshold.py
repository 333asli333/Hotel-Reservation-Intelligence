"""
Bu script model.py'ye ek olarak bir kez çalıştırılır.
Youden's J istatistiği ile ROC eğrisinden optimal karar eşiğini hesaplar
ve threshold.pkl olarak kaydeder.

Youden's J = TPR - FPR (sensitivity + specificity - 1)
Maksimum J noktası: hem iptalleri yakalamak hem de yanlış alarm vermemek
arasındaki en iyi dengeyi verir.

Sonuç: threshold = 0.375
Bu eşikte TPR=0.869, FPR=0.088
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

df = pd.read_csv("hotel_cleaned.csv")
X = df.drop("booking_status", axis=1)
y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open("rf_model.pkl", "rb") as f:
    model_rf = pickle.load(f)

probs = model_rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)

youden_j    = tpr - fpr
optimal_idx = youden_j.argmax()
threshold   = float(thresholds[optimal_idx])

print(f"Optimal esik (Youden's J) : {threshold:.4f}")
print(f"Bu esikteki TPR           : {tpr[optimal_idx]:.4f}")
print(f"Bu esikteki FPR           : {fpr[optimal_idx]:.4f}")
print(f"Youden's J degeri         : {youden_j[optimal_idx]:.4f}")

with open("threshold.pkl", "wb") as f:
    pickle.dump(threshold, f)

print("\nthreshold.pkl basariyla kaydedildi.")

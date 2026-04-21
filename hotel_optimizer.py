# kütüphaneler
import pandas as pd
import numpy as np 
# veriyi içeri aldım
df = pd.read_csv("Hotel Reservations.csv")

# önden göz attım
df.head()
df.shape 
df.info()
df.isnull().sum()

#  gereksiz sütunu düşürdüm (kimlik)
if "Booking_ID" in df.columns:
    df.drop("Booking_ID", axis=1, inplace=True)

# tekrarlayan kayıtlara bakıs
tekrar_sayisi = df.duplicated().sum()
# df[df.duplicated(keep=False)].sort_values(by=['lead_time', 'avg_price_per_room']).head(6)
# tekrarlayan kayıtlar doğal müşter varyansı oldugu kanıtlandığı için silmedim

# aykırı değerleri temizleme (fiyatı 0 olan hatalı/ücretsiz kayıtlar)
bedava_kayitlar = len(df[df["avg_price_per_room"] == 0])
print(f"Fiyatı 0 olan ve silinen kayit sayisi: {bedava_kayitlar}")
df =df[df["avg_price_per_room"] > 0]

# hedef değişkeni sayısallaştırma
df['booking_status'] = df['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})

kategorik_sutunlar = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
df_temiz = pd.get_dummies(df, columns= kategorik_sutunlar, drop_first=True)
print(f"temizlenmis ve modele hazir veri boyutu: {df_temiz.shape}")

df_temiz.to_csv("hotel_cleaned.csv", index=False)


import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pickle, os

# --- DİL DESTEĞİ (i18n) SÖZLÜĞÜ ---
LANGUAGES = {
    "EN": {
        "app_sub": "Cancellation Prediction System",
        "tab_rec": "Reception",
        "tab_mgr": "Manager Panel",
        "sb_m": "Model",
        "sb_acc": "Accuracy",
        "sb_train": "Training data",
        "sb_cx": "Cancellation rate",
        "sb_crit": "Critical factor",
        "sb_err": "⚠ rf_model.pkl not found<br>Rule-based mode active",
        "sb_ok": "✓ Model active · RF 90.34%",
        "r_det": "Reservation details",
        "r_prof": "Guest profile",
        "r_hist": "History & Details",
        "i_lead": "Lead time (days in advance)",
        "i_price": "Room price (₺)",
        "i_seg": "Segment",
        "i_meal": "Meal plan",
        "i_room": "Requested room",
        "i_adults": "Number of adults",
        "i_child": "Number of children",
        "i_week": "Week nights",
        "i_wknd": "Weekend nights",
        "i_spec": "Number of special requests",
        "i_park": "Parking request",
        "i_rep": "Repeated guest",
        "i_pcx": "Previous cancellations",
        "i_pok": "Previous successful bookings",
        "btn_ana": "Analyze reservation",
        "err_file": "Model file not found. Ensure rf_model.pkl and scaler.pkl are in the same folder as app.py.",
        "res_ana": "Analysis result",
        "res_cx": "Cancellation probability",
        "res_gt": "Guest type",
        "res_gl": "Guest level",
        "res_loy": "Loyalty score",
        "res_room": "Room recommendation",
        "res_srv": "Service recommendation",
        "res_rev": "Estimated revenue · {stay} nights",
        "rev_risk": "At risk: ₺{rev:,}",
        "rev_safe": "Safe booking",
        "al_d_t": "Reservation requires follow-up",
        "al_d_d1": "Cancellation probability",
        "al_d_d2": "Contact the guest before check-in. A personal welcome and flexibility might secure this booking.",
        "al_w_t": "Follow-up recommended",
        "al_w_d1": "Cancellation probability",
        "al_w_d2": "Reinforce the booking confirmation with a personal message within 48 hours.",
        "al_o_t": "Reservation is stable",
        "al_o_d1": "Cancellation probability",
        "al_o_d2": "Standard procedure is sufficient. Focus on guest welcome.",
        "m_ds": "Daily summary",
        "m_tot": "Total bookings",
        "m_fi": "Feature importance",
        "m_arr_d": "Arrival date",
        "m_arr_m": "Arrival month",
        "m_srp": "Segment risk profile",
        "m_r": "Risk",
        "m_lr": "LR coefficient",
        "m_h": "High",
        "m_l": "Low",
        "m_m": "Medium",
        "m_vl": "Very low",
        "m_n": "Neutral",
        "m_gs": "Guest segment & service approach",
        "m_gs_d": "Each guest segment requires a different service approach. The optimal strategy is determined by evaluating loyalty score and cancellation risk together.",
        "p_v_l": "Loyal & Safe",
        "p_v_t": "High loyalty · Low risk",
        "p_v_d": "Most valuable segment. High trust in hotel experience, strong probability of rebooking.",
        "p_v_s": "Repeated guest · Special req ≥3 · Short lead time · Offline segment",
        "p_v_a": "Approach: Consistent and complete service. Fulfill expectations and add a small surprise. Comfort focused.",
        "p_r_l": "Priority Attention",
        "p_r_t": "High loyalty · High risk",
        "p_r_d": "Satisfied in the past but uncertainty in this booking. Proactive communication is critical.",
        "p_r_s": "Repeated guest · Very long lead time · Decreasing special requests",
        "p_r_a": "Approach: Personal and warm contact. A message referring to past stays can renew trust.",
        "p_n_l": "Loyalty Opportunity",
        "p_n_t": "Low loyalty · Low risk",
        "p_n_d": "First time visitor with low cancellation. Best window for long-term relationship.",
        "p_n_s": "First stay · Offline/corporate · Special req 1–2",
        "p_n_a": "Approach: First impressions last. A personal touch lays the foundation for loyalty.",
        "p_l_l": "Personal Touch",
        "p_l_t": "Low loyalty · High risk",
        "p_l_d": "Neither frequent nor safe segment. Unexpected warmth can turn them into loyal customers.",
        "p_l_s": "First stay · Online · Special req = 0 · Long lead time",
        "p_l_a": "Approach: Small but sincere gesture. Highest surprise effect. Welcome note or dinner treat.",
        "g_title": "Service guide — special request based",
        "g_3_t": "3 or more special requests",
        "g_3_d": "Values guest experience. Fulfill all requests carefully. Add a personal touch.",
        "g_3_tag": "Comfort focused service",
        "g_1_t": "1–2 special requests",
        "g_1_d": "Go slightly above requests. An extra detail in room preparation increases satisfaction.",
        "g_1_tag": "Personalized service",
        "g_0_t": "No special requests",
        "g_0_d": "Surprise effect is strongest here. Sincere gesture like local treat leaves lasting impression.",
        "g_0_tag": "Warm welcome recommended",
        "c_t": "Model comparison",
        "c_b": "Logistic Regression — baseline",
        "c_s": "Random Forest ✓ selected",
        "f_q": "Quiet",
        "f_q_d": "Upper floors recommended for a calm and comfortable experience",
        "f_fam": "Large Family",
        "f_fam_d": "Lower floors recommended for wide space and garden access",
        "f_mid": "Medium",
        "f_mid_d": "Standard room assignment is appropriate",
        "f_vip": "VIP",
        "f_vip_d": "Loyal guest",
        "f_crit": "CRITICAL",
        "f_crit_d": "Priority attention",
        "f_pot": "POTENTIAL",
        "f_pot_d": "Loyalty opportunity",
        "f_spec": "SPECIAL CARE",
        "f_spec_d": "Personal touch recommended",
        "fs_3_t": "Comfort Focused Service",
        "fs_3_d": "All requests must be carefully fulfilled. Prepare in advance.",
        "fs_3_b": "High expectation · Fulfill completely",
        "fs_1_t": "Personalized Service",
        "fs_1_d": "Add a small surprise to the requests. A personal touch makes a difference.",
        "fs_1_b": "Medium expectation · Offer a little more",
        "fs_0h_t": "Warm Welcome Recommended",
        "fs_0h_d": "Give a sincere and personal welcome at check-in. A welcome note leaves a mark.",
        "fs_0h_b": "Low expectation · High surprise effect",
        "fs_0l_t": "Standard Service + Small Touch",
        "fs_0l_d": "Standard service is enough but a small personal gesture significantly increases satisfaction.",
        "fs_0l_b": "First impression · Beginning of loyalty",
        "fr_h": "High probability",
        "fr_m": "Medium probability",
        "fr_l": "Low probability",
        "rm_f58": "Floors 5–8",
        "rm_f68": "Floors 6–8",
        "rm_f35": "Floors 3–5",
        "rm_f24": "Floors 2–4",
        "rm_f13": "Floors 1–3",
        "rm_fg": "Ground floor",
        "rm_q1": "Calm corridor, with view",
        "rm_q2": "Corner room, large window",
        "rm_m1": "Central location, standard comfort",
        "rm_m2": "Pool view",
        "rm_c1": "Wide space, child friendly",
        "rm_c2": "Garden and pool access",
        "m_rp": "Room price"
    },
    "TR": {
        "app_sub": "İptal Tahmin Sistemi",
        "tab_rec": "Resepsiyon",
        "tab_mgr": "Yönetici Paneli",
        "sb_m": "Model",
        "sb_acc": "Doğruluk",
        "sb_train": "Eğitim verisi",
        "sb_cx": "İptal oranı",
        "sb_crit": "Kritik faktör",
        "sb_err": "⚠ rf_model.pkl bulunamadı<br>Kural tabanlı mod aktif",
        "sb_ok": "✓ Model aktif · RF %90.34",
        "r_det": "Rezervasyon detayı",
        "r_prof": "Misafir profili",
        "r_hist": "Geçmiş & Detay",
        "i_lead": "Lead time (gün önceden)",
        "i_price": "Oda fiyatı (₺)",
        "i_seg": "Segment",
        "i_meal": "Yemek planı",
        "i_room": "Talep edilen oda",
        "i_adults": "Yetişkin sayısı",
        "i_child": "Çocuk sayısı",
        "i_week": "Hafta içi gece",
        "i_wknd": "Hafta sonu gece",
        "i_spec": "Özel istek sayısı",
        "i_park": "Otopark talebi",
        "i_rep": "Tekrar müşteri",
        "i_pcx": "Geçmiş iptal sayısı",
        "i_pok": "Geçmiş tamamlanan rezervasyon",
        "btn_ana": "Rezervasyonu analiz et",
        "err_file": "Model dosyası bulunamadı. rf_model.pkl ve scaler.pkl dosyalarının app.py ile aynı klasörde olduğunu kontrol edin.",
        "res_ana": "Analiz sonucu",
        "res_cx": "İptal olasılığı",
        "res_gt": "Misafir tipi",
        "res_gl": "Misafir seviyesi",
        "res_loy": "Bağlılık skoru",
        "res_room": "Oda önerisi",
        "res_srv": "Hizmet önerisi",
        "res_rev": "Tahmini gelir · {stay} gece",
        "rev_risk": "Risk altı: ₺{rev:,}",
        "rev_safe": "Güvenli rezervasyon",
        "al_d_t": "Rezervasyon takip gerektirir",
        "al_d_d1": "İptal olasılığı",
        "al_d_d2": "Check-in öncesinde misafirle iletişime geçin. Kişisel bir karşılama ve esneklik sunmak bu rezervasyonu güvence altına alabilir.",
        "al_w_t": "Takip önerilir",
        "al_w_d1": "İptal olasılığı",
        "al_w_d2": "Rezervasyon onayını 48 saat içinde kişisel bir mesajla pekiştirin.",
        "al_o_t": "Rezervasyon stabil",
        "al_o_d1": "İptal olasılığı",
        "al_o_d2": "Standart prosedür yeterlidir. Misafir karşılamaya odaklanın.",
        "m_ds": "Günlük özet",
        "m_tot": "Toplam rezervasyon",
        "m_fi": "Feature importance",
        "m_arr_d": "Varış tarihi",
        "m_arr_m": "Varış ayı",
        "m_srp": "Segment risk profili",
        "m_r": "Risk",
        "m_lr": "LR katsayısı",
        "m_h": "Yüksek",
        "m_l": "Düşük",
        "m_m": "Orta",
        "m_vl": "Çok düşük",
        "m_n": "Nötr",
        "m_gs": "Misafir segmenti & hizmet yaklaşımı",
        "m_gs_d": "Her misafir segmenti farklı bir hizmet yaklaşımı gerektirir. Bağlılık skoru ve iptal riski birlikte değerlendirilerek en uygun strateji belirlenir.",
        "p_v_l": "Sadık & Güvenli",
        "p_v_t": "Yüksek bağlılık · Düşük risk",
        "p_v_d": "En değerli misafir segmenti. Otel deneyimine güveni yüksek, tekrar rezervasyon yapma olasılığı güçlü.",
        "p_v_s": "Tekrar müşteri · Özel istek ≥3 · Kısa lead time · Offline segment",
        "p_v_a": "Yaklaşım: Tutarlı ve eksiksiz hizmet. Bu misafir beklenti kurmuş — karşıla ve küçük bir sürpriz ekle. Konfor öncelikli.",
        "p_r_l": "Öncelikli İlgi",
        "p_r_t": "Yüksek bağlılık · Yüksek risk",
        "p_r_d": "Geçmişte memnun kalmış ama bu seferki rezervasyonda belirsizlik var. Proaktif iletişim kritik.",
        "p_r_s": "Tekrar müşteri · Çok uzun lead time · Azalan özel istek",
        "p_r_a": "Yaklaşım: Kişisel ve sıcak bir temas. Geçmiş konaklamaya atıf yapan bir mesaj veya esnek bir teklif güven tazeleyebilir.",
        "p_n_l": "Bağlılık Fırsatı",
        "p_n_t": "Düşük bağlılık · Düşük risk",
        "p_n_d": "İlk kez geliyor ve iptali düşük. Uzun vadeli ilişki için en verimli pencere.",
        "p_n_s": "İlk konaklama · Offline/corporate · Özel istek 1–2",
        "p_n_a": "Yaklaşım: İlk izlenim kalıcıdır. Kişisel bir dokunuş — yerel bir ikram, samimi karşılama — bağlılığın temelini atar.",
        "p_l_l": "Kişisel Dokunuş",
        "p_l_t": "Düşük bağlılık · Yüksek risk",
        "p_l_d": "Ne sık gelen ne de güvenli bir segment. Ancak beklenmedik bir sıcaklık bu misafiri en sadık müşteriye dönüştürebilir.",
        "p_l_s": "İlk konaklama · Online · Özel istek = 0 · Uzun lead time",
        "p_l_a": "Yaklaşım: Küçük ama samimi bir jest. Beklentisi düşük olduğu için sürpriz etkisi en yüksek bu segmentte. Akşam yemeği ikramı veya hoş geldiniz notu.",
        "g_title": "Hizmet rehberi — özel istek bazlı",
        "g_3_t": "3 ve üzeri özel istek",
        "g_3_d": "Misafir deneyimine önem veriyor. Tüm talepleri özenle karşıla, check-in öncesi hazırlık yap. Küçük bir kişisel dokunuş ekle.",
        "g_3_tag": "Konfor öncelikli hizmet",
        "g_1_t": "1–2 özel istek",
        "g_1_d": "Taleplerin biraz üstüne çık. Oda hazırlığında ekstra bir detay veya samimi bir karşılama memnuniyeti belirgin biçimde artırır.",
        "g_1_tag": "Kişiselleştirilmiş hizmet",
        "g_0_t": "Özel istek yok",
        "g_0_d": "Sürpriz etkisi en güçlü bu grupta. Küçük ve samimi bir jest — yerel bir ikram, el yazısı not — kalıcı bir izlenim bırakır.",
        "g_0_tag": "Sıcak karşılama önerilir",
        "c_t": "Model karşılaştırması",
        "c_b": "Logistic Regression — baseline",
        "c_s": "Random Forest ✓ seçilen",
        "f_q": "Sessiz",
        "f_q_d": "Sakin ve konforlu bir deneyim için üst katlar önerilir",
        "f_fam": "Kalabalık Aile",
        "f_fam_d": "Geniş alan ve bahçe erişimi için alt katlar önerilir",
        "f_mid": "Orta",
        "f_mid_d": "Standart oda ataması uygundur",
        "f_vip": "VIP",
        "f_vip_d": "Sadık misafir",
        "f_crit": "KRİTİK",
        "f_crit_d": "Öncelikli ilgi",
        "f_pot": "POTANSİYEL",
        "f_pot_d": "Bağlılık fırsatı",
        "f_spec": "ÖZEL İLGİ",
        "f_spec_d": "Kişisel dokunuş önerilir",
        "fs_3_t": "Konfor Odaklı Hizmet",
        "fs_3_d": "Misafirin tüm talepleri özenle karşılanmalı. Tercihlerini önceden not edin ve check-in öncesi hazırlık yapın.",
        "fs_3_b": "Yüksek beklenti · Eksiksiz karşıla",
        "fs_1_t": "Kişiselleştirilmiş Hizmet",
        "fs_1_d": "Misafirin taleplerine ek olarak küçük bir sürpriz katın. Oda hazırlığında kişisel bir dokunuş fark yaratır.",
        "fs_1_b": "Orta beklenti · Biraz daha fazlasını sun",
        "fs_0h_t": "Sıcak Karşılama Önerilir",
        "fs_0h_d": "Check-in sırasında samimi ve kişisel bir karşılama yapın. Küçük bir ikram veya hoş geldiniz notu unutulmaz bir ilk izlenim bırakır.",
        "fs_0h_b": "Beklenti düşük · Sürpriz etkisi yüksek",
        "fs_0l_t": "Standart Hizmet + Küçük Dokunuş",
        "fs_0l_d": "Standart hizmet yeterli olmakla birlikte küçük kişisel bir jest — yerel bir ikram ya da el yazısı bir not — misafir memnuniyetini belirgin şekilde artırır.",
        "fs_0l_b": "İlk izlenim · Bağlılık başlangıcı",
        "fr_h": "Yüksek olasılık",
        "fr_m": "Orta olasılık",
        "fr_l": "Düşük olasılık",
        "rm_f58": "Katlar 5–8",
        "rm_f68": "Katlar 6–8",
        "rm_f35": "Katlar 3–5",
        "rm_f24": "Katlar 2–4",
        "rm_f13": "Katlar 1–3",
        "rm_fg": "Zemin kat",
        "rm_q1": "Sakin koridor, manzaralı",
        "rm_q2": "Köşe oda, geniş pencere",
        "rm_m1": "Merkezi konum, standart konfor",
        "rm_m2": "Havuz manzarası",
        "rm_c1": "Geniş alan, çocuk dostu",
        "rm_c2": "Bahçe ve havuz erişimi",
        "m_rp": "Oda fiyatı"
    }
}

# Session State içinde dil yoksa, varsayılan olarak EN ata
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'EN'

# Pratik kullanım için t (text) değişkenine aktif sözlüğü ata
t = LANGUAGES[st.session_state['lang']]

st.set_page_config(page_title="Hotel Reservation Intelligence", page_icon="🏨", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Jost:wght@300;400;500&display=swap');

html,body,[class*="css"]{font-family:'Jost',sans-serif!important;}
.stApp{background-color:#f5f0e8!important;}

[data-testid="stSidebar"]{background:#0e1a35!important;border-right:1px solid rgba(201,168,76,0.2)!important;}
[data-testid="stSidebar"] *{color:#fff!important;}

.stButton>button{background:#6b1a2a!important;color:#fff!important;border:none!important;border-radius:2px!important;padding:0.75rem 2rem!important;font-family:'Jost',sans-serif!important;font-size:0.75rem!important;font-weight:500!important;letter-spacing:2.5px!important;text-transform:uppercase!important;width:100%!important;}
.stButton>button:hover{background:#8a2238!important;}



div[data-testid="stCheckbox"] p, 
div[data-testid="stCheckbox"] span,
.stCheckbox label p {
    color: #0e1a35 !important;
    font-weight: 200 !important;
}



.stSelectbox>div>div{background:#f5f0e8!important;border:1px solid #ede6d6!important;border-radius:2px!important;color:#1a1208!important;}
.stSelectbox>div>div>div{color:#1a1208!important;}
.stNumberInput>div>div>input{background:#f5f0e8!important;border:1px solid #ede6d6!important;border-radius:2px!important;color:#1a1208!important;font-family:'Jost',sans-serif!important;}
.stSlider>div>div>div>div{background:#6b1a2a!important;}
.stCheckbox>label>div{border-color:#6b1a2a!important;}

[data-testid="stMetric"]{background:#fff!important;border:1px solid #ede6d6!important;border-radius:2px!important;padding:1rem 1.2rem!important;}
[data-testid="stMetricLabel"]{color:#8a7a62!important;font-size:0.6rem!important;letter-spacing:2px!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:#0e1a35!important;font-family:'Cormorant Garamond',serif!important;font-size:1.8rem!important;font-weight:300!important;}

.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:2px solid #ede6d6!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#8a7a62!important;font-family:'Jost',sans-serif!important;font-size:0.72rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;padding:0.6rem 1.6rem!important;border-radius:2px 2px 0 0!important;}
.stTabs [aria-selected="true"]{background:#0e1a35!important;color:#fff!important;}

.stProgress>div>div>div>div{background:#6b1a2a!important;}
hr{border-color:#ede6d6!important;}
.block-container{padding-top:1.5rem!important;}

.hri-topbar{background:#0e1a35;padding:1rem 2rem;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #c9a84c;margin-bottom:1.5rem;border-radius:2px;}
.hri-logo-name{font-family:'Cormorant Garamond',serif;font-size:1.2rem;font-weight:600;color:#fff;letter-spacing:3px;}
.hri-logo-sub{font-size:0.58rem;color:#c9a84c;letter-spacing:3px;text-transform:uppercase;}
.hri-badge{font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#c9a84c;border:1px solid rgba(201,168,76,0.35);padding:4px 10px;}

.hri-card{background:#fff;border:1px solid #ede6d6;border-radius:2px;padding:1.3rem 1.5rem;margin-bottom:1rem;}
.hri-card-navy{border-top:3px solid #0e1a35;}
.hri-card-bordo{border-top:3px solid #6b1a2a;}
.hri-card-red{border-top:3px solid #8b1a1a;}
.hri-card-title{font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:600;color:#0e1a35;letter-spacing:1px;margin-bottom:1rem;padding-bottom:7px;border-bottom:1px solid #ede6d6;}
.hri-field-label{font-size:0.62rem;letter-spacing:1.5px;text-transform:uppercase;color:#8a7a62;display:block;margin-bottom:4px;}

.hri-result{background:#fff;border:1px solid #ede6d6;border-radius:2px;padding:1.3rem;text-align:center;}
.hri-big{font-family:'Cormorant Garamond',serif;font-size:2.4rem;font-weight:300;line-height:1;}
.hri-rlabel{font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#8a7a62;margin-bottom:8px;}
.hri-pill{display:inline-block;padding:3px 12px;border-radius:2px;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;margin-top:8px;font-weight:500;}
.p-danger{background:#fdf0f2;color:#6b1a2a;border:1px solid rgba(107,26,42,0.2);}
.p-warn{background:#fdf8ee;color:#7a5010;border:1px solid rgba(185,132,58,0.3);}
.p-ok{background:#f0f6f0;color:#1a5535;border:1px solid rgba(26,85,53,0.2);}
.p-navy{background:#eef2f8;color:#0e1a35;border:1px solid rgba(14,26,53,0.2);}
.p-gold{background:#faf6ec;color:#b8943a;border:1px solid rgba(201,168,76,0.3);}

.hri-alert{border-radius:2px;padding:1rem 1.5rem;font-size:0.84rem;line-height:1.9;border-left:3px solid;margin-top:1rem;}
.al-danger{background:#fdf0f2;border-color:#6b1a2a;color:#4a0f1a;}
.al-warn{background:#fdf8ee;border-color:#c9a84c;color:#5a3c0a;}
.al-ok{background:#f0f6f0;border-color:#2d7a50;color:#1a4a2e;}

.hri-room{background:#f5f0e8;border:1px solid #ede6d6;border-radius:2px;padding:9px 10px;margin-bottom:7px;}
.hri-room-name{font-weight:500;font-size:0.85rem;color:#0e1a35;}
.hri-room-floor{font-size:0.68rem;color:#b8943a;letter-spacing:0.5px;margin:2px 0;}
.hri-room-note{font-size:0.72rem;color:#4a3f2f;}

.hri-service-card{background:#fff;border:1px solid #ede6d6;border-radius:2px;padding:1.2rem 1.4rem;margin-bottom:8px;border-left:3px solid #c9a84c;}
.hri-service-title{font-family:'Cormorant Garamond',serif;font-size:0.95rem;font-weight:600;color:#0e1a35;margin-bottom:6px;}
.hri-service-body{font-size:0.8rem;color:#4a3f2f;line-height:1.8;}
.hri-service-tag{font-size:0.62rem;letter-spacing:1px;text-transform:uppercase;color:#8a7a62;margin-top:8px;display:block;}

.hri-panel{background:#fff;border:1px solid #ede6d6;border-radius:2px;padding:1.3rem 1.5rem;margin-bottom:1rem;}
.hri-panel-title{font-family:'Cormorant Garamond',serif;font-size:1rem;font-weight:600;color:#0e1a35;letter-spacing:1px;margin-bottom:1rem;padding-bottom:7px;border-bottom:1px solid #ede6d6;}

.hri-divider{font-size:0.6rem;letter-spacing:2.5px;text-transform:uppercase;color:#8a7a62;margin:1.5rem 0 1rem;display:flex;align-items:center;gap:12px;}
.hri-divider::after{content:'';flex:1;height:1px;background:#ede6d6;}

.hri-action{border-radius:2px;padding:1.2rem 1.4rem;background:#0e1a35;margin-bottom:4px;}
.ac-high{border-top:3px solid #6b1a2a;}
.ac-mid{border-top:3px solid #c9a84c;}
.ac-low{border-top:3px solid #2d7a50;}
.hri-action-title{font-family:'Cormorant Garamond',serif;font-size:1rem;color:#fff;margin-bottom:8px;font-weight:400;letter-spacing:1px;}
.hri-action-body{font-size:0.76rem;color:rgba(255,255,255,0.65);line-height:1.9;}
.hri-action-tag{font-size:0.62rem;letter-spacing:1.5px;text-transform:uppercase;margin-top:10px;font-weight:500;}

.mq{border-radius:2px;padding:1rem 1.2rem;background:#0e1a35;margin-bottom:4px;}
.mq-label{font-size:0.58rem;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;font-weight:500;}
.mq-title{font-family:'Cormorant Garamond',serif;font-size:0.95rem;font-weight:600;color:#fff;margin-bottom:5px;}
.mq-desc{font-size:0.73rem;color:rgba(255,255,255,0.7);line-height:1.7;}
.mq-sigs{font-size:0.68rem;color:rgba(255,255,255,0.5);line-height:1.8;margin-top:6px;}
.mq-approach{font-size:0.72rem;color:rgba(255,255,255,0.75);line-height:1.7;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.1);}
.mq-vip{border-top:3px solid #2d7a50;}.mq-vip .mq-label{color:#5aaa7e;}
.mq-risk{border-top:3px solid #c9a84c;}.mq-risk .mq-label{color:#c9a84c;}
.mq-new{border-top:3px solid #378add;}.mq-new .mq-label{color:#85b7eb;}
.mq-lost{border-top:3px solid #6b1a2a;}.mq-lost .mq-label{color:#e8a0b0;}

.seg-table{width:100%;font-size:0.8rem;border-collapse:collapse;}
.seg-table th{text-align:left;font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#8a7a62;padding:6px 8px;border-bottom:1px solid #ede6d6;}
.seg-table td{padding:7px 8px;border-bottom:1px solid #ede6d6;color:#4a3f2f;}
.seg-table tr:last-child td{border:none;}

.perf-table{width:100%;font-size:0.8rem;border-collapse:collapse;}
.perf-table th{text-align:left;font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#8a7a62;padding:7px 10px;border-bottom:2px solid #ede6d6;}
.perf-table td{padding:9px 10px;border-bottom:1px solid #ede6d6;color:#4a3f2f;}
.perf-table tr.winner td{color:#0e1a35;font-weight:500;}
.perf-table tr:last-child td{border-bottom:none;}

.hri-sidebar-chip{background:rgba(255,255,255,0.06);border:1px solid rgba(201,168,76,0.2);border-radius:2px;padding:8px 12px;margin-bottom:6px;}
.hri-chip-label{font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:#c9a84c!important;margin-bottom:2px;}
.hri-chip-val{font-size:0.9rem;font-weight:500;color:#fff!important;}

.hri-footer{text-align:center;padding:1.2rem;border-top:1px solid #ede6d6;font-size:0.62rem;letter-spacing:2px;text-transform:uppercase;color:#8a7a62;margin-top:2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    mp, sp = os.path.join(base,"rf_model.pkl"), os.path.join(base,"scaler.pkl")
    if not os.path.exists(mp): return None, None
    with open(mp,"rb") as f: model=pickle.load(f)
    with open(sp,"rb") as f: scaler=pickle.load(f)
    return model, scaler

model, scaler = load_model()

FEATURE_ORDER = [
    'no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights',
    'required_car_parking_space','lead_time','arrival_year','arrival_month',
    'arrival_date','repeated_guest','no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests',
    'type_of_meal_plan_Meal Plan 2','type_of_meal_plan_Meal Plan 3',
    'type_of_meal_plan_Not Selected','room_type_reserved_Room_Type 2',
    'room_type_reserved_Room_Type 3','room_type_reserved_Room_Type 4',
    'room_type_reserved_Room_Type 5','room_type_reserved_Room_Type 6',
    'room_type_reserved_Room_Type 7','market_segment_type_Complementary',
    'market_segment_type_Corporate','market_segment_type_Offline','market_segment_type_Online'
]

def get_rooms():
    return {
        "Sessiz":    [{"n":"Room Type 1","f":t["rm_f58"],"d":t["rm_q1"]},
                      {"n":"Room Type 4","f":t["rm_f68"],"d":t["rm_q2"]}],
        "Orta":      [{"n":"Room Type 2","f":t["rm_f35"],"d":t["rm_m1"]},
                      {"n":"Room Type 3","f":t["rm_f24"],"d":t["rm_m2"]}],
        "Kalabalik": [{"n":"Room Type 5","f":t["rm_f13"],"d":t["rm_c1"]},
                      {"n":"Room Type 6","f":t["rm_fg"],"d":t["rm_c2"]}],
    }

def predict(inp):
    if model is None:
        return None
    row = {c: 0 for c in FEATURE_ORDER}
    for k, v in inp.items():
        if k in row:
            row[k] = v
    X = pd.DataFrame([row])[FEATURE_ORDER]
    return model.predict_proba(scaler.transform(X))[0][1]


def noise_profile(adults, children):
    if children == 0 and adults == 1:
        return (t["f_q"], "p-navy", "Sessiz", t["f_q_d"])
    elif children == 2:
        return (t["f_fam"], "p-warn", "Kalabalik", t["f_fam_d"])
    else:
        return (t["f_mid"], "p-gold", "Orta", t["f_mid_d"])


def loyalty_score(repeated, prev_ok, special, parking):
    W_SPECIAL  = 0.6094
    W_REPEATED = 0.2053
    W_PARKING  = 0.1376
    W_PREV_OK  = 0.0479

    special_norm  = min(special, 5) / 5.0
    prev_ok_norm  = min(prev_ok, 5) / 5.0
    repeated_norm = float(repeated)
    parking_norm  = float(parking)

    score = (
        special_norm  * W_SPECIAL  +
        repeated_norm * W_REPEATED +
        parking_norm  * W_PARKING  +
        prev_ok_norm  * W_PREV_OK
    ) * 100

    return round(score, 1)


def quadrant(loyalty, prob):
    LOYALTY_THRESHOLD = 50.0
    RISK_THRESHOLD    = 0.375 

    hl = loyalty >= LOYALTY_THRESHOLD
    hr = prob    >= RISK_THRESHOLD

    if     hl and not hr: return t["f_vip"],    "p-ok",     t["f_vip_d"]
    elif   hl and     hr: return t["f_crit"],   "p-warn",   t["f_crit_d"]
    elif not hl and not hr: return t["f_pot"],  "p-navy",   t["f_pot_d"]
    else:                   return t["f_spec"], "p-danger", t["f_spec_d"]


def service_recommendation(special, prob):
    RISK_THRESHOLD = 0.375

    if special >= 3:
        return (t["fs_3_t"], "ac-low", t["fs_3_d"], t["fs_3_b"])
    elif special >= 1:
        return (t["fs_1_t"], "ac-mid", t["fs_1_d"], t["fs_1_b"])
    else:
        if prob >= RISK_THRESHOLD:
            return (t["fs_0h_t"], "ac-high", t["fs_0h_d"], t["fs_0h_b"])
        else:
            return (t["fs_0l_t"], "ac-mid", t["fs_0l_d"], t["fs_0l_b"])


def risk_info(p):
    if p >= 0.65:  return t["fr_h"], "p-danger", "#6b1a2a", "al-danger"
    if p >= 0.375: return t["fr_m"], "p-warn",   "#b8943a", "al-warn"
    return                t["fr_l"], "p-ok",     "#2d7a50", "al-ok"


# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hri-topbar">
  <div>
    <div class="hri-logo-name">Hotel Reservation Intelligence</div>
    <div class="hri-logo-sub">{t["app_sub"]}</div>
  </div>
  <div class="hri-badge">2026 · v2.0</div>
</div>
""", unsafe_allow_html=True)

# ── DİL SEÇİMİ (SAĞ ÜST KÖŞE) ────────────────────────────────────────────────
col_bosluk, col_dil = st.columns([8, 1.5])
with col_dil:
    secilen_dil_etiket = st.selectbox(
        "",
        options=["EN", "TR"],
        index=0 if st.session_state['lang'] == 'EN' else 1,
        key="lang_selector",
        label_visibility="collapsed"
    )
    
    yeni_dil = "EN" if "EN" in secilen_dil_etiket else "TR"
    if yeni_dil != st.session_state['lang']:
        st.session_state['lang'] = yeni_dil
        st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.4rem;border-bottom:1px solid rgba(201,168,76,0.2);margin-bottom:1.2rem;'>
      <div style='width:48px;height:48px;background:#6b1a2a;border:1.5px solid #c9a84c;display:flex;align-items:center;justify-content:center;margin:0 auto 8px;font-family:Cormorant Garamond,serif;font-size:1.2rem;color:#c9a84c;letter-spacing:2px;'>AT</div>
      <div style='font-size:0.88rem;font-weight:500;color:#fff;letter-spacing:1px;'>Aslı Torun</div>
      <div style='font-size:0.7rem;color:rgba(255,255,255,0.45);margin-top:3px;'>Data Scientist · Hotel Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    for label,val in [(t["sb_m"],"Random Forest"),(t["sb_acc"],"%90.34"),
                      (t["sb_train"],"35,730"), (t["sb_cx"],"%33.2"),
                      (t["sb_crit"],"Lead time")]:
        st.markdown(f'<div class="hri-sidebar-chip"><div class="hri-chip-label">{label}</div>'
                    f'<div class="hri-chip-val">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if model is None:
        st.markdown(f'<div style="background:rgba(139,26,26,0.2);border:1px solid rgba(139,26,26,0.4);border-radius:2px;padding:9px 12px;font-size:0.72rem;color:#e8a0b0;">{t["sb_err"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:rgba(45,122,80,0.2);border:1px solid rgba(45,122,80,0.4);border-radius:2px;padding:9px 12px;font-size:0.72rem;color:#a0e8c0;">{t["sb_ok"]}</div>', unsafe_allow_html=True)

    st.markdown('<div style="color:rgba(255,255,255,0.25);font-size:0.6rem;text-align:center;margin-top:2rem;letter-spacing:1.5px;text-transform:uppercase;">Portfolio · 2026</div>', unsafe_allow_html=True)


# ── TABS ───────────────
tab_rec, tab_mgr = st.tabs([t["tab_rec"], t["tab_mgr"]])


# ════════════════════════════════════════════════════════════════════════════════
# RESEPSIYON
# ════════════════════════════════════════════════════════════════════════════════
with tab_rec:
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f'<div class="hri-card hri-card-navy"><div class="hri-card-title">{t["r_det"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="hri-field-label">{t["i_lead"]}</span>', unsafe_allow_html=True)
        lead_time = st.slider("", 0, 500, 90, key="lead", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_price"]}</span>', unsafe_allow_html=True)
        avg_price = st.number_input("", 50.0, 2000.0, 150.0, 25.0, key="price", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_seg"]}</span>', unsafe_allow_html=True)
        segment = st.selectbox("", ["Online","Offline","Corporate","Complementary","Aviation"], key="seg", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_meal"]}</span>', unsafe_allow_html=True)
        meal = st.selectbox("", ["Meal Plan 1","Meal Plan 2","Meal Plan 3","Not Selected"], key="meal", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_room"]}</span>', unsafe_allow_html=True)
        room_type = st.selectbox("", [f"Room Type {i}" for i in range(1,8)], key="room", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="hri-card hri-card-bordo"><div class="hri-card-title">{t["r_prof"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="hri-field-label">{t["i_adults"]}</span>', unsafe_allow_html=True)
        adults = st.number_input("", 1, 4, 2, key="adults", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_child"]}</span>', unsafe_allow_html=True)
        children = st.number_input("", 0, 10, 0, key="children", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_week"]}</span>', unsafe_allow_html=True)
        week_nights = st.number_input("", 0, 17, 2, key="weekn", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_wknd"]}</span>', unsafe_allow_html=True)
        weekend_nights = st.number_input("", 0, 7, 1, key="weekendn", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_spec"]}</span>', unsafe_allow_html=True)
        special = st.slider("", 0, 5, 1, key="special", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="hri-card hri-card-red"><div class="hri-card-title">{t["r_hist"]}</div>', unsafe_allow_html=True)
        parking  = st.checkbox(t["i_park"], key="parking")
        repeated = st.checkbox(t["i_rep"], key="repeated")
        st.markdown(f'<span class="hri-field-label">{t["i_pcx"]}</span>', unsafe_allow_html=True)
        prev_cx = st.number_input("", 0, 10, 0, key="prevcx", label_visibility="collapsed")
        st.markdown(f'<span class="hri-field-label">{t["i_pok"]}</span>', unsafe_allow_html=True)
        prev_ok = st.number_input("", 0, 20, 0, key="prevok", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    analyze = st.button(t["btn_ana"], type="primary")

    if analyze:
        if model is None:
            st.error(t["err_file"])
            st.stop()

        inp = {
            "no_of_adults": adults,
            "no_of_children": children,
            "no_of_weekend_nights": weekend_nights,
            "no_of_week_nights": week_nights,
            "required_car_parking_space": int(parking),
            "lead_time": lead_time,
            "arrival_year": 2026,
            "arrival_month": 8,
            "arrival_date": 16,
            "repeated_guest": int(repeated),
            "no_of_previous_cancellations": prev_cx,
            "no_of_previous_bookings_not_canceled": prev_ok,
            "avg_price_per_room": avg_price,
            "no_of_special_requests": special,
            f"type_of_meal_plan_{meal}": 1 if meal != "Meal Plan 1" else 0,
            f"room_type_reserved_{room_type.replace(' ', '_')}": 1 if room_type != "Room Type 1" else 0,
            f"market_segment_type_{segment}": 1,
        }

        prob     = predict(inp)
        prob_pct = round(prob*100)
        risk_text, risk_pill, risk_color, alert_cls = risk_info(prob)

        profile_text, profile_pill, room_key, profile_tip = noise_profile(adults, children)
        loyalty = loyalty_score(repeated, prev_ok, special, parking)
        q_code, q_pill, q_label = quadrant(loyalty, prob)
        svc_title, svc_cls, svc_body, svc_tag = service_recommendation(special, prob)

        stay    = week_nights + weekend_nights
        revenue = avg_price * stay

        st.markdown(f'<div class="hri-divider">{t["res_ana"]}</div>', unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4, gap="small")

        with r1:
            st.markdown(f"""<div class="hri-result">
              <div class="hri-rlabel">{t["res_cx"]}</div>
              <div class="hri-big" style="color:{risk_color};">{prob_pct}%</div>
              <div><span class="hri-pill {risk_pill}">{risk_text}</span></div>
            </div>""", unsafe_allow_html=True)
            

        with r2:
            icons = {t["f_q"]:"🤫", t["f_fam"]:"👨‍👩‍👧‍👦", t["f_mid"]:"🔉"}
            st.markdown(f"""<div class="hri-result">
              <div class="hri-rlabel">{t["res_gt"]}</div>
              <div class="hri-big" style="font-size:2rem;">{icons.get(profile_text,'🔉')}</div>
              <div><span class="hri-pill {profile_pill}">{profile_text}</span></div>
              <div style="font-size:0.72rem;color:#8a7a62;margin-top:8px;">{profile_tip}</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            loyalty_color = "#2d7a50" if loyalty>=60 else "#b8943a" if loyalty>=30 else "#6b1a2a"
            st.markdown(f"""<div class="hri-result">
              <div class="hri-rlabel">{t["res_gl"]}</div>
              <div class="hri-big" style="color:{loyalty_color};font-size:1.4rem;padding-top:8px;">{q_label}</div>
              <div><span class="hri-pill {q_pill}">{q_code}</span></div>
              <div style="font-size:0.7rem;color:#8a7a62;margin-top:8px;">{t["res_loy"]}: {loyalty}/100</div>
            </div>""", unsafe_allow_html=True)

        with r4:
            rooms_dict = get_rooms()
            rooms_html = "".join([
                f'<div class="hri-room"><div class="hri-room-name">{r["n"]}</div>'
                f'<div class="hri-room-floor">{r["f"]}</div>'
                f'<div class="hri-room-note">{r["d"]}</div></div>'
                for r in rooms_dict[room_key]
            ])
            st.markdown(f'<div class="hri-result" style="text-align:left;">'
                        f'<div class="hri-rlabel" style="text-align:center;">{t["res_room"]}</div>'
                        f'{rooms_html}</div>', unsafe_allow_html=True)

        # ── HİZMET ÖNERİSİ ──
        st.markdown(f'<div class="hri-divider">{t["res_srv"]}</div>', unsafe_allow_html=True)

        sv1, sv2 = st.columns([2, 1], gap="medium")

        with sv1:
            st.markdown(f"""<div class="hri-action {svc_cls}" style="height:100%;">
              <div class="hri-action-title">{svc_title}</div>
              <div class="hri-action-body">{svc_body}</div>
              <div class="hri-action-tag" style="color:rgba(255,255,255,0.45);">{svc_tag}</div>
            </div>""", unsafe_allow_html=True)

        with sv2:
            rev_color = "#6b1a2a" if prob>=.5 else "#2d7a50"
            rev_label = t["rev_risk"].format(rev=int(revenue * prob)) if prob>=.5 else t["rev_safe"]
            rev_pill  = "p-danger" if prob>=.5 else "p-ok"
            st.markdown(f"""<div class="hri-result" style="text-align:left;">
              <div class="hri-rlabel">{t["res_rev"].format(stay=stay)}</div>
              <div class="hri-big" style="color:#0e1a35;font-size:1.8rem;">₺{int(revenue):,}</div>
              <div><span class="hri-pill {rev_pill}">{rev_label}</span></div>
            </div>""", unsafe_allow_html=True)

        # ── UYARI ──
        if prob >= .65:
            st.markdown(
                f'<div class="hri-alert al-danger"><strong>{t["al_d_t"]}</strong> — '
                f'{t["al_d_d1"]} %{prob_pct}<br>{t["al_d_d2"]}</div>',
                unsafe_allow_html=True)
        elif prob >= .375:
            st.markdown(
                f'<div class="hri-alert al-warn"><strong>{t["al_w_t"]}</strong> — '
                f'{t["al_w_d1"]} %{prob_pct}<br>{t["al_w_d2"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="hri-alert al-ok"><strong>{t["al_o_t"]}</strong> — '
                f'{t["al_o_d1"]} %{prob_pct}<br>{t["al_o_d2"]}</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# YÖNETİCİ PANELİ
# ════════════════════════════════════════════════════════════════════════════════
with tab_mgr:
    st.markdown(f'<div class="hri-divider">{t["m_ds"]}</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4, gap="small")
    with m1: st.metric(t["m_tot"],"35,730","+2.3%")
    with m2: st.metric(t["sb_cx"],"%33.2","-1.1 pp",delta_color="inverse")
    with m3: st.metric(t["sb_acc"],"%90.3")
    with m4: st.metric(t["sb_crit"],"Lead time")

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2, gap="medium")

    with cl:
        st.markdown(f'<div class="hri-panel"><div class="hri-panel-title">{t["m_fi"]}</div>', unsafe_allow_html=True)
        
        # Veriyi hazırla
        fi_data = pd.DataFrame({
            "Feature": ["Lead time", t["m_rp"], t["i_spec"], t["m_arr_d"], t["m_arr_m"]],
            "Importance": [30.9, 14.6, 11.5, 9.2, 8.6]
        }).sort_values(by="Importance", ascending=True)

        # Plotly Yatay Bar Grafiği
        fig_fi = go.Figure(go.Bar(
            x=fi_data["Importance"],
            y=fi_data["Feature"],
            orientation='h',
            marker=dict(color=['#8a7a62', '#b8943a', '#8b1a1a', '#6b1a2a', '#0e1a35']),
            text=[f"%{val}" for val in fi_data["Importance"]],
            textposition='auto',
            insidetextfont=dict(family='Jost', color='white', size=11),
            outsidetextfont=dict(family='Jost', color='#0e1a35', size=11)
        ))
        
        fig_fi.update_layout(
            height=260, 
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(family='Jost', size=12, color='#4a3f2f'))
        )
        
        # Grafiği çizdir
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)

    with cr:
        st.markdown(f"""<div class="hri-panel"><div class="hri-panel-title">{t["m_srp"]}</div>
        <table class="seg-table">
          <tr><th>Segment</th><th>{t["m_r"]}</th><th>{t["m_lr"]}</th></tr>
          <tr><td>Online</td><td><span class="hri-pill p-danger">{t["m_h"]}</span></td><td style="color:#6b1a2a">+0.84</td></tr>
          <tr><td>Offline</td><td><span class="hri-pill p-ok">{t["m_l"]}</span></td><td style="color:#2d7a50">−0.84</td></tr>
          <tr><td>Corporate</td><td><span class="hri-pill p-warn">{t["m_m"]}</span></td><td>−0.10</td></tr>
          <tr><td>Aviation</td><td><span class="hri-pill p-ok">{t["m_vl"]}</span></td><td>{t["m_n"]}</td></tr>
          <tr><td>Complementary</td><td><span class="hri-pill p-ok">{t["m_vl"]}</span></td><td style="color:#2d7a50">−0.34</td></tr>
        </table></div>""", unsafe_allow_html=True)

    # ── MİSAFİR SEGMENTİ MATRİSİ ──
    st.markdown(f'<div class="hri-divider">{t["m_gs"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.72rem;color:#8a7a62;margin-bottom:1rem;line-height:1.7;">{t["m_gs_d"]}</div>', unsafe_allow_html=True)

    mq1, mq2 = st.columns(2, gap="small")
    with mq1:
        st.markdown(f"""<div class="mq mq-vip">
          <div class="mq-label">{t["p_v_l"]}</div>
          <div class="mq-title">{t["p_v_t"]}</div>
          <div class="mq-desc">{t["p_v_d"]}</div>
          <div class="mq-sigs">{t["p_v_s"]}</div>
          <div class="mq-approach">{t["p_v_a"]}</div>
        </div>""", unsafe_allow_html=True)
    with mq2:
        st.markdown(f"""<div class="mq mq-risk">
          <div class="mq-label">{t["p_r_l"]}</div>
          <div class="mq-title">{t["p_r_t"]}</div>
          <div class="mq-desc">{t["p_r_d"]}</div>
          <div class="mq-sigs">{t["p_r_s"]}</div>
          <div class="mq-approach">{t["p_r_a"]}</div>
        </div>""", unsafe_allow_html=True)

    mq3, mq4 = st.columns(2, gap="small")
    with mq3:
        st.markdown(f"""<div class="mq mq-new">
          <div class="mq-label">{t["p_n_l"]}</div>
          <div class="mq-title">{t["p_n_t"]}</div>
          <div class="mq-desc">{t["p_n_d"]}</div>
          <div class="mq-sigs">{t["p_n_s"]}</div>
          <div class="mq-approach">{t["p_n_a"]}</div>
        </div>""", unsafe_allow_html=True)
    with mq4:
        st.markdown(f"""<div class="mq mq-lost">
          <div class="mq-label">{t["p_l_l"]}</div>
          <div class="mq-title">{t["p_l_t"]}</div>
          <div class="mq-desc">{t["p_l_d"]}</div>
          <div class="mq-sigs">{t["p_l_s"]}</div>
          <div class="mq-approach">{t["p_l_a"]}</div>
        </div>""", unsafe_allow_html=True)

    # ── HİZMET REHBERİ ──
    st.markdown(f'<div class="hri-divider">{t["g_title"]}</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3, gap="small")
    with a1:
        st.markdown(f'<div class="hri-action ac-low"><div class="hri-action-title">{t["g_3_t"]}</div>'
                    f'<div class="hri-action-body">{t["g_3_d"]}</div>'
                    f'<div class="hri-action-tag" style="color:#5aaa7e;">{t["g_3_tag"]}</div></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div class="hri-action ac-mid"><div class="hri-action-title">{t["g_1_t"]}</div>'
                    f'<div class="hri-action-body">{t["g_1_d"]}</div>'
                    f'<div class="hri-action-tag" style="color:#c9a84c;">{t["g_1_tag"]}</div></div>', unsafe_allow_html=True)
    with a3:
        st.markdown(f'<div class="hri-action ac-high"><div class="hri-action-title">{t["g_0_t"]}</div>'
                    f'<div class="hri-action-body">{t["g_0_d"]}</div>'
                    f'<div class="hri-action-tag" style="color:#e8a0b0;">{t["g_0_tag"]}</div></div>', unsafe_allow_html=True)

    # ── MODEL KARŞILAŞTIRMA ──
    st.markdown(f'<div class="hri-divider">{t["c_t"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hri-panel"><table class="perf-table">'
                f'<tr><th>Model</th><th>Accuracy</th><th>Precision (iptal)</th><th>Recall (iptal)</th><th>F1 (iptal)</th></tr>'
                f'<tr><td>{t["c_b"]}</td><td>%78.1</td><td>%64</td><td>%77</td><td>%70</td></tr>'
                f'<tr class="winner"><td>{t["c_s"]}</td><td>%90.3</td><td>%88</td><td>%82</td><td>%85</td></tr>'
                f'</table></div>', unsafe_allow_html=True)

    st.markdown('<div class="hri-footer">Aslı Torun · Hotel Reservation Intelligence · 2026 · Data Science Portfolio</div>',
                unsafe_allow_html=True)
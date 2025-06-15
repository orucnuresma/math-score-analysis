# Gerekli kütüphaneleri yüklüyorum
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import scipy.stats as stats

# --- 1. VERİYİ YÜKLEME ---

# CSV dosyasını yüklüyorum
veri = pd.read_csv("StudentsPerformance.csv")

# Sadece "math score" sütununu alıyorum ve eksik değerleri çıkarıyorum
puanlar = veri["math score"].dropna().tolist()


# --- 2. TEMEL İSTATİSTİK FONKSİYONLARI ---

# Ortalama (mean) hesaplayan fonksiyon
def ortalama(dizi):
    toplam = 0
    for eleman in dizi:
        toplam += eleman
    return toplam / len(dizi)  #mean = (x₁ + x₂ + ... + xn) / n”

# Ortanca (median) hesaplayan fonksiyon
def medyan(dizi):
    sirali = sorted(dizi)
    n = len(sirali)
    if n % 2 == 1: # odd length
        return sirali[n // 2]
    else: # even
        return (sirali[n // 2 - 1] + sirali[n // 2]) / 2

# Varyans hesaplayan fonksiyon (n-1 ile bölerek örneklem varyansı)
def varyans(dizi):
    ort = ortalama(dizi)
    toplam_fark_kare = 0
    for eleman in dizi:
        toplam_fark_kare += (eleman - ort) ** 2
    return toplam_fark_kare / (len(dizi) - 1)

# Standart sapma (standard deviation)
def standart_sapma(dizi):
    return math.sqrt(varyans(dizi))

# Standart hata (standard error)
def standart_hata(dizi):
    return standart_sapma(dizi) / math.sqrt(len(dizi))

# Ortalama için güven aralığı (t dağılımı ile)
def guven_araligi_ortalama(dizi, guven=0.95):
    n = len(dizi)
    ort = ortalama(dizi)
    se = standart_hata(dizi)
    t_degeri = stats.t.ppf((1 + guven) / 2, df=n - 1)
    pay = t_degeri * se  #  mean ± t * standard error.
    return ort - pay, ort + pay

# Varyans için güven aralığı (chi- square  dağılımı ile)
def guven_araligi_varyans(dizi, guven=0.95):
    n = len(dizi)
    var = varyans(dizi)
    alpha = 1 - guven
    chi2_alt = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_ust = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    alt = (n - 1) * var / chi2_ust
    ust = (n - 1) * var / chi2_alt
    return alt, ust

# Minimum örneklem büyüklüğü hesaplama (belirli hata ve güven düzeyi için)
def orneklem_buyuklugu(sapma, hata=0.1, guven=0.90):
    z_degeri = stats.norm.ppf((1 + guven) / 2)
    n = (z_degeri * sapma / hata) ** 2 #n = (z * σ / margin of error) squared.
    return math.ceil(n)

# Hipotez testi (tek örneklem t-testi)
def hipotez_testi(dizi, ortalama_hipotez):
    ort = ortalama(dizi)
    se = standart_hata(dizi)
    t_istatistik = (ort - ortalama_hipotez) / se #t = (sample mean - hypothesized mean) / standard error.
    p_degeri = 2 * (1 - stats.t.cdf(abs(t_istatistik), df=len(dizi) - 1))
    return t_istatistik, p_degeri


# --- 3. BETİMSEL İSTATİSTİKLERİ YAZDIRIYORUM ---

print("Ortalama:", ortalama(puanlar))
print("Medyan:", medyan(puanlar))
print("Varyans:", varyans(puanlar))
print("Standart Sapma:", standart_sapma(puanlar))
print("Standart Hata:", standart_hata(puanlar))


# --- 4. AYKIRI DEĞER ANALİZİ (IQR) ---

sirali = sorted(puanlar)
q1_index = int(0.25 * len(sirali))
q3_index = int(0.75 * len(sirali))
Q1 = sirali[q1_index]
Q3 = sirali[q3_index]
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR
aykirilar = [x for x in puanlar if x < alt_sinir or x > ust_sinir]

print("\nAykırı Değer Analizi (IQR):")
print("Alt sınır:", round(alt_sinir, 2))
print("Üst sınır:", round(ust_sinir, 2))
print("Aykırı Değer Sayısı:", len(aykirilar))
print("Aykırı Değerler:", aykirilar)


# --- 5. GÜVEN ARALIKLARI ---

alt_ort, ust_ort = guven_araligi_ortalama(puanlar)
print("\n95% Güven Aralığı (Ortalama):", round(alt_ort,2), "-", round(ust_ort,2))

alt_var, ust_var = guven_araligi_varyans(puanlar)
print("95% Güven Aralığı (Varyans):", round(alt_var,2), "-", round(ust_var,2))


# --- 6. ÖRNEKLEM BÜYÜKLÜĞÜ ---

std = standart_sapma(puanlar)
n_gerekli = orneklem_buyuklugu(std)
print("\nMinimum Örneklem Büyüklüğü (%90 güven, ±0.1 hata):", n_gerekli)


# --- 7. HİPOTEZ TESTİ (ortalama 65 mi?) ---

t_istat, p_deger = hipotez_testi(puanlar, 65)
print("\nHipotez Testi (H0: Ortalama = 65)")
print("t-istatistiği:", round(t_istat, 3))
print("p-değeri:", round(p_deger, 4))
if p_deger < 0.05:
    print("Sonuç: H0 reddedildi. Ortalama 65'ten farklı.")
else:
    print("Sonuç: H0 reddedilemedi. Ortalama 65 olabilir.")


# --- 8. GRAFİKLER ---



# Histogram (Dağılım + Yoğunluk Eğrisi)
plt.figure(figsize=(8, 5))
sns.histplot(puanlar, bins=15, kde=True, color='skyblue')
plt.title("Math Score Histogram")
plt.xlabel("Math Score")
plt.ylabel("Frekans")
plt.grid(True)
plt.show()

# Boxplot (Estetik çizim)
plt.figure(figsize=(6, 4))
sns.boxplot(x=puanlar, color='lightcoral')
plt.title("Math Score Boxplot")
plt.xlabel("Math Score")
plt.grid(True)
plt.show()


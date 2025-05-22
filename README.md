# KiraTahminML – Makine Öğrenmesi ile Kira Fiyatı Tahmini

Bu proje, Delhi bölgesindeki kiralık konutlara ait özellikleri kullanarak, konutların kira fiyatlarını tahmin etmeyi amaçlamaktadır. Farklı regresyon modelleri karşılaştırılarak en iyi sonuç veren model belirlenmiştir.

## 🔍 Proje Amacı

Emlak piyasasında yer alan kira fiyatlarını, konutun fiziksel ve konumsal özelliklerine göre tahmin etmek. Bu sayede hem kullanıcıların hem de emlak danışmanlarının karar destek sistemlerine katkı sağlanması hedeflenmiştir.

## 📊 Kullanılan Veri Kümesi

- **Toplam Gözlem Sayısı:** 17,890
- **Temel Özellikler:**
  - `size_sq_ft`: Konutun metrekaresi
  - `bedrooms`: Oda sayısı
  - `latitude` / `longitude`: Koordinatlar
  - `closest_mtero_station_km`: En yakın metro istasyonuna uzaklık
  - `AP_dist_km`, `Aiims_dist_km`, `NDRLW_dist_km`: Önemli noktalara uzaklıklar
  - `location_cluster`: Lokasyonun kümelenmiş hali

## 🛠️ Kullanılan Yöntemler ve Modeller

- **Veri Ön İşleme:**
  - Eksik verilerin temizlenmesi
  - Sayısal sütunların standardizasyonu (StandardScaler)
  - Özellik mühendisliği (lokasyon kümeleme vb.)

- **Denenen Regresyon Modelleri:**
  | Model                 | RMSE    | MAPE (%) | R² Skoru |
  |----------------------|---------|-----------|-----------|
  | Lineer Regresyon     | 9508.41 | 22.23     | 0.78      |
  | Ridge Regression     | 9508.71 | 22.23     | 0.79      |
  | Random Forest        | 7786.96 | 17.68     | 0.86      |
  | Yapay Sinir Ağı (MLP)| 9183.42 | 21.63     | 0.80      |
  | XGBoost Regressor    | 7991.78 | 17.88     | 0.85      |

- **En Başarılı Modeller:**  
  - Random Forest Regressor  
  - XGBoost Regressor

## 📈 Sonuç ve Değerlendirme

`price` değişkeni yüksek varyansa sahip olup, standart sapması ortalamanın bile üzerindedir. Random Forest Regressor ile elde edilen RMSE değeri (7,786 birim), ortalama kira fiyatı (33,451 birim) ile kıyaslandığında yaklaşık %17 sapmaya denk gelmektedir. Bu sonuç, değişkenliği yüksek bir veri seti için oldukça başarılı kabul edilmektedir.


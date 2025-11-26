# SHAMİR'İN SIR PAYLAŞIM ŞEMASI VE LSB ALGORİTMASININ ETKİLEŞİMLİ BİR UYGULAMASI VE GÖRSELLEŞTİRİLMESİ

**Karadeniz Teknik Üniversitesi Bilgisayar Mühendisliği Bölümü Algoritmalar Dersi Yarıyıl Sonu Serbest Ödevi**

| | |
|---|---|
| **Hazırlayan** | Ege Yolsal |
| **Ödev Sorumlusu** | Prof. Dr. Vasıf Nabiyev |

---

## 1. Projenin Amacı

Bu proje, gizli veri paylaşımı ve güvenli iletişim ihtiyaçlarını karşılamak üzere iki temel teknolojiyi entegre eder:

* **Shamir'in Gizli Paylaşım Şeması:** Veriyi matematiksel olarak güvenli şekilde parçalara böler ve en az *k* parça ile geri çözer.
* **LSB Steganografi:** Payları görsellerin piksellerine gizleyerek tespit edilebilirliği minimize eder.

**Hedefler:**
* Gizli verinin dağıtımında yüksek güvenlik sağlamak.
* Steganografi ile düşük görsel bozulma (yüksek PSNR) elde etmek.
* Esnek parametrelerle (*k*, *n*) kullanıcı dostu bir GUI sunmak.

---

## 2. Yöntem

### 2.1. Sistem Mimarisi

1.  **Kodlama:**
    * Orijinal görsel ham piksellere dönüştürülür ve metadata eklenir.
    * Shamir şeması ile *n* pay oluşturulur.
    * Her pay, LSB kullanılarak ayrı bir cover görsele gömülür.

2.  **Kod Çözme:**
    * En az *k* pay seçilir ve LSB'den veri çıkarılır.
    * Shamir ile sır birleştirilir.
    * Metadata ve piksel verisi orijinal görsele dönüştürülür.

### 2.2. Teknik Araçlar

* **Galois Alanı (GF(251)):** Polinom katsayıları için sonlu alan aritmetiği.
* **OpenCV:** Görsel işleme ve LSB operasyonları.
* **NumPy:** LSB operasyonları için bit düzeyinde işlemler ve dizi manipülasyonları.
* **Tkinter:** Kullanıcı arayüzü tasarımı.

---

## 3. Algoritmalar Hakkında Detaylar

Projenin temelini oluşturan algoritmalar `lastfast.py` dosyasında bulunmaktadır.

### 3.1. Pay Oluşturma Algoritması

**Girdi:** Gizlenecek veri (*secret_bytes*), eşik değeri (*k*), toplam pay sayısı (*n*).

**İşlem:**
1.  Her bir *secret_byte* ∈ *secret_bytes* için:
    a.  Rastgele *k-1* katsayı seç:
        `coeffs = [secret_byte, GF(251)'de rastgele_1, ..., rastgele_k-1]`
    b.  Her $x \in \{1, 2, ..., n\}$ için:
        $y = (coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ... + coeffs[k-1]*x^{k-1}) \mod 251$
    c.  Pay(x) objesine ($x, y$) çiftini ekle.
2.  Tüm payları döndür.

**Çıktı:** *n* adet Share nesnesi listesi (her biri *x* değeri ve hesaplanan *y* değerlerinin byte dizisini içerir).

### 3.2. Sır Birleştirme Algoritması

**Girdi:** Yeniden birleştirme için kullanılacak Share nesneleri listesi (*shares_list*), eşik değeri (*k*).

**İşlem:**
1.  Her orijinal sır byte'ına karşılık gelen pay verisi bloğu için:
    a.  Tüm paylardan ($x, y$) çiftlerini toplanır.
    b.  Lagrange interpolasyonu ile $x=0$'daki *y* değerini (sabit terim) hesapla:
        $L_j(0) = \prod_{m \ne j} (x_m / (x_m - x_j)) \mod 251$
        $secret\_byte = (\Sigma(y_j * L_j(0))) \mod 251$
    c.  *secret_byte*, *secret_bytes*'a eklenir.
2.  *secret_bytes* döndürülür.

**Çıktı:** Yeniden oluşturulmuş orijinal *secret_bytes*.

### 3.3. LSB Gömme Algoritması

**Girdi:** Taşıyıcı görüntü (*cover_image* piksel dizisi), gömülecek veri (*data_to_embed* byte dizisi).

**İşlem:**
1.  *data_to_embed*'in başına 4 byte'lık uzunluk bilgisi eklenir.
2.  Bu toplam veri, bir bit dizisine (*bits_to_embed*) dönüştürülür.
3.  Taşıyıcı görüntünün LSB kapasitesi kontrol edilir. Yetersizse, *bits_to_embed*'i alacak şekilde yeniden boyutlandırılır (NumPy ile verimli şekilde).
4.  Taşıyıcı görüntünün piksel renk bileşenlerinin LSB'leri sıfırlanır (AND 0xFE).
5.  *bits_to_embed* dizisindeki bitler, sırayla bu sıfırlanmış LSB'lere yazılır.

**Çıktı:** Stego-görüntü (içinde veri gizlenmiş piksel dizisi).

### 3.4. LSB Çıkarma Algoritması

**Girdi:** Stego-görüntü (*stego_image* piksel dizisi).

**İşlem:**
1.  Stego-görüntünün tüm piksel renk bileşenlerinden LSB'ler (her birinden 1 bit) çıkarılarak düz bir bit dizisi oluşturulur.
2.  Bu bit dizisi, NumPy'nin `packbits` fonksiyonu kullanılarak byte dizisine dönüştürülür (*extracted_bytes*).
3.  *extracted_bytes*'ın ilk 4 byte'ı okunarak gömülü verinin orijinal uzunluğu (*data_length*) elde edilir.
4.  *extracted_bytes*'tan *data_length* kadar byte (4 byte'lık başlıktan sonraki kısım) alınarak orijinal gömülü veri elde edilir.

**Çıktı:** Orijinal gömülü *data_to_embed* byte dizisi.

---

## Kaynaklar ve Bağlantılar

1.  [Shamir's Secret Sharing - Wikipedia](https://en.wikipedia.org/wiki/Shamir%27s_secret_sharing#Python_code)
2.  [LSB-Steganography](https://www.researchgate.net/publication/371671984_Steganography_in_Images_Using_LSB_Technique)
3.  [Tkinter - GUI](https://docs.python.org/3/library/tkinter.html)

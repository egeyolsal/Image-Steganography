import cv2
import numpy as np
from math import log10, sqrt, ceil
import secrets
import traceback
import json
import hashlib  # Checksum için eklendi

# --- Galois Field (GF257) Optimizasyonları ---
PRIME = 257

# Vektörleştirilmiş GF operasyonları
def gf_add_vec(a, b): return (a + b) % PRIME
def gf_sub_vec(a, b): return (a - b) % PRIME
gf_mul_vec = np.vectorize(lambda a, b: (a * b) % PRIME)
gf_inverse_vec = np.vectorize(lambda a: pow(a, PRIME-2, PRIME) if a != 0 else 0)

# --- Gizli Paylaşım Optimizasyonları ---
class Share:
    __slots__ = ('x', 'data')  # Bellek optimizasyonu
    def __init__(self, x, data):
        self.x = x
        self.data = data

def split_secret(secret_bytes, k, n):
    if not (1 < k <= n):
        raise ValueError("Geçersiz k ve n değerleri")
    
    # Tüm x değerlerini ve kuvvetlerini önceden hesapla
    x_values = np.arange(1, n+1, dtype=np.uint16)
    powers = np.zeros((n, k), dtype=np.uint16)
    for i in range(k):
        powers[:, i] = np.power(x_values, i, dtype=np.uint16) % PRIME
    
    shares = [Share(x, bytearray()) for x in x_values]
    secret_arr = np.frombuffer(secret_bytes, dtype=np.uint8)
    
    for byte in secret_arr:
        # Rastgele katsayıları vektörleştirilmiş olarak oluştur
        coeffs = np.concatenate([[byte], np.random.randint(0, PRIME, k-1, dtype=np.uint16)])
        
        # Polinom değerlerini matris çarpımı ile hesapla
        y_values = (coeffs * powers).sum(axis=1) % PRIME
        
        # Paylara yaz
        for share, y in zip(shares, y_values):
            share.data.extend(y.tobytes())
    
    return shares

def combine_shares(shares, k):
    # Tüm veriyi NumPy array olarak yükle
    share_data = np.frombuffer(shares[0].data, dtype=np.uint16).reshape(-1, 2)
    num_bytes = share_data.shape[0]
    
    # Lagrange interpolasyonunu vektörleştir
    x = np.array([s.x for s in shares], dtype=np.uint16)
    secret = np.zeros(num_bytes, dtype=np.uint8)
    
    for i in range(num_bytes):
        y = np.array([int.from_bytes(s.data[2*i:2*i+2], 'big') for s in shares], dtype=np.uint16)
        
        # Vektörleştirilmiş Lagrange hesabı
        j = np.arange(len(x))
        mask = j[:, None] != j[None, :]
        numerator = np.prod(x * mask, axis=1)
        denominator = np.prod((x - x[:, None]) * np.eye(len(x), dtype=np.uint16) + ~mask, axis=1)
        
        secret[i] = np.sum(y * gf_mul_vec(numerator, gf_inverse_vec(denominator))) % 256
        
    return secret.tobytes()

# --- LSB Steganografi Optimizasyonları ---
def embed_data(img, data):
    # Veriyi hazırla
    data_len = len(data).to_bytes(4, 'big')
    full_data = data_len + data + hashlib.md5(data).digest()  # Checksum eklendi
    
    # Bitleri NumPy ile işle
    bits = np.unpackbits(np.frombuffer(full_data, dtype=np.uint8))
    required_pixels = ceil(len(bits)/3)
    
    # Resmi tek seferde yeniden boyutlandır
    h, w = img.shape[:2]
    if h*w < required_pixels:
        new_size = int(np.sqrt(required_pixels)) + 1
        img = cv2.resize(img, (new_size, new_size))
    
    # Tüm pikselleri tek seferde işle
    flat = img.reshape(-1, 3)
    bits_padded = np.pad(bits, (0, flat.size - len(bits)), 'constant')
    flat = (flat & 0xFE) | bits_padded.reshape(-1, 3)
    
    return flat.reshape(img.shape)

def extract_data(img):
    # Tüm bitleri tek seferde çıkar
    bits = (img.reshape(-1, 3) & 1).ravel()
    
    # Veri uzunluğunu oku
    data_len = int(''.join(bits[:32].astype(str)), 2)
    
    # Veriyi doğrudan NumPy ile paketle
    data_bits = bits[32:32+data_len*8+128]  # 128 bit checksum
    data = np.packbits(data_bits[:data_len*8]).tobytes()
    checksum = np.packbits(data_bits[data_len*8:]).tobytes()
    
    if hashlib.md5(data).digest() != checksum:
        raise ValueError("Veri bozuk! Checksum uyuşmuyor.")
    
    return data

# --- PSNR HESAPLAMA ---
def calculate_psnr(img1, img2):

    # Görüntülerin None olup olmadığını kontrol et
    if img1 is None or img2 is None:
        print("PSNR hesaplaması için bir veya iki görüntü None (boş).")
        return -1 # veya bir hata fırlat

    # Boyut kontrolü
    if img1.shape != img2.shape:
        print(f"PSNR hesaplaması için görüntü boyutları farklı: {img1.shape} vs {img2.shape}")
        # img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        # print("İkinci görüntü, ilk görüntünün boyutuna getirildi.")
        return -1 # Veya boyutları eşitlemeden devam etme

    # Görüntüleri float tipine dönüştür
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf') 

    max_pixel_value = 255.0
    psnr = 10 * log10(max_pixel_value ** 2 / mse)
    return psnr

# --- Ana İş Akışı ---
DEFAULT_K = 3
DEFAULT_N = 6
METADATA_KEY_SHAPE = "shape"
METADATA_KEY_DTYPE = "dtype_str"


# --- Ana İş Akışı Optimizasyonları ---
def encode_image_raw_pixels(image_path, output_prefix, k=DEFAULT_K, n=DEFAULT_N):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    
    # Veriyi hazırla
    raw_pixels = img.tobytes()
    metadata = json.dumps({
        "shape": img.shape,
        "dtype": str(img.dtype),
        "checksum": hashlib.md5(raw_pixels).hexdigest()
    }).encode()
    
    # Tüm veriyi paketle
    secret_data = len(metadata).to_bytes(4, 'big') + metadata + raw_pixels
    
    # Paylaşımları oluştur
    shares = split_secret(secret_data, k, n)
    
    # Görüntüleri paralel oluştur
    base_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)  # Rastgele gürültü
    for share in shares:
        stego = embed_data(base_img.copy(), bytes(share.data))
        cv2.imwrite(f"{output_prefix}_{share.x}.png", stego)

def decode_image_raw_pixels(share_paths, output_path, original_image_for_psnr_path, k_threshold=DEFAULT_K, save_as_jpeg_quality=90):
    # Payları yükle
    shares = []
    for path in share_paths:
        img = cv2.imread(path)
        shares.append(Share(
            int(path.split('_')[-1].split('.')[0]),
            extract_data(img)
        ))
    
    # Sırrı birleştir
    secret_data = combine_shares(shares, k_threshold)
    
    # Metadata'yı parse et
    metadata_len = int.from_bytes(secret_data[:4], 'big')
    metadata = json.loads(secret_data[4:4+metadata_len].decode())
    pixels = secret_data[4+metadata_len:]
    
    # Checksum kontrolü
    if hashlib.md5(pixels).hexdigest() != metadata['checksum']:
        raise ValueError("Veri bütünlüğü bozuk!")
    
    # Görüntüyü yeniden oluştur
    img = np.frombuffer(pixels, dtype=np.dtype(metadata['dtype']))
    cv2.imwrite(output_path, img.reshape(metadata['shape']))


# Kullanım Örneği
if __name__ == "__main__":

    K_PARAM = 3
    N_PARAM = 6
    INPUT_IMAGE_FILENAME = "input.jpg"
    try:
        if cv2.imread(INPUT_IMAGE_FILENAME) is None:
            dummy_img = np.zeros((60, 80, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "Test!", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imwrite("input.png", dummy_img)
            # cv2.imwrite("input.jpg", dummy_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG testi için
            print(f"Test için '{INPUT_IMAGE_FILENAME}' oluşturuldu.")
            # INPUT_IMAGE_FILENAME = "input.jpg" # Eğer JPEG test ediyorsanız
    except Exception as e_img:
        print(f"Test görüntüsü oluşturulurken hata: {e_img}")

    SHARE_OUTPUT_PREFIX = "raw_pixel_share"
    RECONSTRUCTED_OUTPUT_TEMPLATE = "output_reconstructed_raw.{ext}"

    print(f"\n--- HAM PİKSEL KODLAMA (Giriş: {INPUT_IMAGE_FILENAME}) ---")
    try:
        encode_image_raw_pixels(INPUT_IMAGE_FILENAME, SHARE_OUTPUT_PREFIX, k=K_PARAM, n=N_PARAM)
    except FileNotFoundError as e_fnf:
        print(f"Kodlama Hatası: {e_fnf}")
    except Exception as e_encode:
        print(f"Kodlama sırasında genel hata: {e_encode}"); traceback.print_exc()

    print(f"\n--- HAM PİKSEL KOD ÇÖZME ---")
    share_files_for_decoding = [f"{SHARE_OUTPUT_PREFIX}_{i}.png" for i in range(1, K_PARAM + 1)]
    
    if share_files_for_decoding:
        try:
            decode_image_raw_pixels(share_files_for_decoding, 
                                    RECONSTRUCTED_OUTPUT_TEMPLATE, 
                                    INPUT_IMAGE_FILENAME,
                                    k_threshold=K_PARAM,
                                    save_as_jpeg_quality=85)
        except Exception as e_decode:
            print(f"Kod çözme sırasında genel hata: {e_decode}"); traceback.print_exc()
import cv2
import numpy as np
import secrets
from math import ceil

# --- Galois Field İşlemleri ---
def gf_add(a, b): return (a + b) % 257
def gf_sub(a, b): return (a - b) % 257
def gf_mul(a, b): return (a * b) % 257

def gf_inverse(a):
    if a == 0:
        raise ValueError("Sıfırın tersi yoktur")
    for i in range(1, 257):
        if gf_mul(a, i) == 1: 
            return i
    raise ValueError(f"Ters eleman bulunamadı: {a}")

def gf_div(a, b):
    return gf_mul(a, gf_inverse(b))

# --- Geliştirilmiş Gizli Paylaşım ---
class Share:
    def __init__(self, x, data):
        self.x = x
        self.data = data

def split_secret(secret_bytes, k, n):
    shares = [Share(x, bytearray()) for x in range(1, n+1)]
    
    for byte in secret_bytes:
        coeffs = [byte] + [secrets.randbelow(257) for _ in range(k-1)]
        
        for share in shares:
            y = sum(gf_mul(coeff, share.x**i) for i, coeff in enumerate(coeffs)) % 257
            share.data.extend(y.to_bytes(2, 'big'))
    
    return shares

def combine_shares(shares):
    if len(shares) < 3:
        raise ValueError("En az 3 pay gereklidir")
    
    secret_bytes = bytearray()
    share_data = [share.data for share in shares]
    
    for i in range(0, len(share_data[0]), 2):
        points = [(share.x, int.from_bytes(share.data[i:i+2], 'big')) for share in shares]
        secret = 0
        
        for j, (xj, yj) in enumerate(points):
            prod = 1
            for m, (xm, ym) in enumerate(points):
                if m == j: continue
                numerator = xm
                denominator = gf_sub(xm, xj)
                prod = gf_mul(prod, gf_div(numerator, denominator))
            secret = gf_add(secret, gf_mul(yj, prod))
        
        secret_bytes.append(secret % 256)
    
    return bytes(secret_bytes)

# --- LSB Steganografi (Değişmeden) ---
def embed_data(img, data):
    data_len = len(data).to_bytes(4, 'big')
    full_data = data_len + data
    bits = ''.join(f"{byte:08b}" for byte in full_data)
    
    # Görüntüyü veri boyutuna göre yeniden boyutlandır
    required_pixels = ceil(len(bits)/3)
    h, w = img.shape[:2]
    if h*w < required_pixels:
        new_size = int(np.sqrt(required_pixels)) + 1
        img = cv2.resize(img, (new_size, new_size))
    
    # Bitleri yerleştir
    idx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                if idx >= len(bits): return img
                img[i,j,k] = (img[i,j,k] & 0xFE) | int(bits[idx])
                idx += 1
    return img


def extract_data(img):
    bits = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                bits.append(str(img[i,j,k] & 1))
    bits_str = ''.join(bits)
    
    data_len = int(bits_str[:32], 2)
    data = bytes(int(bits_str[i:i+8], 2) for i in range(32, 32+data_len*8, 8))
    return data

# --- Main Flow ---
def encode_image(image_path, output_prefix, k=3, n=6):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("Görüntü bulunamadı")
    
    _, img_encoded = cv2.imencode('.png', img)
    secret = img_encoded.tobytes()
    
    shares = split_secret(secret, k, n)
    
    base_img = np.full((512, 512, 3), 128, dtype=np.uint8)
    for i, share in enumerate(shares):
        stego_img = embed_data(base_img.copy(), share.data)
        cv2.imwrite(f"{output_prefix}_{share.x}.png", stego_img)

def decode_images(share_paths, output_path):
    shares = []
    for path in share_paths:
        img = cv2.imread(path)
        data = extract_data(img)
        x = int(path.split('_')[-1].split('.')[0])  # Dosya adından x değerini al
        shares.append(Share(x, data))
    
    secret = combine_shares(shares)
    img = cv2.imdecode(np.frombuffer(secret, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(output_path, img)

# Kullanım Örneği
if __name__ == "__main__":
    encode_image("input_2.png", "share")
    # Herhangi 3 pay kombinasyonu çalışır:
    decode_images(["share_2.png", "share_4.png", "share_5.png"], "output.png")
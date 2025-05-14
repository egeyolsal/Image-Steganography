import cv2
import numpy as np
from math import log10, sqrt, ceil # PSNR hesaplaması için
import secrets
import traceback
import json

# --- Galois Field İşlemleri ---
PRIME = 251
# ... (gf_add, gf_sub, gf_mul, gf_inverse, gf_div fonksiyonları burada) ...
def gf_add(a, b): return (a + b) % PRIME
def gf_sub(a, b): return (a - b) % PRIME
def gf_mul(a, b): return (a * b) % PRIME
def gf_inverse(a):
    if a == 0: raise ValueError(f"GF({PRIME}) içinde sıfırın tersi yoktur")
    return pow(a, PRIME - 2, PRIME)
def gf_div(a, b):
    if b == 0: raise ValueError(f"GF({PRIME}) içinde sıfıra bölme yapılamaz")
    return gf_mul(a, gf_inverse(b))

# --- Gizli Paylaşım Sınıfı ---
class Share:
    # ... (Share sınıfı burada) ...
    def __init__(self, x, data):
        self.x = x
        self.data = data

# --- Gizli Paylaşım Fonksiyonları ---
def split_secret(secret_bytes, k, n):
    if not (1 < k <= n):
        raise ValueError("k, n'den büyük olamaz ve 1'den büyük olmalıdır (genellikle k >= 2).")
    shares = [Share(x, bytearray()) for x in range(1, n + 1)]
    for original_byte_val in secret_bytes:
        coeffs = [original_byte_val] + [secrets.randbelow(PRIME) for _ in range(k - 1)]
        for share in shares:
            y = 0
            for i, coeff in enumerate(coeffs):
                term_pow_x = pow(share.x, i, PRIME)
                term = gf_mul(coeff, term_pow_x)
                y = gf_add(y, term)
            share.data.extend(y.to_bytes(2, 'big'))
    return shares

def combine_shares(shares_list, k_threshold):
    if len(shares_list) < k_threshold:
        raise ValueError(f"Sırrı yeniden oluşturmak için en az {k_threshold} pay gereklidir.")
    if not shares_list or not shares_list[0].data:
        return b""
    expected_data_len = len(shares_list[0].data)
    if expected_data_len % 2 != 0:
        raise ValueError("Pay veri uzunluğu geçersiz (2'nin katı olmalı).")
    for s_check in shares_list: # Tüm payların aynı uzunlukta olduğunu kontrol et
        if len(s_check.data) != expected_data_len:
            raise ValueError("Tüm paylar aynı uzunlukta veri içermiyor.")

    num_original_bytes = expected_data_len // 2
    secret_reconstructed_bytes = bytearray()
    for i_block in range(num_original_bytes):
        points = []
        current_byte_start_index = i_block * 2
        for share in shares_list:
            y_val_bytes = share.data[current_byte_start_index : current_byte_start_index + 2]
            y_val = int.from_bytes(y_val_bytes, 'big')
            points.append((share.x, y_val))
        current_secret_byte_reconstructed = 0
        for j, (xj, yj) in enumerate(points):
            lagrange_basis_numerator = 1
            lagrange_basis_denominator = 1
            for m, (xm, _) in enumerate(points):
                if m == j: continue
                lagrange_basis_numerator = gf_mul(lagrange_basis_numerator, xm)
                denominator_term = gf_sub(xm, xj)
                if denominator_term == 0: raise ValueError("Lagrange payda sıfır.")
                lagrange_basis_denominator = gf_mul(lagrange_basis_denominator, denominator_term)
            lagrange_basis_at_zero = gf_div(lagrange_basis_numerator, lagrange_basis_denominator)
            term = gf_mul(yj, lagrange_basis_at_zero)
            current_secret_byte_reconstructed = gf_add(current_secret_byte_reconstructed, term)
        secret_reconstructed_bytes.append(current_secret_byte_reconstructed % 256)
    return bytes(secret_reconstructed_bytes)

def embed_data(img_to_embed_in, data_to_embed):
    data_len_bytes = len(data_to_embed).to_bytes(4, 'big')
    full_data_to_embed = data_len_bytes + data_to_embed
    bits_to_embed = np.unpackbits(np.frombuffer(full_data_to_embed, dtype=np.uint8))
    num_bits_to_embed = bits_to_embed.size

    img_h, img_w = img_to_embed_in.shape[:2]
    img_capacity_bits = img_h * img_w * 3

    # Yetersiz kapasite durumunda görseli yeniden boyutlandır
    if img_capacity_bits < num_bits_to_embed:
        required_pixels = np.ceil(num_bits_to_embed / 3).astype(int)
        new_dim = int(np.sqrt(required_pixels)) + 1
        img_to_embed_in = cv2.resize(img_to_embed_in, (new_dim, new_dim), interpolation=cv2.INTER_NEAREST)
        img_h, img_w = img_to_embed_in.shape[:2]

    # Tüm kanalları tek bir düzleme dönüştür ve LSB'leri sıfırla
    flat_img = img_to_embed_in.reshape(-1, 3)
    flat_img = (flat_img & 0xFE).astype(np.uint8)  # 0xFE - 11111110

    # Gömülecek bitleri ekle
    flat_img.flat[:num_bits_to_embed] |= bits_to_embed[:flat_img.size]

    # Orijinal şekle geri dönüştür
    embedded_img = flat_img.reshape(img_h, img_w, 3)
    return embedded_img

# def extract_data(stego_img):
#     # Tüm LSB bitlerini tek seferde çıkar
#     lsb_bits = (stego_img & 1).reshape(-1)
#     lsb_bits_str = lsb_bits.astype(np.uint8).astype(str)
#     all_extracted_bits = ''.join(lsb_bits_str)

#     # İlk 32 biti (veri uzunluğu) oku
#     if len(all_extracted_bits) < 32:
#         raise ValueError("Veri uzunluğu okunamadı.")
#     data_len = int(all_extracted_bits[:32], 2)
#     total_bits_needed = 32 + data_len * 8

#     # Yeterli bit kontrolü
#     if len(all_extracted_bits) < total_bits_needed:
#         raise ValueError("Yetersiz bit.")

#     # Veriyi çıkar
#     payload_bits = all_extracted_bits[32:total_bits_needed]
#     payload_bytes = np.packbits(np.array(list(payload_bits), dtype=np.uint8).reshape(-1, 8), axis=1).tobytes()
#     return bytes(payload_bytes)

def extract_data(stego_img):
    # Tüm LSB'leri tek seferde al
    lsb_bits = (stego_img & 1).astype(np.uint8)
    
    # Bitleri birleştirerek byte'lara dönüştür
    bits_flat = lsb_bits.reshape(-1)
    bytes_flat = np.packbits(bits_flat, axis=0)
    
    # İlk 4 byte (32 bit) veri uzunluğunu oku
    if len(bytes_flat) < 4:
        raise ValueError("Veri uzunluğu okunamadı.")
    data_len = int.from_bytes(bytes_flat[:4].tobytes(), 'big')
    
    # Toplam gerekli byte sayısını hesapla
    total_bytes_needed = 4 + data_len
    if len(bytes_flat) < total_bytes_needed:
        raise ValueError("Yetersiz veri.")
    
    # Veriyi çıkar
    payload_bytes = bytes_flat[4:total_bytes_needed].tobytes()
    return payload_bytes

def calculate_psnr(img1, img2):
    if img1.shape != img2.shape:
        return -1
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse != 0 else float('inf')


# --- Ana İş Akışı ---
DEFAULT_K = 3
DEFAULT_N = 6
METADATA_KEY_SHAPE = "shape"
METADATA_KEY_DTYPE = "dtype_str"

def encode_image_raw_pixels(image_path, output_prefix, k=DEFAULT_K, n=DEFAULT_N, cover_image_path = None):
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
    img_shape = original_img.shape
    img_dtype_str = str(original_img.dtype)
    raw_pixel_bytes = original_img.tobytes()
    metadata = { METADATA_KEY_SHAPE: img_shape, METADATA_KEY_DTYPE: img_dtype_str }
    metadata_json_str = json.dumps(metadata)
    metadata_bytes = metadata_json_str.encode('utf-8')
    metadata_len_bytes = len(metadata_bytes).to_bytes(4, 'big')
    secret_data = metadata_len_bytes + metadata_bytes + raw_pixel_bytes
    print(f"Orijinal: {image_path}, Shape: {img_shape}, Dtype: {img_dtype_str}")
    print(f"Şifrelenecek toplam veri: {len(secret_data)} bayt.")
    list_of_shares = split_secret(secret_data, k, n)
    print(f"{n} pay oluşturuldu. Her payın veri boyutu: {len(list_of_shares[0].data)} bayt.")
    base_cover_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    if cover_image_path:
        base_cover_img = cv2.imread(cover_image_path)
        if base_cover_img is None:
            print(f"Uyarı: Cover image '{cover_image_path}' bulunamadı, varsayılan gri image kullanılıyor.")
            base_cover_img = np.full((100, 100, 3), 128, dtype = np.uint8)
    else:
        base_cover_img = base_cover_img = np.full((100, 100, 3), 128, dtype=np.uint8)

     # Kapasite kontrolü ve otomatik boyutlandırma
    required_capacity = len(secret_data) * 8  # Her byte için 8 bit
    img_h, img_w = base_cover_img.shape[:2]
    available_capacity = img_h * img_w * 3  # 3 kanal (RGB)
    
    if available_capacity < required_capacity:
        new_dim = ceil(sqrt((required_capacity / 3)))
        base_cover_img = cv2.resize(base_cover_img, (new_dim, new_dim), interpolation=cv2.INTER_AREA)
        print(f"Cover imaj {new_dim}x{new_dim} boyutuna yeniden boyutlandırıldı")

    for share_object in list_of_shares:
        stego_image = embed_data(base_cover_img.copy(), bytes(share_object.data))
        output_filename = f"{output_prefix}_{share_object.x}.png"
        cv2.imwrite(output_filename, stego_image)
        print(f"Pay {share_object.x} > Stego: '{output_filename}'")

def decode_image_raw_pixels(list_of_share_paths, output_image_path_template, original_image_for_psnr_path, k_threshold=DEFAULT_K, save_as_jpeg_quality=90):
    collected_shares = []
    if len(list_of_share_paths) < k_threshold:
        print(f"Hata: Yetersiz pay. Gerekli: {k_threshold}, Sağlanan: {len(list_of_share_paths)}")
        return
    for path in list_of_share_paths:
        img = cv2.imread(path)
        if img is None: print(f"Uyarı: '{path}' okunamadı."); continue
        try:
            data = extract_data(img)
            x = int(path.split('_')[-1].split('.')[0])
            collected_shares.append(Share(x, data))
            print(f"Pay okundu: '{path}', x={x}")
        except Exception as e: print(f"'{path}' işlenirken hata: {e}"); traceback.print_exc()
    if len(collected_shares) < k_threshold:
        print(f"Hata: Yeterli geçerli pay okunamadı ({len(collected_shares)}/{k_threshold}).")
        return

    try:
        reconstructed_full_secret_bytes = combine_shares(collected_shares, k_threshold)
        if not reconstructed_full_secret_bytes:
             print("HATA: Yeniden oluşturulan sır boş.")
             return
        print(f"Toplam sır başarıyla yeniden oluşturuldu. Boyut: {len(reconstructed_full_secret_bytes)} bayt.")

        metadata_len = int.from_bytes(reconstructed_full_secret_bytes[:4], 'big')
        metadata_bytes = reconstructed_full_secret_bytes[4 : 4 + metadata_len]
        raw_pixel_bytes_reconstructed = reconstructed_full_secret_bytes[4 + metadata_len :]
        metadata_json_str = metadata_bytes.decode('utf-8')
        metadata = json.loads(metadata_json_str)
        img_shape_reconstructed = tuple(metadata[METADATA_KEY_SHAPE])
        img_dtype_reconstructed_str = metadata[METADATA_KEY_DTYPE]
        img_dtype_reconstructed = np.dtype(img_dtype_reconstructed_str)
        print(f"Yeniden oluşturulan meta: Shape={img_shape_reconstructed}, Dtype={img_dtype_reconstructed_str}")

        reconstructed_image_raw = np.frombuffer(raw_pixel_bytes_reconstructed, dtype=img_dtype_reconstructed)
        try:
            reconstructed_image_raw = reconstructed_image_raw.reshape(img_shape_reconstructed)
        except ValueError as e_reshape:
            print(f"HATA: Yeniden oluşturulan piksel verisi beklenen şekle ({img_shape_reconstructed}) getirilemedi. Hata: {e_reshape}")
            return
        if reconstructed_image_raw is None or reconstructed_image_raw.size == 0 :
            print("HATA: Ham piksellerden görüntü oluşturulamadı.")
            return
        
        # output_png_path = output_image_path_template.format(ext="png")
        # cv2.imwrite(output_png_path, reconstructed_image_raw)
        # print(f"Görüntü PNG olarak kaydedildi: '{output_png_path}'")
        output_jpg_path = output_image_path_template.format(ext="jpg")
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), save_as_jpeg_quality]
        cv2.imwrite(output_jpg_path, reconstructed_image_raw, jpeg_params)
        print(f"Görüntü JPEG olarak (kalite {save_as_jpeg_quality}) kaydedildi: '{output_jpg_path}'")

        original_img_for_psnr = cv2.imread(original_image_for_psnr_path)
        if original_img_for_psnr is None:
            print(f"Uyarı: PSNR için orijinal '{original_image_for_psnr_path}' bulunamadı.")
        else:
            # psnr_png = calculate_psnr(original_img_for_psnr, reconstructed_image_raw) # Ham ile karşılaştır
            # print(f"PSNR (Orijinal vs Yeniden Oluşturulmuş Ham -> PNG): {psnr_png:.2f} dB")
            reconstructed_jpeg_for_psnr = cv2.imread(output_jpg_path)
            if reconstructed_jpeg_for_psnr is not None:
                psnr_jpeg = calculate_psnr(original_img_for_psnr, reconstructed_jpeg_for_psnr)
                print(f"PSNR (Orijinal vs Yeniden Oluşturulmuş JPEG Kalite {save_as_jpeg_quality}): {psnr_jpeg:.2f} dB")
            else:
                print(f"Uyarı: Yeniden oluşturulmuş JPEG ('{output_jpg_path}') PSNR için okunamadı.")
    except ValueError as e:
        print(f"HATA (Decode): {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"HATA (Decode - Genel): {e}")
        traceback.print_exc()

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
        encode_image_raw_pixels(INPUT_IMAGE_FILENAME, SHARE_OUTPUT_PREFIX, k=K_PARAM, n=N_PARAM, cover_image_path = "flower.jpg")
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
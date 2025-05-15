import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import lastfast

class StegoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shamir's Secret Sharing ve Steganography")
        self.root.geometry("1400x800")

        # Stil ayarları
        self.style = ttk.Style()
        self.style.configure("Selected.TFrame", background="#90EE90", relief="solid")
        
        # Değişkenler
        self.k_var = tk.IntVar(value=3)
        self.n_var = tk.IntVar(value=5)
        self.selected_shares = []
        self.share_thumbnails = []
        
        self.cover_image_path = None
        self.original_image_path = None # Orijinal resim yolunu saklamak için
        self.style.configure("CoverSelected.TLabel", background="#ADD8E6")  # Yeni stil
        
        # GUI bileşenleri
        self.create_widgets()

    def create_widgets(self):
        # Ana çerçeve
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Kontrol paneli
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="k:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(control_frame, from_=2, to=10, textvariable=self.k_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="n:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(control_frame, from_=2, to=10, textvariable=self.n_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Resim Seç", command=self.load_image).pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text = "Cover Seç", command = self.load_cover_image).pack(side=tk.LEFT, padx=10)
        self.cover_info_label = ttk.Label(control_frame, text="Cover: Yok")
        self.cover_info_label.pack(side=tk.RIGHT, padx=20)
        ttk.Button(control_frame, text="Encode", command=self.encode_image).pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text="Decode", command=self.decode_image).pack(side=tk.LEFT, padx=10)
        
        # Seçim sayacı
        self.selection_counter = ttk.Label(control_frame, text="Seçilen: 0/0")
        self.selection_counter.pack(side=tk.RIGHT, padx=20)

        # Görüntü panelleri
        image_panel = ttk.Frame(main_frame)
        image_panel.pack(fill=tk.BOTH, expand=True)

        # Input Görüntü
        self.input_frame = ttk.LabelFrame(image_panel, text=" Input Image ", padding=10)
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.input_label = ttk.Label(self.input_frame)
        self.input_label.pack()
        
        # Input image boyut etiketi
        self.input_dims_label = ttk.Label(self.input_frame, text="Boyut: -")
        self.input_dims_label.pack(pady=5)
        
        # Output Görüntü
        self.output_frame = ttk.LabelFrame(image_panel, text=" Output Image ", padding=10)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.output_label = ttk.Label(self.output_frame)
        self.output_label.pack()
        
        # Output image boyut etiketi
        self.output_dims_label = ttk.Label(self.output_frame, text="Boyut: -")
        self.output_dims_label.pack(pady=5)
        
        self.psnr_label = ttk.Label(self.output_frame, text="PSNR: -", font=('Arial', 12))
        self.psnr_label.pack(pady=10)
        
        # Paylaşım Thumbnail Alanı
        shares_frame = ttk.LabelFrame(main_frame, text=" Paylaşımlar ", padding=10)
        shares_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas = tk.Canvas(shares_frame)
        self.scrollbar = ttk.Scrollbar(shares_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )  # Parantez hatası düzeltildi
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    # Değişiklik 3: Cover image yükleme fonksiyonu
    def load_cover_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.cover_image_path = path
            self.cover_info_label.config(
                text=f"Cover: {os.path.basename(path)}",
                style="CoverSelected.TLabel"
            )
            # self.show_image(path, self.input_label, 400)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.original_image_path = path
            self.show_image(path, self.input_label, 400)
        # Input image boyutlarını göster ---
        try:
            img = Image.open(path)
            self.input_dims_label.config(text=f"Boyut: {img.width}x{img.height}")
        except Exception as e:
            self.input_dims_label.config(text="Boyut: Okunamadı")
            print(f"Error getting input image dimensions: {e}")
            
        # Decode sonrası output'u temizle
        self.output_label.config(image='')
        self.output_label.image = None
        self.output_dims_label.config(text="Boyut: -")
        self.psnr_label.config(text="PSNR: -")

    def encode_image(self):
        if not self.original_image_path:
            messagebox.showerror("Hata", "Lütfen önce bir resim seçin!")
            return
        
        if not self.cover_image_path:
            messagebox.showerror("Hata", "Lütfen önce bir cover resim seçin!")
            return
        
        try:
            # # Önceki paylaşımları temizle (dosya sisteminden)
            # for i in range(1, self.n_var.get() + 20): # Olası eski paylaşımları silmek için geniş bir aralık
            #     old_share_path = f"share_{i}.png"
            #     if os.path.exists(old_share_path):
            #         os.remove(old_share_path)

            lastfast.encode_image_raw_pixels(
                self.original_image_path,
                "share",
                k=self.k_var.get(),
                n=self.n_var.get(),
                cover_image_path=self.cover_image_path
            )
            self.load_share_thumbnails()
            messagebox.showinfo("Başarılı", f"{self.n_var.get()} pay oluşturuldu!")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def encode_image_shamir_only(self):
        if not self.original_image_path:
            messagebox.showerror("Hata", "Lütfen önce bir resim seçin!")
            return
        
        try:
            # # Önceki paylaşımları temizle (dosya sisteminden)
            # for i in range(1, self.n_var.get() + 20): # Olası eski paylaşımları silmek için geniş bir aralık
            #     old_share_path = f"share_{i}.png"
            #     if os.path.exists(old_share_path):
            #         os.remove(old_share_path)
            
            lastfast.encode_image_shamir_only(
                self.original_image_path,
                "share",
                k=self.k_var.get(),
                n=self.n_var.get(),
            )
            self.load_share_thumbnails()
            messagebox.showinfo("Başarılı", f"{self.n_var.get()} pay oluşturuldu!")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def load_share_thumbnails(self):
        # Eski thumbnail'leri temizle
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.selected_shares = []
        self.share_thumbnails = []
        share_files = [f"share_{i}.png" for i in range(1, self.n_var.get()+1)]
        
        # Thumbnail'leri oluştur
        row = col = 0
        for share_path in share_files:
            if not os.path.exists(share_path):
                continue
            
            frame = ttk.Frame(self.scrollable_frame, style = "TFrame") # Varsayılan Stil
            frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Görüntüyü yükle
            img = Image.open(share_path)
            # Paylaşım resminin orijinal boyutları
            share_dims_text = f"Boyut: {img.width}x{img.height}"
            img.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img)
            
            # Tıklanabilir label
            label = ttk.Label(frame, image=photo, cursor="hand2")
            label.image = photo
            label.pack()
            label.bind("<Button-1>", lambda e, p=share_path: self.toggle_share(p, frame))
            
            # Dosya adı
            ttk.Label(frame, text=os.path.basename(share_path)).pack()
            
            # Paylaşım boyut etiketini ekle
            ttk.Label(frame, text=share_dims_text).pack()
            
            self.share_thumbnails.append(frame)
            col += 1
            if col > 4:
                col = 0
                row += 1
        
        self.update_selection_counter()

    def toggle_share(self, share_path, frame):
        if share_path in self.selected_shares:
            self.selected_shares.remove(share_path)
            frame.configure(style="TFrame")
        else:
            # k değeri kadar seçim yapılmasına izin ver. Daha fazlasını engelle.
            if len(self.selected_shares) >= self.k_var.get():
                messagebox.showwarning("Uyarı", f"En fazla {self.k_var.get()} paylaşım seçebilirsiniz.")
                return
            
            self.selected_shares.append(share_path)
            frame.configure(style="Selected.TFrame")
        
        self.update_selection_counter()

    def update_selection_counter(self):
        required = self.k_var.get()
        selected = len(self.selected_shares)
        self.selection_counter.config(
            text=f"Seçilen: {selected}/{required}",
            foreground="green" if selected >= required else "red"
        )

    def decode_image(self):
        if len(self.selected_shares) < self.k_var.get():
            messagebox.showerror("Hata", f"En az {self.k_var.get()} pay seçmelisiniz!")
            return
        if not self.original_image_path:
            messagebox.showerror("Hata", "PSNR hesaplaması için orijinal resim seçilmemiş. Lütfen önce 'Resim Seç' ile orijinal resmi belirtin.")
            return
        
         # `lastfast.py` decode fonksiyonu `output_image_path_template` için `.jpg` veya `.png` bekliyor.
        # Biz sabit bir isim kullanabiliriz.
        decoded_image_filename = "decoded_image.jpg" # Sabit isim, uzantısı jpg

        try:
            lastfast.decode_image_raw_pixels(
                self.selected_shares,
                decoded_image_filename.replace(".jpg", ".{ext}"), # lastfast'in format stringine uygun hale getir,
                self.original_image_path, # PSNR hesaplaması için orijinal resim yolu
                k_threshold=self.k_var.get(),
                save_as_jpeg_quality=95 # Yüksek kalite JPEG
            )
            
            self.show_image("decoded_image.jpg", self.output_label, 400)
            # Output image boyutlarını göster ---
            try:
                img = Image.open(decoded_image_filename)
                self.output_dims_label.config(text=f"Boyut: {img.width}x{img.height}")
            except Exception as e:
                self.output_dims_label.config(text="Boyut: Okunamadı")
                print(f"Error getting output image dimensions: {e}")
            
            self.calculate_psnr()
            messagebox.showinfo("Başarılı", "Görüntü başarıyla çözüldü!")
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def show_image(self, path, label, max_size):
        if not os.path.exists(path):
            return
        
        img = Image.open(path)
        img.thumbnail((max_size, max_size))
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo

    def calculate_psnr(self):
        if not os.path.exists("decoded_image.jpg") or not self.original_image_path:
            return
        
        original = cv2.imread(self.original_image_path)
        decoded = cv2.imread("decoded_image.jpg")
        
        if original is None or decoded is None:
            return
        
        try:
            psnr_value = lastfast.calculate_psnr(original, decoded)
            self.psnr_label.config(text=f"PSNR: {psnr_value:.2f} dB")
        except:
            self.psnr_label.config(text="PSNR Hesaplanamadı")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Özel stil for seçili çerçeve
    style = ttk.Style()
    style.configure("Selected.TFrame", background="#90EE90", relief="solid")
    
    app = StegoGUI(root)
    root.mainloop()
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading

# YOLO modelini yükleme
model = YOLO("models/last.pt")  # Model yolu

# Arayüz penceresi
window = tk.Tk()
window.title("Malzeme Tespit Sistemi")
window.attributes('-fullscreen', True)  # Tam ekran modu

# Sol çerçeve, butonlar ve uyarı alanı
left_frame = tk.Frame(window, width=200, bg='lightgrey')
left_frame.pack(side=tk.LEFT, fill=tk.Y)

# Sonuç metin kutusu
result_text = tk.Label(left_frame, text="", font=("Arial", 14), bg='lightgrey')
result_text.pack(pady=20)

# Kamera görüntüsü için canvas
camera_canvas = tk.Canvas(window, bg='black')
camera_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Kapatma fonksiyonu
def close_app():
    global running
    running = False
    window.destroy()

# Kamera yakalayıcı ve iş parçacığı değişkenleri
cap = None
running = False

# Kamera güncelleme fonksiyonu
def update_frame():
    global cap, running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)[0]

        # Bounding boxes, class_id, score
        boxes = np.array(results.boxes.data.tolist())
        detected_butil = detected_kaucuk = detected_takoz = 0

        for box in boxes:
            x1, y1, x2, y2, score, class_id = box[:6]
            box_color = (0, 0, 255)

            if score > 0.3:  # Confidence score
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)

                class_name = results.names[int(class_id)]
                
                # Malzeme isimlerine göre sayma
                if class_name == "BUTIL":
                    detected_butil += 1
                elif class_name == "KAUCUK":
                    detected_kaucuk += 1
                elif class_name == "TAKOZ":
                    detected_takoz += 1
                
                text = f"{class_name}: %{score*100:.2f}"
                label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(frame,
                              (int(x1), int(y1) - 10 - label_size[1]),
                              (int(x1) + label_size[0], int(y1) + base_line - 10),
                              box_color,
                              cv2.FILLED)
                cv2.putText(frame, text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=1)

        # Toplam malzeme sayısı kontrolü
        total_count = detected_butil + detected_kaucuk + detected_takoz

        if total_count < 11:
            result_text.config(text="Eksik malzeme", fg="red")
        elif total_count > 11:
            result_text.config(text="Fazla malzeme", fg="red")
        else:
            result_text.config(text="OK", fg="green")

        # OpenCV görüntüsünü Tkinter Canvas'ta gösterme
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Canvas boyutunu al ve görüntüyü yeniden boyutlandır
        canvas_width = camera_canvas.winfo_width()
        canvas_height = camera_canvas.winfo_height()
        img = img.resize((canvas_width, canvas_height), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        camera_canvas.imgtk = imgtk  # Bu referansı tutmak önemlidir

        # İşlemci yükünü azaltmak için bekleme süresi ekleme
        cv2.waitKey(1)

    cap.release()

# Malzeme kontrol fonksiyonu
def check_malzeme():
    global cap, running

    # Eğer bir önceki kamera açık ise onu kapatıyoruz
    if cap is not None:
        cap.release()

    # Yeni bir kamera yakalayıcı başlatıyoruz
    cap = cv2.VideoCapture(0)
    
    # Kamera FPS ayarlama
    cap.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS olarak sabitleme

    # İş parçacığını başlat
    if not running:
        running = True
        thread = threading.Thread(target=update_frame)
        thread.daemon = True
        thread.start()

# Tek model butonunu oluşturma
button = tk.Button(left_frame, text="B585G06", command=check_malzeme, width=15)
button.pack(pady=10)

# Kapat butonu
close_button = tk.Button(left_frame, text="Kapat", command=close_app, width=15)
close_button.pack(pady=20)

window.mainloop()
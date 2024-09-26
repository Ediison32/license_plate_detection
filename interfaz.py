import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import messagebox


#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
cap = None
video_running = False
model = YOLO("best_placa.pt")  

def iniciar():
    global cap, video_running
    video_running = True
    cap = cv2.VideoCapture("video_moto.mp4")
    visualizar()


def visualizar():
    global cap,video_running
    if video_running and cap is not None:
            status, frame = cap.read()

            if status:
            
                frame = cv2.resize(frame, (640, 640))
                results = model(frame)
            
                for result in results:
                    boxes = result.boxes  # Obtener las cajas delimitadoras
                    for box in boxes:
                    # Obtener las coordenadas de la caja delimitadora
                        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                    
                        conf=box.conf[0]
                        placa_roi = frame[y1:y2, x1:x2]
                        
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                            
                    # Recortar la región de la placa del frame original
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                lbl_video.config(image=img_tk)
                lbl_video.img_tk = img_tk  # Evita que la imagen sea recolectada por el garbage collector
                lbl_video.after(10, visualizar)
                show_plate_image(placa_roi)
                
            else:
                finalizar()
                
def show_plate_image(plate_img):
    # Convertir la imagen de BGR a RGB para mostrar en Tkinter
    plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(plate_img_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Actualizar el Label con la imagen recortada de la placa
    lbl_video2.img_tk = img_tk  # Evita que la imagen sea recolectada por el garbage collector
    lbl_video2.config(image=img_tk)
   
    
def finalizar():
    global cap,video_running
    video_running = False
    if cap is not None:
        cap.release()
    lbl_video.image = None
    messagebox.showinfo("Info", "Video finalizado")
    
    
root = tk.Tk()
root.title("Reproductor de Video con Detección de Placas")

# Crear un Label para mostrar el video
lbl_video = tk.Label(root)
lbl_video.pack()

lbl_video2=tk.Label(root)
lbl_video2.pack(pady=10)

# Botones para iniciar y detener el video
btn_start = tk.Button(root, text="Iniciar Video", command=iniciar)
btn_start.pack(side=tk.LEFT, padx=10, pady=10)

btn_stop = tk.Button(root, text="Finalizar Video", command=finalizar)
btn_stop.pack(side=tk.RIGHT, padx=10, pady=10)

# Iniciar el loop principal de la ventana
root.mainloop()
    
    
    
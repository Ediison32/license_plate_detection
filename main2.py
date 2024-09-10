
# import matplotlib.pyplot as plt
# import cv2
# from ultralytics import YOLO
# import pytesseract

# import numpy

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ctexto=''

# if __name__ == '__main__':

#     cap = cv2.VideoCapture("video_moto3.mp4")  # mostrar video 

#     # cargar modelo 
#     model = YOLO("best_placa.pt")  
#     #mode = YOLO(best.pt) este modelo detecta carros y camiones 
#     """     
#     model.export(format="onnx")
#     onnx_model = YOLO("yolov8m.onnx")
#     results = onnx_model("https://ultralytics.com/images/bus.jpg") 
#     """
#     while cap.isOpened():
#         status, frame = cap.read()

#         if not status:
#             break
        
#         frame = cv2.resize(frame,(900, 800))
#         results = model(frame)
        
        

#         frame2 = results[0].plot()
        
        
        
        
#         config_placa = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                   
                    
#         texto = pytesseract.image_to_string(results,config= config_placa)
        
#         ctexto=texto
        
#         print(ctexto)
        
        
#         cv2.putText(frame2,ctexto,(300,650),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


#         cv2.imshow("frame", frame2)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cap.release()

import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ctexto = ''

if __name__ == '__main__':

    cap = cv2.VideoCapture("video_carro.mp4")  # mostrar video 

    # cargar modelo 
    model = YOLO("best_placa.pt")  
    
    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break
        
        frame = cv2.resize(frame, (900, 800))
        results = model(frame)
        
        # Procesar las detecciones
        for result in results:
            boxes = result.boxes  # Obtener las cajas delimitadoras
            for box in boxes:
                # Obtener las coordenadas de la caja delimitadora
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                
                # Recortar la región de la placa del frame original
                placa_roi = frame[y1:y2, x1:x2]
                
                alp,anp,cp=placa_roi.shape
                
                Mva = np.zeros((alp,anp))
                
                    #normalizamos las matrices
                
                nblue= np.matrix(placa_roi[:,:,0])
                ngreen= np.matrix(placa_roi[:,:,1])
                nred= np.matrix(placa_roi[:,:,2])
                
                    #se crea una mascara
                
                for col in range(0,alp):
                    for fil in range(0,anp):
                        Max= max(nred[col,fil],ngreen[col,fil],nblue[col,fil])
                        Mva[col,fil] = 255 - Max
                        
                    #binarizamos la imagen
                _, bin = cv2.threshold(Mva,150,255,cv2.THRESH_BINARY)

                
                
                    #convertimos la matriz en imagen
                bin = bin.reshape(alp,anp)
                
                bin = Image.fromarray(bin)
                
                
                bin = bin.convert("L")
                
                print(f'alto: {alp} ancho: {anp}')
                
                
                
                # Aplicar Tesseract para realizar OCR en la región de la placa
                config_placa = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                texto = pytesseract.image_to_string(bin, config=config_placa)
                
                # Dibujar la caja delimitadora en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                ctexto = texto  # Limpiar el texto obtenido
                
                if anp==162 and alp==44:
                    
                    cv2.imwrite('imagen4.jpg',placa_roi)
                    img=cv2.imread('imagen4.jpg')
    
                    Mva = np.zeros((alp,anp))
                
                    #normalizamos las matrices
                
                    nblue= np.matrix(img[:,:,0])
                    ngreen= np.matrix(img[:,:,1])
                    nred= np.matrix(img[:,:,2])
                
                    #se crea una mascara
                
                    for col in range(0,alp):
                        for fil in range(0,anp):
                            Max= max(nred[col,fil],ngreen[col,fil],nblue[col,fil])
                            Mva[col,fil] = 255 - Max
                        
                    #binarizamos la imagen
                    _, bin = cv2.threshold(Mva,150,255,cv2.THRESH_BINARY)

                
                
                    #convertimos la matriz en imagen
                    bin = bin.reshape(alp,anp)
                
                    bin = Image.fromarray(bin)
                
                
                    bin = bin.convert("L")
                    
                    texto_fijo = pytesseract.image_to_string(bin, config=config_placa)
                    
                    print(f'la placa para guardar en la base de datos es: {texto_fijo}')
                    
                if len(texto) >= 7:
                    
                    ctexto = texto  # Limpiar el texto obtenido
                    print(f"Texto detectado: {ctexto[0:7]}")
                
                # Mostrar el texto detectado en el frame
                    cv2.putText(frame, ctexto[0:7], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el frame con las cajas y el texto detectado
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
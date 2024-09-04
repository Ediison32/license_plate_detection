
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO


if __name__ == '__main__':

    cap = cv2.VideoCapture("video_moto.mp4")  # mostrar video 

    # cargar modelo 
    model = YOLO("best_placa.pt")  
    #mode = YOLO(best.pt) este modelo detecta carros y camiones 
    """     
    model.export(format="onnx")
    onnx_model = YOLO("yolov8m.onnx")
    results = onnx_model("https://ultralytics.com/images/bus.jpg") 
    """
    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break
        
        #frame = cv2.resize(frame,(1920, 1080))
        results = model(frame)

        frame = results[0].plot()


        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


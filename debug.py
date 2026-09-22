from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")
results = model.predict(r"D:\College Work\Sem 5\SIH\Images\Clean_cow.png", conf=0.05, verbose=True)
results[0].save(filename="debug_annotated.jpg")
print(results[0].boxes)

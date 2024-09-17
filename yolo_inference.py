from ultralytics import YOLO
# Load model
model = YOLO('models\\best.pt')
# Inference
model.predict('C:\\Users\\shive\\OneDrive\\Desktop\\YoLO project\\input_vid\\08fd33_4.mp4', save=True)

#shrey
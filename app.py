import cv2
import supervision as sv
from ultralytics import YOLO


def Test_Live(model, box_annotation):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

    if cam.isOpened():
        loop = True
        while loop:
            success, frame = cam.read()
            if success:
                predictions = model(frame)[0]
                detections = sv.Detections.from_ultralytics(predictions)

                labels = [
                            f"{model.model.names[class_id]}-{confidence:0.2f}"
                            for class_id, confidence
                            in zip(detections.class_id, detections.confidence)
                        ]
            
                annotated_frame = box_annotation.annotate(scene=frame, detections=detections, labels=labels) 

                cv2.imshow("YOLOV8", annotated_frame)

                if cv2.waitKey(1) == ord("q"):
                    loop = False

    else:
        raise "Can't access camera"



if __name__ == "__main__":
    model = YOLO("best.pt", "v8")
    box_annotation = sv.BoxAnnotator()
     
    Test_Live(model=model, box_annotation=box_annotation)

    
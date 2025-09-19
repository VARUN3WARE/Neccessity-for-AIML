import cv2
import time
from ultralytics import YOLO

def main():
    # Load YOLOv8 model with tracking enabled
    model = YOLO("yolov8n.pt")

    # Initialize webcam (0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    # Variables for FPS calculation
    prev_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to grab frame.")
                break

            # Optional: Resize frame for faster processing (comment out if not needed)
            frame = cv2.resize(frame, (640, 480))

            # Run YOLOv8 with tracking enabled on current frame
            results = model.track(source=frame, tracker='bytetrack.yaml', persist=True)

            # Annotate frame with bounding boxes, labels, and unique tracking IDs
            annotated_frame = results[0].plot()

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Display the annotated frame in a window
            cv2.imshow("YOLOv8 Real-Time Detection & Tracking", annotated_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Quitting...")
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")

    finally:
        # Release webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

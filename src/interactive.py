import cv2
import numpy as np
import torch
import os
import argparse
from train import BNN, binarize

drawing = False
pt1_x , pt1_y = None , None
img = np.zeros((400, 400, 1), np.uint8)

def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=20)
            pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of drawing pad')
    args = parser.parse_args()

    device = torch.device("cpu")
    model = BNN().to(device)
    if os.path.exists('models/bnn_mnist.pth'):
        model.load_state_dict(torch.load('models/bnn_mnist.pth', map_location=device))
    model.eval()

    if args.webcam:
        cap = cv2.VideoCapture(0)
        print("Webcam started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and crop center
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            size = min(h, w)
            cropped = gray[h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2]

            # Invert colors (MNIST is white on black)
            inverted = cv2.bitwise_not(cropped)

            # Resize
            resized = cv2.resize(inverted, (28, 28), interpolation=cv2.INTER_AREA)

            # Threshold to make it purely black/white
            _, thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

            normalized = (thresh / 255.0) * 2.0 - 1.0
            tensor_img = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor_img)
                pred = output.argmax(dim=1).item()

            cv2.putText(frame, f"Prediction: {pred}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Webcam BNN', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    else:
        print("Starting interactive recognition...")
        print("Draw a digit. Press 'Enter' to predict, 'c' to clear, 'q' to quit.")

        cv2.namedWindow('Draw Digit')
        cv2.setMouseCallback('Draw Digit', line_drawing)
        global img
        while True:
            cv2.imshow('Draw Digit', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                img = np.zeros((400, 400, 1), np.uint8)
            elif key == 13: # Enter key
                resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                normalized = (resized / 255.0) * 2.0 - 1.0
                tensor_img = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor_img)
                    pred = output.argmax(dim=1).item()
                print(f"Prediction: {pred}")
                cv2.putText(img, f"Prediction: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

model_path = './model/checkpoint.ckpt'

import tensorflow as tf
import cv2
from tensorflow.keras.utils import load_img ,img_to_array


masknet = tf.keras.models.load_model(model_path)
scale_factor = 1.2
min_neighbors = 3
min_size = (50, 50)
webcam=True 
class_names=['WithMask','WithoutMask']

def detect(path):

    cascade = cv2.CascadeClassifier(path)
   
    video_cap = cv2.VideoCapture(0) 
    while True:
        # Capture frame-by-frame
        ret, img = video_cap.read()

        #converting to gray image for faster video processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                         minSize=min_size)
        # if at least 1 face detected
        if len(rects) >= 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in rects:
                image_to_predict = img[y:y+h,x:x+w,:]
                image_to_predict = cv2.resize(image_to_predict,(224,224))
                pred = masknet.predict(tf.expand_dims(image_to_predict,axis=0))                
                pred_class = class_names[tf.argmax(pred[0])]
                percent = max(pred[0])*100

                print(pred_class,pred[0])


                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img=img,
                text=pred_class,
                org=(x,y-4),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.75,
                color=(0,224,0),
                thickness=1,
                lineType=cv2.LINE_AA)
                

            # Display the resulting frame
            cv2.imshow('Face Detection on Video', img)
            #wait for 'c' to close the application
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()

def main():
    cascadeFilePath="./Haarcascade_frontalface_default.xml"
    detect(cascadeFilePath)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
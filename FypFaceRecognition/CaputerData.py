import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

class CaptureData:
    def __init__(self,name):
        self.name=name
        self.cap=None
        self.key=None
        self.count=0

    def frame_Window(self):
        self.cap=cv2.VideoCapture(0)
        return self.cap
    
    def capture(self):
        frames=[]
        shapes=[]
        outputs=[]
        capture=self.frame_Window()
        while True:  
            ret, frame=capture.read()
            if ret:
                cv2.imshow("Screen", frame)
            self.key=cv2.waitKey(1)
            if self.key==ord("c"):
                break
        while self.count < 1:  
            ret, frame=capture.read()
            if ret:
                shapes.append(frame.shape)
                frames.append(frame.flatten())
                outputs.append([self.name])
                self.count+=1
                print(self.count)
        self.handle_capture(frames,outputs,shapes)

    def handle_capture(self,frames,outputs,shapes):
        X=np.array(frames)
        y=np.array(outputs)
        data=np.hstack([y,X])
        print("Press any key to see the captured images")
        for i in range(self.count):
            img=frames[i].reshape(shapes[i])
            cv2.imshow(f"Captured Image {i+1}",img)
            cv2.waitKey(0)
        print("Press c to continue with the data, any key to quit")
        decision=input("Enter your decision: ")
        if decision=="c":
            self.saveData(data, frames, shapes)
            print(data)
            print(type(data))
        self.releaseResource(self.cap)
        
    def saveData(self,data, frames, shapes):
        file_name="face_Data.npy"
        if os.path.exists(file_name):
            old_data=np.load(file_name, allow_pickle=True)
            data=np.vstack([old_data,data])
        np.save(file_name,data)
        if not os.path.exists('images'):
            os.makedirs('images')
        for i in range(self.count):
            img=frames[i].reshape(shapes[i])
            cv2.imwrite(f"images/{self.name}_{i+1}.png", img) 
    
    def releaseResource(self,capture):
        capture.release()
        cv2.destroyAllWindows()
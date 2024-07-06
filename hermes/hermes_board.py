import os

import tkinter as tk
from PIL import ImageGrab
from PIL import Image
import cv2
import numpy as np
import keras

alphabet_list = [chr(65 + i) for i in range(26)]

model = keras.models.load_model(os.path.join(os.path.dirname(__file__), './model/super_hermes.keras'))

class WhiteboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whiteboard")

        self.canvas = tk.Canvas(root, width=1300, height=700, bg="white")
        self.canvas.pack()

        self.radius = 10

        self.draw_button = tk.Button(root, text="Draw", command=self.start_draw)
        self.draw_button.pack(side=tk.LEFT)

        self.erase_button = tk.Button(root, text="Erase", command=self.start_erase)
        self.erase_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.increase_button = tk.Button(root, text="Increase", command=self.increase_thickness)
        self.increase_button.pack(side=tk.LEFT)

        self.decrease_button = tk.Button(root, text="Decrease", command=self.decrease_thickness)
        self.decrease_button.pack(side=tk.LEFT)
        
        self.drawing = True
        self.erasing = False
        self.last_x, self.last_y = None, None

        self.canvas.bind("<Button-1>", self.start_action)
        self.canvas.bind("<B1-Motion>", self.draw_or_erase)
        self.canvas.bind("<ButtonRelease-1>", self.stop_action)
        
        self.save_button = tk.Button(root, text="Predecir", command=self.predict)
        self.save_button.pack(side=tk.LEFT)
        
        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 16), fg="blue")
        self.prediction_label.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.radius_label = tk.Label(root, text="", font=("Helvetica", 16), fg="blue")
        self.radius_label.pack(side=tk.BOTTOM, padx=10, pady=10)
        self.radius_label.config(text=f"Size: {self.radius}")

    def start_action(self, event):
        if self.drawing:
            self.start_draw(event)
        elif self.erasing:
            self.start_erase(event)

    def stop_action(self, event):
        if self.drawing:
            self.stop_draw(event)
        elif self.erasing:
            self.stop_erase(event)

    def start_draw(self, event=None):
        self.drawing = True
        self.erasing = False
        self.draw_button.config(relief=tk.SUNKEN)  # Add this line
        self.erase_button.config(relief=tk.RAISED)  # Add this line
        if event:
            self.last_x, self.last_y = event.x, event.y

    def start_erase(self, event=None):
        self.erasing = True
        self.drawing = False
        self.erase_button.config(relief=tk.SUNKEN)  # Add this line
        self.draw_button.config(relief=tk.RAISED)  # Add this line
        if event:
            self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, event=None):
        pass

    def stop_erase(self, event=None):
        pass

    def increase_thickness(self):
        self.radius += 5
        if self.radius > 50:  # Asegura que el radio nunca sea menor que 1
            self.radius = 50
        self.radius_label.config(text=f"Size: {self.radius}")

    def decrease_thickness(self):
        self.radius -= 5
        if self.radius < 0:  # Asegura que el radio nunca sea menor que 1
            self.radius = 0
        self.radius_label.config(text=f"Size: {self.radius}")


#    def draw_or_erase(self, event):
#        if self.drawing:
#            x, y = event.x, event.y
#            radius = 10  # Radius of the circle
#            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black")
#            self.last_x, self.last_y = x, y
#        elif self.erasing:
#            x, y = event.x, event.y
#            self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill="white", outline="white")

    def draw_or_erase(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_oval(x - self.radius, y - self.radius, x + self.radius, y + self.radius, fill="black")
            self.last_x, self.last_y = x, y
        elif self.erasing:
            x, y = event.x, event.y
            self.canvas.create_rectangle(x - self.radius, y - self.radius, x + self.radius, y + self.radius, fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        
    def predict(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))

        # Convert the PIL image to a grayscale OpenCV image
        cv_image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Convert the PIL image to a colored OpenCV image
        cv_image_color = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply a binary threshold to the image
        _, thresholded = cv2.threshold(cv_image_gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        contours_info, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        padding = 10  # Adjust as needed
        prediction = ""
        
        # Sort the contours from left to right
        contours = sorted(contours_info, key=lambda contour: cv2.boundingRect(contour)[0])
        
        

        for i, contour in enumerate(contours):
            # Find the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
          
            # Draw a rectangle around the contour on the colored image
            cv2.rectangle(cv_image_color, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)

            # Crop the image to the bounding box with padding
            letter_image = cv_image_gray[max(0, y - padding):min(y + h + padding, cv_image_gray.shape[0]), 
                                        max(0, x - padding):min(x + w + padding, cv_image_gray.shape[1])]

            # Convert the OpenCV image back to a PIL image
            letter_image = Image.fromarray(letter_image)

            # Make the image square by adding white padding
            max_size = max(letter_image.size)
            new_image = Image.new('L', (max_size, max_size), (255))  # 'L' for 8-bit pixels, black and white
            new_image.paste(letter_image, ((max_size - letter_image.size[0])//2,
                                           (max_size - letter_image.size[1])//2))

            # Resize the image to 28x28
            letter_image = new_image.resize((28, 28))
            # Convert the PIL Image back to a numpy array
            letter_image = np.array(letter_image)

            # Rotate and flip the image
            letter_image = cv2.rotate(letter_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            letter_image = cv2.flip(letter_image, 0)
            
            letter_image = letter_image.astype('float32') / 255.0
            letter_image = 1 - letter_image
            
            letter_image = letter_image.reshape(-1, 28, 28, 1)

            

            model_prediction = model.predict(letter_image)
            category = np.argmax(model_prediction[0])
            prediction = prediction + alphabet_list[category]
        
        # Show the image with rectangles
        # cv2.imshow('Image with rectangles', cv_image_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Show the prediction
        self.prediction_label.config(text=f"The model predicts: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WhiteboardApp(root)
    root.mainloop()
import tkinter as tk
from PIL import ImageGrab
from PIL import Image
import cv2
import numpy as np
import random

class WhiteboardApp:
    gaming = False
    categories = ['apple', 'orange', 'strawberry', 'pear', 'mango', 'banana', 'pineaple']
    
    def __init__(self, root):
        self.root = root
        self.root.title("Whiteboard")
        background = 'lightblue'

        self.canvas = tk.Canvas(root, width=1000, height=500, bg="white")
        self.canvas.pack()
        self.root.configure(bg=background)
        
        self.clear_button = tk.Button(root, text="Limpiar pizarra", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, anchor="n", padx=10, pady=10)
        self.clear_button.config(bg="#00b4d8", fg="black", borderwidth=2, font=("Helvetica", 10))
        
        self.erase_button = tk.Button(root, text="Borrar", command=self.start_erase)
        self.erase_button.pack(side=tk.RIGHT, anchor="n", padx=10, pady=10)
        self.erase_button.config(bg="#00b4d8", fg="black", borderwidth=2, font=("Helvetica", 10))
     
        self.draw_button = tk.Button(root, text="Dibujar", command=self.start_draw)
        self.draw_button.pack(side=tk.RIGHT, anchor="n", padx=10, pady=10)
        self.draw_button.config(bg="#00b4d8", fg="black", borderwidth=2, font=("Helvetica", 10))
        
        self.save_button = tk.Button(root, text="Empezar juego", command=self.game, width=17, height=2)
        self.save_button.pack(side=tk.LEFT, anchor="n", padx=10, pady=10)
        self.save_button.config(bg="#00b4d8", fg="black", borderwidth=2, font=("Helvetica", 12))

        self.word_to_guess = tk.Label(root, text="Palabra", font=("Helvetica", 16, "bold"), fg="black", bg=background)
        self.word_to_guess.pack(side=tk.TOP, anchor="center")
        
        self.counter = tk.Label(root, text=0, font=("Helvetica", 16), fg="black", bg=background)
        self.counter.pack(side=tk.TOP, anchor="center")
        
        self.predict_word = tk.Label(root, text="Predicción", font=("Helvetica", 14), fg="black", bg=background)
        self.predict_word.pack(side=tk.TOP, anchor="center")

        self.drawing = False
        self.erasing = False
        self.last_x, self.last_y = None, None

        self.canvas.bind("<Button-1>", self.start_action)
        self.canvas.bind("<B1-Motion>", self.draw_or_erase)
        self.canvas.bind("<ButtonRelease-1>", self.stop_action)

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

    def draw_or_erase(self, event):
        if self.drawing:
            x, y = event.x, event.y
            radius = 6  # Radius of the circle
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black")
            self.last_x, self.last_y = x, y
        elif self.erasing:
            x, y = event.x, event.y
            self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        
    def game(self):
        if not self.gaming:
            self.predict_word.config(text="Predicción")
            selected_word = random.choice(self.categories)
            self.word_to_guess.config(text=selected_word.capitalize())
            self.update_counter(counter=30, selected_word=selected_word)
                
    def update_counter(self, counter=20, selected_word=''):
        if counter >= 0:
            self.counter.config(text=counter)
            if counter % 2 == 1:
                predicted_word = random.choice(self.categories)
                self.predict_word.config(text=f"¿La palabra es: {predicted_word.capitalize()}?")
                if selected_word == predicted_word:
                    self.gaming = False
                    self.predict_word.config(text=f"¡Ganaste! La palabra era {predicted_word}")
                    return
            self.root.after(1000, self.update_counter, counter - 1, selected_word)
        else:
            self.predict_word.config(text=f"No logré adivinar la palabra :(")
            self.gaming = False

    def predict(self):
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))

        # Convert the PIL image to a grayscale OpenCV image
        cv_image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Apply a binary threshold to the image
        _, thresholded = cv2.threshold(cv_image_gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        padding = 30  # Adjust as needed
        prediction = ""
        # Find the minimum and maximum x and y coordinates across all contours
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = -float('inf'), -float('inf')

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + w), max(max_y, y + h)
        if min_x != float('inf') & min_y != float('inf') & max_x != -float('inf') & max_y != -float('inf'):   
            min_x, min_y = max(0, min_x - padding), max(0, min_y - padding)
            max_x, max_y = min(cv_image_gray.shape[1], max_x + padding), min(cv_image_gray.shape[0], max_y + padding)
    
        all_drawings_image = cv_image_gray[min_y:max_y, min_x:max_x]
        
        all_drawings_image_pil = Image.fromarray(all_drawings_image)
        
        all_drawings_image_pil = all_drawings_image_pil.resize((28, 28))
        
        all_drawings_image_np = np.array(all_drawings_image_pil)
        
        all_drawings_image_np = cv2.rotate(all_drawings_image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        all_drawings_image_np = cv2.flip(all_drawings_image_np, 0)
        
        all_drawings_image_np = all_drawings_image_np.astype('float32') / 255.0
        all_drawings_image_np = 1 - all_drawings_image_np
    
        prediction = prediction + "H"
        
        # Show the image with rectangles
        #cv2.imshow('Image with rectangles', all_drawings_image_np)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Show the prediction
        return prediction

if __name__ == "__main__":
    root = tk.Tk()
    app = WhiteboardApp(root)
    root.mainloop()
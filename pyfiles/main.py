import tkinter as tk
from tkinter import filedialog, messagebox
import random

class CNNGui:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST CNN Trainer")
        self.root.geometry("1000x600")

        # Main container
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        # Split left / right
        self.left_frame = tk.Frame(self.main_frame, padx=20, pady=20)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.main_frame, padx=20, pady=20)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.build_left_panel()
        self.build_right_panel()

    # -----------------------------
    # LEFT PANEL (Training)
    # -----------------------------
    def build_left_panel(self):
        tk.Label(self.left_frame, text="Training Panel", font=("Arial", 16, "bold")).pack(pady=10)

        tk.Label(self.left_frame, text="Hidden Layers:").pack()
        self.hidden_layers_entry = tk.Entry(self.left_frame)
        self.hidden_layers_entry.pack(pady=5)

        tk.Label(self.left_frame, text="Epochs:").pack()
        self.epochs_entry = tk.Entry(self.left_frame)
        self.epochs_entry.pack(pady=5)

        tk.Button(self.left_frame, text="Load Dataset", command=self.load_dataset).pack(pady=10)
        tk.Button(self.left_frame, text="Load Labels", command=self.load_labels).pack(pady=10)
        tk.Button(self.left_frame, text="Train Model", command=self.train_model).pack(pady=20)

    # -----------------------------
    # RIGHT PANEL (Inference)
    # -----------------------------
    def build_right_panel(self):
        tk.Label(self.right_frame, text="Inference Panel", font=("Arial", 16, "bold")).pack(pady=10)

        tk.Button(self.right_frame, text="Load Model", command=self.load_model).pack(pady=10)
        tk.Button(self.right_frame, text="Load Sample", command=self.load_sample).pack(pady=10)
        tk.Button(self.right_frame, text="Load Sample Labels", command=self.load_sample_labels).pack(pady=10)
        tk.Button(self.right_frame, text="Check Accuracy", command=self.check_accuracy).pack(pady=20)

        tk.Button(self.right_frame, text="Open Drawing Area", command=self.open_drawing_window).pack(pady=20)

        self.prediction_label = tk.Label(self.right_frame, text="Prediction: -", font=("Arial", 14))
        self.prediction_label.pack(pady=10)

    # -----------------------------
    # STUB FUNCTIONS
    # -----------------------------
    def load_dataset(self):
        filedialog.askopenfilename()
        print("Dataset loaded (stub)")

    def load_labels(self):
        filedialog.askopenfilename()
        print("Labels loaded (stub)")

    def train_model(self):
        layers = self.hidden_layers_entry.get()
        epochs = self.epochs_entry.get()
        print(f"Training with {layers} layers for {epochs} epochs (stub)")

    def load_model(self):
        filedialog.askopenfilename()
        print("Model loaded (stub)")

    def load_sample(self):
        filedialog.askopenfilename()
        print("Sample loaded (stub)")

    def load_sample_labels(self):
        filedialog.askopenfilename()
        print("Sample labels loaded (stub)")

    def check_accuracy(self):
        print("Accuracy checked (stub)")
        messagebox.showinfo("Accuracy", "Accuracy: 0.00% (stub)")

    # -----------------------------
    # DRAWING WINDOW
    # -----------------------------
    def open_drawing_window(self):
        self.draw_window = tk.Toplevel(self.root)
        self.draw_window.title("Draw Digit (15x15)")

        self.grid_size = 15
        self.cell_size = 25

        self.canvas = tk.Canvas(
            self.draw_window,
            width=self.grid_size * self.cell_size,
            height=self.grid_size * self.cell_size,
            bg="white"
        )
        self.canvas.pack()

        self.pixels = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        self.canvas.bind("<Button-1>", self.toggle_pixel)

        tk.Button(self.draw_window, text="Clear", command=self.clear_canvas).pack(pady=10)

    def toggle_pixel(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.pixels[row][col] ^= 1  # toggle

            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            color = "black" if self.pixels[row][col] else "white"
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

            self.update_prediction()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pixels = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.prediction_label.config(text="Prediction: -")

    def update_prediction(self):
        # Stub prediction
        fake_prediction = random.randint(0, 9)
        self.prediction_label.config(text=f"Prediction: {fake_prediction}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CNNGui(root)
    root.mainloop()
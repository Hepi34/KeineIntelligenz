from importlib.resources import path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk


class CNNGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Keine Intelligenz by Hepi34, Onatic07 and fritziii")
        self.root.geometry("1100x650")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = tk.Frame(self.main_frame, padx=30, pady=30)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.main_frame, padx=30, pady=30)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.build_left_panel()
        self.build_right_panel()

    # -----------------------------
    # LEFT PANEL (Training)
    # -----------------------------
    def build_left_panel(self):
        tk.Label(self.left_frame, text="Training Panel",
                 font=("Arial", 16, "bold")).pack(pady=10)

        # Hidden Layers
        hidden_layers_label = tk.Label(self.left_frame, text="Hidden Layers:")
        hidden_layers_label.pack()

        self.hidden_layers_var = tk.StringVar()
        self.hidden_layers_entry = tk.Entry(self.left_frame, textvariable=self.hidden_layers_var, width=30)
        self.hidden_layers_entry.pack(pady=5)

        self.hidden_layers_error = tk.Label(
            self.left_frame,
            text="",
            fg="red",
            height=1
        )
        self.hidden_layers_error.pack()

        # Live validation
        self.hidden_layers_var.trace_add("write", lambda *args: self.validate_number_field(
            self.hidden_layers_var, self.hidden_layers_error))

        # Epochs
        epochs_label = tk.Label(self.left_frame, text="Epochs:")
        epochs_label.pack()
        
        self.epochs_var = tk.StringVar()
        self.epochs_entry = tk.Entry(self.left_frame, textvariable=self.epochs_var, width=30)
        self.epochs_entry.pack(pady=5)

        self.epochs_error = tk.Label(
            self.left_frame,
            text="",
            fg="red",
            height=1
        )
        self.epochs_error.pack()

        # Live validation
        self.epochs_var.trace_add("write", lambda *args: self.validate_number_field(
            self.epochs_var, self.epochs_error))

        tk.Button(self.left_frame, text="Load Dataset",
                  command=self.load_dataset).pack(pady=10)

        self.dataset_label = tk.Label(self.left_frame, text="No dataset loaded", fg="gray", wraplength=200)
        self.dataset_label.pack(pady=5)

        tk.Button(self.left_frame, text="Load Labels",
                  command=self.load_labels).pack(pady=10)

        self.labels_label = tk.Label(self.left_frame, text="No labels loaded", fg="gray", wraplength=200)
        self.labels_label.pack(pady=5)

        tk.Button(self.left_frame, text="Train Model",
                  command=self.train_model).pack(pady=20)

        # Progress Bar
        tk.Label(self.left_frame, text="Training Progress:").pack(pady=(20, 5))
        self.progress = ttk.Progressbar(
            self.left_frame,
            orient="horizontal",
            length=250,
            mode="determinate"
        )
        self.progress.pack(pady=5)

        self.training_status = tk.Label(self.left_frame, text="")
        self.training_status.pack(pady=5)

    # -----------------------------
    # RIGHT PANEL (Inference)
    # -----------------------------
    def build_right_panel(self):
        tk.Label(self.right_frame, text="Inference Panel",
                 font=("Arial", 16, "bold")).pack(pady=10)

        tk.Button(self.right_frame, text="Load Model",
                  command=self.load_model).pack(pady=10)

        self.model_status_label = tk.Label(self.right_frame, text="")
        self.model_status_label.pack(pady=5)

        tk.Button(self.right_frame, text="Load Sample",
                  command=self.load_sample).pack(pady=10)

        tk.Button(self.right_frame, text="Load Sample Labels",
                  command=self.load_sample_labels).pack(pady=10)

        tk.Button(self.right_frame, text="Check Accuracy",
                  command=self.check_accuracy).pack(pady=20)

        tk.Button(self.right_frame, text="Open Drawing Area",
                  command=self.open_drawing_window).pack(pady=20)

    # -----------------------------
    # STUB FUNCTIONS
    # -----------------------------
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("All files", "*.*"), ("NumPy files", "*.npy"), ("Ubyte files", "*.ubyte")]
        )
        if file_path:
            self.dataset_path = file_path
            # Show just the filename, or the full path if you prefer
            filename = file_path.split("/")[-1]
            self.dataset_label.config(text=f"✓ {filename}", fg="green")

    def load_labels(self):
        file_path = filedialog.askopenfilename(
            title="Select Labels File",
            filetypes=[("All files", "*.*"), ("NumPy files", "*.npy"), ("Ubyte files", "*.ubyte")]
        )
        if file_path:
            self.labels_path = file_path
            # Show just the filename, or the full path if you prefer
            filename = file_path.split("/")[-1]
            self.labels_label.config(text=f"✓ {filename}", fg="green")
    
    def train_model(self):
        return



    def load_model(self):
        return

    def load_sample(self):
        filedialog.askopenfilename()
        print("Sample loaded (stub)")

    def load_sample_labels(self):
        filedialog.askopenfilename()
        print("Sample labels loaded (stub)")

    def check_accuracy(self):
        messagebox.showinfo("Accuracy", "Accuracy: 0.00% (stub)")

    # -----------------------------
    # DRAWING WINDOW
    # -----------------------------
    def open_drawing_window(self):
        self.draw_window = tk.Toplevel(self.root)
        self.draw_window.title("Draw Digit (10x15 Grid)")

        self.cols = 10
        self.rows = 15
        self.cell_size = 30

        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size

        self.canvas = tk.Canvas(
            self.draw_window,
            width=canvas_width,
            height=canvas_height,
            bg="white",
            highlightthickness=2,
            highlightbackground="black"
        )
        self.canvas.pack(pady=10)

        self.pixels = [[0 for _ in range(self.cols)]
                       for _ in range(self.rows)]

        self.draw_grid()

        self.canvas.bind("<Button-1>", self.toggle_pixel)

        # Space under drawing area
        tk.Label(self.draw_window, text="").pack()

        self.prediction_label = tk.Label(
            self.draw_window,
            text="Prediction: -",
            font=("Arial", 14, "bold")
        )
        self.prediction_label.pack(pady=5)

        self.certainty_label = tk.Label(
            self.draw_window,
            text="Certainty: - %",
            font=("Arial", 12)
        )
        self.certainty_label.pack(pady=5)

        tk.Button(self.draw_window, text="Clear",
                  command=self.clear_canvas).pack(pady=10)

    def draw_grid(self):
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="white",
                    outline="#cccccc"  # subtle grey lines
                )

    def toggle_pixel(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.pixels[row][col] ^= 1

            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            color = "black" if self.pixels[row][col] else "white"

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=color,
                outline="#cccccc"
            )

            self.update_prediction()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pixels = [[0 for _ in range(self.cols)]
                       for _ in range(self.rows)]
        self.draw_grid()
        self.prediction_label.config(text="Prediction: -")
        self.certainty_label.config(text="Certainty: - %")

    def update_prediction(self):
        return
        
    def validate_number_field(self, var, error_label):
        value = var.get()

        if value == "" or value.isdigit():
            error_label.config(text="")
        else:
            error_label.config(text="Only numbers allowed")


if __name__ == "__main__":
    root = tk.Tk()
    app = CNNGui(root)
    root.mainloop()
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import numpy as np
from train_cnn_cpu import train_cnn as train_cnn_cpu, save_model as save_model_cpu


def detect_gpu():
    try:
        import cupy as cp
        cp.array([1])
        return ('cuda', None)
    except Exception:
        pass
    try:
        import mlx.core as mx
        mx.array([1])
        return ('metal', None)
    except Exception:
        pass
    return (None, "No compatible GPU found.\nRequires CUDA (cupy) or Apple Metal (mlx).")


def load_ubyte_images(path):
    with open(path, 'rb') as f:
        np.frombuffer(f.read(4), dtype='>i4')
        n    = int(np.frombuffer(f.read(4), dtype='>i4')[0])
        rows = int(np.frombuffer(f.read(4), dtype='>i4')[0])
        cols = int(np.frombuffer(f.read(4), dtype='>i4')[0])
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def load_ubyte_labels(path):
    with open(path, 'rb') as f:
        np.frombuffer(f.read(4), dtype='>i4')
        np.frombuffer(f.read(4), dtype='>i4')
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_data_file(path, kind):
    if path.endswith('.npy'):
        return np.load(path)
    return load_ubyte_images(path) if kind == 'images' else load_ubyte_labels(path)


class CNNGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Keine Intelligenz by Hepi34, Onatic07 and fritziii")
        self.root.geometry("1200x800")

        # Training state
        self.dataset_path = None
        self.labels_path  = None
        self.model        = None

        # Inference state
        self.infer_model  = None
        self.test_images  = None
        self.test_labels  = None

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame  = tk.Frame(self.main_frame, padx=24, pady=20)
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.main_frame, padx=24, pady=20)
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.build_left_panel()
        self.build_right_panel()

    # =========================================================
    # LEFT PANEL — Training
    # =========================================================
    def build_left_panel(self):
        tk.Label(self.left_frame, text="Training Panel",
                 font=("Arial", 16, "bold")).pack(pady=(0, 8))

        tk.Label(self.left_frame, text="Hidden Layers:").pack()
        self.hidden_layers_var = tk.StringVar()
        tk.Entry(self.left_frame, textvariable=self.hidden_layers_var,
                 width=30).pack(pady=3)
        self.hidden_layers_error = tk.Label(self.left_frame, text="", fg="red", height=1)
        self.hidden_layers_error.pack()
        self.hidden_layers_var.trace_add("write", lambda *a: self.validate_number(
            self.hidden_layers_var, self.hidden_layers_error))

        tk.Label(self.left_frame, text="Epochs:").pack()
        self.epochs_var = tk.StringVar()
        tk.Entry(self.left_frame, textvariable=self.epochs_var, width=30).pack(pady=3)
        self.epochs_error = tk.Label(self.left_frame, text="", fg="red", height=1)
        self.epochs_error.pack()
        self.epochs_var.trace_add("write", lambda *a: self.validate_number(
            self.epochs_var, self.epochs_error))

        tk.Label(self.left_frame, text="Device:", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        device_frame = tk.Frame(self.left_frame)
        device_frame.pack()
        self.device_var = tk.StringVar(value="cpu")
        tk.Checkbutton(device_frame, text="CPU", variable=self.device_var,
                       onvalue="cpu", offvalue="gpu",
                       command=self.on_device_toggle).grid(row=0, column=0, padx=15)
        tk.Checkbutton(device_frame, text="GPU", variable=self.device_var,
                       onvalue="gpu", offvalue="cpu",
                       command=self.on_device_toggle).grid(row=0, column=1, padx=15)

        self.gpu_error_label = tk.Label(self.left_frame, text="", fg="red",
                                        height=1, wraplength=260)
        self.gpu_error_label.pack(pady=(2, 0))

        self.threads_frame = tk.Frame(self.left_frame)
        self.threads_frame.pack()
        tk.Label(self.threads_frame, text="CPU Threads:").pack()
        self.threads_var = tk.StringVar(value="4")
        tk.Entry(self.threads_frame, textvariable=self.threads_var, width=30).pack(pady=3)
        self.threads_error = tk.Label(self.threads_frame, text="", fg="red", height=1)
        self.threads_error.pack()
        self.threads_var.trace_add("write", lambda *a: self.validate_number(
            self.threads_var, self.threads_error))

        tk.Button(self.left_frame, text="Load Dataset",
                  command=self.load_dataset).pack(pady=(10, 2))
        self.dataset_label = tk.Label(self.left_frame, text="No dataset loaded",
                                      fg="gray", wraplength=220)
        self.dataset_label.pack()

        tk.Button(self.left_frame, text="Load Labels",
                  command=self.load_train_labels).pack(pady=(8, 2))
        self.train_labels_label = tk.Label(self.left_frame, text="No labels loaded",
                                           fg="gray", wraplength=220)
        self.train_labels_label.pack()

        self.train_button = tk.Button(self.left_frame, text="Train Model",
                                      command=self.train_model)
        self.train_button.pack(pady=16)

        tk.Label(self.left_frame, text="Training Progress:").pack(pady=(4, 2))
        self.progress = ttk.Progressbar(self.left_frame, orient="horizontal",
                                        length=250, mode="determinate")
        self.progress.pack(pady=3)
        self.training_status = tk.Label(self.left_frame, text="")
        self.training_status.pack(pady=3)

    def on_device_toggle(self):
        if self.device_var.get() == "cpu":
            self.gpu_error_label.config(text="", fg="red")
            self.threads_frame.pack()
        else:
            self.threads_frame.pack_forget()
            gpu_type, error = detect_gpu()
            if error:
                self.gpu_error_label.config(text=error, fg="red")
                self.device_var.set("cpu")
                self.threads_frame.pack()
            else:
                self.gpu_error_label.config(
                    text=f"✓ {gpu_type.upper()} GPU detected", fg="green")

    def load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("All files", "*.*"), ("NumPy", "*.npy"), ("Ubyte", "*.ubyte")])
        if path:
            self.dataset_path = path
            self.dataset_label.config(text=f"✓ {path.split('/')[-1]}", fg="green")

    def load_train_labels(self):
        path = filedialog.askopenfilename(
            title="Select Labels File",
            filetypes=[("All files", "*.*"), ("NumPy", "*.npy"), ("Ubyte", "*.ubyte")])
        if path:
            self.labels_path = path
            self.train_labels_label.config(text=f"✓ {path.split('/')[-1]}", fg="green")

    def train_model(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Please load a dataset first"); return
        if not self.labels_path:
            messagebox.showerror("Error", "Please load labels first"); return

        epochs_str = self.epochs_var.get()
        hidden_str = self.hidden_layers_var.get()
        if not epochs_str or not epochs_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid number for epochs"); return
        if not hidden_str or not hidden_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid number for hidden layers"); return

        epochs = int(epochs_str)
        hidden = int(hidden_str)
        device = self.device_var.get()

        if device == "cpu":
            t = self.threads_var.get()
            if not t or not t.isdigit():
                messagebox.showerror("Error", "Please enter a valid number for threads"); return
            num_threads = int(t)
            gpu_type = None
        else:
            num_threads = None
            gpu_type, err = detect_gpu()
            if err:
                messagebox.showerror("GPU Error", err); return

        self.train_button.config(state="disabled")
        self.progress["value"] = 0
        lbl = f"CPU ({num_threads} threads)" if device == "cpu" else f"{gpu_type.upper()} GPU"
        self.training_status.config(text=f"Training on {lbl}…", fg="blue")
        self.root.update()

        def run():
            try:
                def cb(epoch, total, loss):
                    self.progress["value"] = epoch / total * 100
                    self.training_status.config(
                        text=f"Epoch {epoch}/{total}, Loss: {loss:.4f}", fg="blue")
                    self.root.update()

                if device == "cpu":
                    self.model = train_cnn_cpu(
                        self.dataset_path, self.labels_path,
                        epochs=epochs, hidden_layers=hidden,
                        num_threads=num_threads, callback=cb)
                    save_model_cpu(self.model, "cnn_model.pkl")
                else:
                    from train_gpu_cnn import (train_cnn as train_gpu,
                                               save_model as save_model_gpu)
                    self.model = train_gpu(
                        self.dataset_path, self.labels_path,
                        epochs=epochs, hidden_layers=hidden,
                        gpu_type=gpu_type, callback=cb)
                    save_model_gpu(self.model, "cnn_model.pkl")

                self.progress["value"] = 100
                self.training_status.config(
                    text="Training complete! Model saved as cnn_model.pkl", fg="green")
            except Exception as e:
                self.training_status.config(text=f"Error: {e}", fg="red")
                messagebox.showerror("Training Error", str(e))
            finally:
                self.train_button.config(state="normal")

        threading.Thread(target=run, daemon=True).start()

    # =========================================================
    # RIGHT PANEL — Inference
    # =========================================================
    def build_right_panel(self):
        tk.Label(self.right_frame, text="Inference Panel",
                 font=("Arial", 16, "bold")).pack(pady=(0, 8))

        # Load model
        tk.Button(self.right_frame, text="Load Model (.pkl)",
                  command=self.load_infer_model, width=22).pack(pady=(4, 2))
        self.model_status_label = tk.Label(self.right_frame, text="No model loaded",
                                           fg="gray", wraplength=280)
        self.model_status_label.pack(pady=(0, 8))

        ttk.Separator(self.right_frame, orient="horizontal").pack(fill="x", pady=4)

        # Accuracy tester
        tk.Label(self.right_frame, text="Accuracy Test",
                 font=("Arial", 12, "bold")).pack(pady=(6, 4))

        test_btn_frame = tk.Frame(self.right_frame)
        test_btn_frame.pack(fill="x")

        tk.Button(test_btn_frame, text="Load Test Images",
                  command=self.load_test_images, width=20).grid(
                      row=0, column=0, padx=(0, 8), pady=3, sticky="w")
        self.test_images_label = tk.Label(test_btn_frame, text="Not loaded",
                                          fg="gray", anchor="w")
        self.test_images_label.grid(row=0, column=1, sticky="w")

        tk.Button(test_btn_frame, text="Load Test Labels",
                  command=self.load_test_labels, width=20).grid(
                      row=1, column=0, padx=(0, 8), pady=3, sticky="w")
        self.test_labels_label = tk.Label(test_btn_frame, text="Not loaded",
                                          fg="gray", anchor="w")
        self.test_labels_label.grid(row=1, column=1, sticky="w")

        self.eval_btn = tk.Button(self.right_frame, text="▶  Run Evaluation",
                                  command=self.run_evaluation,
                                  bg="#4a90d9", fg="white",
                                  activebackground="#357abd",
                                  relief="flat", padx=10, pady=4)
        self.eval_btn.pack(pady=8)

        self.eval_progress = ttk.Progressbar(self.right_frame, orient="horizontal",
                                             length=300, mode="determinate")
        self.eval_progress.pack(pady=2)
        self.eval_status = tk.Label(self.right_frame, text="", fg="gray")
        self.eval_status.pack(pady=2)

        self.accuracy_label = tk.Label(self.right_frame, text="",
                                       font=("Arial", 13, "bold"))
        self.accuracy_label.pack(pady=4)

        cols = ("Digit", "Correct", "Total", "%")
        self.tree = ttk.Treeview(self.right_frame, columns=cols,
                                 show="headings", height=10, selectmode="none")
        for c, w in zip(cols, [50, 70, 60, 70]):
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=w)
        self.tree.pack(fill="x", pady=(4, 6))

        ttk.Separator(self.right_frame, orient="horizontal").pack(fill="x", pady=4)

        tk.Button(self.right_frame, text="✏  Open Drawing Area",
                  command=self.open_drawing_window).pack(pady=6)

    def load_infer_model(self):
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        try:
            from inference import load_model
            self.infer_model = load_model(path)
            self.model_status_label.config(
                text=f"✓ {path.split('/')[-1]}", fg="green")
        except Exception as e:
            self.model_status_label.config(text="Error loading model", fg="red")
            messagebox.showerror("Load Error", str(e))

    def load_test_images(self):
        path = filedialog.askopenfilename(
            title="Select Test Images",
            filetypes=[("All files", "*.*"), ("NumPy", "*.npy"), ("Ubyte", "*.ubyte")])
        if not path:
            return
        try:
            self.test_images = load_data_file(path, 'images')
            self.test_images_label.config(
                text=f"✓ {path.split('/')[-1]} ({len(self.test_images)})", fg="green")
        except Exception as e:
            self.test_images_label.config(text="Error", fg="red")
            messagebox.showerror("Load Error", str(e))

    def load_test_labels(self):
        path = filedialog.askopenfilename(
            title="Select Test Labels",
            filetypes=[("All files", "*.*"), ("NumPy", "*.npy"), ("Ubyte", "*.ubyte")])
        if not path:
            return
        try:
            self.test_labels = load_data_file(path, 'labels')
            self.test_labels_label.config(
                text=f"✓ {path.split('/')[-1]} ({len(self.test_labels)})", fg="green")
        except Exception as e:
            self.test_labels_label.config(text="Error", fg="red")
            messagebox.showerror("Load Error", str(e))

    def run_evaluation(self):
        if self.infer_model is None:
            messagebox.showerror("Missing", "Please load a model first."); return
        if self.test_images is None:
            messagebox.showerror("Missing", "Please load test images first."); return
        if self.test_labels is None:
            messagebox.showerror("Missing", "Please load test labels first."); return
        if len(self.test_images) != len(self.test_labels):
            messagebox.showerror("Mismatch",
                f"Images ({len(self.test_images)}) and labels "
                f"({len(self.test_labels)}) count do not match."); return

        self.eval_btn.config(state="disabled")
        self.eval_progress["value"] = 0
        self.eval_status.config(text="Evaluating…", fg="blue")
        self.accuracy_label.config(text="")
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.root.update()

        def run():
            try:
                results = self._evaluate_with_progress(
                    self.infer_model, self.test_images, self.test_labels)
                self._show_results(results)
            except Exception as e:
                self.eval_status.config(text=f"Error: {e}", fg="red")
                messagebox.showerror("Evaluation Error", str(e))
            finally:
                self.eval_btn.config(state="normal")

        threading.Thread(target=run, daemon=True).start()

    def _evaluate_with_progress(self, model, images, labels, batch_size=128):
        from inference import _resize_batch
        imgs = np.array(images, dtype=np.float32)
        if imgs.max() > 1.0:
            imgs /= 255.0
        if imgs.ndim == 3:
            imgs = imgs[:, np.newaxis, :, :]
        if imgs.shape[2] != 28 or imgs.shape[3] != 28:
            imgs = _resize_batch(imgs, 28, 28)
        lbls = np.array(labels, dtype=np.int32)
        n = len(lbls)
        all_preds = []
        nb = (n + batch_size - 1) // batch_size
        for idx, i in enumerate(range(0, n, batch_size)):
            logits = model.forward(imgs[i:i + batch_size])
            all_preds.append(np.argmax(logits, axis=1))
            self.eval_progress["value"] = (idx + 1) / nb * 100
            self.eval_status.config(text=f"Batch {idx+1}/{nb}", fg="blue")
            self.root.update()
        all_preds = np.concatenate(all_preds)
        correct = int((all_preds == lbls).sum())
        per_class = {}
        for c in range(10):
            m = (lbls == c)
            per_class[c] = {'correct': int((all_preds[m] == c).sum()),
                            'total': int(m.sum())}
        return {'accuracy': correct / n * 100, 'correct': correct,
                'total': n, 'per_class': per_class}

    def _show_results(self, r):
        acc = r['accuracy']
        color = "#1a7a1a" if acc >= 90 else ("#d97a00" if acc >= 70 else "red")
        self.accuracy_label.config(
            text=f"Accuracy: {acc:.2f}%  ({r['correct']}/{r['total']})", fg=color)
        self.eval_progress["value"] = 100
        self.eval_status.config(text="Done.", fg="green")
        for digit in range(10):
            pc = r['per_class'][digit]
            t, c = pc['total'], pc['correct']
            pct = f"{c/t*100:.1f}%" if t > 0 else "—"
            tag = "good" if (t > 0 and c/t >= 0.9) else ("ok" if (t > 0 and c/t >= 0.7) else "bad")
            self.tree.insert("", "end", values=(digit, c, t, pct), tags=(tag,))
        self.tree.tag_configure("good", foreground="#1a7a1a")
        self.tree.tag_configure("ok",   foreground="#d97a00")
        self.tree.tag_configure("bad",  foreground="red")

    # =========================================================
    # Drawing window
    # =========================================================
    def open_drawing_window(self):
        win = tk.Toplevel(self.root)
        win.title("Draw Digit")
        self.draw_window = win
        self.cols = 10
        self.rows = 15
        self.cell_size = 30

        self.canvas = tk.Canvas(win,
            width=self.cols * self.cell_size,
            height=self.rows * self.cell_size,
            bg="white", highlightthickness=2, highlightbackground="black")
        self.canvas.pack(pady=10)
        self.pixels = [[0]*self.cols for _ in range(self.rows)]
        self._last_drag_cell = None   # avoid re-drawing same cell mid-drag
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.toggle_pixel)
        self.canvas.bind("<B1-Motion>", self.drag_pixel)

        self.prediction_label = tk.Label(win, text="Prediction: —",
                                         font=("Arial", 14, "bold"))
        self.prediction_label.pack(pady=5)
        self.certainty_label  = tk.Label(win, text="Certainty: — %",
                                         font=("Arial", 12))
        self.certainty_label.pack(pady=2)
        tk.Button(win, text="Clear", command=self.clear_canvas).pack(pady=8)

    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c*self.cell_size, r*self.cell_size
                self.canvas.create_rectangle(
                    x1, y1, x1+self.cell_size, y1+self.cell_size,
                    fill="white", outline="#cccccc")

    def _draw_cell(self, row, col, value):
        self.pixels[row][col] = value
        x1, y1 = col*self.cell_size, row*self.cell_size
        color = "black" if value else "white"
        self.canvas.create_rectangle(
            x1, y1, x1+self.cell_size, y1+self.cell_size,
            fill=color, outline="#cccccc")

    def toggle_pixel(self, event):
        """Single click: toggle the cell on/off."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._last_drag_cell = (row, col)
            self._draw_cell(row, col, 1 - self.pixels[row][col])
            self.update_prediction()

    def drag_pixel(self, event):
        """Click-and-drag: always paints cells black (draw mode only)."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if (row, col) == self._last_drag_cell:
                return   # same cell, skip
            self._last_drag_cell = (row, col)
            if self.pixels[row][col] == 0:   # only paint, never erase on drag
                self._draw_cell(row, col, 1)
                self.update_prediction()

    def clear_canvas(self):
        self.pixels = [[0]*self.cols for _ in range(self.rows)]
        self._last_drag_cell = None
        self.canvas.delete("all")
        self.draw_grid()
        self.prediction_label.config(text="Prediction: —")
        self.certainty_label.config(text="Certainty: — %")

    def update_prediction(self):
        if self.infer_model is None:
            self.prediction_label.config(text="Prediction: (load a model first)")
            self.certainty_label.config(text="")
            return
        from inference import predict
        grid = np.array(self.pixels, dtype=np.float32)
        if grid.max() == 0:
            self.prediction_label.config(text="Prediction: —")
            self.certainty_label.config(text="Certainty: — %")
            return
        cls, conf = predict(self.infer_model, grid)
        self.prediction_label.config(text=f"Prediction: {cls}")
        self.certainty_label.config(
            text=f"Certainty: {conf:.1f}%",
            fg="#1a7a1a" if conf >= 70 else "#d97a00")

    def validate_number(self, var, error_label):
        v = var.get()
        error_label.config(
            text="" if (v == "" or v.isdigit()) else "Only numbers allowed")


if __name__ == "__main__":
    root = tk.Tk()
    app = CNNGui(root)
    root.mainloop()

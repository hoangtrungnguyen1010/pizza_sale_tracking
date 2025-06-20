import tkinter as tk
from tkinter import filedialog, messagebox
from app.video_processing import process_video
import os

class PizzaSalesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pizza Sales Tracking")

        self.video_path = None
        self.output_path = "data/output/processed_video.mp4"
        self.crop_area = None

        self.label = tk.Label(root, text="Select a video file:")
        self.label.pack(pady=5)

        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(pady=5)

        self.start_button = tk.Button(root, text="Process Video", command=self.process_video)
        self.start_button.pack(pady=5)

        self.preview_button = tk.Button(root, text="Preview Processed Video", command=self.preview_video)
        self.preview_button.pack(pady=5)

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:
            messagebox.showinfo("Info", f"Uploaded video: {self.video_path}")

    def process_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video file first.")
            return

        process_video(self.video_path, self.output_path, self.crop_area)
        messagebox.showinfo("Success", f"Video processed and saved to {self.output_path}")

    def preview_video(self):
        if not os.path.exists(self.output_path):
            messagebox.showerror("Error", "No processed video available. Please process a video first.")
            return

        os.system(f"xdg-open {self.output_path}")  # Works on Linux; use 'start' for Windows or 'open' for macOS.


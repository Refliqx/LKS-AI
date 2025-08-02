import tkinter as tk
import tkinter.ttk as ttk
from tkinter.ttk import *
import pandas as pd

data = pd.read_csv("./dataset/Datasset LKS AI Kabupaten Malang 2025.csv")
df = pd.DataFrame(data)

features = df.drop(columns=["target"])
labels = df["target"]

root = tk.Tk()
root.geometry("1200x800")
root.title("GUI Apps")

# Style
heading_font = ("Roboto", 20, "bold")
subheading_font = ("Roboto", 18, "bold")
paragraph_font = ("Roboto", 16)

# Frame components
main_frame = ttk.Frame(root)
main_frame.pack(fill="both")

# main content
app_title = ttk.Label(main_frame, text="Cardiovaskular Classifier", font=heading_font).pack(pady=50, side="top")
sub_title = ttk.Label(main_frame, text="What is Cardiovaskular?", font=subheading_font).pack(padx=50, pady=20, anchor="w")



root.mainloop()
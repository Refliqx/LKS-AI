import tkinter as tk
import tkinter.ttk as ttk
from tkinter.ttk import *
import pandas as pd

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
paragraph_text = ttk.Label(main_frame, text="lorem ipsum dolor sit amet vira conservation ofer ladf lorem ispsum jdksadkakjdkadjaidjiwijjdiisdijs", font=paragraph_font).pack(padx=50, anchor="w", fill="both")

# treeview table for dataset

columns = pd.DataFrame.columns()
dataset_table = ttk.Treeview(main_frame, columns=[])

root.mainloop()
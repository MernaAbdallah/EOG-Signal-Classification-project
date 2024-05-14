import tkinter as tk
import time
from tkinter import filedialog
import preprocessing as ps
import feature_extraction as fe
from PIL import ImageTk, Image
import models as mod


class ArrowWidget(tk.Canvas):
    def __init__(self, master=None, arrow_coord=None, **kwargs):
        super().__init__(master, **kwargs)
        self.arrow = None
        self.__draw_arrow(arrow_coord)
        self.__flash_duration = 2000  # Duration of flashing in milliseconds
        self.__flash_interval = 500  # Interval for flashing in milliseconds
        self.__flashing = False
        self.__flash_start_time = 0  # Initialize flash start time

    def __draw_arrow(self, arrow_coord):
        self.delete("arrow")
        # Draw the arrow
        self.arrow = self.create_polygon(arrow_coord, outline="black", fill="", tags="arrow", width=2)
        self.tag_bind("arrow", "Enter", self.__flash_arrow)
        self.tag_bind("arrow", "Leave", self.stop_flash)

    def start_flash(self):
        if not self.__flashing:
            self.__flashing = True
            self.__flash_start_time = time.time()  # Set flash start time
            self.__flash()

    def stop_flash(self, event=None):
        self.__flashing = False
        self.itemconfig(self.arrow, fill="")

    def __flash_arrow(self, event=None):
        self.__flash_start_time = time.time()
        self.__flash()

    def __flash(self):
        if self.__flashing:
            if time.time() - self.__flash_start_time < self.__flash_duration / 750:
                self.itemconfig(self.arrow, fill="green")
                self.after(self.__flash_interval, self.__un_flash)
            else:
                self.itemconfig(self.arrow, fill="")
        else:
            self.itemconfig(self.arrow, fill="")

    def __un_flash(self):
        if self.__flashing:
            self.itemconfig(self.arrow, fill="")
            self.after(self.__flash_interval, self.__flash)


class Gui:

    def show_arrow(self, direction):
        self.image_path = f'images/${direction}_arrow.png'

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        ps.PreProcessing.add_data_to_csv(folder_path)
        ys, data = ps.PreProcessing.label_encode(['test_data.csv'])
        preprocessed_data = []
        for dt in data:
            preprocessed_data.append(ps.PreProcessing.preprocess_signal(dt))

        x_test, y_test, x_train, y_train = fe.FeatureExtraction.statistical_features(ys, preprocessed_data)
        # models, acc, mse, reports, train_acc = mod.Models.classify(x_test, y_test, x_train, y_train, False)
        # for i in range(len(models)):
        #     print(f'========== {models[i]} ==========')
        #     print(f'Train Accuracy  = {train_acc[i]}')
        #     print(f"Test Accuracy: {acc[i]} %")
        #     print('================================')

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('EOG')
        self.root.geometry('1000x600')
        self.image_path = None
        self.button_frame = tk.Frame(self.root)
        self.button_frame.columnconfigure(0, weight=2)
        self.task1_btn = tk.Button(self.button_frame, text='Test', command=self.load_folder)
        self.task1_btn.grid(row=0, sticky=tk.W + tk.E, padx=10)
        if self.image_path is not None:
            self.img2 = ImageTk.PhotoImage(Image.open(self.image_path), size=50)
            self.panel2 = tk.Label(self.root, image=self.img2)
            self.panel2.pack(side="bottom", fill='both', expand='yes')

        self.button_frame.pack(fill='x', pady=10)
        self.root.mainloop()

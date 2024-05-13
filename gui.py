import tkinter as tk
import time


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

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('EOG')
        self.root.geometry('1000x600')
        # left_arrow = ArrowWidget(self.root, [150, 450, 50, 500, 150, 550])
        # up_arrow = ArrowWidget(self.root, [50, 250, 100, 150, 150, 250])
        # right_arrow = ArrowWidget(self.root, [200, 250, 300, 300, 200, 350])
        # down_arrow = ArrowWidget(self.root, [50, 200, 100, 300, 150, 200])
        # left_arrow.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        # up_arrow.pack(expand=True, fill=tk.BOTH, side=tk.TOP)
        # right_arrow.pack(expand=True, fill=tk.BOTH, side=tk.RIGHT)
        # down_arrow.pack(expand=True, fill=tk.BOTH, side=tk.BOTTOM)
        self.button_frame = tk.Frame(self.root)
        self.button_frame.columnconfigure(0, weight=1)
        self.task1_btn = tk.Button(self.button_frame, text='Test')
        self.task1_btn.grid(row=0, sticky=tk.W + tk.E, padx=10)
        self.button_frame.pack(fill='x', pady=10)
        self.root.mainloop()

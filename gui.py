import tkinter as tk
from tkinter import filedialog, ttk
import preprocessing as ps
import feature_extraction as fe
import models as mod


class Gui:

    def show_arrow(self, index):
        if index < len(self.mapped_predictions):
            image = tk.PhotoImage(file=f"images/{self.mapped_predictions[index]}_arrow.png")
            self.label.config(text='', image=image)
            self.label.image = image
        self.root.after(4000, lambda: self.show_arrow(index + 1))

    def load_folder(self):
        ys, data, le = ps.PreProcessing.label_encode(ps.PreProcessing.add_data_to_csv(filedialog.askdirectory()))
        preprocessed_data = []
        for dt in data:
            preprocessed_data.append(ps.PreProcessing.preprocess_signal(dt))

        x_test, y_test, x_train, y_train = fe.FeatureExtraction.statistical_features(ys, preprocessed_data, False)
        models, acc, mapped_predictions, predictions, _ = mod.Models.classify(x_test, y_test, le, x_train,
                                                                              y_train, False)
        self.mapped_predictions = mapped_predictions
        self.predictions = predictions
        self.models = models
        self.acc = acc
        for i in range(len(models)):
            print(f'========== {models[i]} ==========')
            print(f"Test Accuracy: {acc[i]} %")
            print('================================')
        self.show_arrow(0)
        self.show_directions_table()
        self.show_models_table()

    def show_directions_table(self):
        self.direction_tree.heading('File', text='File')
        self.direction_tree.heading('Direction', text='Direction')
        self.direction_tree.column('File', width=150)
        self.direction_tree.column('Direction', width=150)
        for i, element in enumerate(self.mapped_predictions):
            self.direction_tree.insert('', 'end', values=(f'file number: {i + 1}', element,))
        self.direction_tree.pack(side="bottom", fill='both', expand=tk.YES)

    def show_models_table(self):
        self.models_tree.heading('File', text='#')
        self.models_tree.column('File', width=50)
        for i in range(len(self.models)):
            self.models_tree.heading(i + 1, text=self.models[i])
            self.models_tree.column(i + 1, width=70)
        mapping = {
            'yukari': 'up',
            'yukarı': 'up',
            'asagi': 'down',
            'sag': 'right',
            'sol': 'left',
            'kirp': 'blink'
        }
        mapped_arrays = [[mapping[item.lower()] for item in arr] for arr in self.predictions]

        for i in range(5):
            self.models_tree.insert('', 'end',
                                    values=(f'file number: {i + 1}',
                                            mapped_arrays[0][i],
                                            mapped_arrays[1][i],
                                            mapped_arrays[2][i],
                                            mapped_arrays[3][i],
                                            mapped_arrays[4][i],
                                            mapped_arrays[5][i],
                                            mapped_arrays[6][i],
                                            mapped_arrays[7][i],
                                            ))
        self.models_tree.insert('', 'end', values=('Accuracy',
                                                   f'{self.acc[0]} %',
                                                   f'{self.acc[1]} %',
                                                   f'{self.acc[2]} %',
                                                   f'{self.acc[3]} %',
                                                   f'{self.acc[4]} %',
                                                   f'{self.acc[5]} %',
                                                   f'{self.acc[6]} %',
                                                   f'{self.acc[7]} %',
                                                   ))
        self.models_tree.pack(side="bottom", fill='both', expand=tk.YES)

    def __init__(self):
        self.acc = None
        self.root = tk.Tk()
        self.root.title('EOG')
        self.root.geometry('1000x600')
        self.image_path = None
        self.blink = False
        self.mapped_predictions = None
        self.predictions = None
        self.models = None
        self.button_frame = tk.Frame(self.root)
        self.button_frame.columnconfigure(0, weight=2)
        self.task1_btn = tk.Button(self.button_frame, text='Test', command=self.load_folder)
        self.task1_btn.grid(row=0, sticky=tk.W + tk.E, padx=10)
        self.label = tk.Label(self.root, text="")
        self.direction_frame = ttk.Frame(self.root)
        self.direction_tree = ttk.Treeview(self.direction_frame, columns=('File', 'Direction'), show='headings')
        self.direction_frame.pack(side="bottom", fill='both', expand=True)
        self.models_frame = ttk.Frame(self.root)
        self.models_tree = ttk.Treeview(self.models_frame,
                                        columns=['File', 'Logistic Regression', 'Decision Tree', 'Random Forest',
                                                 'SVM Linear Kernel',
                                                 'SVM RPF Kernel', 'Gaussian Naive Bayes', 'AdaBoost',
                                                 'Gradient Boost'], show='headings')
        self.models_frame.pack(side="bottom", fill='both', expand=True)
        self.label.pack(side="bottom", fill='both', expand=tk.YES)
        self.button_frame.pack(fill='x', pady=10)
        self.root.mainloop()

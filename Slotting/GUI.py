import tkinter as tk


# Must be able to add new item dimensions through GUI
def gui():

    top = tk.Tk()
    top.title("Automated Slotting Tool")
    top.columnconfigure(0, weight = 1)
    top.rowconfigure(0, weight = 1)

    def gen_hashkey_gui():
        win = tk.Toplevel(top)
        lab = tk.Label(master = win,
                       text="Generate Hashkey")\
                           .pack(pady = 15)
        countryvar = tk.StringVar()
        country = tk.ttk.Combobox(win, 
                              textvariable=countryvar)\
                                  .pack(pady = 15)
        country['values'] = ('USA', 'Canada', 'Australia')

        win.mainloop()

    fr_main = tk.Frame(top, width = 500, height = 400)
    fr_main.grid(row = 0, 
                 column = 0, 
                 sticky = (tk.N, tk.W, tk.E, tk.S))

    fr_buttons = tk.Frame(master = fr_main,
                          relief = tk.RIDGE,
                          borderwidth = 1)
    fr_buttons.columnconfigure(0, minsize = 200)
    fr_buttons.rowconfigure([0, 1, 2, 3], minsize = 50)
    fr_buttons.place(x = 10, y = 50)

    fr_text = tk.Frame(fr_main,
                       relief = tk.RIDGE,
                       borderwidth = 3,
                       bg = "white")
    fr_text.place(x = 300, y = 50)

    lab = tk.Label(master = fr_text,
                   text="Automated Slotting Tool")\
                       .pack()

    but_GenHash = tk.Button(master = fr_buttons,
                            text = "Generate Hashkey",
                            command = gen_hashkey_gui)
    but_Slott = tk.Button(master = fr_buttons,
                          text = "Calculate Slotting")
    but_AddItem = tk.Button(master = fr_buttons,
                            text = "Add New Item")
    but_AddCust = tk.Button(master = fr_buttons,
                            text = "Add New Customer")
    but_GenHash.grid(row = 0, column = 0)
    but_Slott.grid(row = 1, column = 0)
    but_AddItem.grid(row = 2, column = 0)
    but_AddCust.grid(row = 3, column = 0)

    top.mainloop()

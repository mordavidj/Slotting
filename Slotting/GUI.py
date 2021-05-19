import tkinter as tk

def gui():

    top = tk.Tk()\
        .title("Automated Slotting Tool")\
        .columnconfigure(0, weight = 1)\
        .rowconfigure(0, weight = 1)

    def gen_hashkey_gui():
        win = tk.Toplevel(top)
        lab = tk.Label(master = win,
                       text="Generate Hashkey")\
                           .pack(pady = 15)
        win.mainloop()

    fr_main = tk.Frame(top, width = 500, height = 400)
    fr_main.grid(row = 0, 
                 column = 0, 
                 sticky = (tk.N, tk.W, tk.E, tk.S))

    fr_buttons = tk.Frame(master = fr_main,
                          relief = tk.RIDGE,
                          borderwidth = 1)\
                              .columnconfigure(0, minsize = 200)\
                              .rowconfigure([0, 1], minsize = 50)\
                              .place(x = 10, y = 50)

    fr_text = tk.Frame(fr_main,
                       relief = tk.RIDGE,
                       borderwidth = 3,
                       bg = "white")\
                           .place(x = 300, y = 50)

    lab = tk.Label(master = fr_text,
                   text="Automated Slotting Tool")\
                       .pack()

    but_GenHash = tk.Button(master = fr_buttons,
                            text = "Generate Hashkey",
                            command = gen_hashkey_gui)
    but_Slott = tk.Button(master = fr_buttons,
                          text = "Calculate Slotting")
    but_GenHash.grid(row = 0, column = 0)
    but_Slott.grid(row = 1, column = 0)

    top.mainloop()

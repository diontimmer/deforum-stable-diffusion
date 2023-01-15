guiwindow = None
import PySimpleGUI as sg
from PIL import Image, ImageTk


def show_login_popup():
    # Create the layout for the login and token input fields
    layout = [[sg.Text('HuggingFace Username:'), sg.InputText()],
              [sg.Text('HuggingFace Token:'), sg.InputText(password_char='*')],
              [sg.Button('Login')]]

    # Create the window and show it
    window = sg.Window('Login', layout=layout, icon='gui/favicon.ico')
    button, values = window.Read()

    # Check which button was clicked and either print the values or a message
    if button == 'Hugging Face Login':
        return ({values[0]}, {values[1]})

    # Close the window
    window.Close()

# This function shows the image in the gui.
def gui_display_img(filepath=None, size=(512, 512), pil_img=None):
    global guiwindow
    data = None
    if pil_img is not None:
        data = ImageTk.PhotoImage(image=pil_img)
    if guiwindow:
        guiwindow['-IMAGE-'].update((filepath), size=size, data=data)
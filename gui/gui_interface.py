import PySimpleGUI as sg
from PIL import Image, ImageTk
import configparser
import io

guiwindow = None
root = None
show_loading = False
reroute_stderr = io.StringIO()


def clean_err_io():
    global reroute_stderr
    reroute_stderr = io.StringIO()

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


def set_ready(ready):
    global show_loading
    show_loading = not ready
    guiwindow['-LOADINGGIF-'].update(visible=not ready)
    disabled = not ready
    guiwindow['-RENDER-'].update(disabled=disabled)
    guiwindow['-MODEL-'].update(disabled=disabled)
    guiwindow['-MODEL_CONFIG-'].update(disabled=disabled)
    if disabled:
        guiwindow['-MENUBAR-'].update(menu_definition=[['!File', ['Open::-OPEN-', 'Save::-SAVE-']]])
    else:
        print('READY!')
        guiwindow['-MENUBAR-'].update(menu_definition=[['File', ['Open::-OPEN-', 'Save::-SAVE-']]])


# gets value from config
def get_config_value(key):
    config = configparser.ConfigParser()
    config.read('gui/gui_config.ini')
    return config.get('settings', f'{key}')
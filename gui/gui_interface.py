import PySimpleGUI as sg
from PIL import Image, ImageTk
import configparser
import io

guiwindow = None
log_ml = None
root = None
show_loading = False
reroute_stderr = io.StringIO()


def clean_err_io():
    global reroute_stderr
    reroute_stderr.flush()
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
def gui_display_img(filepath=None, pil_img=None):
    global guiwindow
    if pil_img is not None:
        data = None
        width, height = pil_img.size
        size = (width, height)
        data = ImageTk.PhotoImage(image=pil_img)
        guiwindow['-IMAGE-'].update(size=size, data=data)
    elif filepath is not None:
        # get size of image at filepath
        width, height = Image.open(filepath).size
        size = (width, height)
        guiwindow['-IMAGE-'].update((filepath), size=size)


def set_ready(ready, override_loading=None):
    global show_loading
    if override_loading is not None:
        show_loading = override_loading
        guiwindow['-LOADINGGIF-'].update(visible=override_loading)
    else:
        show_loading = not ready
        guiwindow['-LOADINGGIF-'].update(visible=not ready)
    guiwindow['-RENDER-'].update(disabled=not ready)
    if ready:
        gui_print('READY!', text_color='lightgreen')


# gets value from config
def get_config_value(key):
    config = configparser.ConfigParser()
    config.read('gui/gui_config.ini')
    return config.get('settings', f'{key}')

def gui_print(text, text_color=None, background_color=None):
    log_ml.print(text, text_color=text_color, background_color=background_color)
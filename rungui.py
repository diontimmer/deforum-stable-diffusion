import PySimpleGUI as sg

# show splash
splash = sg.Window('Window Title', [[sg.Text('d̷̨̗͎̲̟̤̀͆̿͒͆̈́̕e̵̦̓̍̉́̆͂f̵̨͖͙͉͇͊͑͠o̶̹̤͉̼̹͍͇͋̈́r̴̖̾̂͌̆ū̶̳̟͈͕͌̎͑̒͐̏͜m̶̻̭͎͇͔͎̜͐͒̈̓̽', font=("Calibri", 100))]], transparent_color=sg.theme_background_color(), no_titlebar=True, keep_on_top=True)
splash.read(timeout=0)

import random  # noqa: E402
import gui.gui_interface as gui  # noqa: E402
from gui.gui_layout import *  # noqa: E402
from gui.gui_const import *  # noqa: E402
from gui.gui_helpers import *  # noqa: E402
from gui.gui_settings_helpers import *  # noqa: E402
from gui.gui_settings_overrides import DeforumArgs, DeforumAnimArgs  # noqa: E402


# inits
open_file_name = ''
renderprocess = None

# show window  
window = sg.Window('d̷̨̗͎̲̟̤̀͆̿͒͆̈́̕e̵̦̓̍̉́̆͂f̵̨͖͙͉͇͊͑͠o̶̹̤͉̼̹͍͇͋̈́r̴̖̾̂͌̆ū̶̳̟͈͕͌̎͑̒͐̏͜m̶̻̭͎͇͔͎̜͐͒̈̓̽', gui_layout, resizable=True, finalize=True, size=(1500, 1200), font=("Calibri", 11), enable_close_attempted_event=True, icon='gui/favicon.ico')
splash.close()
gui.guiwindow = window

# load older settings
load_settings('saved_settings.pickle')

# disable render and check gpu
window['-RENDER-'].update(disabled=True)
print_gpu()
print('Sweet! Please pick your model and load.')


while True:
    event, values = window.read(timeout=50)

    # refresh loading gif
    if gui.show_loading:
        loading_gif_img.update_animation(LOADING_GIF_B64, time_between_frames=50)

    # handle render button
    if event == '-RENDER-':
        if values['-ANIMATION_MODE-'] == 'None':
            args = DeforumArgs(get_args_from_gui(values, "general"))
            save_settings(values, 'saved_settings.pickle')
            renderprocess = KThread(target=do_render, args=(values, args,), daemon=True)
            renderprocess.start()
        else:
            args = DeforumArgs(get_args_from_gui(values, "general"))
            anim_args = DeforumAnimArgs(get_args_from_gui(values, "animation"))
            save_settings(values, 'saved_settings.pickle')
            renderprocess = KThread(target=do_video_render, args=(values, args, anim_args,), daemon=True)
            renderprocess.start()

    # handle load button
    if event == '-RELOAD-':
        KThread(target=load_root_model, args=(values['-MODEL-'], values['-MODEL_CONFIG-'], values['-OUTPUT_PATH-']), daemon=True).start()
        save_settings(values, 'saved_settings.pickle')

    # handle cancel button
    if event == '-CANCEL-':
        if renderprocess is not None:
            renderprocess.kill()
            gui.set_ready(True)
            print('Process Canceled!')

    # pick random seed
    if event == '-RANDOM_SEED-':
        window['-SEED-'].update(value=random.randint(0, 2**32 - 1))

    # handle open menu button
    if event == 'Open::-OPEN-':
        if sg.running_mac():
            open_file_name = sg.tk.filedialog.askopenfilename(initialdir=values['-OUTPUT_PATH-'], defaultextension='.deforum')  # show the 'get files' dialog box
        else:
            open_file_name = sg.tk.filedialog.askopenfilename(filetypes=[("Deforum File", "*.deforum")], initialdir=values['-OUTPUT_PATH-'], defaultextension='.deforum')  # show the 'get files' dialog box
        load_settings(open_file_name)

    # handle save menu button
    if event == 'Save::-SAVE-':
        if sg.running_mac():
            save_file_name = sg.tk.filedialog.asksaveasfilename(defaultextension='.deforum', initialdir=values['-OUTPUT_PATH-'])
        else:
            save_file_name = sg.tk.filedialog.asksaveasfilename(filetypes=[("Deforum File", "*.deforum")], defaultextension='.deforum', initialdir=values['-OUTPUT_PATH-'])
        save_settings(values, save_file_name)

    # handle exit
    if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, 'Exit'):
        save_settings(values, 'saved_settings.pickle')
        if open_file_name:
            save_settings(values, open_file_name)
        break

window.close()

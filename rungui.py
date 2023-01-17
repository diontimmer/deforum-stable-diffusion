import PySimpleGUI as sg

# show splash
splash = sg.Window('Window Title', [[sg.Text('d̷̨̗͎̲̟̤̀͆̿͒͆̈́̕e̵̦̓̍̉́̆͂f̵̨͖͙͉͇͊͑͠o̶̹̤͉̼̹͍͇͋̈́r̴̖̾̂͌̆ū̶̳̟͈͕͌̎͑̒͐̏͜m̶̻̭͎͇͔͎̜͐͒̈̓̽', font=("Calibri", 100))]], transparent_color=sg.theme_background_color(), no_titlebar=True, keep_on_top=True)
splash.read(timeout=0)

import random  # noqa: E402
import webbrowser # noqa: E402
import gui.gui_interface as gui  # noqa: E402
from gui.gui_layout import *  # noqa: E402
from gui.gui_const import *  # noqa: E402
from gui.gui_helpers import *  # noqa: E402
from gui.gui_settings_helpers import *  # noqa: E402
from gui.gui_settings_overrides import DeforumArgs, DeforumAnimArgs  # noqa: E402

# inits
open_file_name = ''
renderprocess = None
loadrootprocess = None

# show window  
window = sg.Window('d̷̨̗͎̲̟̤̀͆̿͒͆̈́̕e̵̦̓̍̉́̆͂f̵̨͖͙͉͇͊͑͠o̶̹̤͉̼̹͍͇͋̈́r̴̖̾̂͌̆ū̶̳̟͈͕͌̎͑̒͐̏͜m̶̻̭͎͇͔͎̜͐͒̈̓̽', gui_layout, resizable=True, finalize=True, size=(1500, 1200), font=(gui.get_config_value('font'), gui.get_config_value('font_size')), enable_close_attempted_event=True, icon='gui/favicon.ico')
splash.close()
gui.guiwindow = window
gui.log_ml = log_ml

# load older settings
load_settings('saved_settings.pickle')

# disable render and check gpu
window['-RENDER-'].update(disabled=True)
print_gpu()
gui.gui_print('Sweet! Please pick your model and load.')
prog_bar.update_bar(100)


while True:
    event, values = window.read(timeout=10)

    # prog bar
    output = gui.reroute_stderr.getvalue()
    if output:
        percentage = extract_percentage(output)
        if percentage is not None:
            # Update progress bar
            prog_bar.update_bar(percentage)

        gui.reroute_stderr.flush()

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
        loadrootprocess = KThread(target=load_root_model, args=(values['-MODEL-'], values['-MODEL_CONFIG-'], values['-OUTPUT_PATH-']), daemon=True)
        loadrootprocess.start()
        save_settings(values, 'saved_settings.pickle')

    # handle cancel button
    if event == '-CANCEL-':
        if renderprocess is not None:
            renderprocess.kill()
            gui.set_ready(True)
            gui.gui_print('Process Canceled!')
        elif loadrootprocess is not None:
            loadrootprocess.kill()
            gui.set_ready(False, override_loading=False)
            gui.gui_print('Process Canceled!')


    # pick random seed
    if event == '-RANDOM_SEED-':
        window['-SEED-'].update(value=random.randint(0, 2**32 - 1))

    # handle open menu button
    if event == 'Open::-OPEN-':
        if sg.running_mac():
            open_file_name = sg.tk.filedialog.askopenfilename(initialdir=values['-OUTPUT_PATH-'], defaultextension='.deforum')  # show the 'get files' dialog box
        else:
            open_file_name = sg.tk.filedialog.askopenfilename(filetypes=[("Deforum File", "*.deforum")], initialdir=values['-OUTPUT_PATH-'], defaultextension='.deforum')  # show the 'get files' dialog box
        load_settings(open_file_name, from_user_file=True)

    # handle save menu button
    if event == 'Save::-SAVE-':
        if sg.running_mac():
            save_file_name = sg.tk.filedialog.asksaveasfilename(defaultextension='.deforum', initialdir=values['-OUTPUT_PATH-'])
        else:
            save_file_name = sg.tk.filedialog.asksaveasfilename(filetypes=[("Deforum File", "*.deforum")], defaultextension='.deforum', initialdir=values['-OUTPUT_PATH-'])
        save_settings(values, save_file_name, for_user_file=True)

    # handle open batch
    if event == 'Open Batch Folder::-OPEN_BATCH-':
        open_batch_folder(values)

    # handle clean button
    if event == 'Clean Batch Folder::-CLEAN-':
        clean_batch_folder(values)

    if event == 'Audio Keyframe Generator::-AUDIO_KEYFRAME_TOOL-':
        webbrowser.open('https://www.chigozie.co.uk/audio-keyframe-generator/')

    if event == 'Graph Keyframe Generator::-GRAPH_KEYFRAME_TOOL-':
        webbrowser.open('https://www.chigozie.co.uk/keyframe-string-generator/')

    if event == '-MODELS_PATH-':
        lastmodel = values['-MODEL-']
        window['-MODEL-'].update(value=lastmodel, values=getmodels())

    # handle exit
    if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, 'Exit'):
        save_settings(values, 'saved_settings.pickle')
        if open_file_name:
            save_settings(values, open_file_name)
        break

window.close()

# Deforum Stable Diffusion GUI

Install normally using instructions provided in main repo.
launch rungui.py to open the GUI.
To have the GUI output to the regular console window for debugging, swap 'app_log' to False in the rungui.py file.

Regular batch image prompts should be separated by a newline in the prompts box:

"A blue fox"
"A yellow fox"

Animation keyframes should note the keyframe before the prompt, these should be seperated by newlines aswell:

"0: A blue fox"
"10: A red fox"
"20: A yellow fox"

The gui allows for saving and loading .deforum files which save and load the settings.
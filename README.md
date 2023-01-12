# Deforum Stable Diffusion GUI

This GUI allows you to use Deforum locally without the need for google colab. This fork comes with a couple extra settings such as being able to remove animation frames after export and the saving and loading of settings. Settings will be saved upon close, to initialize the gui again please close the gui and remove "saved_settings.pickle". Credits go to the deforum devs and community for the magic behind the program. Ive tried my best to wrap around the original code with minimal hooks, but issues should still be brought up here as well in case some variables are incorrectly casted from the GUIs string types.


## Install
Install normally using instructions provided in main repo: https://github.com/deforum-art/deforum-stable-diffusion.
launch rungui.py to open the GUI.
To have the GUI output to the regular console window for debugging, swap 'app_log' to False in the rungui.py file.

## Prompt Syntax
Regular batch image prompts (anim=None) should be separated by a newline in the prompts box ie:

A blue fox<br>
A yellow fox<br>

Animation keyframes (anim=2D, 3D, Interp etc..) should note the keyframe before the prompt, these should be seperated by newlines aswell ie:

0: A blue fox<br>
10: A red fox<br>
20: A yellow fox<br>

## Issues
Text progress bar is not parsing correctly to the in-app console log. if this bothers you feel free to toggle 'app_log' to False in the rungui.py script to enable output to the regular console.

![Alt text](https://www.dropbox.com/s/4tms1vloi4kg72a/deforumgui1.png?raw=1 "Optional title")

@echo off
call "D:\mambaforge\Scripts\activate.bat" deforum
cd G:\DevWkspc\External\deforum-wrapper\deforum
python rungui.py
pause
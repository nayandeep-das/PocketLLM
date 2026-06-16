@echo off
title PocketLLM

call "C:\ProgramData\anaconda3\Scripts\activate.bat" pocketllm

echo Loading PocketLLM...
echo Please wait while models are initialized.
echo.

python app.py

pause
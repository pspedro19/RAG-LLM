@echo off
cd /d "%~dp0"
wsl.exe --cd "%CD:\=/%"
CALL C:\Users\MSI-NB\Anaconda3\Scripts\activate.bat C:\Users\MSI-NB\Anaconda3
CALL conda activate tensor_try

@echo off
for %%i in (*.json) do labelme_json_to_dataset "%%i"
pause


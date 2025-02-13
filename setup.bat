@echo off

:: Check if the virtual environment folder exists
if not exist ".venv" (
    :: Create the virtual environment named .venv
    py -3.10 -m venv .venv
    echo Virtual environment .venv created successfully!
) else (
    echo Virtual environment .venv already exists.
)

:: Create the .gitignore file inside .venv if it doesn't exist
if not exist ".venv\.gitignore" (
    echo * > .venv\.gitignore
    echo .gitignore file created inside .venv folder.
) else (
    echo .gitignore file already exists inside .venv folder.
)

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Check if the requirements.txt file exists before trying to install libraries
if exist "requirements.txt" (
    :: Install the dependencies from requirements.txt
    pip install -r requirements.txt
    echo Libraries from requirements.txt installed successfully!
) else (
    echo requirements.txt not found. Skipping library installation.
)
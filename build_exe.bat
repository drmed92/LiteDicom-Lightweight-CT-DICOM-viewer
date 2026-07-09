@echo off
REM ===========================================================================
REM  build_exe.bat  -  builds LiteDicom.exe on Windows 10/11
REM  Put this next to LiteDicom.py and double-click it (or run in a terminal).
REM  Output:  dist\LiteDicom.exe   (a single, self-contained executable)
REM ===========================================================================
setlocal

echo ============================================================
echo   Building LiteDicom.exe
echo ============================================================

REM 1) Check that Python is available
where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo   Python was not found on your PATH.
  echo   Install it from https://www.python.org/downloads/ ^(tick "Add to PATH"^) and retry.
  echo.
  pause
  exit /b 1
)

REM 2) Install the build tool + runtime dependencies
echo.
echo   Installing build dependencies...
python -m pip install --upgrade pip
python -m pip install pyinstaller pydicom numpy pillow
if errorlevel 1 (
  echo.
  echo   Dependency installation failed.
  pause
  exit /b 1
)

REM 3) Build a single-file, windowed (no console) executable.
REM    --collect-all pydicom bundles the DICOM dictionary + encoders.
echo.
echo   Building the executable ^(this can take a minute^)...
python -m PyInstaller --noconfirm --clean --onefile --windowed --name LiteDicom --collect-all pydicom LiteDicom.py
if errorlevel 1 (
  echo.
  echo   Build failed.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo   Done!  Your executable is here:
echo       dist\LiteDicom.exe
echo ============================================================
echo.
echo   To support compressed DICOM as well, re-run after:
echo       pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
echo   and add:  --collect-all pylibjpeg  to the PyInstaller line.
echo.
pause

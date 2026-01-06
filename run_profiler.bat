@echo off
setlocal
set PATH=%PATH%;%cd%\build\bin

:: Lock-Prevention: Kill any lingering processes
taskkill /F /IM npu_profiler.exe /FI "STATUS eq RUNNING" >nul 2>&1
taskkill /F /IM demo_gemm.exe /FI "STATUS eq RUNNING" >nul 2>&1
taskkill /F /IM verify_accuracy.exe /FI "STATUS eq RUNNING" >nul 2>&1

:menu
cls
echo ================================================================
echo           SoftAccelNPU: 4D-V Management Console
echo ================================================================
echo  1. Run Interactive Profiler (C++)
echo  2. Run LLM 5B Simulator (Python)
echo  3. Run CPU vs NPU (Head-to-Head)
echo  4. Rebuild Project (Clean Build)
echo  5. Run Accuracy Suite
echo  6. Exit
echo.
set /p choice="Selection: "

if "%choice%"=="1" goto run
if "%choice%"=="2" goto py_llm
if "%choice%"=="3" goto head_to_head
if "%choice%"=="4" goto build
if "%choice%"=="5" goto verify
if "%choice%"=="6" goto end

:head_to_head
if exist "build\bin\head_to_head.exe" (
    "build\bin\head_to_head.exe"
) else (
    echo [ERROR] head_to_head.exe not found. Please build first.
)
pause
goto menu

:py_llm
echo [Python] Launching 5B LLM Simulator...
python scripts\llm_5b_tester.py
pause
goto menu

:run
if exist "build\bin\npu_profiler.exe" (
    "build\bin\npu_profiler.exe"
) else (
    echo [ERROR] npu_profiler.exe not found. Please build first.
    pause
)
goto menu

:build
echo [Build] Starting Clean Build...
taskkill /F /IM libsoftaccelnpu_v5.dll >nul 2>&1
if not exist build mkdir build
cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make clean
mingw32-make -j8
cd ..
echo [Build] Complete.
pause
goto menu

:verify
if exist "build\bin\verify_accuracy.exe" (
    "build\bin\verify_accuracy.exe"
) else (
    echo [ERROR] verify_accuracy.exe not found.
    pause
)
goto menu

:end
endlocal

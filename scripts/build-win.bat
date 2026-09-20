@echo off
REM KVMem Windows dev build: MSVC + CUDA 13.3 (nvcc 13.3.73 >= 13.2.86 correctness baseline)
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

set PATH=N:\cmake-3.31.6-windows-x86_64\bin;N:\llama.cpp-master;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3

set ROOT=N:\llama.cpp-master\kvmem-llama.cpp
set BUILD_DIR=%ROOT%\build-win

cd /d %ROOT%

echo [1/2] Configure...
cmake -S %ROOT% -B %BUILD_DIR% -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" ^
    -DCMAKE_CUDA_ARCHITECTURES=75;86 ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DKVMEM_BUILD_LLAMA=ON ^
    -DLLAMA_KVMEM=ON ^
    -DLLAMA_KVMEM_ROOT=%ROOT%
if errorlevel 1 exit /b 1

echo [2/2] Build llama-kvmem-server...
cmake --build %BUILD_DIR% --target llama-kvmem-server -j48
if errorlevel 1 exit /b 1
echo DONE: %BUILD_DIR%\bin\llama-kvmem-server.exe

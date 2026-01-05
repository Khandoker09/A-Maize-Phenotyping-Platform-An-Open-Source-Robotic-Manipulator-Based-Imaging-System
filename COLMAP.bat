@echo off
setlocal EnableExtensions

rem ==========================================================
rem  COLMAP FAST + ROBUST (NO if (...) ( ... ) blocks)
rem  - Fixes "... was unexpected at this time."
rem  - WORK folder in user profile (writable)
rem  - Auto-opens live log window
rem  - GPU-first, CPU fallback
rem  - Sequential matching (fast)
rem  - Dense downscaled
rem ==========================================================

rem ---- EDIT if needed ----
set "COLMAP_EXE=C:\Users\yolzo57\Desktop\paper2\bin\colmap.exe"
rem ------------------------

if not exist "%COLMAP_EXE%" (
  echo [ERROR] colmap.exe not found: "%COLMAP_EXE%"
  pause
  exit /b 9
)

set "IMAGES=%CD%\rgbimages"

set "WORK=%USERPROFILE%\colmap_work\COLMAP"
set "DB=%WORK%\colmap.db"
set "SPARSE=%WORK%\sparse"
set "SPARSE_A=%SPARSE%\A"
set "DENSE=%WORK%\dense"
set "LOG=%WORK%\run.log"

set "THREADS=%NUMBER_OF_PROCESSORS%"
if "%THREADS%"=="" set "THREADS=8"
if %THREADS% GTR 16 set "THREADS=16"

set "MATCH_THREADS=8"

rem 1 = do dense, 0 = sparse only
set "DO_DENSE=1"

rem 1 = open live log window
set "FOLLOW_LOG=1"

echo.
echo === DIAGNOSTICS ===
echo COLMAP:   "%COLMAP_EXE%"
echo IMAGES:   "%IMAGES%"
echo WORK:     "%WORK%"
echo THREADS:  %THREADS%
echo MATCH_T:  %MATCH_THREADS%
echo LOG:      "%LOG%"
echo.

if not exist "%IMAGES%" (
  echo [ERROR] Folder not found: "%IMAGES%"
  pause
  exit /b 1
)

rem ---- Count images (simple) ----
set "COUNT=0"
for %%E in (jpg jpeg png JPG JPEG PNG) do (
  for /f "delims=" %%F in ('dir /b "%IMAGES%\*.%%E" 2^>nul') do set /a COUNT+=1
)
echo Found %COUNT% image(s).
if %COUNT% LSS 2 (
  echo [ERROR] Need at least 2 images.
  pause
  exit /b 2
)

rem ---- Create workspace ----
if not exist "%WORK%" mkdir "%WORK%"
if errorlevel 1 (
  echo [ERROR] Cannot create WORK: "%WORK%"
  pause
  exit /b 10
)

if not exist "%SPARSE%" mkdir "%SPARSE%"
if not exist "%SPARSE_A%" mkdir "%SPARSE_A%"
if not exist "%DENSE%" mkdir "%DENSE%"

if exist "%DB%" del /q "%DB%"

rem ---- Create log and verify writable ----
> "%LOG%" echo ===== COLMAP RUN %DATE% %TIME% =====
if errorlevel 1 (
  echo [ERROR] Cannot write log: "%LOG%"
  pause
  exit /b 11
)

>> "%LOG%" echo IMAGES="%IMAGES%"
>> "%LOG%" echo WORK="%WORK%"
>> "%LOG%" echo THREADS=%THREADS%  MATCH_THREADS=%MATCH_THREADS%
>> "%LOG%" echo.

rem ---- Live log tail window ----
if "%FOLLOW_LOG%"=="1" start "COLMAP LOG LIVE" powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -LiteralPath '%LOG%' -Tail 80 -Wait"

echo.
echo ================== 1) FEATURE EXTRACTION ==================
echo [TRY GPU extraction...]
"%COLMAP_EXE%" feature_extractor ^
  --database_path "%DB%" ^
  --image_path "%IMAGES%" ^
  --SiftExtraction.use_gpu 1 ^
  --SiftExtraction.num_threads %THREADS% ^
  --SiftExtraction.max_image_size 3200 ^
  --SiftExtraction.max_num_features 12000 ^
  --ImageReader.single_camera 1 ^
  --ImageReader.camera_model SIMPLE_RADIAL ^
  --ImageReader.default_focal_length_factor 1.2 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FE_CPU
echo [OK] Feature extraction complete.
goto MATCHING

:FE_CPU
echo [WARN] GPU extraction failed -> CPU extraction...
"%COLMAP_EXE%" feature_extractor ^
  --database_path "%DB%" ^
  --image_path "%IMAGES%" ^
  --SiftExtraction.use_gpu 0 ^
  --SiftExtraction.num_threads %THREADS% ^
  --SiftExtraction.max_image_size 3200 ^
  --SiftExtraction.max_num_features 12000 ^
  --ImageReader.single_camera 1 ^
  --ImageReader.camera_model SIMPLE_RADIAL ^
  --ImageReader.default_focal_length_factor 1.2 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL
echo [OK] Feature extraction complete.
goto MATCHING

:MATCHING
echo.
echo ================== 2) MATCHING (SEQUENTIAL) ==================
echo [TRY GPU sequential matching...]
"%COLMAP_EXE%" sequential_matcher ^
  --database_path "%DB%" ^
  --SiftMatching.use_gpu 1 ^
  --SiftMatching.num_threads %MATCH_THREADS% ^
  --SiftMatching.guided_matching 1 ^
  --SequentialMatching.overlap 4 ^
  --SequentialMatching.loop_detection 0 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto MATCH_CPU
echo [OK] Matching complete.
goto MAPPER

:MATCH_CPU
echo [WARN] GPU sequential failed -> CPU sequential...
"%COLMAP_EXE%" sequential_matcher ^
  --database_path "%DB%" ^
  --SiftMatching.use_gpu 0 ^
  --SiftMatching.num_threads %MATCH_THREADS% ^
  --SiftMatching.guided_matching 1 ^
  --SequentialMatching.overlap 4 ^
  --SequentialMatching.loop_detection 0 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL
echo [OK] Matching complete.
goto MAPPER

:MAPPER
echo.
echo ================== 3) SPARSE MAPPING ==================
"%COLMAP_EXE%" mapper ^
  --database_path "%DB%" ^
  --image_path "%IMAGES%" ^
  --output_path "%SPARSE_A%" ^
  --Mapper.min_num_matches 8 ^
  --Mapper.abs_pose_min_num_inliers 10 ^
  --Mapper.abs_pose_max_error 12 ^
  --Mapper.ba_global_max_num_iterations 30 ^
  --Mapper.tri_ignore_two_view_tracks 0 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL

rem Find the model folder (don’t assume it's "0")
set "MODEL="
for /d %%D in ("%SPARSE_A%\*") do (
  if exist "%%D\cameras.bin" set "MODEL=%%D"
)

if "%MODEL%"=="" goto ADVICE

echo.
echo ================== EXPORT SPARSE TO PLY ==================
"%COLMAP_EXE%" model_converter ^
  --input_path "%MODEL%" ^
  --output_path "%SPARSE%\sparse_points.ply" ^
  --output_type PLY ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL

if "%DO_DENSE%"=="0" goto DONE_SPARSE

echo.
echo ================== 4) DENSE (DOWNSCALED) ==================
"%COLMAP_EXE%" image_undistorter ^
  --image_path "%IMAGES%" ^
  --input_path "%MODEL%" ^
  --output_path "%DENSE%" ^
  --output_type COLMAP ^
  --max_image_size 2000 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL

echo [TRY GPU PatchMatch...]
"%COLMAP_EXE%" patch_match_stereo ^
  --workspace_path "%DENSE%" ^
  --workspace_format COLMAP ^
  --PatchMatchStereo.geom_consistency 1 ^
  --PatchMatchStereo.max_image_size 2000 ^
  --PatchMatchStereo.gpu_index 0 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto PM_CPU

"%COLMAP_EXE%" stereo_fusion ^
  --workspace_path "%DENSE%" ^
  --workspace_format COLMAP ^
  --input_type geometric ^
  --output_path "%DENSE%\fused.ply" ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL
goto DONE_ALL

:PM_CPU
echo [WARN] GPU PatchMatch failed -> CPU PatchMatch...
"%COLMAP_EXE%" patch_match_stereo ^
  --workspace_path "%DENSE%" ^
  --workspace_format COLMAP ^
  --PatchMatchStereo.geom_consistency 1 ^
  --PatchMatchStereo.max_image_size 2000 ^
  --PatchMatchStereo.gpu_index -1 ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL

"%COLMAP_EXE%" stereo_fusion ^
  --workspace_path "%DENSE%" ^
  --workspace_format COLMAP ^
  --input_type geometric ^
  --output_path "%DENSE%\fused.ply" ^
  --log_to_stderr 1 ^
  1>> "%LOG%" 2>&1

if errorlevel 1 goto FATAL
goto DONE_ALL

:DONE_SPARSE
echo.
echo -------- DONE (SPARSE ONLY) --------
echo Model:      "%MODEL%"
echo Sparse PLY: "%SPARSE%\sparse_points.ply"
echo Log:        "%LOG%"
exit /b 0

:DONE_ALL
echo.
echo -------- DONE --------
echo Model:      "%MODEL%"
echo Sparse PLY: "%SPARSE%\sparse_points.ply"
echo Dense PLY:  "%DENSE%\fused.ply"
echo Log:        "%LOG%"
exit /b 0

:FATAL
echo.
echo [ERROR] A COLMAP step failed.
echo Log: "%LOG%"
pause
exit /b 3

:ADVICE
echo.
echo [ERROR] Mapper produced no model folder with cameras.bin.
echo Log: "%LOG%"
pause
exit /b 4

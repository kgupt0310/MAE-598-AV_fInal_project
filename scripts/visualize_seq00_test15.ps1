$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"

# IMPORTANT:
# This must be an extracted folder, not the Windows Explorer path inside the .zip file.
$DataRoot = "C:\Users\Admin\Downloads\data_odometry_velodyne\dataset\sequences"
$ModelPath = "experiments\dgcnn_seq00_first51_fps30.pth"

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:NUMEXPR_NUM_THREADS = "8"
$env:TORCH_NUM_THREADS = "8"

# Test frames: 15 frames after the first 51 training frames.
# These are frame indices 51 through 65 from sequence 00.
foreach ($Frame in 51..65) {
    & $PythonExe test/test_dgcnn_full_cloud_visualization.py `
        --data-root $DataRoot `
        --model-path $ModelPath `
        --sequence 00 `
        --frame-index $Frame `
        --max-chunks 30 `
        --center-selection fps_xy `
        --save-prefix "experiments\seq00_test_frame_$Frame" `
        --no-show
}

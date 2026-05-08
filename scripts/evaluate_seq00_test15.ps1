$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"
$DataRoot = "C:\Users\Admin\Downloads\data_odometry_velodyne\dataset\sequences"
$ModelPath = "experiments\dgcnn_seq00_first51_fps30.pth"

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:NUMEXPR_NUM_THREADS = "8"
$env:TORCH_NUM_THREADS = "8"

& $PythonExe test/evaluate_dgcnn_test_frames.py `
    --data-root $DataRoot `
    --model-path $ModelPath `
    --sequence 00 `
    --start-frame 51 `
    --num-frames 15 `
    --chunks-per-frame 30 `
    --center-selection fps_xy `
    --save-csv experiments\seq00_test15_metrics.csv

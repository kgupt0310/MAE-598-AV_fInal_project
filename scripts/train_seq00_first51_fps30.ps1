$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"
$DataRoot = "C:\Users\Admin\Downloads\data_odometry_velodyne\dataset\sequences"

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:NUMEXPR_NUM_THREADS = "8"
$env:TORCH_NUM_THREADS = "8"

# Train only on KITTI sequence 00, first 51 frames: 000000.bin through 000050.bin.
# Uses 30 BEV-spread kNN chunks per frame.
# Full-frame training is avoided because kNN/Chamfer/cdist are too memory-heavy
# for ~100k-point raw KITTI frames.
& $PythonExe src/train/train_dgcnn.py `
    --data-root $DataRoot `
    --sequences "00" `
    --epochs 30 `
    --max-frames-per-seq 51 `
    --chunks-per-frame 30 `
    --center-selection fps_xy `
    --max-batches 0 `
    --num-workers 2 `
    --num-points 2048 `
    --save-name dgcnn_seq00_first51_fps30.pth `
    --history-name dgcnn_seq00_first51_fps30_history.csv

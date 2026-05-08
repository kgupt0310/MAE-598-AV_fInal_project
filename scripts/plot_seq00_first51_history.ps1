$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"

$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:NUMEXPR_NUM_THREADS = "8"
$env:TORCH_NUM_THREADS = "8"

& $PythonExe scripts\plot_training_history.py `
    --history experiments\dgcnn_seq00_first51_fps30_history.csv `
    --save experiments\dgcnn_seq00_first51_fps30_training_history.png

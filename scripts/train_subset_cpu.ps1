$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"
$DataRoot = "C:\Semester 3_Spring 2026\MAE 598_AVE\Project\Data\dataset\dataset\sequences"
$Sequences = "00,01,05,07,09"

& $PythonExe src/train/train_dgcnn.py `
    --data-root $DataRoot `
    --sequences $Sequences `
    --epochs 34 `
    --max-frames-per-seq 20 `
    --max-batches 40 `
    --num-points 2048 `
    --save-name dgcnn_motion_attention_subset.pth `
    --history-name dgcnn_motion_attention_subset_history.csv

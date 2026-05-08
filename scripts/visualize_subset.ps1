$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"
$DataRoot = "C:\Semester 3_Spring 2026\MAE 598_AVE\Project\Data\dataset\dataset\sequences"
$ModelPath = "experiments\dgcnn_motion_attention_subset.pth"

& $PythonExe test/test_dgcnn_full_cloud_visualization.py `
    --data-root $DataRoot `
    --model-path $ModelPath `
    --sequence 09 `
    --frame-index 1 `
    --max-chunks 8 `
    --save-prefix experiments\subset_seq09 `
    --no-show

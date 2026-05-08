$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"
$DataRoot = "C:\Semester 3_Spring 2026\MAE 598_AVE\Project\Data\dataset\dataset\sequences"

& $PythonExe test/test_raw_kitti_only.py `
    --data-root $DataRoot `
    --sequence 09 `
    --frame-index 0 `
    --save experiments\raw_seq09_frame0.png `
    --no-show

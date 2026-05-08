$ZipPath = "C:\Users\Admin\Downloads\data_odometry_velodyne.zip"
$ExtractRoot = "C:\Users\Admin\Downloads\data_odometry_velodyne"

New-Item -ItemType Directory -Force $ExtractRoot | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem

$zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $entries = $zip.Entries | Where-Object {
        $_.FullName -like "dataset/sequences/00/velodyne/*.bin"
    }

    Write-Host "Found $($entries.Count) sequence 00 velodyne frames in zip."
    Write-Host "Extracting to $ExtractRoot ..."

    foreach ($entry in $entries) {
        $target = Join-Path $ExtractRoot $entry.FullName
        $targetDir = Split-Path $target -Parent
        New-Item -ItemType Directory -Force $targetDir | Out-Null
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
    }

    Write-Host "Done."
    Write-Host "Data root for scripts:"
    Write-Host "$ExtractRoot\dataset\sequences"
}
finally {
    $zip.Dispose()
}

param(
    [string]$ImagePath = "C:\Users\rahul\Desktop\Projects\MNIST_BNNv2\my_digit.png",
    [string]$ConverterPath = "C:\Users\rahul\Desktop\Projects\MNIST_BNNv2\image_to_mem.py",
    [string]$OutputName = "input.mem",
    [switch]$Once
)

function Invoke-Conversion {
    param(
        [string]$ImagePath,
        [string]$ConverterPath,
        [string]$OutputName
    )

    if (-not (Test-Path $ConverterPath)) {
        throw "Converter script not found: $ConverterPath"
    }

    if (-not (Test-Path $ImagePath)) {
        Write-Warning "Image not found: $ImagePath"
        return
    }

    python $ConverterPath $ImagePath $OutputName
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Conversion failed with exit code $LASTEXITCODE"
    } else {
        Write-Host "Updated $OutputName from $ImagePath at $(Get-Date -Format 'HH:mm:ss')"
    }
}

Invoke-Conversion -ImagePath $ImagePath -ConverterPath $ConverterPath -OutputName $OutputName

if ($Once) {
    return
}

$watchDir = Split-Path -Parent $ImagePath
$watchFile = Split-Path -Leaf $ImagePath

if (-not (Test-Path $watchDir)) {
    throw "Watch directory does not exist: $watchDir"
}

$watcher = New-Object System.IO.FileSystemWatcher $watchDir, $watchFile
$watcher.NotifyFilter =
    [System.IO.NotifyFilters]::LastWrite -bor
    [System.IO.NotifyFilters]::Size -bor
    [System.IO.NotifyFilters]::FileName
$watcher.EnableRaisingEvents = $true

$lastWrite = if (Test-Path $ImagePath) { (Get-Item $ImagePath).LastWriteTimeUtc } else { [DateTime]::MinValue }

Write-Host "Watching: $ImagePath"
Write-Host "Press Ctrl+C to stop."

try {
    while ($true) {
        $null = $watcher.WaitForChanged([System.IO.WatcherChangeTypes]::All)

        if (-not (Test-Path $ImagePath)) {
            continue
        }

        $currentWrite = (Get-Item $ImagePath).LastWriteTimeUtc
        if ($currentWrite -ne $lastWrite) {
            $lastWrite = $currentWrite
            Invoke-Conversion -ImagePath $ImagePath -ConverterPath $ConverterPath -OutputName $OutputName
        }
    }
}
finally {
    $watcher.Dispose()
}

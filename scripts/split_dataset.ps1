param(
    [Parameter(Mandatory = $true)][string]$SourcePath,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$PartSizeMB = 1900
)

$ErrorActionPreference = "Stop"

$partSize = [int64]$PartSizeMB * 1MB
$bufferSize = 8MB
$buffer = New-Object byte[] $bufferSize

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Get-ChildItem -Path $OutputDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$inStream = [System.IO.File]::OpenRead($SourcePath)
try {
    $part = 1
    while ($inStream.Position -lt $inStream.Length) {
        $partPath = Join-Path $OutputDir ("conase_distributable_a.part{0:D3}.bin" -f $part)
        $outStream = [System.IO.File]::Create($partPath)
        try {
            $written = 0L
            while ($written -lt $partSize -and $inStream.Position -lt $inStream.Length) {
                $remainingPart = $partSize - $written
                $toRead = if ($remainingPart -lt $bufferSize) { [int]$remainingPart } else { $bufferSize }
                $read = $inStream.Read($buffer, 0, $toRead)
                if ($read -le 0) { break }
                $outStream.Write($buffer, 0, $read)
                $written += $read
            }
        }
        finally {
            $outStream.Dispose()
        }
        $part += 1
    }
}
finally {
    $inStream.Dispose()
}

Get-ChildItem -Path $OutputDir -File |
    Sort-Object Name |
    Select-Object Name, @{Name = "MB"; Expression = { [math]::Round($_.Length / 1MB, 2) }}

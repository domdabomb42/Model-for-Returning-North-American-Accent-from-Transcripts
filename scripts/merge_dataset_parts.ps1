param(
    [Parameter(Mandatory = $true)][string]$PartsDir,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = "Stop"

$parts = Get-ChildItem -Path $PartsDir -File |
    Where-Object { $_.Name -like "conase_distributable_a.part*.bin" } |
    Sort-Object Name

if (-not $parts) {
    throw "No dataset parts found in $PartsDir"
}

$outStream = [System.IO.File]::Create($OutputPath)
try {
    $buffer = New-Object byte[] (8MB)
    foreach ($part in $parts) {
        $inStream = [System.IO.File]::OpenRead($part.FullName)
        try {
            while (($read = $inStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $outStream.Write($buffer, 0, $read)
            }
        }
        finally {
            $inStream.Dispose()
        }
    }
}
finally {
    $outStream.Dispose()
}

Get-Item $OutputPath | Select-Object FullName, @{Name = "MB"; Expression = { [math]::Round($_.Length / 1MB, 2) }}

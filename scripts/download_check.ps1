# ============================================================================
# ECG Research Project - Dataset Verification Script
# ============================================================================
# This script verifies the existence of all required datasets and reports
# their sizes, file counts, and basic statistics.
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ECG Research Project - Dataset Verification" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = "D:\ecg-research"
$DatasetRoot = Join-Path $ProjectRoot "dataset"

# Define expected datasets
$datasets = @(
    @{Name="ptb-xl"; Description="PTB-XL Database"},
    @{Name="PTB_Diagnostic"; Description="PTB Diagnostic ECG Database"},
    @{Name="CinC2017"; Description="CinC Challenge 2017 Dataset"},
    @{Name="Chapman_Shaoxing"; Description="Chapman-Shaoxing ECG Database"}
)

$allFound = $true
$results = @()

foreach ($dataset in $datasets) {
    $dsName = $dataset.Name
    $dsDesc = $dataset.Description
    $dsPath = Join-Path $DatasetRoot $dsName

    Write-Host "Checking: $dsDesc ($dsName)..." -ForegroundColor Yellow

    if (Test-Path $dsPath) {
        # Calculate size and file count
        $files = Get-ChildItem -Path $dsPath -Recurse -File -ErrorAction SilentlyContinue
        $fileCount = ($files | Measure-Object).Count
        $totalSize = ($files | Measure-Object -Property Length -Sum).Sum

        # Convert bytes to appropriate unit
        $sizeStr = ""
        if ($totalSize -gt 1GB) {
            $sizeStr = "{0:N2} GB" -f ($totalSize / 1GB)
        } elseif ($totalSize -gt 1MB) {
            $sizeStr = "{0:N2} MB" -f ($totalSize / 1MB)
        } elseif ($totalSize -gt 1KB) {
            $sizeStr = "{0:N2} KB" -f ($totalSize / 1KB)
        } else {
            $sizeStr = "{0:N0} bytes" -f $totalSize
        }

        Write-Host "  ✓ Status:     Found" -ForegroundColor Green
        Write-Host "  ✓ Location:   $dsPath" -ForegroundColor Gray
        Write-Host "  ✓ Files:      $fileCount" -ForegroundColor Gray
        Write-Host "  ✓ Total Size: $sizeStr" -ForegroundColor Gray

        $results += [PSCustomObject]@{
            Dataset = $dsName
            Status = "Found"
            Files = $fileCount
            Size = $sizeStr
            Path = $dsPath
        }
    } else {
        Write-Host "  ✗ Status:     MISSING" -ForegroundColor Red
        Write-Host "  ✗ Expected:   $dsPath" -ForegroundColor Red
        $allFound = $false

        $results += [PSCustomObject]@{
            Dataset = $dsName
            Status = "MISSING"
            Files = 0
            Size = "N/A"
            Path = $dsPath
        }
    }

    Write-Host ""
}

# Summary Table
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$results | Format-Table -AutoSize

# Final status
Write-Host ""
if ($allFound) {
    Write-Host "✓ All datasets are present and ready for processing!" -ForegroundColor Green
} else {
    Write-Host "✗ Some datasets are missing. Please download them before continuing." -ForegroundColor Red
    Write-Host ""
    Write-Host "Dataset sources:" -ForegroundColor Yellow
    Write-Host "  - ptb-xl:           https://physionet.org/content/ptb-xl/" -ForegroundColor White
    Write-Host "  - PTB_Diagnostic:   https://physionet.org/content/ptbdb/" -ForegroundColor White
    Write-Host "  - CinC2017:         https://physionet.org/content/challenge-2017/" -ForegroundColor White
    Write-Host "  - Chapman_Shaoxing: https://physionet.org/content/ecg-arrhythmia/" -ForegroundColor White
}
Write-Host ""

# Check for common data files
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Key Files Check" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$keyFiles = @(
    @{Dataset="ptb-xl"; File="ptbxl_database.csv"},
    @{Dataset="PTB_Diagnostic"; File="RECORDS"},
    @{Dataset="CinC2017"; File="REFERENCE-v3.csv"},
    @{Dataset="Chapman_Shaoxing"; File="ConditionNames_SNOMED-CT.csv"}
)

foreach ($keyFile in $keyFiles) {
    $filePath = Join-Path $DatasetRoot (Join-Path $keyFile.Dataset $keyFile.File)
    if (Test-Path $filePath) {
        Write-Host "  ✓ $($keyFile.Dataset)/$($keyFile.File)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($keyFile.Dataset)/$($keyFile.File)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Verification complete." -ForegroundColor Cyan
Write-Host ""


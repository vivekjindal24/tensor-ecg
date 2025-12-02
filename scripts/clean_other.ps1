# ============================================================================
# ECG Research Project - Clean "OTHER" Labels Script
# ============================================================================
# This script identifies records with excessive "OTHER" labels and suggests
# candidates for re-mapping or removal.
# ============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ECG Research Project - Clean OTHER Labels" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = "D:\ecg-research"
$LogsDir = Join-Path $ProjectRoot "logs"
$MappingFile = Join-Path $LogsDir "unified_label_mapping.csv"

# Check if mapping file exists
if (-Not (Test-Path $MappingFile)) {
    Write-Host "✗ Error: Label mapping file not found at: $MappingFile" -ForegroundColor Red
    Write-Host "Please run bootstrap.ps1 first to create the file." -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/3] Loading label mapping..." -ForegroundColor Yellow

# Load and analyze the mapping file
try {
    $mapping = Import-Csv $MappingFile
    $totalLabels = $mapping.Count
    $otherLabels = ($mapping | Where-Object { $_.unified_label -eq "Other" }).Count
    $otherPercentage = [math]::Round(($otherLabels / $totalLabels) * 100, 2)

    Write-Host "  ✓ Total label mappings: $totalLabels" -ForegroundColor Green
    Write-Host "  ✓ OTHER labels: $otherLabels ($otherPercentage%)" -ForegroundColor $(if ($otherPercentage -gt 20) { "Yellow" } else { "Green" })
    Write-Host ""

    if ($otherPercentage -gt 20) {
        Write-Host "  ⚠ Warning: More than 20% of labels are mapped to OTHER!" -ForegroundColor Yellow
        Write-Host "  Consider reviewing and re-mapping these labels." -ForegroundColor Yellow
        Write-Host ""
    }

} catch {
    Write-Host "✗ Error reading mapping file: $_" -ForegroundColor Red
    exit 1
}

# Display OTHER labels
Write-Host "[2/3] Analyzing OTHER labels by dataset..." -ForegroundColor Yellow
Write-Host ""

$otherByDataset = $mapping | Where-Object { $_.unified_label -eq "Other" } | Group-Object -Property dataset

foreach ($group in $otherByDataset) {
    $dataset = $group.Name
    $count = $group.Count

    Write-Host "Dataset: $dataset ($count OTHER labels)" -ForegroundColor Cyan
    Write-Host "─────────────────────────────────────────────────────────" -ForegroundColor Gray

    $group.Group | ForEach-Object {
        Write-Host "  • $($_.original_label)" -ForegroundColor White -NoNewline
        if ($_.description) {
            Write-Host " - $($_.description)" -ForegroundColor Gray
        } else {
            Write-Host " - [No description]" -ForegroundColor DarkGray
        }
    }
    Write-Host ""
}

# Top 20 candidates for re-mapping
Write-Host "[3/3] Top 20 candidates for re-mapping..." -ForegroundColor Yellow
Write-Host ""

# This is a simulation - in reality, you would scan processed records
# For now, we'll list the OTHER labels as candidates
Write-Host "The following labels are currently mapped to OTHER:" -ForegroundColor White
Write-Host "Consider re-mapping them to specific clinical categories:" -ForegroundColor White
Write-Host ""

$candidates = $mapping | Where-Object { $_.unified_label -eq "Other" } | Select-Object -First 20

$rank = 1
foreach ($candidate in $candidates) {
    Write-Host "  $rank. " -ForegroundColor Yellow -NoNewline
    Write-Host "$($candidate.original_label)" -ForegroundColor White -NoNewline
    Write-Host " [$($candidate.dataset)]" -ForegroundColor Gray -NoNewline
    if ($candidate.description) {
        Write-Host " - $($candidate.description)" -ForegroundColor DarkGray
    } else {
        Write-Host ""
    }
    $rank++
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Recommendations" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

if ($otherPercentage -gt 30) {
    Write-Host "✗ HIGH: $otherPercentage% OTHER labels detected!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Yellow
    Write-Host "  1. Review logs\unified_label_mapping.csv" -ForegroundColor White
    Write-Host "  2. Map ambiguous labels to specific categories" -ForegroundColor White
    Write-Host "  3. Consider excluding truly ambiguous records" -ForegroundColor White
    Write-Host "  4. Document decisions in your preprocessing notebook" -ForegroundColor White
} elseif ($otherPercentage -gt 15) {
    Write-Host "⚠ MODERATE: $otherPercentage% OTHER labels detected" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Actions:" -ForegroundColor Yellow
    Write-Host "  1. Review the most frequent OTHER labels" -ForegroundColor White
    Write-Host "  2. Consider consolidating similar categories" -ForegroundColor White
    Write-Host "  3. Re-run preprocessing after updates" -ForegroundColor White
} else {
    Write-Host "✓ GOOD: Only $otherPercentage% OTHER labels" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your label mapping appears well-structured!" -ForegroundColor Green
    Write-Host "Proceed with preprocessing and training." -ForegroundColor White
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Generate a report file
$reportPath = Join-Path $LogsDir "other_labels_report.txt"
$reportContent = @"
ECG Research Project - OTHER Labels Report
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
═══════════════════════════════════════════════════════════

Summary:
- Total Labels: $totalLabels
- OTHER Labels: $otherLabels ($otherPercentage%)

By Dataset:
$($otherByDataset | ForEach-Object { "- $($_.Name): $($_.Count) labels" } | Out-String)

Top Candidates for Re-mapping:
$($candidates | ForEach-Object { $idx = 1 } { "$idx. $($_.original_label) [$($_.dataset)]"; $idx++ } | Out-String)

Recommendation:
$(if ($otherPercentage -gt 30) { "HIGH priority - Review and re-map immediately" }
  elseif ($otherPercentage -gt 15) { "MODERATE priority - Review when possible" }
  else { "LOW priority - Mapping is acceptable" })

═══════════════════════════════════════════════════════════
"@

Set-Content -Path $reportPath -Value $reportContent
Write-Host "✓ Report saved to: $reportPath" -ForegroundColor Green
Write-Host ""


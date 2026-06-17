param(
    [string]$OutputDir = "review_bundles",
    [string]$NamePrefix = "FSPT_MMP_GPT_REVIEW"
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stageDir = Join-Path $repoRoot "$OutputDir\${NamePrefix}_$timestamp"
$zipPath = "$stageDir.zip"

New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

$includePaths = @(
    'README.md',
    'requirements.txt',
    'train.py',
    'evaluate.py',
    'projects\mmp_tracker',
    'models',
    'utils',
    'datasets',
    'tests',
    'configs',
    'docs',
    'scripts',
    'baselines',
    'papers'
)

$excludeDirNames = @(
    '.git', '.pytest_cache', '__pycache__',
    'checkpoints', 'outputs', 'logs', 'weights',
    'datasets_data', 'datasets_code', 'review_bundles'
)

foreach ($rel in $includePaths) {
    $src = Join-Path $repoRoot $rel
    if (-not (Test-Path $src)) { continue }

    $dst = Join-Path $stageDir $rel
    if (Test-Path $src -PathType Container) {
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Get-ChildItem -Path $src -Recurse -Force | ForEach-Object {
            $full = $_.FullName
            $name = $_.Name
            if ($excludeDirNames -contains $name) { return }
            if ($_.PSIsContainer) {
                $targetDir = $full.Replace($repoRoot, $stageDir)
                New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
            } else {
                $targetFile = $full.Replace($repoRoot, $stageDir)
                $parentDir = Split-Path -Parent $targetFile
                New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
                Copy-Item -Force $full $targetFile
            }
        }
    } else {
        $parentDir = Split-Path -Parent $dst
        if ($parentDir) {
            New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
        }
        Copy-Item -Force $src $dst
    }
}

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Compress-Archive -Path (Join-Path $stageDir '*') -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host "Stage dir: $stageDir"
Write-Host "Zip path:  $zipPath"


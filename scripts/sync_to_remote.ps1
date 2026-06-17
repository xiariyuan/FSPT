# =============================================================================
# FSPT - 同步本地项目到远程服务器 (PowerShell)
# 在Windows本地运行
# =============================================================================

param(
    [string]$Server = "your-server",
    [string]$RemotePath = "/path/to/FSPT"
)

$ProjectRoot = $null
try {
    $ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..") -ErrorAction Stop
} catch {
    Write-Error "Failed to resolve project root: $_"
    exit 1
}
$LocalPath = $ProjectRoot.Path

if ($Server -eq "your-server" -or $RemotePath -eq "/path/to/FSPT") {
    Write-Error "Please configure -Server and -RemotePath before syncing."
    Write-Host "Example: .\sync_to_remote.ps1 -Server 'user@host' -RemotePath '/home/user/FSPT'"
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "FSPT Sync to Remote Server" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Local:  $LocalPath"
Write-Host "Remote: $Server`:$RemotePath"
Write-Host ""

# 确认
$confirm = Read-Host "Proceed with sync? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit
}

# 创建远程目录
Write-Host ""
Write-Host "[1/2] Creating remote directory..." -ForegroundColor Green
ssh $Server "mkdir -p $RemotePath"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create remote directory."
    exit 1
}

# 同步文件
Write-Host "[2/2] Syncing files..." -ForegroundColor Green

# 使用scp同步各个目录（不含datasets等大文件）
$dirs = @("docs", "configs", "scripts", "models", "utils", "tests")
foreach ($dir in $dirs) {
    $localDir = Join-Path $LocalPath $dir
    if (Test-Path $localDir) {
        Write-Host "  Syncing $dir..."
        scp -r "$localDir" "${Server}:${RemotePath}/"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to sync $dir."
            exit 1
        }
    }
}

# 同步单个文件
$files = @("README.md", "requirements.txt", "train.py", "evaluate.py", "verify_project.py")
foreach ($file in $files) {
    $localFile = Join-Path $LocalPath $file
    if (Test-Path $localFile) {
        Write-Host "  Syncing $file..."
        scp "$localFile" "${Server}:${RemotePath}/"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to sync $file."
            exit 1
        }
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Sync Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To initialize on remote server:"
Write-Host "  ssh $Server"
Write-Host "  cd $RemotePath"
Write-Host "  bash scripts/init_remote.sh"

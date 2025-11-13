
# Upload KAIST Dataset using WSL and rsync
# This script requires WSL (Windows Subsystem for Linux) to be installed

$SERVER_IP = Read-Host "Enter server IP address"
$USERNAME = Read-Host "Enter username (default: msi-kklt)" -Default "msi-kklt"
$TARGET_PATH = Read-Host "Enter target path on server (default: /data/kaist)" -Default "/data/kaist"

Write-Host "`nPreparing to upload C:\KAIST_processed to $USERNAME@$SERVER_IP`:$TARGET_PATH"
Write-Host "This may take a while depending on your network speed...`n"

# Convert Windows path to WSL path
$WSL_SOURCE = "/mnt/c/KAIST_processed/"

# Create rsync command
$RSYNC_CMD = "rsync -avz --progress '$WSL_SOURCE' '$USERNAME@$SERVER_IP`:$TARGET_PATH'"

Write-Host "Command: wsl $RSYNC_CMD`n"
Write-Host "Press any key to start upload or Ctrl+C to cancel..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Execute via WSL
wsl bash -c $RSYNC_CMD

Write-Host "`nUpload complete!"
Write-Host "You can now run on the server:"
Write-Host "  bash setup_planC.sh $TARGET_PATH ./work_dirs/stage1/epoch_48.pth"

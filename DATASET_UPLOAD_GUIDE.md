
# KAIST Dataset Upload Guide

## Your Local Dataset Location
C:\KAIST_processed\
  ├── Annotations/
  ├── ImageSets/
  ├── infrared/
  └── visible/

## Method 1: WinSCP (Recommended - No command line needed)

1. Download WinSCP: https://winscp.net/eng/download.php
2. Install and open WinSCP
3. Create new connection:
   - File protocol: SFTP
   - Host name: <your server IP>
   - User name: msi-kklt
   - Password: <your password>
4. Click "Login"
5. Navigate to /data/ (or create it if not exists)
6. Drag and drop C:\KAIST_processed from left panel to right panel
7. Wait for upload to complete

## Method 2: Command Line (Requires WSL or Git Bash)

### If you have WSL installed:
```powershell
# In PowerShell
wsl
# Now in WSL:
cd /mnt/c/KAIST_processed
rsync -avz --progress . msi-kklt@<SERVER_IP>:/data/kaist/
```

### If you have Git Bash:
```bash
cd /c/KAIST_processed
scp -r . msi-kklt@<SERVER_IP>:/data/kaist/
```

## Method 3: Compress first (Faster for slow connections)

### In PowerShell:
```powershell
# Compress
Compress-Archive -Path C:\KAIST_processed -DestinationPath C:\kaist.zip

# Then upload the zip file using WinSCP or:
scp C:\kaist.zip msi-kklt@<SERVER_IP>:/tmp/
```

### On Linux server:
```bash
cd /data
sudo unzip /tmp/kaist.zip
sudo mv KAIST_processed kaist
```

## After Upload - Verify on Server

```bash
# SSH to server
ssh msi-kklt@<SERVER_IP>

# Check uploaded data
ls -la /data/kaist/
# Should see: Annotations/, ImageSets/, infrared/, visible/

# Count images
find /data/kaist/visible -name "*.jpg" | wc -l
find /data/kaist/infrared -name "*.jpg" | wc -l

# Check annotations
ls /data/kaist/Annotations/ | head
ls /data/kaist/ImageSets/
```

## Finally - Run Setup Script

```bash
cd ~/xyz/mmdetection/linux_planC_package
bash setup_planC.sh /data/kaist ./work_dirs/stage1/epoch_48.pth
```

## Troubleshooting

### Permission Denied
```bash
sudo mkdir -p /data
sudo chown -R $USER:$USER /data
```

### Disk Space Check
```bash
df -h /data
# Make sure you have at least 10GB free space
```

### Network Too Slow?
Consider uploading only the training split first:
1. Upload ImageSets/train.txt
2. Read the file to get list of needed images
3. Upload only those images


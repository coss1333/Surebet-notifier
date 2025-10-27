# PowerShell run script
Get-Content .env | ForEach-Object {
  if ($_ -and $_ -notmatch '^#') {
    $name,$value = $_.split('=',2)
    [Environment]::SetEnvironmentVariable($name,$value,"Process")
  }
}
python main.py

param(
    [string]$EnvName = "camp",
    [string]$OutputFile = "environment.yml"
)

$ErrorActionPreference = "Stop"
conda export -n $EnvName --from-history --file $OutputFile
Write-Host "Exported $EnvName to $OutputFile"

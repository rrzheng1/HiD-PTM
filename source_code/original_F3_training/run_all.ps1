param(
    [string]$Device='cuda',
    [int]$BatchSize=64,
    [int]$Epochs=40,
    [int]$Patience=5,
    [int]$Seed=42,
    [ValidateSet('F0','F1','F2','F3','F4','F5','F6','all')][string]$Variant='all',
    [double]$MinDelta=0.0001,
    [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
$env:LOKY_MAX_CPU_COUNT='4'
function Invoke-PythonChecked {
    param([string[]]$PythonArguments)
    & python @PythonArguments
    if ($LASTEXITCODE -ne 0) {throw "Python failed with exit $LASTEXITCODE. Completed folds and checkpoints are preserved."}
}
$LogDirectory=Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
Start-Transcript -Path (Join-Path $LogDirectory "${Variant}_seed${Seed}.log") -Append | Out-Null
try {
    Invoke-PythonChecked -PythonArguments @((Join-Path $PSScriptRoot 'check_environment.py'))
    if ($CheckOnly) {return}
    Invoke-PythonChecked -PythonArguments @('-m','pip','install','-r',(Join-Path $PSScriptRoot 'requirements.txt'))
    Invoke-PythonChecked -PythonArguments @((Join-Path $PSScriptRoot 'smoke_test.py'),'--device',$Device)
    Invoke-PythonChecked -PythonArguments @((Join-Path $PSScriptRoot 'prepare.py'),'--device',$Device)
    $Experiments=if ($Variant -eq 'all') {@('F0','F1','F2','F3','F4','F5','F6')} else {@($Variant)}
    foreach ($Experiment in $Experiments) {
        Invoke-PythonChecked -PythonArguments @((Join-Path $PSScriptRoot 'run_cv.py'),'--device',$Device,'--batch-size',"$BatchSize",'--epochs',"$Epochs",'--patience',"$Patience",'--seed',"$Seed",'--variant',$Experiment,'--min-delta',"$MinDelta")
    }
    if ($Variant -eq 'all') {Invoke-PythonChecked -PythonArguments @((Join-Path $PSScriptRoot 'compare_results.py'),'--seed',"$Seed")}
} finally {Stop-Transcript | Out-Null}

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$distro = if ($env:V1_WSL_DISTRO) { $env:V1_WSL_DISTRO } else { "Ubuntu-22.04-Bio" }
$python = if ($env:V1_WSL_PYTHON) { $env:V1_WSL_PYTHON } else { "/opt/miniconda3/envs/pytorch/bin/python" }
$linuxRoot = (wsl.exe -d $distro -- wslpath -a (Resolve-Path $root).Path).Trim()
wsl.exe -d $distro -- bash -lc "cd '$linuxRoot' && '$python' -m vestibular_fusion reproduce --config configs/paths.local.json --datasets monifeixing vrq city --device cuda"
if ($LASTEXITCODE -ne 0) { throw "v1 reproduction failed with exit code $LASTEXITCODE" }

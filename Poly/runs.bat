@echo off
setlocal enabledelayedexpansion

:: ==================================================
:: Experiment 1
:: ==================================================

set Batch_size=8
set tau=0.02
set init_point=-1.2

set noise_levels=0 0.5 1
set sigma_Beq=0.05
set sigma_Bal=0.35

set project_name=Poly_B%Batch_size%

for %%n in (%noise_levels%) do (
    set nl=%%n
    :: Usiamo replace anche qui per sicurezza sebbene i valori siano hardcoded
    set nl_p=!nl:,=.!
    
    echo Running: Ballistic ^| RMSProp ^| Sigma %sigma_Bal%
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma %sigma_Bal% --batch-size-simulation %Batch_size% --noise_level !nl_p! --tau-list %tau% --final-time 4 --name-project %project_name% --initial_points %init_point%

    echo Running: Batch Equivalent ^| RMSProp ^| Sigma %sigma_Beq%
    python -m Poly.main --regime batch_equivalent --optimizer RMSProp --batch-size-simulation %Batch_size% --noise_level !nl_p! --tau-list %tau% --sigma %sigma_Beq% --final-time 4 --name-project %project_name% --initial_points %init_point%

    echo Running: Ballistic ^| Adam ^| Sigma %sigma_Bal%
    python -m Poly.main --regime balistic --optimizer Adam --sigma %sigma_Bal% --batch-size-simulation %Batch_size% --noise_level !nl_p! --tau-list %tau% --final-time 4 --name-project %project_name% --initial_points %init_point%
    
    echo Running: Batch Equivalent ^| Adam ^| Sigma %sigma_Beq%
    python -m Poly.main --regime batch_equivalent --optimizer Adam --batch-size-simulation %Batch_size% --noise_level !nl_p! --tau-list %tau% --sigma %sigma_Beq% --final-time 4 --name-project %project_name% --initial_points %init_point%

    echo Finished all runs for alpha !nl!
    echo.
)

:: ==================================================
:: Experiment 2
:: ==================================================

set Batch_size=32
set tau=0.02
set init_point=1.5

set noise_levels=0 2 4

:: Calcolo sigma con replace virgola -> punto
for /f "delims=" %%a in ('powershell -command "([math]::sqrt(%tau%/%Batch_size%)) -split ',' -join '.'"') do set sigma_Beq=%%a
for /f "delims=" %%a in ('powershell -command "(1/[math]::sqrt(%Batch_size%)) -split ',' -join '.'"') do set sigma_Bal=%%a

set project_name=Poly_B%Batch_size%

for %%n in (%noise_levels%) do (
    set nl=%%n
    echo Running: Ballistic ^| RMSProp ^| Sigma %sigma_Bal%
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma %sigma_Bal% --batch-size-simulation %Batch_size% --noise_level !nl! --tau-list %tau% --final-time 10 --name-project %project_name% --initial_points %init_point%

    echo Running: Batch Equivalent ^| RMSProp ^| Sigma %sigma_Beq%
    python -m Poly.main --regime batch_equivalent --optimizer RMSProp --batch-size-simulation %Batch_size% --noise_level !nl! --tau-list %tau% --sigma %sigma_Beq% --final-time 10 --name-project %project_name% --initial_points %init_point%

    echo Running: Ballistic ^| Adam ^| Sigma %sigma_Bal%
    python -m Poly.main --regime balistic --optimizer Adam --sigma %sigma_Bal% --batch-size-simulation %Batch_size% --noise_level !nl! --tau-list %tau% --final-time 10 --name-project %project_name% --initial_points %init_point%
    
    echo Running: Batch Equivalent ^| Adam ^| Sigma %sigma_Beq%
    python -m Poly.main --regime batch_equivalent --optimizer Adam --batch-size-simulation %Batch_size% --noise_level !nl! --tau-list %tau% --sigma %sigma_Beq% --final-time 10 --name-project %project_name% --initial_points %init_point%

    echo Finished all runs for alpha !nl!
    echo.
)

:: ==================================================
:: Experiment 3
:: ==================================================

set scaling_rule_exp_alpha=0 1
set scaling_rule_exp_k_values=1 2 4 8
set Batch_size_0=8
set tau_0=0.02
set project_name=Poly_scaling_rule_batch_tau

for %%a in (%scaling_rule_exp_alpha%) do (
    set nl=%%a
    for %%k in (%scaling_rule_exp_k_values%) do (
        set k=%%k
        echo ==================================================
        echo Running Batch equivalent experiments with alpha: !nl!, k: !k!
        echo ==================================================

        :: Calcolo tau e batch_size con replace
        for /f %%t in ('powershell -command "(%tau_0%/!k!) -split ',' -join '.'"') do set tau_calc=%%t
        for /f %%b in ('powershell -command "[int](%Batch_size_0%/!k!)"') do set b_size=%%b

        echo Running: Batch Equivalent ^| RMSProp 
        python -m Poly.main --regime batch_equivalent --optimizer RMSProp --sigma %sigma_Beq% --batch-size-simulation !b_size! --noise_level !nl! --tau-list !tau_calc! --name-project %project_name% --final-time 4

        echo Running: Batch Equivalent ^| Adam 
        python -m Poly.main --regime batch_equivalent --optimizer Adam --sigma %sigma_Beq% --batch-size-simulation !b_size! --noise_level !nl! --tau-list !tau_calc! --name-project %project_name% --final-time 4
    )
)

echo All experiments completed successfully.
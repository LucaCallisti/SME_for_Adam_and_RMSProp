@echo off
setlocal enabledelayedexpansion

:: ==================================================
:: Neural Network Experiments
:: ==================================================

set NN_type=ShallowNN MLP ResNet

echo Starting Batch Experiments...

for %%n in (%NN_type%) do (
    set model=%%n
    echo ==================================================
    echo Running experiments with: !model!
    echo ==================================================

    echo Running: Batch Equivalent ^| RMSProp 
    python -m NeuralNetwork.main --regime batch_equivalent --optimizer RMSProp --model !model!

    echo Running: Batch Equivalent ^| Adam 
    python -m NeuralNetwork.main --regime batch_equivalent --optimizer Adam --model !model!

    echo Running: Ballistic ^| RMSProp 
    python -m NeuralNetwork.main --regime balistic --optimizer RMSProp --model !model!

    echo Running: Ballistic ^| Adam 
    python -m NeuralNetwork.main --regime balistic --optimizer Adam --model !model!

    echo Finished all runs for Model !model!
    echo.
)

echo All experiments completed successfully.
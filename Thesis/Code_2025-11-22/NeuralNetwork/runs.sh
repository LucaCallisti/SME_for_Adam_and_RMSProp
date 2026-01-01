#!/bin/bash

# Lista dei noise levels da testare
NN_type=("MLP" "ShallowNN")

echo "Starting Batch Experiments..."

# Ciclo su ogni livello di rumore
for nl in "${NN_type[@]}"
do
    echo "=================================================="
    echo "Running experiments with: $nl"
    echo "=================================================="

    # 1. Batch Equivalent - RMSProp
    echo "Running: Batch Equivalent | RMSProp "
    python -m NeuralNetwork.main --regime batch_equivalent --optimizer RMSProp --model $nl 

    # 2. Batch Equivalent - Adam
    echo "Running: Batch Equivalent | Adam "
    python -m NeuralNetwork.main --regime batch_equivalent --optimizer Adam --model $nl 

    # 3. Ballistic - RMSProp
    echo "Running: Ballistic | RMSProp "
    python -m NeuralNetwork.main --regime balistic --optimizer RMSProp --model $nl

    # 4. Ballistic - Adam
    echo "Running: Ballistic | Adam "
    python -m NeuralNetwork.main --regime balistic --optimizer Adam --model $nl 


    echo "Finished all runs for Noise Level $nl"
    echo ""
done

echo "All experiments completed successfully."

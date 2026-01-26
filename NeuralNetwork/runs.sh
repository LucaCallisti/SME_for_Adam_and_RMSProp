# !/bin/bash

NN_type=("ShallowNN" "MLP" "ResNet")

echo "Starting Batch Experiments..."

for nl in "${NN_type[@]}"
do
    echo "=================================================="
    echo "Running experiments with: $nl"
    echo "=================================================="

    echo "Running: Batch Equivalent | RMSProp "
    python -m NeuralNetwork.main --regime batch_equivalent --optimizer RMSProp --model $nl 

    echo "Running: Batch Equivalent | Adam "
    python -m NeuralNetwork.main --regime batch_equivalent --optimizer Adam --model $nl 

    echo "Running: Ballistic | RMSProp "
    python -m NeuralNetwork.main --regime balistic --optimizer RMSProp --model $nl

    echo "Running: Ballistic | Adam "
    python -m NeuralNetwork.main --regime balistic --optimizer Adam --model $nl 


    echo "Finished all runs for Model $nl"
    echo ""
done

echo "All experiments completed successfully."

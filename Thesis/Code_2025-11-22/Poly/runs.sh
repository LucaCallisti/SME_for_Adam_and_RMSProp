#!/bin/bash

# Lista dei noise levels da testare
noise_levels=(0 0.25 1 4)

echo "Starting Batch Experiments..."

# Ciclo su ogni livello di rumore
for nl in "${noise_levels[@]}"
do
    echo "=================================================="
    echo "Running experiments with Noise Level: $nl"
    echo "=================================================="

    # 1. Batch Equivalent - RMSProp
    echo "Running: Batch Equivalent | RMSProp | Sigma -1"
    python -m Poly.main --regime batch_equivalent --optimizer RMSProp --sigma -1 --batch-size-simulation 10 --noise_level $nl

    # 2. Batch Equivalent - Adam
    echo "Running: Batch Equivalent | Adam | Sigma -1"
    python -m Poly.main --regime batch_equivalent --optimizer Adam --sigma -1 --batch-size-simulation 10 --noise_level $nl

    # 3. Ballistic - RMSProp - Sigma 0.07
    echo "Running: Ballistic | RMSProp | Sigma 0.07"
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma 0.07 --batch-size-simulation -1 --noise_level $nl

    # 4. Ballistic - RMSProp - Sigma -1
    echo "Running: Ballistic | RMSProp | Sigma -1"
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma -1 --batch-size-simulation 10 --noise_level $nl

    # 5. Ballistic - Adam - Sigma 0.07
    echo "Running: Ballistic | Adam | Sigma 0.07"
    python -m Poly.main --regime balistic --optimizer Adam --sigma 0.07 --batch-size-simulation -1 --noise_level $nl

    # 6. Ballistic - Adam - Sigma -1
    echo "Running: Ballistic | Adam | Sigma -1"
    python -m Poly.main --regime balistic --optimizer Adam --sigma -1 --batch-size-simulation 10 --noise_level $nl

    echo "Finished all runs for Noise Level $nl"
    echo ""
done

echo "All experiments completed successfully."
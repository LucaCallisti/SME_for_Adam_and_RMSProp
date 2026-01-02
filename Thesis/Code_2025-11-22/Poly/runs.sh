#!/bin/bash

# Lista dei noise levels da testare
noise_levels=(0 1 2 4 6 8 10)

Batch_size_Beq=16
sigma_Beq=0.01
tau_Beq=0.0016

echo "Starting Batch Experiments..."

# Ciclo su ogni livello di rumore
for nl in "${noise_levels[@]}"
do
    echo "=================================================="
    echo "Running Batch equivalent experiments with Noise Level: $nl"
    echo "=================================================="

    # 1. Batch Equivalent - RMSProp
    echo "Running: Batch Equivalent | RMSProp | Sigma -1"
    python -m Poly.main --regime batch_equivalent --optimizer RMSProp --sigma $sigma_Beq --batch-size-simulation $Batch_size_Beq --noise_level $nl --tau-list $tau_Beq 

    # 2. Batch Equivalent - Adam
    echo "Running: Batch Equivalent | Adam | Sigma -1"
    python -m Poly.main --regime batch_equivalent --optimizer Adam --sigma $sigma_Beq --batch-size-simulation $Batch_size_Beq --noise_level $nl --tau-list $tau_Beq 

    echo "=================================================="
    echo "Running Balistic experiments with Noise Level: $nl"
    echo "=================================================="

    # 3. Ballistic - RMSProp - Sigma 0.07
    echo "Running: Ballistic | RMSProp | Sigma 0.07"
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma $sigma_Beq --batch-size-simulation -1 --noise_level $nl --tau-list $tau_Beq 

    # 4. Ballistic - RMSProp - Sigma -1
    echo "Running: Ballistic | RMSProp | Sigma -1"
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma -1 --batch-size-simulation $Batch_size_Beq --noise_level $nl --tau-list $tau_Beq 

    # 5. Ballistic - Adam - Sigma 0.07
    echo "Running: Ballistic | Adam | Sigma 0.07"
    python -m Poly.main --regime balistic --optimizer Adam --sigma $sigma_Beq --batch-size-simulation -1 --noise_level $nl --tau-list $tau_Beq 
    # 6. Ballistic - Adam - Sigma -1
    echo "Running: Ballistic | Adam | Sigma -1"
    python -m Poly.main --regime balistic --optimizer Adam --sigma -1 --batch-size-simulation $Batch_size_Beq --noise_level $nl --tau-list $tau_Beq 

    # wait
    echo "Finished all runs for Noise Level $nl"
    echo ""
done


scaling_rule_exp_alpha=(0 1)
scaling_rule_exp_k_values=(1 2 4 8 16)

for nl in "${scaling_rule_exp_alpha[@]}"
do
    for k in "${scaling_rule_exp_k_values[@]}"
    do
        echo "=================================================="
        echo "Running Batch equivalent experiments with Noise Level: $nl"
        echo "=================================================="

        # 1. Batch Equivalent - RMSProp
        echo "Running: Batch Equivalent | RMSProp | Sigma -1"

        tau=$(awk "BEGIN {print $tau_Beq/$k}")
        batch_size=$(awk "BEGIN {print int($Batch_size_Beq/$k)}")
        python -m Poly.main --regime batch_equivalent --optimizer RMSProp --sigma $sigma_Beq --batch-size-simulation $batch_size --noise_level $nl --tau-list $tau --name-project "Poly_scaling_rule_batch_tau" 

        # 2. Batch Equivalent - Adam
        echo "Running: Batch Equivalent | Adam | Sigma -1"
        python -m Poly.main --regime batch_equivalent --optimizer Adam --sigma $sigma_Beq --batch-size-simulation $batch_size --noise_level $nl --tau-list $tau --name-project "Poly_scaling_rule_batch_tau"
    
        # wait
    done
done

echo "All experiments completed successfully."
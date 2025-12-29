# #!/bin/bash

# # Lista dei noise levels da testare
# noise_levels_Balistic=(0 1 2  4 6 8 10)
# noise_levels_BatchEquivalent=(0 0.25 1 2 3 4)

# echo "Starting Batch Experiments..."

# # Ciclo su ogni livello di rumore
# for nl in "${noise_levels_BatchEquivalent[@]}"
# do
#     echo "=================================================="
#     echo "Running Batch equivalent experiments with Noise Level: $nl"
#     echo "=================================================="

#     # 1. Batch Equivalent - RMSProp
#     echo "Running: Batch Equivalent | RMSProp | Sigma -1"
#     python -m Poly.main --regime batch_equivalent --optimizer RMSProp --sigma 0.02 --batch-size-simulation -1 --noise_level $nl

#     # 2. Batch Equivalent - Adam
#     echo "Running: Batch Equivalent | Adam | Sigma -1"
#     python -m Poly.main --regime batch_equivalent --optimizer Adam --sigma 0.02 --batch-size-simulation -1 --noise_level $nl
# done

# for nl in "${noise_levels_Balistic[@]}"
# do
#     echo "=================================================="
#     echo "Running Balistic experiments with Noise Level: $nl"
#     echo "=================================================="

#     # 3. Ballistic - RMSProp - Sigma 0.07
#     echo "Running: Ballistic | RMSProp | Sigma 0.07"
#     python -m Poly.main --regime balistic --optimizer RMSProp --sigma 0.02 --batch-size-simulation -1 --noise_level $nl

#     # 4. Ballistic - RMSProp - Sigma -1
#     echo "Running: Ballistic | RMSProp | Sigma -1"
#     python -m Poly.main --regime balistic --optimizer RMSProp --sigma -1 --batch-size-simulation 12 --noise_level $nl

#     # 5. Ballistic - Adam - Sigma 0.07
#     echo "Running: Ballistic | Adam | Sigma 0.07"
#     python -m Poly.main --regime balistic --optimizer Adam --sigma 0.02 --batch-size-simulation -1 --noise_level $nl

#     # 6. Ballistic - Adam - Sigma -1
#     echo "Running: Ballistic | Adam | Sigma -1"
#     python -m Poly.main --regime balistic --optimizer Adam --sigma -1 --batch-size-simulation 12 --noise_level $nl

#     echo "Finished all runs for Noise Level $nl"
#     echo ""
# done


scaling_rule_exp_alpha=(0 1)
scaling_rule_exp_k_values=(1 5 10 50 100)

for nl in "${scaling_rule_exp_alpha[@]}"
do
    for k in "${scaling_rule_exp_k_values[@]}"
    do
        echo "=================================================="
        echo "Running Batch equivalent experiments with Noise Level: $nl"
        echo "=================================================="

        # 1. Batch Equivalent - RMSProp
        echo "Running: Batch Equivalent | RMSProp | Sigma -1"

        tau=$(awk "BEGIN {print $k*0.005}")
        python -m Poly.main --regime batch_equivalent --optimizer RMSProp --sigma -1 --batch-size-simulation $((k*12)) --noise_level $nl --tau-list $tau --name-project "Poly_scaling_rule_batch_tau"

        # 2. Batch Equivalent - Adam
        echo "Running: Batch Equivalent | Adam | Sigma -1"
        tau=$(awk "BEGIN {print $k*0.005}")
        python -m Poly.main --regime batch_equivalent --optimizer Adam --sigma -1 --batch-size-simulation $((k*12)) --noise_level $nl --tau-list $tau --name-project "Poly_scaling_rule_batch_tau"
    done
done

echo "All experiments completed successfully."
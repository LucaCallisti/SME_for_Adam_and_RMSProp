# !/bin/bash

# Experiment 1

Batch_size=8
tau=0.02
init_point=1.5

noise_levels=(0 0.5 1)
sigma_Beq=0.05
sigma_Bal=0.35

project_name="Poly_with_additional_noise_changing_noise_B$Batch_size"

for nl in "${noise_levels[@]}"
do

    echo "Running: Ballistic | RMSProp | Sigma ${sigma_Bal}"
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma $sigma_Bal --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --final-time 4 --name-project $project_name --initial_points $init_point

    echo "Running: Batch Equivalent | RMSProp | Sigma ${sigma_Beq}"
    python -m Poly.main --regime batch_equivalent --optimizer RMSProp --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --sigma $sigma_Beq --final-time 4 --name-project $project_name --initial_points $init_point

    echo "Running: Ballistic | Adam | Sigma ${sigma_Bal}"
    python -m Poly.main --regime balistic --optimizer Adam --sigma $sigma_Bal --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --final-time 4 --name-project $project_name --initial_points $init_point
    
    echo "Running: Batch Equivalent | Adam | Sigma ${sigma_Beq}"
    python -m Poly.main --regime batch_equivalent --optimizer Adam --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --sigma $sigma_Beq --final-time 4 --name-project $project_name --initial_points $init_point

    echo "Finished all runs for alpha $nl"
    echo ""
done


# Experiment 2

Batch_size=32
tau=0.02
init_point=1.5

noise_levels=(0 2 4)
sigma_Beq=$(awk "BEGIN {print sqrt($tau/$Batch_size)}")
sigma_Bal=$(awk "BEGIN {print 1/sqrt($Batch_size)}")

project_name="Poly_with_additional_noise_changing_noise_B$Batch_size"

for nl in "${noise_levels[@]}"
do

    echo "Running: Ballistic | RMSProp | Sigma ${sigma_Bal}"
    python -m Poly.main --regime balistic --optimizer RMSProp --sigma $sigma_Bal --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --final-time 10 --name-project $project_name --initial_points $init_point

    echo "Running: Batch Equivalent | RMSProp | Sigma ${sigma_Beq}"
    python -m Poly.main --regime batch_equivalent --optimizer RMSProp --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --sigma $sigma_Beq --final-time 10 --name-project $project_name --initial_points $init_point

    echo "Running: Ballistic | Adam | Sigma ${sigma_Bal}"
    python -m Poly.main --regime balistic --optimizer Adam --sigma $sigma_Bal --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --final-time 10 --name-project $project_name --initial_points $init_point
    
    echo "Running: Batch Equivalent | Adam | Sigma ${sigma_Beq}"
    python -m Poly.main --regime batch_equivalent --optimizer Adam --batch-size-simulation $Batch_size --noise_level $nl --tau-list $tau --sigma $sigma_Beq --final-time 10 --name-project $project_name --initial_points $init_point


    echo "Finished all runs for alpha $nl"
    echo ""
done


# Experiment 3

scaling_rule_exp_alpha=(0 1)
scaling_rule_exp_k_values=(1 2 4 8)
Batch_size_0=8
tau_0=0.02

for nl in "${scaling_rule_exp_alpha[@]}"
do
    for k in "${scaling_rule_exp_k_values[@]}"
    do
        echo "=================================================="
        echo "Running Batch equivalent experiments with alpha: $nl, k: $k"
        echo "=================================================="


        tau=$(awk "BEGIN {print $tau_0/$k}")
        batch_size=$(awk "BEGIN {print int($Batch_size_0\/$k)}")

        echo "Running: Batch Equivalent | RMSProp "
        python -m Poly.main --regime batch_equivalent --optimizer RMSProp --sigma $sigma_Beq --batch-size-simulation $batch_size --noise_level $nl --tau-list $tau --name-project "Poly_scaling_rule_batch_tau" --final-time 4

        echo "Running: Batch Equivalent | Adam "
        python -m Poly.main --regime batch_equivalent --optimizer Adam --sigma $sigma_Beq --batch-size-simulation $batch_size --noise_level $nl --tau-list $tau --name-project "Poly_scaling_rule_batch_tau" --final-time 4
    
    done
done

echo "All experiments completed successfully."

#!/bin/bash

# Weak Scaling GMRES Test Script
# Usage: ./weakScaling.sh

# Configuration parameters
NUM_RUNS=5  # Number of runs

# Fixed parameters - only both mode, s=8
TEST_CLASSICAL=true
TEST_CA=true
S_VALUE=8

 

# Weak Scaling test configuration
declare -A WEAK_SCALING_CONFIG
WEAK_SCALING_CONFIG[1]="matrices/weakScaling/1p_nasa4704.mtx:matrices/weakScaling/1p_nasa4704_rhs.mtx"
WEAK_SCALING_CONFIG[4]="matrices/weakScaling/4p_bodyy5.mtx:matrices/weakScaling/4p_bodyy5_rhs.mtx"
WEAK_SCALING_CONFIG[8]="matrices/weakScaling/8p_pdb1HYS.mtx:matrices/weakScaling/8p_pdb1HYS_rhs.mtx"
WEAK_SCALING_CONFIG[16]="matrices/weakScaling/16p_F2.mtx:matrices/weakScaling/16p_F2_rhs.mtx"
WEAK_SCALING_CONFIG[20]="matrices/weakScaling/20p_s3dkt3m2.mtx:matrices/weakScaling/20p_s3dkt3m2_rhs.mtx"

# GMRES parameters - Enhanced convergence
ITERATIONS=30  # Increase iterations to ensure convergence
PRECOND=1       # Enable preconditioning
RESTARTS=30     # Increase restart count

# Process number configuration
PROCESSES=(1 4 8 16 20)

echo "============================================================"
echo "                Weak Scaling Performance Analysis           "
echo "============================================================"
echo "Test mode: both"
echo "GMRES parameters: iterations=$ITERATIONS, restarts=$RESTARTS, precond=$PRECOND"
echo "Number of runs: $NUM_RUNS times average"
echo "CA-GMRES s value: $S_VALUE"

echo ""

# Function: Execute single test
run_test() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    local matrix_file=$4
    local rhs_file=$5
    local log_file=$6
    
    echo "      Execute: mpirun -np $processes ./gmres $matrix_file $rhs_file $ITERATIONS $PRECOND $RESTARTS $s_param"
    
    # Execute the command
    mpirun -np $processes ./gmres "$matrix_file" "$rhs_file" $ITERATIONS $PRECOND $RESTARTS $s_param > "$log_file" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "      Successfully completed"
        return 0
    else
        echo "      Failed (exit code: $exit_code)"
        return 1
    fi
}

# Function: Extract performance data
extract_performance_data() {
    local log_file=$1
    
    if [ ! -f "$log_file" ]; then
        echo "N/A N/A N/A N/A N/A"
        return
    fi
    
    local total_time=$(grep "Total Execution Time:" "$log_file" | awk '{print $4}' | tail -1)
    local comm_time=$(grep "Total Communication Time:" "$log_file" | awk '{print $4}' | tail -1)
    local comm_percent=$(grep "Total Communication Time:" "$log_file" | sed -n 's/.*(\([0-9.]*\)%).*/\1/p' | tail -1)
    local inner_count=$(grep "Inner Product Operations:" "$log_file" | awk '{print $4}' | tail -1)
    local converged=$(grep "converged successfully" "$log_file" > /dev/null && echo "Yes" || echo "No")
    
    # Handle empty values
    total_time=${total_time:-"N/A"}
    comm_time=${comm_time:-"N/A"}
    comm_percent=${comm_percent:-"N/A"}
    inner_count=${inner_count:-"N/A"}
    
    echo "$total_time $comm_time $comm_percent $inner_count $converged"
}

# Function: Calculate average
calculate_average() {
    local values=("$@")
    local sum=0
    local count=0
    local has_valid_data=false
    
    for value in "${values[@]}"; do
        if [[ "$value" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            sum=$(echo "$sum + $value" | bc -l)
            count=$((count + 1))
            has_valid_data=true
        fi
    done
    
    if [ "$has_valid_data" = true ] && [ $count -gt 0 ]; then
        echo "scale=4; $sum / $count" | bc -l
    else
        echo "N/A"
    fi
}

# Function: Multiple test runs and calculate average
run_multiple_tests() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    local matrix_file=$4
    local rhs_file=$5
    local base_log_name=$6
    
    # Extract matrix name for display
    local matrix_name=$(basename "$matrix_file" .mtx | sed 's/^[0-9]*p_//')
    
    echo "    Running $algorithm (s=$s_param, np=$processes, matrix=$matrix_name) - $NUM_RUNS tests:"
    
    # Store results from each run
    local total_times=()
    local comm_times=()
    local comm_percents=()
    local inner_counts=()
    local converged_count=0
    local successful_runs=0
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "      Run $run..."
        local log_file="${base_log_name}_run${run}.log"
        
        if run_test "$algorithm" $s_param $processes "$matrix_file" "$rhs_file" "$log_file"; then
            # Extract data
            local result=$(extract_performance_data "$log_file")
            read -r total_time comm_time comm_percent inner_count converged <<< "$result"
            
            # Collect data
            if [ "$total_time" != "N/A" ]; then
                total_times+=("$total_time")
            fi
            if [ "$comm_time" != "N/A" ]; then
                comm_times+=("$comm_time")
            fi
            if [ "$comm_percent" != "N/A" ]; then
                comm_percents+=("$comm_percent")
            fi
            if [ "$inner_count" != "N/A" ]; then
                inner_counts+=("$inner_count")
            fi
            if [ "$converged" = "Yes" ]; then
                converged_count=$((converged_count + 1))
            fi
            
            successful_runs=$((successful_runs + 1))
        else
            echo "      Run $run failed"
        fi
    done
    
    # Calculate averages
    local avg_total_time=$(calculate_average "${total_times[@]}")
    local avg_comm_time=$(calculate_average "${comm_times[@]}")
    local avg_comm_percent=$(calculate_average "${comm_percents[@]}")
    
    # Inner product count is usually integer, round after calculating average
    local avg_inner_count=$(calculate_average "${inner_counts[@]}")
    if [ "$avg_inner_count" != "N/A" ]; then
        avg_inner_count=$(printf "%.0f" "$avg_inner_count")
    fi
    
    # Convergence rate
    local convergence_rate="$converged_count/$successful_runs"
    
    # Store results in temporary array (for final display)
    RESULTS["$algorithm,$s_param,$processes,$matrix_name"]="$avg_total_time,$avg_comm_time,$avg_comm_percent,$avg_inner_count,$convergence_rate"
    
    echo "    Completed $NUM_RUNS runs, averages calculated"
}

# Store results in associative array
declare -A RESULTS

# Main test loop
for np in "${PROCESSES[@]}"; do
    echo "Testing process count: $np"
    echo "  Start time: $(date +'%H:%M:%S')"
    
    # Get corresponding matrix and RHS files
    config=${WEAK_SCALING_CONFIG[$np]}
    matrix_file=$(echo "$config" | cut -d':' -f1)
    rhs_file=$(echo "$config" | cut -d':' -f2)
    matrix_name=$(basename "$matrix_file" .mtx | sed 's/^[0-9]*p_//')
    
    echo "  Using matrix: $matrix_name ($(echo "$matrix_file" | sed 's/.*\///g'))"
    
    # Test Classical GMRES
    echo "  Testing Classical GMRES (s=1)..."
    classical_log="/tmp/classical_${matrix_name}_np${np}"
    run_multiple_tests "Classical" 1 $np "$matrix_file" "$rhs_file" "$classical_log"
    
    # Test CA-GMRES
    echo "  Testing CA-GMRES (s=$S_VALUE)..."
    ca_log="/tmp/ca_gmres_s${S_VALUE}_${matrix_name}_np${np}"
    run_multiple_tests "CA-GMRES" $S_VALUE $np "$matrix_file" "$rhs_file" "$ca_log"
    
    echo "  End time: $(date +'%H:%M:%S')"
    echo ""
done

# Generate performance report
echo "============================================================"
echo "               Weak Scaling Performance Analysis Results    "
echo "============================================================"

# Print formatted results table
echo "Algorithm  | S   | Procs  | Matrix      | Total(s)  | Comm(s)     | Comm(%)     | InnerProd | ConvRate"
echo "-----------|-----|--------|-------------|-----------|-------------|-------------|-----------|----------"

# Display results sorted by algorithm and process count
for algorithm in "Classical" "CA-GMRES"; do
    for np in "${PROCESSES[@]}"; do
        for key in "${!RESULTS[@]}"; do
            IFS=',' read -r alg s_val processes matrix_name <<< "$key"
            if [[ "$alg" == "$algorithm" && "$processes" == "$np" ]]; then
                IFS=',' read -r total_time comm_time comm_percent inner_count convergence_rate <<< "${RESULTS[$key]}"
                printf "%-10s | %3s | %6s | %-11s | %9s | %11s | %11s | %9s | %8s\n" \
                       "$algorithm" "$s_val" "$processes" "$matrix_name" "$total_time" "$comm_time" "$comm_percent" "$inner_count" "$convergence_rate"
                break
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "Test Complete!"
echo "============================================================"

echo ""
echo "Weak Scaling test summary:"
echo "- Test algorithms: Classical GMRES CA-GMRES(s=$S_VALUE)"
echo "- Test process counts: ${PROCESSES[*]}"
echo "- Runs per configuration: $NUM_RUNS"
echo "- Total test configurations: $((${#PROCESSES[@]} * 2))"
echo ""

exit 0
#!/bin/bash

# Configuration parameters
NUM_RUNS=5  # Number of runs for averaging
S_VALUE=8   # Fixed s-value for CA-GMRES
 

# Test configuration
MATRIX="matrices/strongScaling/s3dkt3m2.mtx"
RHS="matrices/strongScaling/s3dkt3m2_rhs.mtx"

ITERATIONS=30
PRECOND=1
RESTARTS=3

# Process numbers
PROCESSES=(1 4 8 12 16 20)

echo "============================================================"
echo "                strongScaling Performance Analysis"
echo "============================================================"
echo "Test Matrix: s3dkt3m2.mtx (90,448 x 90,448, 1,921,942 non-zeros)"
echo "GMRES Parameters: iterations=$ITERATIONS, restarts=$RESTARTS, precond=$PRECOND"
echo "Number of runs: $NUM_RUNS  will calculate average "
echo "CA-GMRES s-value: $S_VALUE"
echo "Starting tests..."
echo ""

# Function: Execute single test
run_test() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    
    mpirun -np $processes ./gmres "$MATRIX" "$RHS" $ITERATIONS $PRECOND $RESTARTS $s_param >/dev/null 2>&1
    return $?
}

# Function: Extract performance data
extract_performance_data() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    
    local output=$(mpirun -np $processes ./gmres "$MATRIX" "$RHS" $ITERATIONS $PRECOND $RESTARTS $s_param 2>/dev/null)
    
    local total_time=$(echo "$output" | grep "Total Execution Time:" | awk '{print $4}' | tail -1)
    local comm_time=$(echo "$output" | grep "Total Communication Time:" | awk '{print $4}' | tail -1)
    # 专门从通信时间行提取百分比，避免取到其他行的百分比
    local comm_percent=$(echo "$output" | grep "Total Communication Time:" | sed -n 's/.*(\([0-9.]*\)%)/\1/p')
    local inner_count=$(echo "$output" | grep "Inner Product Operations:" | awk '{print $4}' | tail -1)
    local converged=$(echo "$output" | grep "converged successfully" >/dev/null && echo "Yes" || echo "No")
    
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

# Function: Run multiple tests and calculate averages
run_multiple_tests() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    
    echo "    Running $algorithm (s=$s_param) - $NUM_RUNS tests:"
    
    # Store results from each run
    local total_times=()
    local comm_times=()
    local comm_percents=()
    local inner_counts=()
    local converged_count=0
    local successful_runs=0
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "      Run $run..."
        echo "       mpirun -np $processes ./gmres $MATRIX $RHS $ITERATIONS $PRECOND $RESTARTS $s_param"
        
        if run_test "$algorithm" $s_param $processes; then
            echo "        Success"
            # Extract data
            local result=$(extract_performance_data "$algorithm" $s_param $processes)
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
            echo "        Failed"
        fi
    done
    
    # Calculate averages
    local avg_total_time=$(calculate_average "${total_times[@]}")
    local avg_comm_time=$(calculate_average "${comm_times[@]}")
    local avg_comm_percent=$(calculate_average "${comm_percents[@]}")
    
    # Inner product count is usually integer, round after averaging
    local avg_inner_count=$(calculate_average "${inner_counts[@]}")
    if [ "$avg_inner_count" != "N/A" ]; then
        avg_inner_count=$(printf "%.0f" "$avg_inner_count")
    fi
    
    # Convergence rate
    local convergence_rate="$converged_count/$successful_runs"
    
    # Store results for final display
    RESULTS["$algorithm,$s_param,$processes"]="$avg_total_time,$avg_comm_time,$avg_comm_percent,$avg_inner_count,$convergence_rate"
    
    echo "    Completed $NUM_RUNS runs, averages calculated"
}

# Storage array for results
declare -A RESULTS

# Main test loop
for np in "${PROCESSES[@]}"; do
    echo "Testing process count: $np"
    echo "  Start time: $(date +'%H:%M:%S')"
    
    # Test Classical GMRES
    echo "  Testing Classical GMRES (s=1)..."
    run_multiple_tests "Classical" 1 $np
    
    # Test CA-GMRES
    echo "  Testing CA-GMRES (s=$S_VALUE)..."
    run_multiple_tests "CA-GMRES" $S_VALUE $np
    
    echo "  End time: $(date +'%H:%M:%S')"
    echo ""
done

# Generate performance report
echo "============================================================"
echo "               strongScaling Performance  Analysis Results  "
echo "============================================================"

# Print formatted results table
echo "Algorithm  | S   | Procs  | Total(s)  | Comm(s)     | Comm(%)     | InnerProd | Conv Rate"
echo "-----------|-----|--------|-----------|-------------|-------------|-----------|----------"

# Display Classical results first, then CA-GMRES
for algorithm in "Classical" "CA-GMRES"; do
    for np in "${PROCESSES[@]}"; do
        for key in "${!RESULTS[@]}"; do
            IFS=',' read -r alg s_val processes <<< "$key"
            if [[ "$alg" == "$algorithm" && "$processes" == "$np" ]]; then
                IFS=',' read -r total_time comm_time comm_percent inner_count convergence_rate <<< "${RESULTS[$key]}"
                printf "%-10s | %3s | %6s | %9s | %11s | %11s | %9s | %8s\n" \
                       "$algorithm" "$s_val" "$processes" "$total_time" "$comm_time" "$comm_percent" "$inner_count" "$convergence_rate"
                break
            fi
        done
    done
done



echo ""
echo "============================================================"
echo "Test Complete!"
echo "============================================================"

# Generate test summary
echo ""
echo "Test Summary:"
echo "- Test algorithms: Classical GMRES CA-GMRES(s=$S_VALUE)"
echo "- Test process counts: ${PROCESSES[*]}"
echo "- Runs per configuration: $NUM_RUNS"
echo "- Matrix size: 90,448 × 90,448"
 

exit 0
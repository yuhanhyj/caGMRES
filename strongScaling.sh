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
echo "Number of runs: $NUM_RUNS (will calculate average)"
echo "CA-GMRES s-value: $S_VALUE"
echo "Data source: PERFORMANCE TABLE"
echo "Comm % calculation: (Average Comm Time / Average Total Time) × 100"
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

# Function: Extract performance data from PERFORMANCE TABLE
extract_performance_data() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    
    local output=$(mpirun -np $processes ./gmres "$MATRIX" "$RHS" $ITERATIONS $PRECOND $RESTARTS $s_param 2>/dev/null)
    
    # Extract data from PERFORMANCE TABLE
    local table_line=$(echo "$output" | grep -A 20 "PERFORMANCE TABLE" | grep "^\s*$processes\s*|" | tail -1)
    
    if [ -n "$table_line" ]; then
        # Parse the table line
        local total_time=$(echo "$table_line" | awk -F'|' '{print $2}' | xargs)
        local comm_time=$(echo "$table_line" | awk -F'|' '{print $3}' | xargs)
        local inner_count=$(echo "$table_line" | awk -F'|' '{print $5}' | xargs)
    else
        local total_time="N/A"
        local comm_time="N/A"
        local inner_count="N/A"
    fi
    
    local converged=$(echo "$output" | grep "converged successfully" >/dev/null && echo "Yes" || echo "No")
    
    # Handle empty values
    total_time=${total_time:-"N/A"}
    comm_time=${comm_time:-"N/A"}
    inner_count=${inner_count:-"N/A"}
    
    echo "$total_time $comm_time $inner_count $converged"
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
        echo "scale=6; $sum / $count" | bc -l
    else
        echo "N/A"
    fi
}

# Function: Run multiple tests and calculate averages
run_multiple_tests() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    
    echo "    Running $algorithm (s=$s_param) with $processes processes - $NUM_RUNS tests:"
    
    # Store results from each run
    local total_times=()
    local comm_times=()
    local inner_counts=()
    local converged_count=0
    local successful_runs=0
    
    for run in $(seq 1 $NUM_RUNS); do
        echo -n "      Run $run... "
        
        if run_test "$algorithm" $s_param $processes; then
            echo "Success"
            
            # Extract data from PERFORMANCE TABLE
            local result=$(extract_performance_data "$algorithm" $s_param $processes)
            read -r total_time comm_time inner_count converged <<< "$result"
            
            # Collect data if valid
            if [ "$total_time" != "N/A" ]; then
                total_times+=("$total_time")
            fi
            if [ "$comm_time" != "N/A" ]; then
                comm_times+=("$comm_time")
            fi
            if [ "$inner_count" != "N/A" ]; then
                inner_counts+=("$inner_count")
            fi
            if [ "$converged" = "Yes" ]; then
                converged_count=$((converged_count + 1))
            fi
            
            successful_runs=$((successful_runs + 1))
        else
            echo "Failed"
        fi
    done
    
    # Calculate average Total Time
    local avg_total_time=$(calculate_average "${total_times[@]}")
    
    # Calculate average Comm Time  
    local avg_comm_time=$(calculate_average "${comm_times[@]}")
    
    # Calculate Comm % = (avg_comm_time / avg_total_time) * 100
    local avg_comm_percent
    if [[ "$avg_total_time" != "N/A" && "$avg_comm_time" != "N/A" && "$avg_total_time" != "0" ]]; then        
        avg_comm_percent=$(echo "scale=4; ($avg_comm_time / $avg_total_time) * 100" | bc -l)
    else
        avg_comm_percent="N/A"
    fi
    
    # Calculate average inner product count
    local avg_inner_count=$(calculate_average "${inner_counts[@]}")
    if [ "$avg_inner_count" != "N/A" ]; then
        avg_inner_count=$(printf "%.0f" "$avg_inner_count")
    fi
    
    # Calculate convergence rate
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
echo "               strongScaling Performance Analysis Results  "
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

echo ""
echo "Test Summary:"
echo "- Data source: PERFORMANCE TABLE"
echo "- Comm % calculation: (Average Comm Time / Average Total Time) × 100"
echo "- Test algorithms: Classical GMRES, CA-GMRES(s=$S_VALUE)"
echo "- Test process counts: ${PROCESSES[*]}"
echo "- Runs per configuration: $NUM_RUNS (averaged)"
echo "- Matrix size: 90,448 × 90,448"
 
exit 0

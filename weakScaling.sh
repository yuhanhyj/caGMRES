#!/bin/bash

NUM_RUNS=10

TEST_CLASSICAL=true
TEST_CA=true
S_VALUE=8

# GMRES parameters
ITERATIONS=500
PRECOND=0
RESTARTS=100

# Process configuration
PROCESSES=(1 2 4 8 16) 

GRID_SIZE=100  
N_LOCAL=$((GRID_SIZE * GRID_SIZE))  # 10000 unknowns per processor
NNZ_LOCAL=$((5 * N_LOCAL - 4 * GRID_SIZE))  # ~49600 NNZ per processor for 5-point stencil

echo "============================================================"
echo "               Weak Scaling Performance Test                 "
echo "============================================================"
echo ""
echo "Process counts: ${PROCESSES[*]}"
echo "Problem size per processor: ${GRID_SIZE}x${GRID_SIZE} grid"
echo "Runs per test: $NUM_RUNS"
echo ""
echo "------------------------------------------------------------"

# Function: Get matrix files for process count
get_matrix_files() {
    local np=$1
    echo "matrices/weakScaling/P${np}.mtx matrices/weakScaling/P${np}_rhs.mtx"
}

# Function: Execute single test
run_test() {
    local algorithm=$1
    local s_param=$2
    local processes=$3
    local matrix_file=$4
    local rhs_file=$5
    local log_file=$6
    
    mpirun -np $processes ./gmres "$matrix_file" "$rhs_file" $ITERATIONS $PRECOND $RESTARTS $s_param > "$log_file" 2>&1
    return $?
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
    local inner_count=$(grep "Inner Product Operations:" "$log_file" | awk '{print $4}' | tail -1)
    local iterations=$(grep -E "converged successfully after|iterations:" "$log_file" | sed 's/.*after \([0-9]*\).*/\1/' | tail -1)
    
    local converged="No"
    if grep -q "converged successfully" "$log_file"; then
        converged="Yes"
    fi
    
    total_time=${total_time:-"N/A"}
    comm_time=${comm_time:-"N/A"}
    inner_count=${inner_count:-"N/A"}
    iterations=${iterations:-"N/A"}
    
    echo "$total_time $comm_time $inner_count $iterations $converged"
}

# Function: Calculate statistics
calculate_statistics() {
    local -a values=("$@")
    local n=${#values[@]}
    
    if [ $n -eq 0 ]; then
        echo "N/A N/A N/A"
        return
    fi
    
    # Calculate mean
    local sum=0
    for val in "${values[@]}"; do
        sum=$(echo "$sum + $val" | bc -l)
    done
    local mean=$(echo "scale=6; $sum / $n" | bc -l)
    
    # Calculate standard deviation
    local sq_diff_sum=0
    for val in "${values[@]}"; do
        local diff=$(echo "$val - $mean" | bc -l)
        local sq_diff=$(echo "$diff * $diff" | bc -l)
        sq_diff_sum=$(echo "$sq_diff_sum + $sq_diff" | bc -l)
    done
    local variance=$(echo "scale=6; $sq_diff_sum / $n" | bc -l)
    local std_dev=$(echo "scale=6; sqrt($variance)" | bc -l)
    
    # Calculate 95% confidence interval
    local std_err=$(echo "scale=6; $std_dev / sqrt($n)" | bc -l)
    local ci_95=$(echo "scale=6; 1.96 * $std_err" | bc -l)
    
    echo "$mean $std_dev $ci_95"
}

# Main test loop
echo "Starting weak scaling tests..."
echo ""

# Store all raw data for statistical analysis
declare -A all_times_classical
declare -A all_times_ca
declare -A all_comm_classical
declare -A all_comm_ca
declare -A all_eff_classical
declare -A all_eff_ca
declare -A all_iter_classical
declare -A all_iter_ca
declare -A all_inner_classical
declare -A all_inner_ca

baseline_classical=""
baseline_ca=""

for np in "${PROCESSES[@]}"; do
    echo "Testing with $np process(es)..."
    
    # Calculate problem dimensions for weak scaling
    total_unknowns=$((N_LOCAL * np))
    total_nnz=$((NNZ_LOCAL * np + (np - 1) * GRID_SIZE * 2 * 100))  # Including interface
    interface_elements=$((np > 1 ? (np - 1) * GRID_SIZE * 2 : 0))
    
    # Get matrix files
    files=$(get_matrix_files $np)
    matrix_file=$(echo $files | cut -d' ' -f1)
    rhs_file=$(echo $files | cut -d' ' -f2)
    
    # Arrays to store run data
    classical_times=()
    classical_comm=()
    classical_iter=()
    classical_inner=()
    ca_times=()
    ca_comm=()
    ca_iter=()
    ca_inner=()
    
    # Test Classical GMRES
    if [ "$TEST_CLASSICAL" = true ]; then
        echo "  Classical GMRES:"
        
        for run in $(seq 1 $NUM_RUNS); do
            log_file="/tmp/classical_P${np}_run${run}.log"
            
            if run_test "Classical" 1 $np "$matrix_file" "$rhs_file" "$log_file"; then
                result=$(extract_performance_data "$log_file")
                read -r time comm inner iter conv <<< "$result"
                
                if [ "$time" != "N/A" ]; then
                    classical_times+=("$time")
                    [ "$comm" != "N/A" ] && classical_comm+=("$comm")
                    [ "$iter" != "N/A" ] && classical_iter+=("$iter")
                    [ "$inner" != "N/A" ] && classical_inner+=("$inner")
                fi
            fi
        done
        
        # Calculate statistics for Classical
        if [ ${#classical_times[@]} -gt 0 ]; then
            stats=$(calculate_statistics "${classical_times[@]}")
            read -r mean_time std_time ci_time <<< "$stats"
            
            # Store baseline for P=1
            if [ "$np" = "1" ]; then
                baseline_classical="$mean_time"
            fi
            
            # Save to results arrays
            all_times_classical[$np]="$mean_time $std_time $ci_time"
            
            if [ ${#classical_comm[@]} -gt 0 ]; then
                comm_stats=$(calculate_statistics "${classical_comm[@]}")
                all_comm_classical[$np]="$comm_stats"
            fi
            
            if [ ${#classical_iter[@]} -gt 0 ]; then
                iter_stats=$(calculate_statistics "${classical_iter[@]}")
                all_iter_classical[$np]="$iter_stats"
            fi
            
            if [ ${#classical_inner[@]} -gt 0 ]; then
                inner_stats=$(calculate_statistics "${classical_inner[@]}")
                all_inner_classical[$np]="$inner_stats"
            fi
            
            # Calculate efficiency if baseline exists
            if [ -n "$baseline_classical" ] && [ "$mean_time" != "N/A" ]; then
                eff=$(echo "scale=4; $baseline_classical / $mean_time" | bc -l)
                all_eff_classical[$np]="$eff"
            fi
            
            echo "    Time: $(echo "scale=3; $mean_time * 1000" | bc -l) ms"
        fi
    fi
    
    # Test CA-GMRES
    if [ "$TEST_CA" = true ]; then
        echo "  CA-GMRES (s=$S_VALUE):"
        
        for run in $(seq 1 $NUM_RUNS); do
            log_file="/tmp/ca_P${np}_run${run}.log"
            
            if run_test "CA-GMRES" $S_VALUE $np "$matrix_file" "$rhs_file" "$log_file"; then
                result=$(extract_performance_data "$log_file")
                read -r time comm inner iter conv <<< "$result"
                
                if [ "$time" != "N/A" ]; then
                    ca_times+=("$time")
                    [ "$comm" != "N/A" ] && ca_comm+=("$comm")
                    [ "$iter" != "N/A" ] && ca_iter+=("$iter")
                    [ "$inner" != "N/A" ] && ca_inner+=("$inner")
                fi
            fi
        done
        
        # Calculate statistics for CA-GMRES
        if [ ${#ca_times[@]} -gt 0 ]; then
            stats=$(calculate_statistics "${ca_times[@]}")
            read -r mean_time std_time ci_time <<< "$stats"
            
            # Store baseline for P=1
            if [ "$np" = "1" ]; then
                baseline_ca="$mean_time"
            fi
            
            # Save to results arrays
            all_times_ca[$np]="$mean_time $std_time $ci_time"
            
            if [ ${#ca_comm[@]} -gt 0 ]; then
                comm_stats=$(calculate_statistics "${ca_comm[@]}")
                all_comm_ca[$np]="$comm_stats"
            fi
            
            if [ ${#ca_iter[@]} -gt 0 ]; then
                iter_stats=$(calculate_statistics "${ca_iter[@]}")
                all_iter_ca[$np]="$iter_stats"
            fi
            
            if [ ${#ca_inner[@]} -gt 0 ]; then
                inner_stats=$(calculate_statistics "${ca_inner[@]}")
                all_inner_ca[$np]="$inner_stats"
            fi
            
            # Calculate efficiency if baseline exists
            if [ -n "$baseline_ca" ] && [ "$mean_time" != "N/A" ]; then
                eff=$(echo "scale=4; $baseline_ca / $mean_time" | bc -l)
                all_eff_ca[$np]="$eff"
            fi
            
            echo "    Time: $(echo "scale=3; $mean_time * 1000" | bc -l) ms"
        fi
    fi
    
    echo ""
done

# Generate final report with statistics
echo ""
echo "============================================================"
echo "                         RESULTS                            "
echo "============================================================"
echo ""

# Display Classical GMRES results
if [ "$TEST_CLASSICAL" = true ]; then
    echo "Classical GMRES:"
    echo "----------------------------------------------------------------------------------------------------"
    echo "P  | Time(ms)        | Comm(ms)       | Iterations     | Inner Products | Efficiency    | Speedup"
    echo "----------------------------------------------------------------------------------------------------"
    
    for np in "${PROCESSES[@]}"; do
        if [[ -n "${all_times_classical[$np]}" ]]; then
            read -r mean_time std_time ci_time <<< "${all_times_classical[$np]}"
            
            # Convert to milliseconds
            mean_ms=$(echo "scale=3; $mean_time * 1000" | bc -l)
            std_ms=$(echo "scale=3; $std_time * 1000" | bc -l)
            
            # Get communication time if available
            comm_str="N/A"
            if [[ -n "${all_comm_classical[$np]}" ]]; then
                read -r comm_mean comm_std comm_ci <<< "${all_comm_classical[$np]}"
                comm_mean_ms=$(echo "scale=3; $comm_mean * 1000" | bc -l)
                comm_std_ms=$(echo "scale=3; $comm_std * 1000" | bc -l)
                comm_str=$(printf "%.2f±%.2f" "$comm_mean_ms" "$comm_std_ms")
            fi
            
            # Get efficiency
            eff_str="N/A"
            if [[ -n "${all_eff_classical[$np]}" ]]; then
                eff="${all_eff_classical[$np]}"
                eff_pct=$(echo "scale=1; $eff * 100" | bc -l)
                eff_str=$(printf "%.1f%%" "$eff_pct")
            fi
            
            # Calculate speedup (for weak scaling, should ideally be 1.0)
            speedup="N/A"
            if [ -n "$baseline_classical" ]; then
                speedup=$(echo "scale=2; $baseline_classical / $mean_time * $np" | bc -l)
            fi
            
            # Get iterations
            iter_str="N/A"
            if [[ -n "${all_iter_classical[$np]}" ]]; then
                read -r iter_mean iter_std iter_ci <<< "${all_iter_classical[$np]}"
                iter_str=$(printf "%.0f±%.0f" "$iter_mean" "$iter_std")
            fi
            
            # Get inner products
            inner_str="N/A"
            if [[ -n "${all_inner_classical[$np]}" ]]; then
                read -r inner_mean inner_std inner_ci <<< "${all_inner_classical[$np]}"
                inner_str=$(printf "%.0f±%.0f" "$inner_mean" "$inner_std")
            fi
            
            printf "%-2s | %7.3f±%-6.3f | %-14s | %-14s | %-14s | %-13s | %.2f\n" \
                   "$np" "$mean_ms" "$std_ms" "$comm_str" "$iter_str" "$inner_str" "$eff_str" "$speedup"
        fi
    done
    echo ""
fi

# Display CA-GMRES results
if [ "$TEST_CA" = true ]; then
    echo "CA-GMRES (s=$S_VALUE):"
    echo "----------------------------------------------------------------------------------------------------"
    echo "P  | Time(ms)        | Comm(ms)       | Iterations     | Inner Products | Efficiency    | Speedup"
    echo "----------------------------------------------------------------------------------------------------"
    
    for np in "${PROCESSES[@]}"; do
        if [[ -n "${all_times_ca[$np]}" ]]; then
            read -r mean_time std_time ci_time <<< "${all_times_ca[$np]}"
            
            # Convert to milliseconds
            mean_ms=$(echo "scale=3; $mean_time * 1000" | bc -l)
            std_ms=$(echo "scale=3; $std_time * 1000" | bc -l)
            
            # Get communication time if available
            comm_str="N/A"
            if [[ -n "${all_comm_ca[$np]}" ]]; then
                read -r comm_mean comm_std comm_ci <<< "${all_comm_ca[$np]}"
                comm_mean_ms=$(echo "scale=3; $comm_mean * 1000" | bc -l)
                comm_std_ms=$(echo "scale=3; $comm_std * 1000" | bc -l)
                comm_str=$(printf "%.2f±%.2f" "$comm_mean_ms" "$comm_std_ms")
            fi
            
            # Get efficiency
            eff_str="N/A"
            if [[ -n "${all_eff_ca[$np]}" ]]; then
                eff="${all_eff_ca[$np]}"
                eff_pct=$(echo "scale=1; $eff * 100" | bc -l)
                eff_str=$(printf "%.1f%%" "$eff_pct")
            fi
            
            # Calculate speedup (for weak scaling, should ideally be 1.0)
            speedup="N/A"
            if [ -n "$baseline_ca" ]; then
                speedup=$(echo "scale=2; $baseline_ca / $mean_time * $np" | bc -l)
            fi
            
            # Get iterations
            iter_str="N/A"
            if [[ -n "${all_iter_ca[$np]}" ]]; then
                read -r iter_mean iter_std iter_ci <<< "${all_iter_ca[$np]}"
                iter_str=$(printf "%.0f±%.0f" "$iter_mean" "$iter_std")
            fi
            
            # Get inner products
            inner_str="N/A"
            if [[ -n "${all_inner_ca[$np]}" ]]; then
                read -r inner_mean inner_std inner_ci <<< "${all_inner_ca[$np]}"
                inner_str=$(printf "%.0f±%.0f" "$inner_mean" "$inner_std")
            fi
            
            printf "%-2s | %7.3f±%-6.3f | %-14s | %-14s | %-14s | %-13s | %.2f\n" \
                   "$np" "$mean_ms" "$std_ms" "$comm_str" "$iter_str" "$inner_str" "$eff_str" "$speedup"
        fi
    done
    echo ""
fi

exit 0
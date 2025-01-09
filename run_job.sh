#!/bin/bash

# Log file
LOG_FILE="execution_times.log"

# Number of times to sample
repeats=5

# Clear the log file and write header
echo "Execution Time Log" > "$LOG_FILE"
echo "===================" >> "$LOG_FILE"

# Loop over num_threads from 1 to 30
for num_threads in {1..30}; do
    echo "Running simulation with num_threads=$num_threads..."
    total=0
    for ((count=1; count<=repeats; count++)); do
        # Get the start time in milliseconds
        start=$(date +%s%3N)
        
        # Run the Julia simulation
        julia -t "$num_threads" simulate_well.jl

        # Get the end time in milliseconds
        end=$(date +%s%3N)

        # Calculate elapsed time for this run
        elapsed=$((end - start))
        total=$((total + elapsed))
    done

    # Calculate the average elapsed time
    average=$((total / repeats))
    log_entry="num_threads=$num_threads: $average ms"

    echo "$log_entry"
    echo "$log_entry" >> "$LOG_FILE"
done

echo "Execution times logged in $LOG_FILE"
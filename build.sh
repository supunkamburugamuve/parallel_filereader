read(3, "\217-\243C\275\\~\\\2453\343\247\234#\215\355\304\352\3\234Zz\351\rR\\\237\343\250\377\243\275"..., 62277111808) = 2147479552 <0.859589>
#!/bin/bash

# Compile the parallel file reader
g++ -std=c++17 -O2 -pthread -o parallel_reader reader.cpp

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo ""
    echo "Usage: ./parallel_reader <filename> [num_threads]"
    echo ""
    echo "Creating a test file (100MB)..."
    dd if=/dev/urandom of=test_file.bin bs=1M count=100 2>/dev/null
    echo "Test file created: test_file.bin"
    echo ""
    echo "Running parallel reader with 2 threads, 1GB chunks and O_DIRECT..."
    ./parallel_reader test_file.bin 2 1000000 1
else
    echo "Compilation failed!"
    exit 1
fi
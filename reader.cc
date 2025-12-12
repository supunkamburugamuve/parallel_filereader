#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <cstring>
#include <sys/stat.h>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>

class ParallelFileReader {
private:
    std::string filename;
    size_t file_size;
    char* buffer;
    size_t num_threads;
    size_t read_chunk_size;  // Size of each read operation (e.g., 1MB)
    size_t block_size = 4096;  // Filesystem block size for O_DIRECT alignment
    bool use_odirect;  // Whether to use O_DIRECT for file I/O

    // Get file size
    size_t getFileSize(const std::string& filename) {
        struct stat stat_buf;
        int rc = stat(filename.c_str(), &stat_buf);
        return rc == 0 ? stat_buf.st_size : 0;
    }

    // Thread worker function to read a portion of the file in configurable chunks
    void readChunk(size_t thread_id, size_t section_start, size_t section_size) {
        auto thread_start = std::chrono::high_resolution_clock::now();

        int flags = O_RDONLY;
        if (use_odirect) {
            flags |= O_DIRECT;
        }
        int fd = open(filename.c_str(), flags);
        if (fd == -1) {
            std::cerr << "Thread " << thread_id << ": Failed to open file\n";
            return;
        }

        size_t bytes_completed = 0;

        if (use_odirect) {
            // O_DIRECT path: requires aligned reads
            // Allocate a temporary aligned buffer for reading
            auto temp_alloc_start = std::chrono::high_resolution_clock::now();
            char* temp_buffer;
            if (posix_memalign(reinterpret_cast<void**>(&temp_buffer), block_size, read_chunk_size) != 0) {
                std::cerr << "Thread " << thread_id << ": Failed to allocate aligned temp buffer\n";
                close(fd);
                return;
            }
            auto temp_alloc_end = std::chrono::high_resolution_clock::now();
            auto temp_alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(temp_alloc_end - temp_alloc_start);
            std::cout << "Thread " << thread_id << " temp buffer allocation: " << temp_alloc_duration.count() << " μs\n";

            size_t bytes_processed = 0;
            size_t current_offset = section_start;

            // Read the section in aligned chunks
            while (bytes_processed < section_size) {
                // Calculate aligned read parameters
                size_t aligned_offset = (current_offset / block_size) * block_size;
                size_t offset_in_block = current_offset - aligned_offset;
                size_t remaining_in_section = section_size - bytes_processed;
                size_t bytes_to_read = std::min(read_chunk_size, remaining_in_section + offset_in_block);

                // Ensure read size is multiple of block_size
                bytes_to_read = ((bytes_to_read + block_size - 1) / block_size) * block_size;

                // Seek to aligned position
                if (lseek(fd, aligned_offset, SEEK_SET) == -1) {
                    std::cerr << "Thread " << thread_id << ": Seek error at offset "
                              << aligned_offset << "\n";
                    break;
                }

                // Read aligned chunk into temp buffer
                ssize_t actually_read = ::read(fd, temp_buffer, bytes_to_read);

                if (actually_read == -1) {
                    std::cerr << "Thread " << thread_id << ": Read error at offset "
                              << aligned_offset << "\n";
                    break;
                }

                // Copy only the needed portion to the main buffer
                size_t bytes_to_copy = std::min(static_cast<size_t>(actually_read) - offset_in_block, remaining_in_section);
                std::memcpy(buffer + current_offset, temp_buffer + offset_in_block, bytes_to_copy);

                bytes_processed += bytes_to_copy;
                current_offset += bytes_to_copy;

                // Break if we reach EOF or read less than expected
                if (actually_read < static_cast<ssize_t>(bytes_to_read)) {
                    break;
                }
            }

            free(temp_buffer);
            bytes_completed = bytes_processed;
        } else {
            // Regular I/O path: simpler logic
            size_t bytes_read = 0;
            size_t current_offset = section_start;

            // Read the section in chunks of read_chunk_size
            while (bytes_read < section_size) {
                size_t bytes_to_read = std::min(read_chunk_size, section_size - bytes_read);

                // Seek to the current position
                if (lseek(fd, current_offset, SEEK_SET) == -1) {
                    std::cerr << "Thread " << thread_id << ": Seek error at offset "
                              << current_offset << "\n";
                    break;
                }

                // Read chunk directly into buffer at correct offset
                ssize_t actually_read = ::read(fd, buffer + current_offset, bytes_to_read);

                if (actually_read == -1) {
                    std::cerr << "Thread " << thread_id << ": Read error at offset "
                              << current_offset << "\n";
                    break;
                }

                bytes_read += actually_read;
                current_offset += actually_read;

                // If we couldn't read as much as expected and we're not at EOF, something's wrong
                if (actually_read < static_cast<ssize_t>(bytes_to_read) && actually_read > 0) {
                    std::cerr << "Thread " << thread_id << ": Short read at offset "
                              << current_offset << "\n";
                    break;
                }

                // Break if we reach EOF
                if (actually_read == 0) {
                    break;
                }
            }

            bytes_completed = bytes_read;
        }

        close(fd);

        auto thread_end = std::chrono::high_resolution_clock::now();
        auto thread_duration = std::chrono::duration_cast<std::chrono::milliseconds>(thread_end - thread_start);

        std::cout << "Thread " << thread_id << " completed: processed " << bytes_completed
                  << " bytes in " << (bytes_completed + read_chunk_size - 1) / read_chunk_size
                  << " chunks (" << thread_duration.count() << " ms)\n";
    }

public:
    ParallelFileReader(const std::string& fname,
                      size_t threads = std::thread::hardware_concurrency(),
                      size_t chunk_size = 1024 * 1024,  // Default 1MB chunks
                      bool odirect = false)  // Default: don't use O_DIRECT
        : filename(fname), num_threads(threads), read_chunk_size(chunk_size), buffer(nullptr), use_odirect(odirect) {
        // Ensure read_chunk_size is a multiple of block_size for O_DIRECT
        if (use_odirect && read_chunk_size % block_size != 0) {
            read_chunk_size = ((read_chunk_size + block_size - 1) / block_size) * block_size;
        }
        file_size = getFileSize(filename);
        if (file_size == 0) {
            throw std::runtime_error("File not found or empty: " + filename);
        }
    }

    ~ParallelFileReader() {
        if (buffer != nullptr) {
            if (use_odirect) {
                free(buffer);
            } else {
                delete[] buffer;
            }
        }
    }

    // Main function to read file in parallel
    void read() {

        // Allocate buffer equal to file size (aligned if using O_DIRECT)
        auto alloc_start = std::chrono::high_resolution_clock::now();
        if (use_odirect) {
            if (posix_memalign(reinterpret_cast<void**>(&buffer), block_size, file_size) != 0) {
                throw std::runtime_error("Failed to allocate aligned buffer");
            }
        } else {
            buffer = new char[file_size];
        }
        auto alloc_end = std::chrono::high_resolution_clock::now();
        auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start);
        std::cout << "Buffer allocation: " << alloc_duration.count() << " μs\n";

        // Parallel memset using dedicated threads
        auto memset_start = std::chrono::high_resolution_clock::now();

        size_t memset_chunk_size = file_size / num_threads;
        size_t memset_remainder = file_size % num_threads;

        std::vector<std::thread> memset_threads;
        size_t current_memset_offset = 0;

        for (size_t i = 0; i < num_threads; ++i) {
            size_t current_chunk_size = memset_chunk_size;

            // Give the last thread any remaining bytes
            if (i == num_threads - 1) {
                current_chunk_size += memset_remainder;
            }

            memset_threads.emplace_back([this, i, current_memset_offset, current_chunk_size]() {
                auto thread_memset_start = std::chrono::high_resolution_clock::now();
                std::memset(buffer + current_memset_offset, 0, current_chunk_size);
                auto thread_memset_end = std::chrono::high_resolution_clock::now();
                auto thread_memset_duration = std::chrono::duration_cast<std::chrono::milliseconds>(thread_memset_end - thread_memset_start);
                std::cout << "Memset thread " << i << ": " << current_chunk_size << " bytes in " << thread_memset_duration.count() << " ms\n";
            });

            current_memset_offset += current_chunk_size;
        }

        // Wait for all memset threads to complete
        for (auto& thread : memset_threads) {
            thread.join();
        }

        auto memset_end = std::chrono::high_resolution_clock::now();
        auto memset_duration = std::chrono::duration_cast<std::chrono::milliseconds>(memset_end - memset_start);
        std::cout << "Parallel memset total: " << memset_duration.count() << " ms\n";

        std::cout << "Reading file: " << filename << "\n";
        std::cout << "File size: " << file_size << " bytes ("
                  << (file_size / (1024.0 * 1024.0)) << " MB)\n";
        std::cout << "Using " << num_threads << " threads\n";
        std::cout << "Read chunk size: " << read_chunk_size << " bytes ("
                  << (read_chunk_size / 1024.0) << " KB)\n";
        std::cout << "O_DIRECT: " << (use_odirect ? "enabled" : "disabled");
        if (use_odirect) {
            std::cout << " (bypasses page cache - shows TRUE storage performance)";
        } else {
            std::cout << " (uses page cache - may show cached performance on repeat runs)";
        }
        std::cout << "\n";

        // Calculate chunk size for each thread
        size_t chunk_size = file_size / num_threads;
        size_t remainder = file_size % num_threads;

        std::vector<std::thread> threads;
        size_t current_offset = 0;

        auto start = std::chrono::high_resolution_clock::now();
        // Create and launch threads
        for (size_t i = 0; i < num_threads; ++i) {
            size_t current_chunk_size = chunk_size;

            // Give the last thread any remaining bytes
            if (i == num_threads - 1) {
                current_chunk_size += remainder;
            }

            threads.emplace_back(&ParallelFileReader::readChunk, this,
                               i, current_offset, current_chunk_size);

            current_offset += current_chunk_size;
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nRead completed in " << duration.count() << " ms\n";
        double throughput = (file_size / (1024.0 * 1024.0)) / (duration.count() / 1000.0);
        std::cout << "Throughput: " << throughput << " MB/s\n";
    }

    // Get the buffer (for verification or further processing)
    const char* getBuffer() const {
        return buffer;
    }

    size_t getFileSize() const {
        return file_size;
    }

    // Verify the read by comparing with sequential read
    bool verify() {
        std::cout << "\nVerifying parallel read...\n";

        // Use regular file I/O for verification (not O_DIRECT) to avoid alignment issues
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for verification\n";
            return false;
        }

        char* verify_buffer = new char[file_size];
        file.read(verify_buffer, file_size);
        file.close();

        bool success = file.gcount() == static_cast<std::streamsize>(file_size);
        bool match = success && (std::memcmp(buffer, verify_buffer, file_size) == 0);
        delete[] verify_buffer;

        if (match) {
            std::cout << "Verification PASSED: Parallel read matches sequential read\n";
        } else {
            std::cout << "Verification FAILED: Data mismatch detected\n";
        }

        return match;
    }
};

int main(int argc, char* argv[]) {
    try {
        std::string filename;
        size_t num_threads = std::thread::hardware_concurrency();
        size_t read_chunk_size = 1024 * 1024; // Default 1MB
        bool use_odirect = false; // Default: don't use O_DIRECT

        if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <filename> [num_threads] [read_chunk_size_KB] [use_odirect]\n";
            std::cout << "Example: " << argv[0] << " large_file.bin 8 1024 1\n";
            std::cout << "  - filename: file to read\n";
            std::cout << "  - num_threads: number of parallel threads (default: CPU cores)\n";
            std::cout << "  - read_chunk_size_KB: size of each read operation in KB (default: 1024 = 1MB)\n";
            std::cout << "  - use_odirect: 1 to use O_DIRECT, 0 to use regular I/O (default: 0)\n";
            return 1;
        }

        filename = argv[1];

        if (argc >= 3) {
            num_threads = std::stoul(argv[2]);
        }

        if (argc >= 4) {
            read_chunk_size = std::stoul(argv[3]) * 1024; // Convert KB to bytes
        }

        if (argc >= 5) {
            use_odirect = std::stoul(argv[4]) != 0;
        }

        if (num_threads == 0) {
            num_threads = 1;
        }

        if (read_chunk_size == 0) {
            read_chunk_size = 1024 * 1024; // Default to 1MB
        }

        ParallelFileReader reader(filename, num_threads, read_chunk_size, use_odirect);
        reader.read();

        // Optional: Verify the read
        reader.verify();

        // Optional: Print first few bytes
        std::cout << "\nFirst 64 bytes of buffer (hex):\n";
        const char* buf = reader.getBuffer();
        for (size_t i = 0; i < std::min(size_t(64), reader.getFileSize()); ++i) {
            printf("%02x ", static_cast<unsigned char>(buf[i]));
            if ((i + 1) % 16 == 0) std::cout << "\n";
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

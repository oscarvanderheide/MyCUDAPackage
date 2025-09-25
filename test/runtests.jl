using MyCUDAPackage
using CUDA
using Test

if !CUDA.functional()
    @error("CUDA is not functional. Cannot run the test.")
else
    println("Starting Test for MyCUDAPackage")

    # 1. SETUP: Define vector size and generate data on the CPU (Host)
    # -----------------------------------------------------------------
    println("1. Generating host data...")
    n = 1000
    T = Float32

    # Create two random vectors on the CPU
    a_host = rand(T, n)
    b_host = rand(T, n)
    c_ground_truth = a_host .+ b_host

    # CuArray() copies the host data to a new GPU array
    a_device = CuArray(a_host)
    b_device = CuArray(b_host)
    
    # Allocate an output array on the GPU.
    c_device = CUDA.zeros(T, n)

    # This is the call to our compiled CUDA code
    MyCUDAPackage.add_vectors!(c_device, a_device, b_device)

    # Array() copies the GPU data back to a new host array
    c_from_gpu = Array(c_device)
    
    # Test whether the vector addition was succesful
    @test c_from_gpu ≈ c_ground_truth
end

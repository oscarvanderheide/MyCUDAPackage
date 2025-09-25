using MyCUDAPackage
using CUDA
using Test

if !CUDA.functional()
    @error("CUDA is not functional. Cannot run the test.")
else
    @testset "Test for MyCUDAPackage" begin

        println("1. Generating data on CPU")
        n = 1000
        T = Float32

        # Create two random vectors on the CPU
        a_host = rand(T, n)
        b_host = rand(T, n)
        c_ground_truth = a_host .+ b_host

        println("2. Copying data to GPU")
        a_device = CuArray(a_host)
        b_device = CuArray(b_host)
        
        # Allocate an output array on the GPU.
        c_device = CUDA.zeros(T, n)

        println("3. Call CUDA kernel")
        MyCUDAPackage.add_vectors!(c_device, a_device, b_device)

        println("4. Copy result back to CPU")
        c_from_gpu = Array(c_device)
        
        println("5. Validate results")
        @test c_from_gpu ≈ c_ground_truth
    end
end

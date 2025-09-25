module MyCUDAPackage

using CUDA

# Check if the build script found a GPU and built the library
function __init__()
    if libcudakernels === nothing
        @warn "MyCUDAPackage was built without a CUDA-capable GPU. CUDA functionality will not be available."
    end
end

# Load the compiled library 
const deps_file = joinpath(@__DIR__, "..", "deps", "deps.jl")
if !isfile(deps_file)
    error("MyCUDAPackage is not properly built. Please run `using Pkg; Pkg.build(\"MyCUDAPackage\")`.")
end
include(deps_file)


# Julia wrapper for CUDA function 
"""
    add_vectors!(c, a, b)

Computes `c = a + b` on the GPU using a custom CUDA kernel.
`a`, `b`, and `c` must be `CuVector{Float32}` of the same length.
"""
function add_vectors!(c::CuVector{Float32}, a::CuVector{Float32}, b::CuVector{Float32})

    n = length(a)
    if length(b) != n || length(c) != n
        throw(ArgumentError("All vectors must have the same length."))
    end

    # ccall((function_name, library_path), return_type, (arg_types...), args...)
    ccall(
        (:add_vectors, libcudakernels), # Function name and library path
        Cvoid,                         # Return type (void)
        (CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32}, Cint), # Argument types
        a, b, c, n                     # Arguments
    )
    
    return c
end

end

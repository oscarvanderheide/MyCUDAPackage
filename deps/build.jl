using Pkg
using CUDA
using Libdl

const PKG_ROOT = dirname(@__DIR__)
const CUDA_SRC_DIR = joinpath(PKG_ROOT, "deps", "cuda_source")
const LIB_DIR = joinpath(PKG_ROOT, "deps", "lib")

function build_library()

    # Ensure the output directory exists
    mkpath(LIB_DIR)

    if !CUDA.functional()
        @warn "MyCUDAPackage: No functional CUDA setup found. Skipping library build."
        open(joinpath(PKG_ROOT, "deps", "deps.jl"), "w") do f
            write(f, "const libcudakernels = nothing\n")
        end
        return
    end

    # Find the g++ compiler on the system
    gxx_path = Sys.which("g++")
    if gxx_path === nothing
        @error "Could not find g++ compiler. Please install it (e.g., `sudo apt-get install build-essential`)."
        exit(1) 
    end
    # -----------------------

    # Define source and output paths
    source_file = joinpath(CUDA_SRC_DIR, "kernels.cu")
    lib_name = "libcudakernels.$(Libdl.dlext)"
    output_path = joinpath(LIB_DIR, lib_name)

    @info "Building CUDA library for MyCUDAPackage..."
    @info "Host C++ compiler (g++): $(gxx_path)"
    @info "Source file: $(source_file)"
    @info "Output library: $(output_path)"

    # Add GPU Architecture Flags
    arch_flags = try
        dev = CUDA.device()
        cap = CUDA.capability(dev)
        @info "Compiling for local GPU architecture: sm_$(cap.major)$(cap.minor)"
        "-gencode=arch=compute_$(cap.major)$(cap.minor),code=sm_$(cap.major)$(cap.minor)"
    catch e
        @warn "Could not detect local GPU architecture. Falling back to a default set."
        ["-gencode=arch=compute_70,code=sm_70",
         "-gencode=arch=compute_75,code=sm_75",
         "-gencode=arch=compute_86,code=sm_86"]
    end

    # Construct the nvcc command
    cmd = `nvcc -ccbin $(gxx_path) -allow-unsupported-compiler -shared -Xcompiler -fPIC $(arch_flags) -o $(output_path) $(source_file)`
    @info "Build command: $cmd"

    try
        run(cmd)
        @info "Successfully built $lib_name"
    catch e
        @error "Failed to build CUDA library. Please check your build log."
        rethrow(e)
    end

    # Write deps.jl file
    deps_file = joinpath(PKG_ROOT, "deps", "deps.jl")
    open(deps_file, "w") do f
        write(f, "const libcudakernels = \"$(escape_string(output_path))\"\n")
    end

    @info "MyCUDAPackage build process complete."
end

# Run the build function
build_library()
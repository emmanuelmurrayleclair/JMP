# All CUDA julia functions required to simulate and estimate model in Murray Leclair (2023, JMP)
# Author: Emmanuel Murray Leclair 
# (Latest version): August 2023


##############################################################
#----  DYNAMIC SIMULATION FUNCTIONS (VFI)               -----#
##############################################################

    #--------------------------------------------------
    #--------- ALL FUNCTIONS THAT USE CUDA (GPU) ------
    #--------------------------------------------------

### Functions that gets the Emax given value function grid and transition matrix
    # Main version
        function Emax_GPU_faster(Emax,vec_Wold,Π_transition,β)
            is = (blockIdx().x-1)*blockDim().x + threadIdx().x
            isp = (blockIdx().y-1)*blockDim().y + threadIdx().y
            t = (blockIdx().z-1)*blockDim().z + threadIdx().z

            @inbounds Emax[is,isp,t] = β*vec_Wold[isp,t]*Π_transition[is,isp];
            return 
        end
    #
    # Version with fuel price/productivity as state var
        function Emax_GPU_faster_alt(Emax_oe,Emax_oge,Emax_oce,Emax_ogce,vec_Wold,Π_transition_oe,Π_transition_oge,Π_transition_oce,Π_transition_ogce,β)
            is = (blockIdx().x-1)*blockDim().x + threadIdx().x
            isp = (blockIdx().y-1)*blockDim().y + threadIdx().y
            t = (blockIdx().z-1)*blockDim().z + threadIdx().z
            # Starting from (in order): oe, oge, oce, ogce
            @inbounds Emax_oe[is,isp,t] = β*vec_Wold[isp,t]*Π_transition_oe[is,isp];
            @inbounds Emax_oge[is,isp,t] = β*vec_Wold[isp,t]*Π_transition_oge[is,isp];
            @inbounds Emax_oce[is,isp,t] = β*vec_Wold[isp,t]*Π_transition_oce[is,isp];
            @inbounds Emax_ogce[is,isp,t] = β*vec_Wold[isp,t]*Π_transition_ogce[is,isp];
            return 
        end
    #
    # Version that vectorizes state space (demands more memory)
        function Emax_GPU_vec(Emax,vec_Wold,Π_transition,β,SIZE_GRID)
            is = (blockIdx().x-1)*blockDim().x + threadIdx().x

            temp1 = vec_Wold.*Π_transition[is,:];
            @inbounds Emax[is] = β*sum(temp1);
            return 
        end
    #
#

### Function that returns new choice-specific value functions given Emax 
    # Slow version, but demands less GPU memory
        function T_W_GPU(vec_Wnew,vec_π,Emax,κ_g,κ_c,γ_g,γ_c)
            is = (blockIdx().x-1)*blockDim().x + threadIdx().x
            # Compute new value function 
            @inbounds vec_Wnew[is,1] = vec_π[is,1]+Emax[4,is]+CUDA.log(CUDA.exp(Emax[1,is]-Emax[4,is])+CUDA.exp(- κ_g + Emax[2,is]-Emax[4,is])+CUDA.exp(- κ_c + Emax[3,is]-Emax[4,is]) + CUDA.exp(-κ_g - κ_c + Emax[4,is]-Emax[4,is]));
            @inbounds vec_Wnew[is,2] = vec_π[is,2]+Emax[4,is]+CUDA.log(γ_g+CUDA.exp(Emax[1,is]-Emax[4,is])+CUDA.exp(Emax[2,is]-Emax[4,is])+CUDA.exp(γ_g-κ_c + Emax[3,is]-Emax[4,is]) + CUDA.exp(-κ_c + Emax[4,is]-Emax[4,is]));
            @inbounds vec_Wnew[is,3] = vec_π[is,3]+Emax[4,is]+CUDA.log(γ_c+CUDA.exp(Emax[1,is]-Emax[4,is])+CUDA.exp(γ_c-κ_g+Emax[2,is]-Emax[4,is])+CUDA.exp(Emax[3,is]-Emax[4,is]) + CUDA.exp(-κ_g+Emax[4,is]-Emax[4,is]));
            @inbounds vec_Wnew[is,4] = vec_π[is,4]+Emax[4,is]+CUDA.log(γ_g+γ_c+CUDA.exp(Emax[1,is]-Emax[4,is])+CUDA.exp(γ_c + Emax[2,is]-Emax[4,is])+CUDA.exp(γ_g + Emax[3,is]-Emax[4,is]) + CUDA.exp(Emax[4,is]-Emax[4,is]));
            return
        end
    #
    # Faster version, but demands more GPU memory 
        function T_W_GPU_faster(vec_Wnew,vec_π,Emax,κ_g,κ_c,γ_g,γ_c,index)
            is = (blockIdx().x-1)*blockDim().x + threadIdx().x
            t = (blockIdx().y-1)*blockDim().y + threadIdx().y
            is_e = index[is];
            #is_e=is;
            # Compute new value function 
            # @inbounds vec_Wnew[is,1,t] = vec_π[is,1,t]+Emax[t,4,is]+CUDA.log(CUDA.exp(Emax[t,1,is]-Emax[t,4,is])+CUDA.exp(- κ_g + Emax[t,2,is]-Emax[t,4,is])+CUDA.exp(- κ_c + Emax[t,3,is]-Emax[t,4,is]) + CUDA.exp(-κ_g - κ_c));
            # @inbounds vec_Wnew[is,2,t] = vec_π[is,2,t]+Emax[t,4,is]+CUDA.log(CUDA.exp(γ_g+Emax[t,1,is]-Emax[t,4,is])+CUDA.exp(Emax[t,2,is]-Emax[t,4,is])+CUDA.exp(γ_g-κ_c + Emax[t,3,is]-Emax[t,4,is]) + CUDA.exp(-κ_c));
            # @inbounds vec_Wnew[is,3,t] = vec_π[is,3,t]+Emax[t,4,is]+CUDA.log(CUDA.exp(γ_c+Emax[t,1,is]-Emax[t,4,is])+CUDA.exp(γ_c-κ_g+Emax[t,2,is]-Emax[t,4,is])+CUDA.exp(Emax[t,3,is]-Emax[t,4,is]) + CUDA.exp(-κ_g));
            # @inbounds vec_Wnew[is,4,t] = vec_π[is,4,t]+Emax[t,4,is]+CUDA.log(CUDA.exp(γ_g+γ_c+Emax[1,is]-Emax[t,4,is])+CUDA.exp(γ_c + Emax[t,2,is]-Emax[t,4,is])+CUDA.exp(γ_g + Emax[t,3,is]-Emax[t,4,is]) + 1);
            @inbounds vec_Wnew[is,1,t] = vec_π[is,1,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(- κ_g + Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(- κ_c + Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_g - κ_c));
            @inbounds vec_Wnew[is,2,t] = vec_π[is,2,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_g+Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_g-κ_c + Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_c));
            @inbounds vec_Wnew[is,3,t] = vec_π[is,3,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_c+Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_c-κ_g+Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_g));
            @inbounds vec_Wnew[is,4,t] = vec_π[is,4,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_g+γ_c+Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_c + Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_g + Emax[t,3,is_e]-Emax[t,4,is_e]) + 1);
            return
        end
    #
    # Faster version allowing for fixed costs to depend on productivity (size)
        function T_W_GPU_faster_zfc(vec_Wnew,vec_π,vec_lnz,Emax,κ_g,κ_c,γ_g,γ_c,κ_z,γ_z,index)
            is = (blockIdx().x-1)*blockDim().x + threadIdx().x
            t = (blockIdx().y-1)*blockDim().y + threadIdx().y
            is_e = index[is];
            #is_e=is;
            # Compute new value function 
            @inbounds vec_Wnew[is,1,t] = vec_π[is,1,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(- κ_g - κ_z*vec_lnz[is,t] + Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(- κ_c - κ_z*vec_lnz[is,t] + Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_g - κ_c - κ_z*vec_lnz[is,t]));
            @inbounds vec_Wnew[is,2,t] = vec_π[is,2,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_g + γ_z*vec_lnz[is,t] + Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_g + γ_z*vec_lnz[is,t] -κ_c - κ_z*vec_lnz[is,t] + Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_c - κ_z*vec_lnz[is,t]));
            @inbounds vec_Wnew[is,3,t] = vec_π[is,3,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_c + γ_z*vec_lnz[is,t] + Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_c + γ_z*vec_lnz[is,t] -κ_g - κ_z*vec_lnz[is,t] +Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_g - κ_z*vec_lnz[is,t]));
            @inbounds vec_Wnew[is,4,t] = vec_π[is,4,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_g + γ_z*vec_lnz[is,t] + γ_c+ Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_c + γ_z*vec_lnz[is,t] + Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_g + γ_z*vec_lnz[is,t] + Emax[t,3,is_e]-Emax[t,4,is_e]) + 1);
            return
        end
    #
    # Faster version allowing for fixed costs to depend on productivity (size) - fuel price/productivity as state var
        function T_W_GPU_faster_zfc_alt(vec_Wnew,vec_π,vec_lnz,Emax,κ_g,κ_c,γ_g,γ_c,κ_z,γ_z,index)
            is = (blockIdx().x-1)*blockDim().x + threadIdx().x
            t = (blockIdx().y-1)*blockDim().y + threadIdx().y
            is_e = index[is];
            #is_e=is;
            # Compute new value function 
            @inbounds vec_Wnew[is,1,t] = vec_π[4,is,1,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(- κ_g - κ_z*vec_lnz[4,is,t] + Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(- κ_c - κ_z*vec_lnz[4,is,t] + Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_g - κ_c - κ_z*vec_lnz[4,is,t]));
            @inbounds vec_Wnew[is,2,t] = vec_π[4,is,2,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_g + γ_z*vec_lnz[4,is,t] + Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_g + γ_z*vec_lnz[4,is,t] -κ_c - κ_z*vec_lnz[4,is,t] + Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_c - κ_z*vec_lnz[4,is,t]));
            @inbounds vec_Wnew[is,3,t] = vec_π[4,is,3,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_c + γ_z*vec_lnz[4,is,t] + Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_c + γ_z*vec_lnz[4,is,t] -κ_g - κ_z*vec_lnz[4,is,t] +Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(Emax[t,3,is_e]-Emax[t,4,is_e]) + CUDA.exp(-κ_g - κ_z*vec_lnz[4,is,t]));
            @inbounds vec_Wnew[is,4,t] = vec_π[4,is,4,t]+Emax[t,4,is_e]+CUDA.log(CUDA.exp(γ_g + γ_z*vec_lnz[4,is,t] + γ_c+ Emax[t,1,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_c + γ_z*vec_lnz[4,is,t] + Emax[t,2,is_e]-Emax[t,4,is_e])+CUDA.exp(γ_g + γ_z*vec_lnz[4,is,t] + Emax[t,3,is_e]-Emax[t,4,is_e]) + 1);
            return
        end
    #
#

### Bellman operator (updates value function)
    # Slow version, but demands less GPU memory
        function T_EVF_gpu_faster(M::Model,vec_π,Κ,Π_transition) 
            # This function iterates over the expected value function
            @unpack p,Wind0_year,n_c,n_g,Π_g,Π_c,Πs = M; 
            @unpack β,T = p;
            # Use previous value function from model (already vectorized)
            vec_Wold_oe = Wind0_year[:,1,:];
            vec_Wold_oge = Wind0_year[:,2,:];
            vec_Wold_oce = Wind0_year[:,3,:];
            vec_Wold_ogce = Wind0_year[:,4,:];
            # Vectorize static profits
            # Size of vectorized grid
            SIZE_GRID = size(vec_Wold_oe,1);
            ## Create GPU arrays for all relevant objects
            # New value functions
                vec_Wnew = CuArray(zeros(Float32,SIZE_GRID,4,T));
            #
            # Old value functions
                vec_Wold_oe = CuArray{Float32}(vec_Wold_oe);
                vec_Wold_oge = CuArray{Float32}(vec_Wold_oge);
                vec_Wold_oce = CuArray{Float32}(vec_Wold_oce);
                vec_Wold_ogce = CuArray{Float32}(vec_Wold_ogce);
                #vec_Wold = CuArray{Float32}([vec_Wold_oe vec_Wold_oge vec_Wold_oce vec_Wold_ogce]);
            #
            # Profit functions
                vec_π = CuArray{Float32}(vec_π);
            #
            #Κgpu = CuArray(Κ);
            κ_g = Κ[1] ;
            κ_c = Κ[2] ;
            γ_g = Κ[3] ;
            γ_c = Κ[4] ;
            # Get Emax for each combination of choices (turns out every 1024 rows, Emax is the same)
            #rep=1024;
            rep=Int(M.ngrid^M.nstate);
            #Emax = CUDA.zeros(rep,4);
            Emax = fill(CuArray{Float32}(zeros(rep,T)),4);
            Π_transition0 = CuArray{Float32}(Π_transition[1:rep,:]);
            #block_size= 32;
            block_size = 256;
            #block_size = 1024;
            nthreads_3d = (ceil(Int,rep/block_size),ceil(Int,SIZE_GRID/block_size),1);
            block_3d = (block_size,block_size,T);
            # oe
                Emax_wide = CUDA.zeros(rep,SIZE_GRID,T);
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oe,Π_transition0,Float32(β));
                Emax[1] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # oge
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oge,Π_transition0,Float32(β));
                Emax[2] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # oce
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oce,Π_transition0,Float32(β));
                Emax[3] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # ogce
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_ogce,Π_transition0,Float32(β));
                Emax[4] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            index = repeat(collect(1:rep),ceil(Int,SIZE_GRID/rep));
            index = CuArray(index);
            Emax = reshape(mapreduce(permutedims,vcat,Emax),T,4,rep);
            # Emax[1] = repeat(Emax[1],ceil(Int,SIZE_GRID/rep));
            # Emax[2] = repeat(Emax[2],ceil(Int,SIZE_GRID/rep));
            # Emax[3] = repeat(Emax[3],ceil(Int,SIZE_GRID/rep));
            # Emax[4] = repeat(Emax[4],ceil(Int,SIZE_GRID/rep));
            # Emax = reshape(mapreduce(permutedims,vcat,Emax),T,4,SIZE_GRID);
            # Get new value function 
            nthreads_2d = (ceil(Int,SIZE_GRID/block_size),1);
            block_2d = (block_size,T);
            @cuda threads=nthreads_2d blocks=block_2d T_W_GPU_faster(vec_Wnew,vec_π,Emax,Float32(κ_g),Float32(κ_c),Float32(γ_g),Float32(γ_c),index);
            # Return value function 
            return Array{Float64}(vec_Wnew);
        end
    #
    # Faster version, but demands more GPU memory 
        function T_EVF_gpu_evenfaster(M::Model,vec_π,Κ,Π_transition)
            # This function iterates over the expected value function
            @unpack p,Wind0_full,n_c,n_g,Π_g,Π_c,Πs,nc_re,ng_re = M; 
            @unpack β,T = p;
            # Use previous value function from model (already vectorized)
            vec_Wold_oe = Wind0_full[:,1,:];
            vec_Wold_oge = Wind0_full[:,2,:];
            vec_Wold_oce = Wind0_full[:,3,:];
            vec_Wold_ogce = Wind0_full[:,4,:];
            # Vectorize static profits
            # Size of vectorized grid
            SIZE_GRID = size(vec_Wold_oe,1);
            ## Create GPU arrays for all relevant objects
            # New value functions
                vec_Wnew = CuArray(zeros(Float32,SIZE_GRID,4,T*nc_re*ng_re));
            #
            # Old value functions
                vec_Wold_oe = CuArray{Float32}(vec_Wold_oe);
                vec_Wold_oge = CuArray{Float32}(vec_Wold_oge);
                vec_Wold_oce = CuArray{Float32}(vec_Wold_oce);
                vec_Wold_ogce = CuArray{Float32}(vec_Wold_ogce);
                #vec_Wold = CuArray{Float32}([vec_Wold_oe vec_Wold_oge vec_Wold_oce vec_Wold_ogce]);
            #
            # Profit functions
                vec_π = CuArray{Float32}(vec_π);
            #
            #Κgpu = CuArray(Κ);
            κ_g = Κ[1] ;
            κ_c = Κ[2] ;
            γ_g = Κ[3] ;
            γ_c = Κ[4] ;
            # Get Emax for each combination of choices (turns out every 1024 rows, Emax is the same because some states are not persistent)
            #rep=1024;
            rep=Int(M.ngrid^M.nstate);
            #Emax = CUDA.zeros(rep,4);
            Emax = fill(CuArray{Float32}(zeros(rep,T*nc_re*ng_re)),4);
            Π_transition0 = CuArray{Float32}(Π_transition[1:rep,:]);
            #block_size= 32;
            block_size = 256;
            #block_size = 1024;
            nthreads_3d = (ceil(Int,rep/block_size),ceil(Int,SIZE_GRID/block_size),1);
            block_3d = (block_size,block_size,T*nc_re*ng_re);
            # oe
                Emax_wide = CUDA.zeros(rep,SIZE_GRID,T*nc_re*ng_re);
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oe,Π_transition0,Float32(β));
                Emax[1] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # oge
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oge,Π_transition0,Float32(β));
                Emax[2] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # oce
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oce,Π_transition0,Float32(β));
                Emax[3] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # ogce
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_ogce,Π_transition0,Float32(β));
                Emax[4] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            index = repeat(collect(1:rep),ceil(Int,SIZE_GRID/rep));
            index = CuArray(index);
            Emax = reshape(mapreduce(permutedims,vcat,Emax),T*nc_re*ng_re,4,rep);
            # Emax[1] = repeat(Emax[1],ceil(Int,SIZE_GRID/rep));
            # Emax[2] = repeat(Emax[2],ceil(Int,SIZE_GRID/rep));
            # Emax[3] = repeat(Emax[3],ceil(Int,SIZE_GRID/rep));
            # Emax[4] = repeat(Emax[4],ceil(Int,SIZE_GRID/rep));
            # Emax = reshape(mapreduce(permutedims,vcat,Emax),T,4,SIZE_GRID);
            # Get new value function 
            nthreads_2d = (ceil(Int,SIZE_GRID/block_size),1);
            block_2d = (block_size,T*nc_re*ng_re);
            @cuda threads=nthreads_2d blocks=block_2d T_W_GPU_faster(vec_Wnew,vec_π,Emax,Float32(κ_g),Float32(κ_c),Float32(γ_g),Float32(γ_c),index);
            # Return value function 
            return Array{Float64}(vec_Wnew);
        end
    #
    # Faster version allowing for fixed costs to depend on productivity (size)
        function T_EVF_gpu_evenfaster_zfc(M::Model,vec_π,Κ,Π_transition,vec_lnz)
            # This function iterates over the expected value function
            @unpack p,Wind0_full,n_c,n_g,Π_g,Π_c,Πs,nc_re,ng_re = M; 
            @unpack β,T = p;
            # Use previous value function from model (already vectorized)
            vec_Wold_oe = Wind0_full[:,1,:];
            vec_Wold_oge = Wind0_full[:,2,:];
            vec_Wold_oce = Wind0_full[:,3,:];
            vec_Wold_ogce = Wind0_full[:,4,:];
            # Vectorize static profits
            # Size of vectorized grid
            SIZE_GRID = size(vec_Wold_oe,1);
            ## Create GPU arrays for all relevant objects
            # New value functions
                vec_Wnew = CuArray(zeros(Float32,SIZE_GRID,4,T*nc_re*ng_re));
            #
            # Old value functions
                vec_Wold_oe = CuArray{Float32}(vec_Wold_oe);
                vec_Wold_oge = CuArray{Float32}(vec_Wold_oge);
                vec_Wold_oce = CuArray{Float32}(vec_Wold_oce);
                vec_Wold_ogce = CuArray{Float32}(vec_Wold_ogce);
                #vec_Wold = CuArray{Float32}([vec_Wold_oe vec_Wold_oge vec_Wold_oce vec_Wold_ogce]);
            #
            # Profit functions
                vec_π = CuArray{Float32}(vec_π);
                vec_lnz = CuArray{Float32}(vec_lnz);
            #
            #Κgpu = CuArray(Κ);
            κ_g = Κ[1] ;
            κ_c = Κ[2] ;
            γ_g = Κ[3] ;
            γ_c = Κ[4] ;
            κ_z = Κ[5] ;
            γ_z = Κ[6] ;
            # Get Emax for each combination of choices (turns out every 1024 rows, Emax is the same because some states are not persistent)
            #rep=1024;
            rep=Int(M.ngrid^M.nstate);
            #Emax = CUDA.zeros(rep,4);
            Emax = fill(CuArray{Float32}(zeros(rep,T*nc_re*ng_re)),4);
            Π_transition0 = CuArray{Float32}(Π_transition[1:rep,:]);
            #block_size= 32;
            block_size = 256;
            #block_size = 1024;
            nthreads_3d = (ceil(Int,rep/block_size),ceil(Int,SIZE_GRID/block_size),1);
            block_3d = (block_size,block_size,T*nc_re*ng_re);
            # oe
                Emax_wide = CUDA.zeros(rep,SIZE_GRID,T*nc_re*ng_re);
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oe,Π_transition0,Float32(β));
                Emax[1] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # oge
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oge,Π_transition0,Float32(β));
                Emax[2] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # oce
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oce,Π_transition0,Float32(β));
                Emax[3] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # ogce
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_ogce,Π_transition0,Float32(β));
                Emax[4] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            index = repeat(collect(1:rep),ceil(Int,SIZE_GRID/rep));
            index = CuArray(index);
            Emax = reshape(mapreduce(permutedims,vcat,Emax),T*nc_re*ng_re,4,rep);
            # Emax[1] = repeat(Emax[1],ceil(Int,SIZE_GRID/rep));
            # Emax[2] = repeat(Emax[2],ceil(Int,SIZE_GRID/rep));
            # Emax[3] = repeat(Emax[3],ceil(Int,SIZE_GRID/rep));
            # Emax[4] = repeat(Emax[4],ceil(Int,SIZE_GRID/rep));
            # Emax = reshape(mapreduce(permutedims,vcat,Emax),T,4,SIZE_GRID);
            # Get new value function 
            nthreads_2d = (ceil(Int,SIZE_GRID/block_size),1);
            block_2d = (block_size,T*nc_re*ng_re);
            @cuda threads=nthreads_2d blocks=block_2d T_W_GPU_faster_zfc(vec_Wnew,vec_π,vec_lnz,Emax,Float32(κ_g),Float32(κ_c),Float32(γ_g),Float32(γ_c),Float32(κ_z),Float32(γ_z),index);
            # Return value function 
            return Array{Float64}(vec_Wnew);
        end
    #
    # Faster version allowing for fixed costs to depend on productivity (size) - fuel price/productivity as state
        function T_EVF_gpu_evenfaster_zfc_alt(M::Model,vec_π,Κ,Π_transition_ogce,vec_lnz)
            # This function iterates over the expected value function
            @unpack p,Wind0_full,Πs_oe,Πs_oge,Πs_oce,Πs_ogce,nc_re,ng_re = M; 
            @unpack β,T,R = p;
            # Use previous value function from model (already vectorized)
            vec_Wold_oe = Wind0_full[:,1,:];
            vec_Wold_oge = Wind0_full[:,2,:];
            vec_Wold_oce = Wind0_full[:,3,:];
            vec_Wold_ogce = Wind0_full[:,4,:];
            # Vectorize static profits
            # Size of vectorized grid
            SIZE_GRID = size(vec_Wold_oe,1);
            ## Create GPU arrays for all relevant objects
            # New value functions
                vec_Wnew = CuArray(zeros(Float32,SIZE_GRID,4,T*R*nc_re*ng_re));
            #
            # Old choice-specific value functions
                vec_Wold_oe = CuArray{Float32}(vec_Wold_oe);
                vec_Wold_oge = CuArray{Float32}(vec_Wold_oge); 
                vec_Wold_oce = CuArray{Float32}(vec_Wold_oce);
                vec_Wold_ogce = CuArray{Float32}(vec_Wold_ogce);
                #vec_Wold = CuArray{Float32}([vec_Wold_oe vec_Wold_oge vec_Wold_oce vec_Wold_ogce]);
            #
            # Profit functions
                vec_π = CuArray{Float32}(vec_π);
                vec_lnz = CuArray{Float32}(vec_lnz);
            #
            #Κgpu = CuArray(Κ);
            κ_g = Κ[1] ;
            κ_c = Κ[2] ;
            γ_g = Κ[3] ;
            γ_c = Κ[4] ;
            κ_z = Κ[5] ;
            γ_z = Κ[6] ;
            # Get Emax for each combination of choices (turns out every 1024 rows, Emax is the same because some states are not persistent)
            #rep=1024;
            rep=Int(M.ngrid^M.nstate);
            #Emax = CUDA.zeros(rep,4);
            Emax = fill(CuArray{Float32}(zeros(rep,T*R*nc_re*ng_re)),4);
            Π_transition0 = CuArray{Float32}(Π_transition_ogce[1:rep,:]);
            #block_size= 32;
            # block_size = 256;
            #block_size = 1024;
            if M.ngrid == 3
                block_size = 81;
            elseif M.ngrid == 4
                block_size = 256;
            end
            nthreads_3d = (ceil(Int,rep/block_size),ceil(Int,SIZE_GRID/block_size),1);
            block_3d = (block_size,block_size,T*R*nc_re*ng_re);
            # transition to: oe
                # Starting from (in order): oe, oge, oce, ogce 
                Emax_wide = CUDA.zeros(rep,SIZE_GRID,T*R*nc_re*ng_re);
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oe,Π_transition0,Float32(β));
                Emax[1] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # transition to: oge
                # Starting from (in order): oe, oge, oce, ogce 
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oge,Π_transition0,Float32(β));
                Emax[2] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # transition to: oce
                # Starting from (in order): oe, oge, oce, ogce 
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_oce,Π_transition0,Float32(β));
                Emax[3] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            # transition to: ogce
                # Starting from (in order): oe, oge, oce, ogce 
                @cuda threads=nthreads_3d blocks=block_3d Emax_GPU_faster(Emax_wide,vec_Wold_ogce,Π_transition0,Float32(β));
                Emax[4] = dropdims(sum(Emax_wide,dims=2),dims=2);
            #
            index = repeat(collect(1:rep),ceil(Int,SIZE_GRID/rep));
            index = CuArray(index);
            Emax = reshape(mapreduce(permutedims,vcat,Emax),T*R*nc_re*ng_re,4,rep);
            # Emax[1] = repeat(Emax[1],ceil(Int,SIZE_GRID/rep));
            # Emax[2] = repeat(Emax[2],ceil(Int,SIZE_GRID/rep));
            # Emax[3] = repeat(Emax[3],ceil(Int,SIZE_GRID/rep));
            # Emax[4] = repeat(Emax[4],ceil(Int,SIZE_GRID/rep));
            # Emax = reshape(mapreduce(permutedims,vcat,Emax),T,4,SIZE_GRID);
            # Get new value function 
            nthreads_2d = (ceil(Int,SIZE_GRID/block_size),1);
            block_2d = (block_size,T*R*nc_re*ng_re);
            @cuda threads=nthreads_2d blocks=block_2d T_W_GPU_faster_zfc_alt(vec_Wnew,vec_π,vec_lnz,Emax,Float32(κ_g),Float32(κ_c),Float32(γ_g),Float32(γ_c),Float32(κ_z),Float32(γ_z),index);
            # Return value function 
            return Array{Float64}(vec_Wnew);
        end
    #
#

### Main functions for value function iteration (VFI) using GPU - iteratively calls Bellman operator until convergence
    ####-------------------------------------------------------------------####
    ###  ALL FUNCTIONS ALLOW FOR FIXED COSTS TO VARY WITH PIPELINE NETWORK  ###
    ####-------------------------------------------------------------------####
    # Slow version, but demands less GPU memory
        function VFI_discrete_faster(M::Model,Κ,W_old_bench::VF_bench)
            ## Given the following, this function returns the expected value function (W)
                #1. Guess of the fixed cost parameters
                #2. Individual firm indices
                #3. Current fuel set 𝓕
            @unpack p,ng_re,nc_re = M;
            @unpack β,T = p;
            SIZE_GRID = (M.ngrid^M.nstate)*M.n_c*M.n_g;
            # aux = Array{Float64}(undef,SIZE_GRID,p.F_tot,T,nc_re,ng_re);
            W_old_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_old_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_converged = Array{Float64}(undef,p.F_tot,p.T,M.ngrid^M.nstate,M.n_c,M.n_g,nc_re,ng_re);
            Π_transition=Array{Float64}(kronecker(M.Π_g,M.Π_c,M.Πs));
            # Fixed costs based on pipeline proximity
                κg_nopipe   = Κ[1];
                κc          = Κ[2];
                γg_nopipe   = Κ[3];
                γc          = Κ[4];
                κg_pipe     = Κ[5];
                γg_pipe     = Κ[6];
                # Gas pipeline connection 
                Κ_pipe = [κg_pipe,κc,γg_pipe,γc];
                # No gas pipeline connection
                Κ_nopipe = [κg_nopipe,κc,γg_nopipe,γc];
            #
            println("--------------------------------------------------------------")
            println("--------     FIXED POINT ITERATION BEGINS      ---------------")
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                # Initialize value function
                if W_old_bench.W == nothing
                    for t = 1:T
                        W_old_nopipe[:,1,t] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]);
                        W_old_nopipe[:,2,t] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]);
                        W_old_nopipe[:,3,t] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]);
                        W_old_nopipe[:,4,t] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,1,t] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,2,t] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,3,t] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,4,t] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]);
                    end
                else
                    for t = 1:T
                        W_old_nopipe[:,1,t] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]);
                        W_old_nopipe[:,2,t] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]);
                        W_old_nopipe[:,3,t] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]);
                        W_old_nopipe[:,4,t] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,1,t] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,2,t] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,3,t] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]);
                        W_old_pipe[:,4,t] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]);
                    end
                end 
                vec_π = zeros(SIZE_GRID,p.F_tot,T);
                for t = 1:T
                    vec_π[:,:,t] = [vec(M.πgrid_oe[t,:,:,:,ic_re,ig_re]) vec(M.πgrid_oge[t,:,:,:,ic_re,ig_re]) vec(M.πgrid_oce[t,:,:,:,ic_re,ig_re]) vec(M.πgrid_ogce[t,:,:,:,ic_re,ig_re])];
                end

                # No gas pipeline connection
                    # Initialized distance 
                    println("      ")
                    println("--------------------------------------------------------------")
                    println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                    println("-------------  NO GAS PIPELINE CONNECTION  ------------------")
                    println("-------------------------------------------------------------")
                    println("      ")
                    W_dist_new = 100 ;
                    for iter=1:p.max_iter
                        # Update old distance and iterations
                        W_new_nopipe = T_EVF_gpu_evenfaster(Model(M,Wind0_full=copy(W_old_nopipe)),vec_π,Κ_nopipe,Π_transition)  ;
                        # Update distance
                        W_dist_new = sqrt(norm(W_new_nopipe.-W_old_nopipe,2))         ;
                        #println("iter = $iter, Distance = $W_dist_new")
                        if mod(iter,250)==0
                            println("iter = $iter, Distance = $W_dist_new")
                            for t = 1:T
                                W_new_converged[1,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,1,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[1,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,2,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[1,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,3,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[1,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,4,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                            end
                            break
                        end
                        # Update value function
                        W_old_nopipe .= deepcopy(W_new_nopipe) ;
                        # Check if converged 
                        if W_dist_new<=p.dist_tol
                            println("iter = $iter, Distance = $W_dist_new") 
                            for t = 1:T
                                W_new_converged[1,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,1,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[1,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,2,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[1,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,3,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[1,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,4,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                            end
                            break
                        end
                    end
                #
                # Gas pipeline connection
                    # Initialized distance 
                    println("      ")
                    println("--------------------------------------------------------------")
                    println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                    println("---------------   GAS PIPELINE CONNECTION  ------------------")
                    println("-------------------------------------------------------------")
                    println("      ")
                    W_dist_new = 100 ;
                    for iter=1:p.max_iter
                        # Update old distance and iterations
                        W_new_pipe = T_EVF_gpu_evenfaster(Model(M,Wind0_full=copy(W_old_pipe)),vec_π,Κ_pipe,Π_transition)  ;
                        # Update distance
                        W_dist_new = sqrt(norm(W_new_pipe.-W_old_pipe,2))         ;
                        #println("iter = $iter, Distance = $W_dist_new")
                        if mod(iter,250)==0
                            println("iter = $iter, Distance = $W_dist_new")
                            for t = 1:T
                                W_new_converged[2,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,1,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[2,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,2,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[2,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,3,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[2,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,4,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                            end
                            break
                        end
                        # Update value function
                        W_old_pipe .= deepcopy(W_new_pipe) ;
                        # Check if converged 
                        if W_dist_new<=p.dist_tol
                            println("iter = $iter, Distance = $W_dist_new") 
                            for j = 1:(T*nc_re*ng_re)
                                t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                                W_new_converged[2,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,1,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[2,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,2,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[2,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,3,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                                W_new_converged[2,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,4,t],M.ngrid^M.nstate,M.n_c,M.n_g);
                            end
                            break
                        end
                    end
                #
            end
            println("--------     FIXED POINT ITERATION ENDS     ------------------")
            println("--------------------------------------------------------------")
            return W_new_converged;
        end
    #
    # Faster version, but demands more GPU memory
        function VFI_discrete_evenfaster(M::Model,Κ,σ,W_old_bench::VF_bench)
            ## Given the following, this function returns the expected value function (W)
                #1. Guess of the fixed cost parameters
                #2. Individual firm indices
                #3. Current fuel set 𝓕
            @unpack p,ng_re,nc_re,nconnect = M;
            @unpack β,T = p;
            SIZE_GRID = (M.ngrid^M.nstate)*M.n_c*M.n_g;
            W_old_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_old_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_converged = Array{Float64}(undef,nconnect,p.F_tot,p.T,M.ngrid^M.nstate,M.n_c,M.n_g,nc_re,ng_re);
            Π_transition=Array{Float64}(kronecker(M.Π_g,M.Π_c,M.Πs));
            # Fixed costs based on pipeline proximity
                κg_nopipe   = Κ[1];
                κc          = Κ[2];
                γg_nopipe   = Κ[3];
                γc          = Κ[4];
                κg_pipe     = Κ[5];
                γg_pipe     = Κ[6];
                # Gas pipeline connection 
                Κ_pipe = [κg_pipe,κc,γg_pipe,γc];
                # No gas pipeline connection
                Κ_nopipe = [κg_nopipe,κc,γg_nopipe,γc];
            #
            # Initialize value function
            if W_old_bench.W == nothing
                for j = 1:(T*nc_re*ng_re)
                    t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                    # vec(aux) and W_old should match here
                    W_old_nopipe[:,1,j] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,2,j] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,3,j] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,4,j] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,1,j] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,2,j] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,3,j] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,4,j] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]/σ);
                end
            else
                for j = 1:(T*nc_re*ng_re)
                    t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                    W_old_nopipe[:,1,j] = vec(W_old_bench.W[1,1,t,:,:,:,ic_re,ig_re]);
                    W_old_nopipe[:,2,j] = vec(W_old_bench.W[1,2,t,:,:,:,ic_re,ig_re]);
                    W_old_nopipe[:,3,j] = vec(W_old_bench.W[1,3,t,:,:,:,ic_re,ig_re]);
                    W_old_nopipe[:,4,j] = vec(W_old_bench.W[1,4,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,1,j] = vec(W_old_bench.W[2,1,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,2,j] = vec(W_old_bench.W[2,2,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,3,j] = vec(W_old_bench.W[2,3,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,4,j] = vec(W_old_bench.W[2,4,t,:,:,:,ic_re,ig_re]);
                end
            end 
            vec_π = zeros(SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            for j = 1:(T*nc_re*ng_re)
                t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                vec_π[:,:,j] = [vec(M.πgrid_oe[t,:,:,:,ic_re,ig_re])/σ vec(M.πgrid_oge[t,:,:,:,ic_re,ig_re])/σ vec(M.πgrid_oce[t,:,:,:,ic_re,ig_re])/σ vec(M.πgrid_ogce[t,:,:,:,ic_re,ig_re])/σ];
            end
            # No gas pipeline connection
                # Initialized distance 
                println("      ")
                println("--------------------------------------------------------------")
                println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                println("-------------  NO GAS PIPELINE CONNECTION  ------------------")
                println("-------------------------------------------------------------")
                println("      ")
                W_dist_new = 100 ;
                for iter=1:p.max_iter
                    # Update old distance and iterations
                    W_new_nopipe = T_EVF_gpu_evenfaster(Model(M,Wind0_full=copy(W_old_nopipe)),vec_π,Κ_nopipe,Π_transition)  ;
                    # Update distance
                    W_dist_new = sqrt(norm(W_new_nopipe.-W_old_nopipe,2))         ;
                    #println("iter = $iter, Distance = $W_dist_new")
                    if mod(iter,250)==0
                        println("iter = $iter, Distance = $W_dist_new")
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[1,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                    # Update value function
                    W_old_nopipe .= deepcopy(W_new_nopipe) ;
                    # Check if converged 
                    if W_dist_new<=p.dist_tol
                        println("iter = $iter, Distance = $W_dist_new") 
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[1,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                end
            #
            # Gas pipeline connection
                # Initialized distance 
                println("      ")
                println("--------------------------------------------------------------")
                println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                println("---------------   GAS PIPELINE CONNECTION  ------------------")
                println("-------------------------------------------------------------")
                println("      ")
                W_dist_new = 100 ;
                for iter=1:p.max_iter
                    # Update old distance and iterations
                    W_new_pipe = T_EVF_gpu_evenfaster(Model(M,Wind0_full=copy(W_old_pipe)),vec_π,Κ_pipe,Π_transition)  ;
                    # Update distance
                    W_dist_new = sqrt(norm(W_new_pipe.-W_old_pipe,2))         ;
                    #println("iter = $iter, Distance = $W_dist_new")
                    if mod(iter,250)==0
                        println("iter = $iter, Distance = $W_dist_new")
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[2,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                    # Update value function
                    W_old_pipe .= deepcopy(W_new_pipe) ;
                    # Check if converged 
                    if W_dist_new<=p.dist_tol
                        println("iter = $iter, Distance = $W_dist_new") 
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[2,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                end
            #
            println("      ")
            println("--------------------------------------------------------------")
            println("--------     FIXED POINT ITERATION ENDS     ------------------")
            println("--------------------------------------------------------------")
            println("      ")
            return W_new_converged;
        end
    #
    # Faster version allowing for fixed costs to depend on productivity (size)
        function VFI_discrete_evenfaster_zfc(M::Model,Κ,σ,W_old_bench::VF_bench)
            ## Given the following, this function returns the expected value function (W)
                #1. Guess of the fixed cost parameters
                #2. Individual firm indices
                #3. Current fuel set 𝓕
            @unpack p,ng_re,nc_re,nconnect = M;
            @unpack β,T = p;
            SIZE_GRID = (M.ngrid^M.nstate)*M.n_c*M.n_g;
            W_old_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_old_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            W_new_converged = Array{Float64}(undef,nconnect,p.F_tot,p.T,M.ngrid^M.nstate,M.n_c,M.n_g,nc_re,ng_re);
            Π_transition=Array{Float64}(kronecker(M.Π_g,M.Π_c,M.Πs));
            # Fixed costs based on pipeline proximity and productivity (size)
                κg_nopipe   = Κ[1];
                κc          = Κ[2];
                γg_nopipe   = Κ[3];
                γc          = Κ[4];
                κg_pipe     = Κ[5];
                γg_pipe     = Κ[6];
                κz          = Κ[7];
                γz          = Κ[8];
                # Gas pipeline connection 
                Κ_pipe = [κg_pipe,κc,γg_pipe,γc,κz,γz];
                # No gas pipeline connection
                Κ_nopipe = [κg_nopipe,κc,γg_nopipe,γc,κz,γz];
            #
            # Initialize value function
            if W_old_bench.W == nothing
                for j = 1:(T*nc_re*ng_re)
                    t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                    # vec(aux) and W_old should match here
                    W_old_nopipe[:,1,j] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,2,j] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,3,j] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,4,j] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,1,j] = vec((1/(1-β))*M.πgrid_oe[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,2,j] = vec((1/(1-β))*M.πgrid_oge[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,3,j] = vec((1/(1-β))*M.πgrid_oce[t,:,:,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,4,j] = vec((1/(1-β))*M.πgrid_ogce[t,:,:,:,ic_re,ig_re]/σ);
                end
            else
                for j = 1:(T*nc_re*ng_re)
                    t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                    W_old_nopipe[:,1,j] = vec(W_old_bench.W[1,1,t,:,:,:,ic_re,ig_re]);
                    W_old_nopipe[:,2,j] = vec(W_old_bench.W[1,2,t,:,:,:,ic_re,ig_re]);
                    W_old_nopipe[:,3,j] = vec(W_old_bench.W[1,3,t,:,:,:,ic_re,ig_re]);
                    W_old_nopipe[:,4,j] = vec(W_old_bench.W[1,4,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,1,j] = vec(W_old_bench.W[2,1,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,2,j] = vec(W_old_bench.W[2,2,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,3,j] = vec(W_old_bench.W[2,3,t,:,:,:,ic_re,ig_re]);
                    W_old_pipe[:,4,j] = vec(W_old_bench.W[2,4,t,:,:,:,ic_re,ig_re]);
                end
            end 
            vec_π = zeros(SIZE_GRID,p.F_tot,T*nc_re*ng_re);
            vec_lnz = zeros(SIZE_GRID,T*nc_re*ng_re);
            for j = 1:(T*nc_re*ng_re)
                t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                vec_π[:,:,j] = [vec(M.πgrid_oe[t,:,:,:,ic_re,ig_re])/σ vec(M.πgrid_oge[t,:,:,:,ic_re,ig_re])/σ vec(M.πgrid_oce[t,:,:,:,ic_re,ig_re])/σ vec(M.πgrid_ogce[t,:,:,:,ic_re,ig_re])/σ];
                vec_lnz[:,j] = vec(mapreduce(permutedims,vcat,vec(fill(vec(M.lnSgrid[:,1]),M.n_c*M.n_g)))') .+ p.μz_t[t+1]
            end
            # No gas pipeline connection
                # Initialized distance
                println("      ")
                println("--------------------------------------------------------------")
                println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                println("-------------  NO GAS PIPELINE CONNECTION  ------------------")
                println("-------------------------------------------------------------")
                println("      ")
                W_dist_new = 100 ;
                for iter=1:p.max_iter
                    # Update old distance and iterations
                    W_new_nopipe = T_EVF_gpu_evenfaster_zfc(Model(M,Wind0_full=copy(W_old_nopipe)),vec_π,Κ_nopipe,Π_transition,vec_lnz)  ;
                    # Update distance
                    W_dist_new = sqrt(norm(W_new_nopipe.-W_old_nopipe,2))         ;
                    #println("iter = $iter, Distance = $W_dist_new")
                    if mod(iter,250)==0
                        println("iter = $iter, Distance = $W_dist_new")
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[1,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                    # Update value function
                    W_old_nopipe .= deepcopy(W_new_nopipe) ;
                    # Check if converged 
                    if W_dist_new<=p.dist_tol
                        println("iter = $iter, Distance = $W_dist_new") 
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[1,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[1,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_nopipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                end
            #
            # Gas pipeline connection
                # Initialized distance 
                println("      ")
                println("--------------------------------------------------------------")
                println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                println("---------------   GAS PIPELINE CONNECTION  ------------------")
                println("-------------------------------------------------------------")
                println("      ")
                W_dist_new = 100 ;
                for iter=1:p.max_iter
                    # Update old distance and iterations
                    W_new_pipe = T_EVF_gpu_evenfaster_zfc(Model(M,Wind0_full=copy(W_old_pipe)),vec_π,Κ_pipe,Π_transition,vec_lnz)  ;
                    # Update distance
                    W_dist_new = sqrt(norm(W_new_pipe.-W_old_pipe,2))         ;
                    #println("iter = $iter, Distance = $W_dist_new")
                    if mod(iter,250)==0
                        println("iter = $iter, Distance = $W_dist_new")
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[2,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                    # Update value function
                    W_old_pipe .= deepcopy(W_new_pipe) ;
                    # Check if converged 
                    if W_dist_new<=p.dist_tol
                        println("iter = $iter, Distance = $W_dist_new") 
                        for j = 1:(T*nc_re*ng_re)
                            t,ic_re,ig_re = Tuple(CartesianIndices((p.T,M.nc_re,M.ng_re))[j]);
                            W_new_converged[2,1,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,1,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,2,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,2,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,3,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,3,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                            W_new_converged[2,4,t,:,:,:,ic_re,ig_re] = reshape(W_new_pipe[:,4,j],M.ngrid^M.nstate,M.n_c,M.n_g);
                        end
                        break
                    end
                end
            #
            println("      ")
            println("--------------------------------------------------------------")
            println("--------     FIXED POINT ITERATION ENDS     ------------------")
            println("--------------------------------------------------------------")
            println("      ")
            return W_new_converged;
        end
    #
    # Faster version allowing for fixed costs to depend on productivity (size) - fuel price/productivity as state
        function VFI_discrete_evenfaster_zfc_alt(M::Model,Κ,σ,W_old_bench::VF_bench)
            ## Given the following, this function returns the expected value function (W)
                #1. Guess of the fixed cost parameters
                #2. Individual firm indices
                #3. Current fuel set 𝓕
            @unpack p,ng_re,nc_re,nconnect = M;
            @unpack β,T,R = p;
            SIZE_GRID = M.ngrid^M.nstate;
            W_old_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*R*nc_re*ng_re);
            W_new_nopipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*R*nc_re*ng_re);
            W_old_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*R*nc_re*ng_re);
            W_new_pipe = Array{Float64}(undef,SIZE_GRID,p.F_tot,T*R*nc_re*ng_re);
            W_new_converged = Array{Float64}(undef,nconnect,p.F_tot,p.T,R,M.ngrid^M.nstate,nc_re,ng_re);
            Π_transition_ogce=Array{Float64}(kronecker(M.Πs_ogce));
            # Fixed costs based on pipeline proximity and productivity (size)
                κg_nopipe   = Κ[1];
                κc          = Κ[2];
                γg_nopipe   = Κ[3];
                γc          = Κ[4];
                κg_pipe     = Κ[5];
                γg_pipe     = Κ[6];
                κz          = Κ[7];
                γz          = Κ[8];
                # Gas pipeline connection 
                Κ_pipe = [κg_pipe,κc,γg_pipe,γc,κz,γz];
                # No gas pipeline connection
                Κ_nopipe = [κg_nopipe,κc,γg_nopipe,γc,κz,γz];
            #
            # Initialize value function
            if W_old_bench.W == nothing
                for j = 1:(T*R*nc_re*ng_re)
                    t,r,ic_re,ig_re = Tuple(CartesianIndices((T,R,nc_re,ng_re))[j]);
                    # vec(aux) and W_old should match here
                    W_old_nopipe[:,1,j] = vec((1/(1-β))*M.πgrid_oe[4,t,r,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,2,j] = vec((1/(1-β))*M.πgrid_oge[4,t,r,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,3,j] = vec((1/(1-β))*M.πgrid_oce[4,t,r,:,ic_re,ig_re]/σ);
                    W_old_nopipe[:,4,j] = vec((1/(1-β))*M.πgrid_ogce[4,t,r,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,1,j] = vec((1/(1-β))*M.πgrid_oe[4,t,r,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,2,j] = vec((1/(1-β))*M.πgrid_oge[4,t,r,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,3,j] = vec((1/(1-β))*M.πgrid_oce[4,t,r,:,ic_re,ig_re]/σ);
                    W_old_pipe[:,4,j] = vec((1/(1-β))*M.πgrid_ogce[4,t,r,:,ic_re,ig_re]/σ);
                end
            else
                for j = 1:(T*R*nc_re*ng_re)
                    t,r,ic_re,ig_re = Tuple(CartesianIndices((T,R,nc_re,ng_re))[j]);
                    W_old_nopipe[:,1,j] = vec(W_old_bench.W[1,1,t,r,:,ic_re,ig_re]);
                    W_old_nopipe[:,2,j] = vec(W_old_bench.W[1,2,t,r,:,ic_re,ig_re]);
                    W_old_nopipe[:,3,j] = vec(W_old_bench.W[1,3,t,r,:,ic_re,ig_re]);
                    W_old_nopipe[:,4,j] = vec(W_old_bench.W[1,4,t,r,:,ic_re,ig_re]);
                    W_old_pipe[:,1,j] = vec(W_old_bench.W[2,1,t,r,:,ic_re,ig_re]);
                    W_old_pipe[:,2,j] = vec(W_old_bench.W[2,2,t,r,:,ic_re,ig_re]);
                    W_old_pipe[:,3,j] = vec(W_old_bench.W[2,3,t,r,:,ic_re,ig_re]);
                    W_old_pipe[:,4,j] = vec(W_old_bench.W[2,4,t,r,:,ic_re,ig_re]);
                end
            end 
            vec_π = zeros(p.F_tot,SIZE_GRID,p.F_tot,T*R*nc_re*ng_re);
            vec_lnz = zeros(p.F_tot,SIZE_GRID,T*R*nc_re*ng_re);
            for j = 1:(T*R*nc_re*ng_re)
                t,r,ic_re,ig_re = Tuple(CartesianIndices((p.T,p.R,M.nc_re,M.ng_re))[j]);
                for f = 1:p.F_tot 
                    vec_π[f,:,:,j] = [vec(M.πgrid_oe[f,t,r,:,ic_re,ig_re])/σ vec(M.πgrid_oge[f,t,r,:,ic_re,ig_re])/σ vec(M.πgrid_oce[f,t,r,:,ic_re,ig_re])/σ vec(M.πgrid_ogce[f,t,r,:,ic_re,ig_re])/σ];
                end 
                vec_lnz[1,:,j] = M.lnSgrid_oe[:,1] .+ p.μz_t[t+1] .+ p.μz_r[r];
                vec_lnz[2,:,j] = M.lnSgrid_oge[:,1] .+ p.μz_t[t+1] .+ p.μz_r[r];
                vec_lnz[3,:,j] = M.lnSgrid_oce[:,1] .+ p.μz_t[t+1] .+ p.μz_r[r];
                vec_lnz[4,:,j] = M.lnSgrid_ogce[:,1] .+ p.μz_t[t+1] .+ p.μz_r[r];
            end
            # No gas pipeline connection
                # Initialized distance
                println("      ")
                println("--------------------------------------------------------------")
                println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                println("-------------  NO GAS PIPELINE CONNECTION  ------------------")
                println("-------------------------------------------------------------")
                println("      ")
                W_dist_new = 100 ;
                for iter=1:p.max_iter
                    # Update old distance and iterations
                    W_new_nopipe = T_EVF_gpu_evenfaster_zfc_alt(Model(M,Wind0_full=copy(W_old_nopipe)),vec_π,Κ_nopipe,Π_transition_ogce,vec_lnz)  ;
                    # Update distance
                    W_dist_new = sqrt(norm(W_new_nopipe.-W_old_nopipe,2))         ;
                    #println("iter = $iter, Distance = $W_dist_new")
                    if mod(iter,250)==0
                        println("iter = $iter, Distance = $W_dist_new")
                        for j = 1:(T*R*nc_re*ng_re)
                            t,r,ic_re,ig_re = Tuple(CartesianIndices((p.T,p.R,M.nc_re,M.ng_re))[j]);
                            W_new_converged[1,1,t,r,:,ic_re,ig_re] = W_new_nopipe[:,1,j];
                            W_new_converged[1,2,t,r,:,ic_re,ig_re] = W_new_nopipe[:,2,j];
                            W_new_converged[1,3,t,r,:,ic_re,ig_re] = W_new_nopipe[:,3,j];
                            W_new_converged[1,4,t,r,:,ic_re,ig_re] = W_new_nopipe[:,4,j];
                        end
                        break
                    end
                    # Update value function
                    W_old_nopipe .= deepcopy(W_new_nopipe) ;
                    # Check if converged 
                    if W_dist_new<=p.dist_tol
                        println("iter = $iter, Distance = $W_dist_new") 
                        for j = 1:(T*R*nc_re*ng_re)
                            t,r,ic_re,ig_re = Tuple(CartesianIndices((p.T,p.R,M.nc_re,M.ng_re))[j]);
                            W_new_converged[1,1,t,r,:,ic_re,ig_re] = W_new_nopipe[:,1,j];
                            W_new_converged[1,2,t,r,:,ic_re,ig_re] = W_new_nopipe[:,2,j];
                            W_new_converged[1,3,t,r,:,ic_re,ig_re] = W_new_nopipe[:,3,j];
                            W_new_converged[1,4,t,r,:,ic_re,ig_re] = W_new_nopipe[:,4,j];
                        end
                        break
                    end
                end
            #
            # Gas pipeline connection
                # Initialized distance 
                println("      ")
                println("--------------------------------------------------------------")
                println("--------     FIXED POINT ITERATION BEGINS     ---------------")
                println("---------------   GAS PIPELINE CONNECTION  ------------------")
                println("-------------------------------------------------------------")
                println("      ")
                W_dist_new = 100 ;
                for iter=1:p.max_iter
                    # Update old distance and iterations
                    W_new_pipe = T_EVF_gpu_evenfaster_zfc_alt(Model(M,Wind0_full=copy(W_old_pipe)),vec_π,Κ_pipe,Π_transition_ogce,vec_lnz)  ;
                    # Update distance
                    W_dist_new = sqrt(norm(W_new_pipe.-W_old_pipe,2))         ;
                    #println("iter = $iter, Distance = $W_dist_new")
                    if mod(iter,250)==0
                        println("iter = $iter, Distance = $W_dist_new")
                        for j = 1:(T*R*nc_re*ng_re)
                            t,r,ic_re,ig_re = Tuple(CartesianIndices((p.T,p.R,M.nc_re,M.ng_re))[j]);
                            W_new_converged[2,1,t,r,:,ic_re,ig_re] = W_new_pipe[:,1,j];
                            W_new_converged[2,2,t,r,:,ic_re,ig_re] = W_new_pipe[:,2,j];
                            W_new_converged[2,3,t,r,:,ic_re,ig_re] = W_new_pipe[:,3,j];
                            W_new_converged[2,4,t,r,:,ic_re,ig_re] = W_new_pipe[:,4,j];
                        end
                        break
                    end
                    # Update value function
                    W_old_pipe .= deepcopy(W_new_pipe) ;
                    # Check if converged 
                    if W_dist_new<=p.dist_tol
                        println("iter = $iter, Distance = $W_dist_new") 
                        for j = 1:(T*R*nc_re*ng_re)
                            t,r,ic_re,ig_re = Tuple(CartesianIndices((p.T,p.R,M.nc_re,M.ng_re))[j]);
                            W_new_converged[2,1,t,r,:,ic_re,ig_re] = W_new_pipe[:,1,j];
                            W_new_converged[2,2,t,r,:,ic_re,ig_re] = W_new_pipe[:,2,j];
                            W_new_converged[2,3,t,r,:,ic_re,ig_re] = W_new_pipe[:,3,j];
                            W_new_converged[2,4,t,r,:,ic_re,ig_re] = W_new_pipe[:,4,j];
                        end
                        break
                    end
                end
            #
            println("      ")
            println("--------------------------------------------------------------")
            println("--------     FIXED POINT ITERATION ENDS     ------------------")
            println("--------------------------------------------------------------")
            println("      ")
            return W_new_converged;
        end
    #
#

##############################################################
#----           ESTIMATION FUNCTIONS                    -----#
##############################################################

####-------------------------------------------------------------------####
###  ALL FUNCTIONS ALLOW FOR FIXED COSTS TO VARY WITH PIPELINE NETWORK  ###
####-------------------------------------------------------------------####

### Functions that return fuel set choice probability probability given value function, observed choice, state variables and pre-determined comparative advantages
    # Main version
        function choicePR_func(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,ig_re,ic_re,W_new_inner=nothing)
            @unpack W_new,p,ngrid,nstate,n_c,n_g = M;
            @unpack T,β = p;
            N = size(Data,1);
            #Prnext_long = Array{Float64}(undef,N);
            Prnext_long = zeros(N);
            vec_Wold_nopipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_nopipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_nopipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_nopipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            for t=1:T 
                if W_new_inner == nothing 
                    vec_Wold_nopipe_oe[t] = vec(W_new[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[t] = vec(W_new[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[t] = vec(W_new[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[t] = vec(W_new[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[t] = vec(W_new[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[t] = vec(W_new[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[t] = vec(W_new[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[t] = vec(W_new[2,4,t,:,:,:,ic_re,ig_re])   ;
                else
                    vec_Wold_nopipe_oe[t] = vec(W_new_inner[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[t] = vec(W_new_inner[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[t] = vec(W_new_inner[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[t] = vec(W_new_inner[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[t] = vec(W_new_inner[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[t] = vec(W_new_inner[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[t] = vec(W_new_inner[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[t] = vec(W_new_inner[2,4,t,:,:,:,ic_re,ig_re])   ;
                end
            end
            # Fixed cost parameters
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            for i = 1:N
                # Get state indice
                is = grid_indices.s[i];
                # Transition probabilities
                aux_1 = kronecker(M.Π_g[1,:],M.Π_c[1,:],M.Πs[is,:]);
                t = Data.year[i]-2009;
                # Choice probabilities
                if Data.Connection[i] == "3" # No gas pipeline connection 
                    v_ogce = β*sum(vec_Wold_nopipe_ogce[t].*aux_1);
                    v_oge = β*sum(vec_Wold_nopipe_oge[t].*aux_1);
                    v_oce = β*sum(vec_Wold_nopipe_oce[t].*aux_1);
                    v_oe = β*sum(vec_Wold_nopipe_oe[t].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_nopipe + v_oge;
                        v_oce = - κc + v_oce;
                        v_ogce = - κg_nopipe - κc + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + v_oe;
                        v_oge = γc - κg_nopipe + v_oge;
                        v_ogce = - κg_nopipe + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_nopipe + v_oe;
                        v_oce = γg_nopipe - κc + v_oce;
                        v_ogce = - κc + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_nopipe + v_oe;
                        v_oce = γg_nopipe + v_oce;
                        v_oge = γc + v_oge;
                    end
                    vtilde = maximum([v_oe,v_oge,v_oce,v_ogce])
                    aux_2 = exp(v_oe-vtilde) + exp(v_oge-vtilde) + exp(v_oce-vtilde) + exp(v_ogce-vtilde) ;
                    if aux_2 == 0
                        Prnext_long[i] = 0
                        println(i)
                    else
                        if Data.FcombineF[i] == 12
                            Prnext_long[i] = exp(v_oe-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 123
                            Prnext_long[i] = exp(v_oce-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 124
                            Prnext_long[i] = exp(v_oge-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 1234
                            Prnext_long[i] = exp(v_ogce-vtilde)/aux_2; 
                        end
                    end
                    if Prnext_long[i] == 0;
                        Prnext_long[i] = 0.00000000000000000001;
                    end
                    if isnan(Prnext_long[i]) == true
                        println(i)
                        println(vtilde)
                    end
                elseif Data.Connection[i] == "direct" || Data.Connection[i] == "indirect" # gas pipeline connection
                    v_ogce = β*sum(vec_Wold_pipe_ogce[t].*aux_1);
                    v_oge = β*sum(vec_Wold_pipe_oge[t].*aux_1);
                    v_oce = β*sum(vec_Wold_pipe_oce[t].*aux_1);
                    v_oe = β*sum(vec_Wold_pipe_oe[t].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_pipe + v_oge;
                        v_oce = - κc + v_oce;
                        v_ogce = - κg_pipe - κc + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + v_oe;
                        v_oge = γc - κg_pipe + v_oge;
                        v_ogce = - κg_pipe + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_pipe + v_oe;
                        v_oce = γg_pipe - κc + v_oce;
                        v_ogce = - κc + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_pipe + v_oe;
                        v_oce = γg_pipe + v_oce;
                        v_oge = γc + v_oge;
                    end
                    vtilde = maximum([v_oe,v_oge,v_oce,v_ogce])
                    aux_2 = exp(v_oe-vtilde) + exp(v_oge-vtilde) + exp(v_oce-vtilde) + exp(v_ogce-vtilde) ;
                    if aux_2 == 0
                        Prnext_long[i] = 0
                        println(i)
                    else
                        if Data.FcombineF[i] == 12;
                            Prnext_long[i] = exp(v_oe-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 123;
                            Prnext_long[i] = exp(v_oce-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 124;
                            Prnext_long[i] = exp(v_oge-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 1234;
                            Prnext_long[i] = exp(v_ogce-vtilde)/aux_2; 
                        end
                    end
                    if Prnext_long[i] == 0;
                        Prnext_long[i] = 0.00000000000000000001;
                    end
                    if isnan(Prnext_long[i]) == true
                        println(i)
                        println(vtilde)
                    end
                end
            end
            return Prnext_long;
        end
    #
    # Version allowing for fixed costs to depend on productivity
        function choicePR_func_zfc(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,ig_re,ic_re,W_new_inner=nothing)
            @unpack W_new,p,ngrid,nstate,n_c,n_g = M;
            @unpack T,β = p;
            N = size(Data,1);
            #Prnext_long = Array{Float64}(undef,N);
            Prnext_long = zeros(N);
            vec_Wold_nopipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_nopipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_nopipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_nopipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            vec_Wold_pipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g)),T);
            for t=1:T 
                if W_new_inner == nothing 
                    vec_Wold_nopipe_oe[t] = vec(W_new[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[t] = vec(W_new[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[t] = vec(W_new[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[t] = vec(W_new[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[t] = vec(W_new[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[t] = vec(W_new[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[t] = vec(W_new[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[t] = vec(W_new[2,4,t,:,:,:,ic_re,ig_re])   ;
                else
                    vec_Wold_nopipe_oe[t] = vec(W_new_inner[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[t] = vec(W_new_inner[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[t] = vec(W_new_inner[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[t] = vec(W_new_inner[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[t] = vec(W_new_inner[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[t] = vec(W_new_inner[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[t] = vec(W_new_inner[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[t] = vec(W_new_inner[2,4,t,:,:,:,ic_re,ig_re])   ;
                end
            end
            # Fixed cost parameters
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            κz          = Κ[7];
            γz          = Κ[8];
            for i = 1:N
                # Get state indice
                is = grid_indices.s[i];
                # Transition probabilities
                aux_1 = kronecker(M.Π_g[1,:],M.Π_c[1,:],M.Πs[is,:]);
                t = Data.year[i]-2009;
                # Choice probabilities
                if Data.Connection[i] == "3" # No gas pipeline connection 
                    v_ogce = β*sum(vec_Wold_nopipe_ogce[t].*aux_1);
                    v_oge = β*sum(vec_Wold_nopipe_oge[t].*aux_1);
                    v_oce = β*sum(vec_Wold_nopipe_oce[t].*aux_1);
                    v_oe = β*sum(vec_Wold_nopipe_oe[t].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_nopipe -κz*Data.lnz[i] + v_oge;
                        v_oce = - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κg_nopipe - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + γz*Data.lnz[i] + v_oe;
                        v_oge = γc + γz*Data.lnz[i] - κg_nopipe -κz*Data.lnz[i] + v_oge;
                        v_ogce = - κg_nopipe -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_nopipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_nopipe + γz*Data.lnz[i] - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_nopipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_nopipe + γz*Data.lnz[i] + v_oce;
                        v_oge = γc + γz*Data.lnz[i] + v_oge;
                    end
                    vtilde = maximum([v_oe,v_oge,v_oce,v_ogce])
                    aux_2 = exp(v_oe-vtilde) + exp(v_oge-vtilde) + exp(v_oce-vtilde) + exp(v_ogce-vtilde) ;
                    if aux_2 == 0
                        Prnext_long[i] = 0
                        println(i)
                    else
                        if Data.FcombineF[i] == 12
                            Prnext_long[i] = exp(v_oe-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 123
                            Prnext_long[i] = exp(v_oce-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 124
                            Prnext_long[i] = exp(v_oge-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 1234
                            Prnext_long[i] = exp(v_ogce-vtilde)/aux_2; 
                        end
                    end
                    if Prnext_long[i] == 0;
                        Prnext_long[i] = 0.00000000000000000001;
                    end
                    if isnan(Prnext_long[i]) == true
                        println(i)
                        println(vtilde)
                    end
                elseif Data.Connection[i] == "direct" || Data.Connection[i] == "indirect" # gas pipeline connection
                    v_ogce = β*sum(vec_Wold_pipe_ogce[t].*aux_1);
                    v_oge = β*sum(vec_Wold_pipe_oge[t].*aux_1);
                    v_oce = β*sum(vec_Wold_pipe_oce[t].*aux_1);
                    v_oe = β*sum(vec_Wold_pipe_oe[t].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_pipe - κz*Data.lnz[i] + v_oge;
                        v_oce = - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κg_pipe - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + γz*Data.lnz[i] + v_oe;
                        v_oge = γc + γz*Data.lnz[i] - κg_pipe -κz*Data.lnz[i] + v_oge;
                        v_ogce = - κg_pipe -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_pipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_pipe + γz*Data.lnz[i] - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_pipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_pipe + γz*Data.lnz[i] + v_oce;
                        v_oge = γc + γz*Data.lnz[i] + v_oge;
                    end
                    vtilde = maximum([v_oe,v_oge,v_oce,v_ogce])
                    aux_2 = exp(v_oe-vtilde) + exp(v_oge-vtilde) + exp(v_oce-vtilde) + exp(v_ogce-vtilde) ;
                    if aux_2 == 0
                        Prnext_long[i] = 0
                        println(i)
                    else
                        if Data.FcombineF[i] == 12;
                            Prnext_long[i] = exp(v_oe-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 123;
                            Prnext_long[i] = exp(v_oce-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 124;
                            Prnext_long[i] = exp(v_oge-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 1234;
                            Prnext_long[i] = exp(v_ogce-vtilde)/aux_2; 
                        end
                    end
                    if Prnext_long[i] == 0;
                        Prnext_long[i] = 0.00000000000000000001;
                    end
                    if isnan(Prnext_long[i]) == true
                        println(i)
                        println(vtilde)
                    end
                end
            end
            return Prnext_long;
        end
    #
    # Version allowing for fixed costs to depend on productivity - fuel price/productivity as state var
        function choicePR_func_zfc_alt(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,ig_re,ic_re,W_new_inner=nothing)
            @unpack W_new,p,ngrid,nstate = M;
            @unpack T,R,β = p;
            N = size(Data,1);
            #Prnext_long = Array{Float64}(undef,N);
            Prnext_long = zeros(N);
            vec_Wold_nopipe_oe = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            vec_Wold_nopipe_oge = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            vec_Wold_nopipe_oce = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            vec_Wold_nopipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            vec_Wold_pipe_oe = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            vec_Wold_pipe_oge = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            vec_Wold_pipe_oce = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            vec_Wold_pipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate))),T,R);
            for t=1:T 
            for r = 1:R
                if W_new_inner == nothing 
                    vec_Wold_nopipe_oe[t,r] = vec(W_new[1,1,t,r,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[t,r] = vec(W_new[1,2,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[t,r] = vec(W_new[1,3,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[t,r] = vec(W_new[1,4,t,r,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[t,r] = vec(W_new[2,1,t,r,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[t,r] = vec(W_new[2,2,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[t,r] = vec(W_new[2,3,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[t,r] = vec(W_new[2,4,t,r,:,ic_re,ig_re])   ;
                else
                    vec_Wold_nopipe_oe[t,r] = vec(W_new_inner[1,1,t,r,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[t,r] = vec(W_new_inner[1,2,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[t,r] = vec(W_new_inner[1,3,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[t,r] = vec(W_new_inner[1,4,t,r,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[t,r] = vec(W_new_inner[2,1,t,r,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[t,r] = vec(W_new_inner[2,2,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[t,r] = vec(W_new_inner[2,3,t,r,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[t,r] = vec(W_new_inner[2,4,t,r,:,ic_re,ig_re])   ;
                end
            end
            end
            # Fixed cost parameters
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            κz          = Κ[7];
            γz          = Κ[8];
            for i = 1:N
                # Get state indice
                is = grid_indices.s[i];
                # Transition probabilities
                aux_1 = kronecker(M.Πs_oe[is,:]);
                t = Data.year[i]-2009;
                r = Data.region[i];
                # Choice probabilities
                if Data.Connection[i] == 3 # No gas pipeline connection 
                    v_ogce = β*sum(vec_Wold_nopipe_ogce[t,r].*aux_1);
                    v_oge = β*sum(vec_Wold_nopipe_oge[t,r].*aux_1);
                    v_oce = β*sum(vec_Wold_nopipe_oce[t,r].*aux_1);
                    v_oe = β*sum(vec_Wold_nopipe_oe[t,r].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_nopipe -κz*Data.lnz[i] + v_oge;
                        v_oce = - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κg_nopipe - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + γz*Data.lnz[i] + v_oe;
                        v_oge = γc + γz*Data.lnz[i] - κg_nopipe -κz*Data.lnz[i] + v_oge;
                        v_ogce = - κg_nopipe -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_nopipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_nopipe + γz*Data.lnz[i] - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_nopipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_nopipe + γz*Data.lnz[i] + v_oce;
                        v_oge = γc + γz*Data.lnz[i] + v_oge;
                    end
                    vtilde = maximum([v_oe,v_oge,v_oce,v_ogce])
                    aux_2 = exp(v_oe-vtilde) + exp(v_oge-vtilde) + exp(v_oce-vtilde) + exp(v_ogce-vtilde) ;
                    if aux_2 == 0
                        Prnext_long[i] = 0
                        println(i)
                    else
                        if Data.FcombineF[i] == 12
                            Prnext_long[i] = exp(v_oe-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 123
                            Prnext_long[i] = exp(v_oce-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 124
                            Prnext_long[i] = exp(v_oge-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 1234
                            Prnext_long[i] = exp(v_ogce-vtilde)/aux_2; 
                        end
                    end
                    if Prnext_long[i] == 0;
                        Prnext_long[i] = 0.00000000000000000001;
                    end
                    if isnan(Prnext_long[i]) == true
                        println(i)
                        println(vtilde)
                    end
                elseif Data.Connection[i] == 2 || Data.Connection[i] == 1 # gas pipeline connection
                    v_ogce = β*sum(vec_Wold_pipe_ogce[t,r].*aux_1);
                    v_oge = β*sum(vec_Wold_pipe_oge[t,r].*aux_1);
                    v_oce = β*sum(vec_Wold_pipe_oce[t,r].*aux_1);
                    v_oe = β*sum(vec_Wold_pipe_oe[t,r].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_pipe - κz*Data.lnz[i] + v_oge;
                        v_oce = - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κg_pipe - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + γz*Data.lnz[i] + v_oe;
                        v_oge = γc + γz*Data.lnz[i] - κg_pipe -κz*Data.lnz[i] + v_oge;
                        v_ogce = - κg_pipe -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 124 
                        v_oe = γg_pipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_pipe + γz*Data.lnz[i] - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_pipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_pipe + γz*Data.lnz[i] + v_oce;
                        v_oge = γc + γz*Data.lnz[i] + v_oge;
                    end
                    vtilde = maximum([v_oe,v_oge,v_oce,v_ogce]);
                    aux_2 = exp(v_oe-vtilde) + exp(v_oge-vtilde) + exp(v_oce-vtilde) + exp(v_ogce-vtilde) ;
                    if aux_2 == 0
                        Prnext_long[i] = 0
                        println(i)
                    else
                        if Data.FcombineF[i] == 12;
                            Prnext_long[i] = exp(v_oe-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 123;
                            Prnext_long[i] = exp(v_oce-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 124;
                            Prnext_long[i] = exp(v_oge-vtilde)/aux_2; 
                        elseif Data.FcombineF[i] == 1234;
                            Prnext_long[i] = exp(v_ogce-vtilde)/aux_2; 
                        end
                    end
                    if Prnext_long[i] == 0;
                        Prnext_long[i] = 0.00000000000000000001;
                    end
                    if isnan(Prnext_long[i]) == true
                        println(i)
                        println(vtilde)
                    end
                end
            end
            return Prnext_long;
        end
    #
#

### Functions that return choice-specific value functions
# Returns value function for all choices given observed state variables
    # Basline
        function choiceVF_func(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,W_new_inner=nothing)
            @unpack W_new,p,ngrid,nstate,n_c,n_g,nc_re,ng_re = M;
            @unpack T,β = p;
            N = size(Data,1);
            Prnext = Array{Float64}(undef,p.F_tot,N,nc_re,ng_re);
            vchoicef = Array{Float64}(undef,N,4);
            # Initialize value function
            vec_Wold_nopipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_nopipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_nopipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_nopipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            # Fixed cost parameters
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                for t=1:T 
                    if W_new_inner == nothing 
                        vec_Wold_nopipe_oe[t][:,ic_re,ig_re] = vec(W_new[1,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_nopipe_oge[t][:,ic_re,ig_re] = vec(W_new[1,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_oce[t][:,ic_re,ig_re] = vec(W_new[1,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_ogce[t][:,ic_re,ig_re] = vec(W_new[1,4,t,:,:,:,ic_re,ig_re])   ;
                        vec_Wold_pipe_oe[t][:,ic_re,ig_re] = vec(W_new[2,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_pipe_oge[t][:,ic_re,ig_re] = vec(W_new[2,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_oce[t][:,ic_re,ig_re] = vec(W_new[2,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_ogce[t][:,ic_re,ig_re] = vec(W_new[2,4,t,:,:,:,ic_re,ig_re])   ;
                    else
                        vec_Wold_nopipe_oe[t][:,ic_re,ig_re] = vec(W_new_inner[1,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_nopipe_oge[t][:,ic_re,ig_re] = vec(W_new_inner[1,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_oce[t][:,ic_re,ig_re] = vec(W_new_inner[1,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_ogce[t][:,ic_re,ig_re] = vec(W_new_inner[1,4,t,:,:,:,ic_re,ig_re])   ;
                        vec_Wold_pipe_oe[t][:,ic_re,ig_re] = vec(W_new_inner[2,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_pipe_oge[t][:,ic_re,ig_re] = vec(W_new_inner[2,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_oce[t][:,ic_re,ig_re] = vec(W_new_inner[2,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_ogce[t][:,ic_re,ig_re] = vec(W_new_inner[2,4,t,:,:,:,ic_re,ig_re])   ;
                    end
                end
            end
            for i = 1:N
                # Get state indice
                is = grid_indices.s[i];
                ic_re = grid_indices.c_re[i];
                ig_re = grid_indices.g_re[i];
                # Transition probabilities
                aux_1 = kronecker(M.Π_g[1,:],M.Π_c[1,:],M.Πs[is,:]);
                t = Data.year[i]-2009;
                # Choice probabilities
                if Data.Connection[i] == "3" # No gas pipeline connection
                    v_ogce = β*sum(vec_Wold_nopipe_ogce[t][:,ic_re,ig_re].*aux_1);
                    v_oge = β*sum(vec_Wold_nopipe_oge[t][:,ic_re,ig_re].*aux_1);
                    v_oce = β*sum(vec_Wold_nopipe_oce[t][:,ic_re,ig_re].*aux_1);
                    v_oe = β*sum(vec_Wold_nopipe_oe[t][:,ic_re,ig_re].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_nopipe + v_oge;
                        v_oce = - κc + v_oce;
                        v_ogce = - κg_nopipe - κc + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + v_oe;
                        v_oge = γc - κg_nopipe + v_oge;
                        v_ogce = - κg_nopipe + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_nopipe + v_oe;
                        v_oce = γg_nopipe - κc + v_oce;
                        v_ogce = - κc + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_nopipe + v_oe;
                        v_oce = γg_nopipe + v_oce;
                        v_oge = γc + v_oge;
                    end
                    # Expected value functions
                    vchoicef[i,1] = v_oe;
                    vchoicef[i,2] = v_oge;
                    vchoicef[i,3] = v_oce;
                    vchoicef[i,4] = v_ogce;
                elseif Data.Connection[i] == "direct" || Data.Connection[i] == "indirect" # gas pipeline connection
                    v_ogce = β*sum(vec_Wold_pipe_ogce[t][:,ic_re,ig_re].*aux_1);
                    v_oge = β*sum(vec_Wold_pipe_oge[t][:,ic_re,ig_re].*aux_1);
                    v_oce = β*sum(vec_Wold_pipe_oce[t][:,ic_re,ig_re].*aux_1);
                    v_oe = β*sum(vec_Wold_pipe_oe[t][:,ic_re,ig_re].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_pipe + v_oge;
                        v_oce = - κc + v_oce;
                        v_ogce = - κg_pipe - κc + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + v_oe;
                        v_oge = γc - κg_pipe + v_oge;
                        v_ogce = - κg_pipe + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_pipe + v_oe;
                        v_oce = γg_pipe - κc + v_oce;
                        v_ogce = - κc + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_pipe + v_oe;
                        v_oce = γg_pipe + v_oce;
                        v_oge = γc + v_oge;
                    end
                    # Expected value functions
                    vchoicef[i,1] = v_oe;
                    vchoicef[i,2] = v_oge;
                    vchoicef[i,3] = v_oce;
                    vchoicef[i,4] = v_ogce;
                end
            end
            return vchoicef;
        end
    # Allowing for fixed costs to depend on productivity
        function choiceVF_func_zfc(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,W_new_inner=nothing)
            @unpack W_new,p,ngrid,nstate,n_c,n_g,nc_re,ng_re,lnSgrid = M;
            @unpack T,β = p;
            N = size(Data,1);
            Prnext = Array{Float64}(undef,p.F_tot,N,nc_re,ng_re);
            vchoicef = Array{Float64}(undef,N,4);
            # Initialize value function
            vec_Wold_nopipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_nopipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_nopipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_nopipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_oe = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_oge = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_oce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            vec_Wold_pipe_ogce = fill(Array{Float64}(zeros((ngrid^nstate)*n_c*n_g,nc_re,ng_re)),T);
            # Fixed cost parameters
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            κz          = Κ[7];
            γz          = Κ[8];
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                for t=1:T 
                    if W_new_inner == nothing 
                        vec_Wold_nopipe_oe[t][:,ic_re,ig_re] = vec(W_new[1,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_nopipe_oge[t][:,ic_re,ig_re] = vec(W_new[1,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_oce[t][:,ic_re,ig_re] = vec(W_new[1,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_ogce[t][:,ic_re,ig_re] = vec(W_new[1,4,t,:,:,:,ic_re,ig_re])   ;
                        vec_Wold_pipe_oe[t][:,ic_re,ig_re] = vec(W_new[2,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_pipe_oge[t][:,ic_re,ig_re] = vec(W_new[2,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_oce[t][:,ic_re,ig_re] = vec(W_new[2,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_ogce[t][:,ic_re,ig_re] = vec(W_new[2,4,t,:,:,:,ic_re,ig_re])   ;
                    else
                        vec_Wold_nopipe_oe[t][:,ic_re,ig_re] = vec(W_new_inner[1,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_nopipe_oge[t][:,ic_re,ig_re] = vec(W_new_inner[1,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_oce[t][:,ic_re,ig_re] = vec(W_new_inner[1,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_nopipe_ogce[t][:,ic_re,ig_re] = vec(W_new_inner[1,4,t,:,:,:,ic_re,ig_re])   ;
                        vec_Wold_pipe_oe[t][:,ic_re,ig_re] = vec(W_new_inner[2,1,t,:,:,:,ic_re,ig_re])     ;
                        vec_Wold_pipe_oge[t][:,ic_re,ig_re] = vec(W_new_inner[2,2,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_oce[t][:,ic_re,ig_re] = vec(W_new_inner[2,3,t,:,:,:,ic_re,ig_re])    ;
                        vec_Wold_pipe_ogce[t][:,ic_re,ig_re] = vec(W_new_inner[2,4,t,:,:,:,ic_re,ig_re])   ;
                    end
                end
            end
            for i = 1:N
                # Get state indice
                is = grid_indices.s[i];
                ic_re = grid_indices.c_re[i];
                ig_re = grid_indices.g_re[i];
                # Transition probabilities
                aux_1 = kronecker(M.Π_g[1,:],M.Π_c[1,:],M.Πs[is,:]);
                t = Data.year[i]-2009;
                # Choice probabilities
                if Data.Connection[i] == "3" # No gas pipeline connection
                    v_ogce = β*sum(vec_Wold_nopipe_ogce[t][:,ic_re,ig_re].*aux_1);
                    v_oge = β*sum(vec_Wold_nopipe_oge[t][:,ic_re,ig_re].*aux_1);
                    v_oce = β*sum(vec_Wold_nopipe_oce[t][:,ic_re,ig_re].*aux_1);
                    v_oe = β*sum(vec_Wold_nopipe_oe[t][:,ic_re,ig_re].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_nopipe -κz*Data.lnz[i] + v_oge;
                        v_oce = - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κg_nopipe - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + γz*Data.lnz[i] + v_oe;
                        v_oge = γc + γz*Data.lnz[i] - κg_nopipe -κz*Data.lnz[i] + v_oge;
                        v_ogce = - κg_nopipe -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_nopipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_nopipe + γz*Data.lnz[i] - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_nopipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_nopipe + γz*Data.lnz[i] + v_oce;
                        v_oge = γc + γz*Data.lnz[i] + v_oge;
                    end
                # Expected value functions
                vchoicef[i,1] = v_oe;
                vchoicef[i,2] = v_oge;
                vchoicef[i,3] = v_oce;
                vchoicef[i,4] = v_ogce;
                elseif Data.Connection[i] == "direct" || Data.Connection[i] == "indirect" # gas pipeline connection
                    v_ogce = β*sum(vec_Wold_pipe_ogce[t][:,ic_re,ig_re].*aux_1);
                    v_oge = β*sum(vec_Wold_pipe_oge[t][:,ic_re,ig_re].*aux_1);
                    v_oce = β*sum(vec_Wold_pipe_oce[t][:,ic_re,ig_re].*aux_1);
                    v_oe = β*sum(vec_Wold_pipe_oe[t][:,ic_re,ig_re].*aux_1);
                    if Data.combineF[i] == 12
                        v_oge = - κg_pipe - κz*Data.lnz[i] + v_oge;
                        v_oce = - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κg_pipe - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 123
                        v_oe = γc + γz*Data.lnz[i] + v_oe;
                        v_oge = γc + γz*Data.lnz[i] - κg_pipe -κz*Data.lnz[i] + v_oge;
                        v_ogce = - κg_pipe -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 124
                        v_oe = γg_pipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_pipe + γz*Data.lnz[i] - κc -κz*Data.lnz[i] + v_oce;
                        v_ogce = - κc -κz*Data.lnz[i] + v_ogce;
                    elseif Data.combineF[i] == 1234
                        v_oe = γc + γg_pipe + γz*Data.lnz[i] + v_oe;
                        v_oce = γg_pipe + γz*Data.lnz[i] + v_oce;
                        v_oge = γc + γz*Data.lnz[i] + v_oge;
                    end
                    # Expected value functions
                    vchoicef[i,1] = v_oe;
                    vchoicef[i,2] = v_oge;
                    vchoicef[i,3] = v_oce;
                    vchoicef[i,4] = v_ogce;
                end
            end
            return vchoicef;
        end
    #
#
# Returns value function for all choices given all combination of state variables in the grid
    # Baseline
        function choiceVF_func_grid(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,W_new_inner=nothing)
            @unpack W_new,p,ngrid,nstate,n_c,n_g,nc_re,ng_re = M;
            @unpack T,β = p;
            N = size(Data,1);
            vchoicef = Array{Float64}(undef,N,4,nc_re,ng_re);
            vec_Wold_nopipe_oe = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_nopipe_oge = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_nopipe_oce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_nopipe_ogce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_oe = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_oge = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_oce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_ogce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            for j = 1:(T*nc_re*ng_re)
                t,ic_re,ig_re = Tuple(CartesianIndices((T,nc_re,ng_re))[j]);
                if W_new_inner == nothing 
                    vec_Wold_nopipe_oe[:,t,ic_re,ig_re] = vec(W_new[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[:,t,ic_re,ig_re] = vec(W_new[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[:,t,ic_re,ig_re] = vec(W_new[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[:,t,ic_re,ig_re] = vec(W_new[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[:,t,ic_re,ig_re] = vec(W_new[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[:,t,ic_re,ig_re] = vec(W_new[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[:,t,ic_re,ig_re] = vec(W_new[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[:,t,ic_re,ig_re] = vec(W_new[2,4,t,:,:,:,ic_re,ig_re])   ;
                else
                    vec_Wold_nopipe_oe[:,t,ic_re,ig_re] = vec(W_new_inner[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[:,t,ic_re,ig_re] = vec(W_new_inner[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[:,t,ic_re,ig_re] = vec(W_new_inner[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[:,t,ic_re,ig_re] = vec(W_new_inner[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[:,t,ic_re,ig_re] = vec(W_new_inner[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[:,t,ic_re,ig_re] = vec(W_new_inner[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[:,t,ic_re,ig_re] = vec(W_new_inner[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[:,t,ic_re,ig_re] = vec(W_new_inner[2,4,t,:,:,:,ic_re,ig_re])   ;
                end
            end
            # Fixed cost parameters
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            for i = 1:N
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    # Get state indices
                    is = grid_indices.s[i];
                    # Transition probabilities
                    aux_1 = kronecker(M.Π_g[1,:],M.Π_c[1,:],M.Πs[is,:]);
                    t = Data.year[i]-2009;
                    if Data.Connection[i] == "3" # No gas pipeline connection 
                        # Choice probabilities
                        v_ogce = β*sum(vec_Wold_nopipe_ogce[:,t,ic_re,ig_re].*aux_1);
                        v_oge = β*sum(vec_Wold_nopipe_oge[:,t,ic_re,ig_re].*aux_1);
                        v_oce = β*sum(vec_Wold_nopipe_oce[:,t,ic_re,ig_re].*aux_1);
                        v_oe = β*sum(vec_Wold_nopipe_oe[:,t,ic_re,ig_re].*aux_1);
                        if Data.combineF[i] == 12
                            v_oge = - κg_nopipe + v_oge;
                            v_oce = - κc + v_oce;
                            v_ogce = - κg_nopipe - κc + v_ogce;
                        elseif Data.combineF[i] == 123
                            v_oe = γc + v_oe;
                            v_oge = γc - κg_nopipe + v_oge;
                            v_ogce = - κg_nopipe + v_ogce;
                        elseif Data.combineF[i] == 124
                            v_oe = γg_nopipe + v_oe;
                            v_oce = γg_nopipe - κc + v_oce;
                            v_ogce = - κc + v_ogce;
                        elseif Data.combineF[i] == 1234
                            v_oe = γc + γg_nopipe + v_oe;
                            v_oce = γg_nopipe + v_oce;
                            v_oge = γc + v_oge;
                        end
                        vchoicef[i,1,ic_re,ig_re] = v_oe;
                        vchoicef[i,2,ic_re,ig_re] = v_oge;
                        vchoicef[i,3,ic_re,ig_re] = v_oce;
                        vchoicef[i,4,ic_re,ig_re] = v_ogce;
                    elseif Data.Connection[i] == "direct" || Data.Connection[i] == "indirect" # gas pipeline connection
                        # Choice probabilities
                        v_ogce = β*sum(vec_Wold_pipe_ogce[:,t,ic_re,ig_re].*aux_1);
                        v_oge = β*sum(vec_Wold_pipe_oge[:,t,ic_re,ig_re].*aux_1);
                        v_oce = β*sum(vec_Wold_pipe_oce[:,t,ic_re,ig_re].*aux_1);
                        v_oe = β*sum(vec_Wold_pipe_oe[:,t,ic_re,ig_re].*aux_1);
                        if Data.combineF[i] == 12
                            v_oge = - κg_pipe + v_oge;
                            v_oce = - κc + v_oce;
                            v_ogce = - κg_pipe - κc + v_ogce;
                        elseif Data.combineF[i] == 123
                            v_oe = γc + v_oe;
                            v_oge = γc - κg_pipe + v_oge;
                            v_ogce = - κg_pipe + v_ogce;
                        elseif Data.combineF[i] == 124
                            v_oe = γg_pipe + v_oe;
                            v_oce = γg_pipe - κc + v_oce;
                            v_ogce = - κc + v_ogce;
                        elseif Data.combineF[i] == 1234
                            v_oe = γc + γg_pipe+ v_oe;
                            v_oce = γg_pipe + v_oce;
                            v_oge = γc + v_oge;
                        end
                        vchoicef[i,1,ic_re,ig_re] = v_oe;
                        vchoicef[i,2,ic_re,ig_re] = v_oge;
                        vchoicef[i,3,ic_re,ig_re] = v_oce;
                        vchoicef[i,4,ic_re,ig_re] = v_ogce;
                    end 
                end
            end
            return vchoicef;
        end
    #
    # Allowing for fixed costs to depend on productivity
        function choiceVF_func_grid_zfc(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,W_new_inner=nothing)
            @unpack W_new,p,ngrid,nstate,n_c,n_g,nc_re,ng_re,lnSgrid = M;
            @unpack T,β = p;
            N = size(Data,1);
            vchoicef = Array{Float64}(undef,N,4,nc_re,ng_re);
            vec_Wold_nopipe_oe = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_nopipe_oge = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_nopipe_oce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_nopipe_ogce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_oe = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_oge = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_oce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            vec_Wold_pipe_ogce = zeros((ngrid^nstate)*n_c*n_g,T,nc_re,ng_re);
            for j = 1:(T*nc_re*ng_re)
                t,ic_re,ig_re = Tuple(CartesianIndices((T,nc_re,ng_re))[j]);
                if W_new_inner == nothing 
                    vec_Wold_nopipe_oe[:,t,ic_re,ig_re] = vec(W_new[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[:,t,ic_re,ig_re] = vec(W_new[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[:,t,ic_re,ig_re] = vec(W_new[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[:,t,ic_re,ig_re] = vec(W_new[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[:,t,ic_re,ig_re] = vec(W_new[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[:,t,ic_re,ig_re] = vec(W_new[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[:,t,ic_re,ig_re] = vec(W_new[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[:,t,ic_re,ig_re] = vec(W_new[2,4,t,:,:,:,ic_re,ig_re])   ;
                else
                    vec_Wold_nopipe_oe[:,t,ic_re,ig_re] = vec(W_new_inner[1,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_nopipe_oge[:,t,ic_re,ig_re] = vec(W_new_inner[1,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_oce[:,t,ic_re,ig_re] = vec(W_new_inner[1,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_nopipe_ogce[:,t,ic_re,ig_re] = vec(W_new_inner[1,4,t,:,:,:,ic_re,ig_re])   ;
                    vec_Wold_pipe_oe[:,t,ic_re,ig_re] = vec(W_new_inner[2,1,t,:,:,:,ic_re,ig_re])     ;
                    vec_Wold_pipe_oge[:,t,ic_re,ig_re] = vec(W_new_inner[2,2,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_oce[:,t,ic_re,ig_re] = vec(W_new_inner[2,3,t,:,:,:,ic_re,ig_re])    ;
                    vec_Wold_pipe_ogce[:,t,ic_re,ig_re] = vec(W_new_inner[2,4,t,:,:,:,ic_re,ig_re])   ;
                end
            end
            # Fixed cost parameters
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            κz          = Κ[7];
            γz          = Κ[8];
            for i = 1:N
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    # Get state indices
                    is = grid_indices.s[i];
                    # Transition probabilities
                    aux_1 = kronecker(M.Π_g[1,:],M.Π_c[1,:],M.Πs[is,:]);
                    t = Data.year[i]-2009;
                    if Data.Connection[i] == "3" # No gas pipeline connection 
                        # Choice probabilities
                        v_ogce = β*sum(vec_Wold_nopipe_ogce[:,t,ic_re,ig_re].*aux_1);
                        v_oge = β*sum(vec_Wold_nopipe_oge[:,t,ic_re,ig_re].*aux_1);
                        v_oce = β*sum(vec_Wold_nopipe_oce[:,t,ic_re,ig_re].*aux_1);
                        v_oe = β*sum(vec_Wold_nopipe_oe[:,t,ic_re,ig_re].*aux_1);
                        if Data.combineF[i] == 12
                            v_oge = - κg_nopipe - κz*lnSgrid[is,1] + v_oge;
                            v_oce = - κc - κz*lnSgrid[is,1] + v_oce;
                            v_ogce = - κg_nopipe - κc - κz*lnSgrid[is,1] + v_ogce;
                        elseif Data.combineF[i] == 123
                            v_oe = γc + γz*lnSgrid[is,1] + v_oe;
                            v_oge = γc + γz*lnSgrid[is,1] - κg_nopipe - κz*lnSgrid[is,1] + v_oge;
                            v_ogce = - κg_nopipe - κz*lnSgrid[is,1] + v_ogce;
                        elseif Data.combineF[i] == 124
                            v_oe = γg_nopipe + γz*lnSgrid[is,1] + v_oe;
                            v_oce = γg_nopipe + γz*lnSgrid[is,1] - κc - κz*lnSgrid[is,1] + v_oce;
                            v_ogce = - κc - κz*lnSgrid[is,1] + v_ogce;
                        elseif Data.combineF[i] == 1234
                            v_oe = γc + γg_nopipe + γz*lnSgrid[is,1] + v_oe;
                            v_oce = γg_nopipe + γz*lnSgrid[is,1] + v_oce;
                            v_oge = γc + γz*lnSgrid[is,1] + v_oge;
                        end
                        vchoicef[i,1,ic_re,ig_re] = v_oe;
                        vchoicef[i,2,ic_re,ig_re] = v_oge;
                        vchoicef[i,3,ic_re,ig_re] = v_oce;
                        vchoicef[i,4,ic_re,ig_re] = v_ogce;
                    elseif Data.Connection[i] == "direct" || Data.Connection[i] == "indirect" # gas pipeline connection
                        # Choice probabilities
                        v_ogce = β*sum(vec_Wold_pipe_ogce[:,t,ic_re,ig_re].*aux_1);
                        v_oge = β*sum(vec_Wold_pipe_oge[:,t,ic_re,ig_re].*aux_1);
                        v_oce = β*sum(vec_Wold_pipe_oce[:,t,ic_re,ig_re].*aux_1);
                        v_oe = β*sum(vec_Wold_pipe_oe[:,t,ic_re,ig_re].*aux_1);
                        if Data.combineF[i] == 12
                            v_oge = - κg_pipe - κz*lnSgrid[is,1] + v_oge;
                            v_oce = - κc - κz*lnSgrid[is,1] + v_oce;
                            v_ogce = - κg_pipe - κc - κz*lnSgrid[is,1] + v_ogce;
                        elseif Data.combineF[i] == 123
                            v_oe = γc + γz*lnSgrid[is,1] + v_oe;
                            v_oge = γc + γz*lnSgrid[is,1] - κg_pipe - κz*lnSgrid[is,1] + v_oge;
                            v_ogce = - κg_pipe - κz*lnSgrid[is,1] + v_ogce;
                        elseif Data.combineF[i] == 124
                            v_oe = γg_pipe + γz*lnSgrid[is,1] + v_oe;
                            v_oce = γg_pipe + γz*lnSgrid[is,1] - κc - κz*lnSgrid[is,1] + v_oce;
                            v_ogce = - κc - κz*lnSgrid[is,1] + v_ogce;
                        elseif Data.combineF[i] == 1234
                            v_oe = γc + γg_pipe + γz*lnSgrid[is,1] + v_oe;
                            v_oce = γg_pipe + γz*lnSgrid[is,1] + v_oce;
                            v_oge = γc + γz*lnSgrid[is,1] + v_oge;
                        end
                        vchoicef[i,1,ic_re,ig_re] = v_oe;
                        vchoicef[i,2,ic_re,ig_re] = v_oge;
                        vchoicef[i,3,ic_re,ig_re] = v_oce;
                        vchoicef[i,4,ic_re,ig_re] = v_ogce;
                    end 
                end
            end
            return vchoicef;
        end
    #
#
#

### Functions that update posterior conditional and unconditional probabilities of gas/coal comparative advantages given fixed costs
    # Baseline
        function PosteriorRE_func(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ)
            @unpack p,π_uncond,nc_re,ng_re = M;
            @unpack T = p;
            N = size(Data,1);
            Nfirms = size(unique(Data.IDnum))[1];
            Prnext_long = Array{Float64}(undef,N,nc_re,ng_re);
            π_cond = Array{Float64}(undef,Nfirms,nc_re,ng_re);
            π_uncond_new = Array{Float64}(undef,nc_re,ng_re);
            π_uncond_marginal_c = Array{Float64}(undef,nc_re);
            π_uncond_marginal_g = Array{Float64}(undef,ng_re); 
            numer = Array{Float64}(undef,Nfirms,nc_re,ng_re);
            # Get choice probabilities given all combination of random effects
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                Prnext_long[:,ic_re,ig_re] = choicePR_func(M,Data,grid_indices,Κ,ig_re,ic_re);
            end
            # Update conditional probability of each random effect by unique plant
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                Data.Prnext = Prnext_long[:,ic_re,ig_re];
                Data_ID = groupby(Data,:IDnum);
                for i = 1:Nfirms
                    numer[i,ic_re,ig_re] = M.π_uncond[ic_re,ig_re]*prod(Data_ID[i].Prnext);
                end
            end
            for i = 1:Nfirms
                denom = sum(numer[i,:,:]);
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    π_cond[i,ic_re,ig_re] =  numer[i,ic_re,ig_re]/denom;
                end
            end
            # Update unconditional probability of each random effects
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                π_uncond_new[ic_re,ig_re] = sum(π_cond[:,ic_re,ig_re])/Nfirms;
            end
            # Get new unconditional probabilities (marginal of c,g separately) - law of total probability
            for ic_re = 1:M.nc_re
                π_uncond_marginal_c[ic_re] = sum(π_uncond_new[ic_re,:]);
            end
            for ig_re = 1:M.ng_re
                π_uncond_marginal_g[ig_re] = sum(π_uncond_new[:,ig_re]);
            end
            # Compute new mean and variance, and print
            μg_new = sum(π_uncond_marginal_g.*M.lng_re_grid);
            μc_new = sum(π_uncond_marginal_c.*M.lnc_re_grid);
            σg_new = sum(((M.lng_re_grid.-μg_new).^2).*π_uncond_marginal_g);
            σc_new = sum(((M.lnc_re_grid.-μc_new).^2).*π_uncond_marginal_c);
            println("---------------------------------------------------------------------------------")
            println("-----------------  UPDATING DISTRIBUTION OF RANDOM EFFECTS  ----------------------")
            println("old μg_re = $(M.μg_re), new μg_re = $μg_new")
            println("old μc_re = $(M.μc_re), new μc_re = $μc_new")
            println("old σg_re = $(M.σg_re), new σg_re = $σg_new")
            println("old σc_re = $(M.σc_re), new σc_re = $σc_new")
            println("Old unconditional distribution of fixed effects = $(π_uncond)")
            println("New unconditional distribution of fixed effects = $(π_uncond_new)")
            #println("L2 distance between iterations = $(sqrt((M.μg_re-μg_new)^2 + (M.μc_re-μc_new)^2 + (M.σg_re-σg_new)^2 + (M.σc_re-σc_new)^2))")
            println("---------------------------------------------------------------------------------")
            # Update model
            M = Model(M; μg_re = copy(μg_new), μc_re = copy(μc_new), σg_re = copy(σg_new), σc_re = copy(σc_new), Πg_re = copy(π_uncond_marginal_c), Πc_re = copy(π_uncond_marginal_c), π_uncond = copy(π_uncond_new));
            return M,π_cond,Prnext_long;
        end
    #
    # Allowing for fixed costs to depend on productivity
        # Function for estimation
            function PosteriorRE_func_zfc(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ)
                @unpack p,π_uncond,nc_re,ng_re = M;
                @unpack T = p;
                N = size(Data,1);
                Nfirms = size(unique(Data.IDnum))[1];
                Prnext_long = Array{Float64}(undef,N,nc_re,ng_re);
                π_cond = Array{Float64}(undef,Nfirms,nc_re,ng_re);
                π_uncond_new = Array{Float64}(undef,nc_re,ng_re);
                π_uncond_marginal_c = Array{Float64}(undef,nc_re);
                π_uncond_marginal_g = Array{Float64}(undef,ng_re); 
                numer = Array{Float64}(undef,Nfirms,nc_re,ng_re);
                # Get choice probabilities given all combination of random effects
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    Prnext_long[:,ic_re,ig_re] = choicePR_func_zfc(M,Data,grid_indices,Κ,ig_re,ic_re);
                end
                # Update conditional probability of each random effect by unique plant
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    Data.Prnext = Prnext_long[:,ic_re,ig_re];
                    Data_ID = groupby(Data,:IDnum);
                    for i = 1:Nfirms
                        numer[i,ic_re,ig_re] = M.π_uncond[ic_re,ig_re]*prod(Data_ID[i].Prnext);
                    end
                end
                for i = 1:Nfirms
                    denom = sum(numer[i,:,:]);
                    for j = 1:(nc_re*ng_re)
                        ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                        π_cond[i,ic_re,ig_re] =  numer[i,ic_re,ig_re]/denom;
                    end
                end
                # Update unconditional probability of each random effects
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    π_uncond_new[ic_re,ig_re] = sum(π_cond[:,ic_re,ig_re])/Nfirms;
                end
                # Get new unconditional probabilities (marginal of c,g separately) - law of total probability
                for ic_re = 1:M.nc_re
                    π_uncond_marginal_c[ic_re] = sum(π_uncond_new[ic_re,:]);
                end
                for ig_re = 1:M.ng_re
                    π_uncond_marginal_g[ig_re] = sum(π_uncond_new[:,ig_re]);
                end
                # Compute new mean and variance, and print
                μg_new = sum(π_uncond_marginal_g.*M.lng_re_grid);
                μc_new = sum(π_uncond_marginal_c.*M.lnc_re_grid);
                σg_new = sum(((M.lng_re_grid.-μg_new).^2).*π_uncond_marginal_g);
                σc_new = sum(((M.lnc_re_grid.-μc_new).^2).*π_uncond_marginal_c);
                println("---------------------------------------------------------------------------------")
                println("-----------------  UPDATING DISTRIBUTION OF RANDOM EFFECTS  ----------------------")
                println("old μg_re = $(M.μg_re), new μg_re = $μg_new")
                println("old μc_re = $(M.μc_re), new μc_re = $μc_new")
                println("old σg_re = $(M.σg_re), new σg_re = $σg_new")
                println("old σc_re = $(M.σc_re), new σc_re = $σc_new")
                println("Old unconditional distribution of fixed effects = $(π_uncond)")
                println("New unconditional distribution of fixed effects = $(π_uncond_new)")
                #println("L2 distance between iterations = $(sqrt((M.μg_re-μg_new)^2 + (M.μc_re-μc_new)^2 + (M.σg_re-σg_new)^2 + (M.σc_re-σc_new)^2))")
                println("---------------------------------------------------------------------------------")
                # Update model
                M = Model(M; μg_re = copy(μg_new), μc_re = copy(μc_new), σg_re = copy(σg_new), σc_re = copy(σc_new), Πg_re = copy(π_uncond_marginal_c), Πc_re = copy(π_uncond_marginal_c), π_uncond = copy(π_uncond_new));
                return M,π_cond,Prnext_long;
            end
            # fuel price/productivity as state var 
            function PosteriorRE_func_zfc_alt(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ)
                @unpack p,π_uncond,nc_re,ng_re = M;
                @unpack T = p;
                N = size(Data,1);
                Nfirms = size(unique(Data.IDnum))[1];
                Prnext_long = Array{Float64}(undef,N,nc_re,ng_re);
                π_cond = Array{Float64}(undef,Nfirms,nc_re,ng_re);
                π_uncond_new = Array{Float64}(undef,nc_re,ng_re);
                π_uncond_marginal_c = Array{Float64}(undef,nc_re);
                π_uncond_marginal_g = Array{Float64}(undef,ng_re); 
                numer = Array{Float64}(undef,Nfirms,nc_re,ng_re);
                # Get choice probabilities given all combination of random effects
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    Prnext_long[:,ic_re,ig_re] = choicePR_func_zfc_alt(M,Data,grid_indices,Κ,ig_re,ic_re);
                end
                # Update conditional probability of each random effect by unique plant
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    Data.Prnext = Prnext_long[:,ic_re,ig_re];
                    Data_ID = groupby(Data,:IDnum);
                    for i = 1:Nfirms
                        numer[i,ic_re,ig_re] = M.π_uncond[ic_re,ig_re]*prod(Data_ID[i].Prnext);
                    end
                end
                for i = 1:Nfirms
                    denom = sum(numer[i,:,:]);
                    for j = 1:(nc_re*ng_re)
                        ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                        π_cond[i,ic_re,ig_re] =  numer[i,ic_re,ig_re]/denom;
                    end
                end
                # Update unconditional probability of each random effects
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    π_uncond_new[ic_re,ig_re] = sum(π_cond[:,ic_re,ig_re])/Nfirms;
                end
                # Get new unconditional probabilities (marginal of c,g separately) - law of total probability
                for ic_re = 1:M.nc_re
                    π_uncond_marginal_c[ic_re] = sum(π_uncond_new[ic_re,:]);
                end
                for ig_re = 1:M.ng_re
                    π_uncond_marginal_g[ig_re] = sum(π_uncond_new[:,ig_re]);
                end
                # Compute new mean and variance, and print
                μg_new = sum(π_uncond_marginal_g.*M.lng_re_grid);
                μc_new = sum(π_uncond_marginal_c.*M.lnc_re_grid);
                σg_new = sum(((M.lng_re_grid.-μg_new).^2).*π_uncond_marginal_g);
                σc_new = sum(((M.lnc_re_grid.-μc_new).^2).*π_uncond_marginal_c);
                println("---------------------------------------------------------------------------------")
                println("-----------------  UPDATING DISTRIBUTION OF RANDOM EFFECTS  ----------------------")
                println("old μg_re = $(M.μg_re), new μg_re = $μg_new")
                println("old μc_re = $(M.μc_re), new μc_re = $μc_new")
                println("old σg_re = $(M.σg_re), new σg_re = $σg_new")
                println("old σc_re = $(M.σc_re), new σc_re = $σc_new")
                println("Old unconditional distribution of fixed effects = $(π_uncond)")
                println("New unconditional distribution of fixed effects = $(π_uncond_new)")
                #println("L2 distance between iterations = $(sqrt((M.μg_re-μg_new)^2 + (M.μc_re-μc_new)^2 + (M.σg_re-σg_new)^2 + (M.σc_re-σc_new)^2))")
                println("---------------------------------------------------------------------------------")
                # Update model
                M = Model(M; μg_re = copy(μg_new), μc_re = copy(μc_new), σg_re = copy(σg_new), σc_re = copy(σc_new), Πg_re = copy(π_uncond_marginal_c), Πc_re = copy(π_uncond_marginal_c), π_uncond = copy(π_uncond_new));
                return M,π_cond,Prnext_long;
            end
        #
        # Function for counterfactual
        function PosteriorRE_func_zfc_cf(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ)
            @unpack p = M;
            @unpack T,π_uncond,nc_re,ng_re = p;
            N = size(Data,1);
            Nfirms = size(unique(Data.IDnum))[1];
            Prnext_long = Array{Float64}(undef,N,nc_re,ng_re);
            π_cond = Array{Float64}(undef,Nfirms,nc_re,ng_re);
            π_uncond_new = Array{Float64}(undef,nc_re,ng_re);
            π_uncond_marginal_c = Array{Float64}(undef,nc_re);
            π_uncond_marginal_g = Array{Float64}(undef,ng_re); 
            numer = Array{Float64}(undef,Nfirms,nc_re,ng_re);
            # Get choice probabilities given all combination of random effects
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                Prnext_long[:,ic_re,ig_re] = choicePR_func_zfc(M,Data,grid_indices,Κ,ig_re,ic_re);
            end
            # Update conditional probability of each random effect by unique plant
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                Data.Prnext = Prnext_long[:,ic_re,ig_re];
                Data_ID = groupby(Data,:IDnum);
                for i = 1:Nfirms
                    numer[i,ic_re,ig_re] = p.π_uncond[ic_re,ig_re]*prod(Data_ID[i].Prnext);
                end
            end
            for i = 1:Nfirms
                denom = sum(numer[i,:,:]);
                for j = 1:(nc_re*ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                    π_cond[i,ic_re,ig_re] =  numer[i,ic_re,ig_re]/denom;
                end
            end
            # Update unconditional probability of each random effects
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                π_uncond_new[ic_re,ig_re] = sum(π_cond[:,ic_re,ig_re])/Nfirms;
            end
            # Get new unconditional probabilities (marginal of c,g separately) - law of total probability
            for ic_re = 1:M.nc_re
                π_uncond_marginal_c[ic_re] = sum(π_uncond_new[ic_re,:]);
            end
            for ig_re = 1:M.ng_re
                π_uncond_marginal_g[ig_re] = sum(π_uncond_new[:,ig_re]);
            end
            # Compute new mean and variance, and print
            μg_new = sum(π_uncond_marginal_g.*p.lng_re_grid);
            μc_new = sum(π_uncond_marginal_c.*p.lnc_re_grid);
            σg_new = sum(((p.lng_re_grid.-μg_new).^2).*π_uncond_marginal_g);
            σc_new = sum(((p.lnc_re_grid.-μc_new).^2).*π_uncond_marginal_c);
            println("---------------------------------------------------------------------------------")
            println("-----------------  UPDATING DISTRIBUTION OF RANDOM EFFECTS  ----------------------")
            println("old μg_re = $(p.μg_re), new μg_re = $μg_new")
            println("old μc_re = $(p.μc_re), new μc_re = $μc_new")
            println("old σg_re = $(p.σg_re), new σg_re = $σg_new")
            println("old σc_re = $(p.σc_re), new σc_re = $σc_new")
            println("Old unconditional distribution of fixed effects = $(π_uncond)")
            println("New unconditional distribution of fixed effects = $(π_uncond_new)")
            #println("L2 distance between iterations = $(sqrt((M.μg_re-μg_new)^2 + (M.μc_re-μc_new)^2 + (M.σg_re-σg_new)^2 + (M.σc_re-σc_new)^2))")
            println("---------------------------------------------------------------------------------")
            # Update model
            #M = Model(M; μg_re = copy(μg_new), μc_re = copy(μc_new), σg_re = copy(σg_new), σc_re = copy(σc_new), Πg_re = copy(π_uncond_marginal_c), Πc_re = copy(π_uncond_marginal_c), π_uncond = copy(π_uncond_new));
            M = Model(M; π_cond = copy(π_cond), π_uncond = copy(π_uncond_new));
            return M,π_cond,Prnext_long;
        end
    #
#

### Function that returns the likelihood given fixed costs 
    function loglik_func(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ)
        @unpack p,nc_re,ng_re = M;
        N = size(Data,1);
        Nfirms = size(unique(Data.IDnum))[1];
        # Update choice probability conditional on random effects
        Prnext_long = Array{Float64}(undef,N,nc_re,ng_re);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            Prnext_long[:,ic_re,ig_re] = choicePR_func(M,Data,grid_indices,Κ,ig_re,ic_re,M.W_new);
        end
        # Evalutate the current conditional log-likelihood
        indloglik = Array{Float64}(undef,Nfirms,nc_re,ng_re);
        # Get conditional distribution of random effects
        π_uncond = p.π_uncond;
        M,π_cond,Prnext_long = PosteriorRE_func(M,Data,grid_indices,Κ)
        #Data_ID = groupby(Data,:IDnum);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            Data.Prnext = Prnext_long[:,ic_re,ig_re];
            Data_ID = groupby(Data,:IDnum);
            for ifirm = 1:Nfirms
                # Individual likelihood (by unique plants)
                indloglik[ifirm,ic_re,ig_re] = π_cond[ifirm,ic_re,ig_re]*sum(log.(Data_ID[ifirm].Prnext));
            end
        end
        loglik = sum(indloglik);
        # Evaluate the current unconditional log-likelihood
        π_uncond = zeros(nc_re,ng_re);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            π_uncond[ic_re,ig_re] = (1/Nfirms)*sum(π_cond[:,ic_re,ig_re]);
        end
        # if sum(π_uncond) ≈ 1 != true
        #     error("Probability do not sum to 1")
        # end
        indloglik_uncond = zeros(Nfirms,nc_re,ng_re); 
        Data_ID = groupby(Data,:IDnum);
        for i = 1:Nfirms
            itotal = Data_ID[i].id;
            pr_aux = zeros(size(itotal,1),nc_re,ng_re);
            for ii in eachindex(itotal)
                pr_aux[ii,:,:] = Prnext_long[itotal[ii],:,:];
            end
            aux_1 = zeros(nc_re,ng_re);
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                aux_1[ic_re,ig_re] = π_uncond[ic_re,ig_re]*prod(pr_aux[:,ic_re,ig_re]);
            end
            indloglik_uncond[i] = log(sum(aux_1));
        end
        loglik_uncond = sum(indloglik_uncond);
        println("----------------------------------------------------------------------------")
        println("--------------------  CURRENT LIKELIHOOD EVALUATION ------------------------")
        println("Current conditional log likelihood = $loglik")
        println("Current unconditional log likelihood = $loglik_uncond")
        println("Current fixed costs = $Κ")
        println("----------------------------------------------------------------------------")
        return -loglik;
    end
    # For counterfactual
    function loglik_func_zfc_cf(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ)
        @unpack p,nc_re,ng_re = M;
        N = size(Data,1);
        Nfirms = size(unique(Data.IDnum))[1];
        # Update choice probability conditional on random effects
        Prnext_long = Array{Float64}(undef,N,nc_re,ng_re);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            Prnext_long[:,ic_re,ig_re] = choicePR_func_zfc(M,Data,grid_indices,Κ,ig_re,ic_re,M.W_new);
        end
        # Evalutate the current conditional log-likelihood
        indloglik = Array{Float64}(undef,Nfirms,nc_re,ng_re);
        # Get conditional distribution of random effects
        π_uncond = p.π_uncond;
        M,π_cond,Prnext_long = PosteriorRE_func_zfc_cf(M,Data,grid_indices,Κ)
        #Data_ID = groupby(Data,:IDnum);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            Data.Prnext = Prnext_long[:,ic_re,ig_re];
            Data_ID = groupby(Data,:IDnum);
            for ifirm = 1:Nfirms
                # Individual likelihood (by unique plants)
                indloglik[ifirm,ic_re,ig_re] = π_cond[ifirm,ic_re,ig_re]*sum(log.(Data_ID[ifirm].Prnext));
            end
        end
        loglik = sum(indloglik);
        # Evaluate the current unconditional log-likelihood
        π_uncond = zeros(nc_re,ng_re);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            π_uncond[ic_re,ig_re] = (1/Nfirms)*sum(π_cond[:,ic_re,ig_re]);
        end
        # if sum(π_uncond) ≈ 1 != true
        #     error("Probability do not sum to 1")
        # end
        indloglik_uncond = zeros(Nfirms,nc_re,ng_re); 
        Data_ID = groupby(Data,:IDnum);
        for i = 1:Nfirms
            itotal = Data_ID[i].id;
            pr_aux = zeros(size(itotal,1),nc_re,ng_re);
            for ii in eachindex(itotal)
                pr_aux[ii,:,:] = Prnext_long[itotal[ii],:,:];
            end
            aux_1 = zeros(nc_re,ng_re);
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                aux_1[ic_re,ig_re] = π_uncond[ic_re,ig_re]*prod(pr_aux[:,ic_re,ig_re]);
            end
            indloglik_uncond[i] = log(sum(aux_1));
        end
        loglik_uncond = sum(indloglik_uncond);
        println("----------------------------------------------------------------------------")
        println("--------------------  CURRENT LIKELIHOOD EVALUATION ------------------------")
        println("Current conditional log likelihood = $loglik")
        println("Current unconditional log likelihood = $loglik_uncond")
        println("Current fixed costs = $Κ")
        println("----------------------------------------------------------------------------")
        return -loglik;
    end
#
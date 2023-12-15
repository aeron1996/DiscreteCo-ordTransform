module TransformM
export Transform, map_point, Transform_Const

using LinearAlgebra
using Delaunay
using Shuffle
using Base.Threads
using Infinity
using ChunkSplitters

mutable struct Transform
    dimensions::Int64
    values0::Matrix{Float64}
    values1::Matrix{Float64}
    n_points::Int64
    mesh::Triangulation
    n_simplices::Int64
    mapping::Dict{Set{Float64}, Matrix{Float64}}
end

function Transform_Const(dimensions::Int64,values0::Matrix{Float64},values1::Matrix{Float64})

    values0_a = nothing
    values1_a = nothing
    n_points = nothing
    mesh = nothing
    mapping = Dict{Tuple{Array{Float64}}, Matrix{Float64}}()
    if(size(values0) != size(values1))
        throw(ArgumentError("Both input grids must be of same size"))
    end
    cols0, rows0 = size(values0)
    if(cols0 == dimensions)
        values0_a = Transpose(values0)
        values1_a = Transpose(values1)
        n_points = cols0
    elseif(rows0 == dimensions)
        values0_a = values0
        values1_a = values1
        n_points = rows0
    else
        throw(ArgumentError("array inputs must be n_dimensions x N"))
    end
    
    mesh=delaunay(values0_a)
    n_simplices = size(mesh.simplices)[1]

    #=
    if(n_simplices==dimensions+1 || n_simplices=dimensions)
        throw(Error("simplicies should be more than number of dimensions"))
    end
    =#

    return Transform(dimensions,values0_a,values1_a,n_points,mesh,n_simplices,mapping)
end

function map_point(trans::Transform,loc::Vector{Float64},parallel::Bool = true)

    simp_ind = Vector{Int64}(undef, 0)
    if(parallel)
        simp_ind = find_simp_parallel(trans,loc)
    else
        simp_ind = find_simp(trans,loc)
    end

    simp_key = Set(simp_ind)
    T_mat = nothing
    if(haskey(trans.mapping,simp_key))
        T_mat = trans.mapping[simp_key]

    else
        ones_a = ones(Float64,(1,trans.dimensions+1))
        simp0 = reshape(trans.values0[simp_ind,:],trans.dimensions+1,trans.dimensions)
        mat0 = cat(Transpose(simp0),ones_a,dims=1)
        simp1 = reshape(trans.values1[simp_ind,:],trans.dimensions+1,trans.dimensions)
        mat1 = cat(Transpose(simp1),ones_a,dims=1)

        mat0Inv = inv(mat0)
        T_mat = *(mat1*mat0Inv)
        trans.mapping[simp_key] = T_mat
    end

    loc_v = push!(loc,1)
    loc_t = *(T_mat,loc_v)

    return loc_t[1:end-1]
end

function find_simp(trans::Transform,loc::Array{Float64})
    simp_ind = trans.mesh.simplices
    #simp_points = trans.values0[simp_ind]
    
    for i in 1:trans.n_simplices
        simp_ind_i = simp_ind[i,:]
        simp_p = trans.values0[simp_ind_i,:]
        maxs = maximum(simp_p,dims=1)
        mins = minimum(simp_p,dims=1)
        within_bounds = all((loc.>=mins) .& (loc.<=maxs))
        if(within_bounds)
            #println("within bounds: $(i)")
            if(is_within(loc,simp_p))
                #println("within simp: $(i)")
                return Transpose(simp_ind_i)
            end
        end
    end

    min_dist = Inf
    min_simp_id = -1

    for i in 1:trans.n_simplices
        simp_ind_i = simp_ind[i,:]
        simp_p = trans.values0[simp_ind_i,:]
        simp_centre = sum(simp_p,dims=1)/(trans.dimensions+1)
        println(simp_centre)
        distance = sqrt(sum((simp_centre-Transpose(loc)).^2))
        if(distance<min_dist)
            min_dist = distance
            min_simp_id = i
        end
    end

    return Transpose(simp_ind[min_simp_id,:])
end

function find_simp_parallel(trans::Transform,loc::Array{Float64})
    
    simp_ind = trans.mesh.simplices
    #simp_points = trans.values0[simp_ind]
    n_simplices = trans.n_simplices
    nchunks = Threads.nthreads()

    lk = ReentrantLock()

    i_found = nothing
    #start parallel computation
    Threads.@threads for (i_range, _) in chunks(1:n_simplices, nchunks)
        for i in i_range
            #if simplex has already been found break 
            if !isnothing(i_found)
                break
            end
            # complex computation here
            lock(lk) do
                if isnothing(i_found)
                    #get simplex points
                    simp_ind_i = simp_ind[i,:]
                    simp_p = trans.values0[simp_ind_i,:]
                    #do basic intial check to see if point is within bound of siplex
                    maxs = maximum(simp_p,dims=1)
                    mins = minimum(simp_p,dims=1)
                    within_bounds = all((loc.>=mins) .& (loc.<=maxs))
                    if(within_bounds)
                        #println("within bounds: $(i)")
                        if(is_within(loc,simp_p))
                            #return Transpose(simp_ind[i,:])
                            #println("within simp: $(i)")
                            i_found = simp_ind_i
                        end
                    end
                end
            end
        end
    end

    if !isnothing(i_found)
        return i_found
    end

    distance_list = Vector{Float64}(undef,n_simplices)
    Threads.@threads for i in 1:n_simplices
        simp_ind_i = simp_ind[i,:]
        simp_p = trans.values0[simp_ind_i,:]
        simp_centre = sum(simp_p,dims=1)/(trans.dimensions+1)
        distance = sqrt(sum((simp_centre-Transpose(loc)).^2))
        distance_list[i] = distance
    end

    min_dist,ind_choice = findmin(distance_list)
    simp_choice = Transpose(simp_ind[ind_choice,:])

    return simp_choice
end

function is_within(point::Array{Float64},simplex::Matrix{Float64})
    facets = get_facets(simplex)
    n_facets,dim,ign = size(facets)
    C = fill(NaN, dim,n_facets)
    
    a = rand(Float64,dim)
    #a = [0.32,0.64]
    b = point
    count=0
    #facet_matches = zeros(Bool, n_facets)

    for i in 1:n_facets
        facet = facets[i,:,:]
        S = Transpose(facet[2:end,:]).-facet[1,:]
        N = externalproduct(S)
        c = -sum(facet[1,:].*N)
        t = (-c-Transpose(N)*b)/(Transpose(N)*a)
        #println("t: $(t)")
        C[:,i] = vec(a*t + b)
        SI = [zeros(dim,1) Matrix(I, dim, dim); ones(1,dim+1)]
        IM = [Transpose(facet) vec(N); ones(1,dim+1)]
        while(det(IM)==0)
            facet = facets[i,:,:]
            facet = facet[shuffle(1:end), :]
            S = Transpose(facet[2:end,:])-facet[1,:]
            N = externalproduct(S)
            c = -sum(facet[1,:].*N)
            t = (-c-Transpose(N)*b)/(Transpose(N)*a)
            C[:,i] = vec(a*t + b)
            SI = [zeros(dim,1) Matrix(I, dim, dim); ones(1,dim+1)]
            IM = [Transpose(facet) vec(N); ones(1,dim+1)]
        end
        M=SI*inv(IM)
        temp=M*push!(C[:,i], 1)
        cp = temp[1:dim-1]

        if(all(cp.>=0) && sum(cp)<=1 && t>0)
            count+=1
            #=
            if(i>1 && !(any(sum(abs.(C[:,2:i-1].-C[:,i]),dims=2).==0)))
                count+=1
            elseif(i==1 && !(any(sum(abs.(C[:,i]),dims=2).==0)))
                count+=1
            end
            =#
        end
    end
    
    #println(typeof(count))
    return mod(count,2)==1
end

function get_facets(points)
    n,d = size(points)
    facets = zeros(n,n-1,d)
    for i in 1:n
        facets[i,:,:] = points[1:n.!=i,:]
    end
    return facets
end

function externalproduct(A)
    n,cols = size(A)
    if(n != cols+1)
        throw(ArgumentError("external_product: A n by n-1 matrix is required."))
    end
    w=zeros(n)
    for i in 1:n
        w[i] = ((-1)^(i-1))*det(A[1:n.!=i,:])
    end
    return w
end


function is_point_inside_simplex(point, simplex_points)
    if length(simplex_points) == 0
        throw(ArgumentError("Simplex must contain at least one point"))
    end

    n = length(point)
    if n != length(simplex_points[1,:])
        throw(ArgumentError("Point and simplex dimensions must match"))
    end

    for i in 1:length(simplex_points[1,:])
        if length(simplex_points[i,:]) != n
            throw(ArgumentError("All simplex points must have the same dimensions"))
        end
    end

    # Check if the point is inside the convex hull of the simplex points
    v = [point - simplex_points[1,:]]
    for i in 2:length(simplex_points[1,:])
        v = [v; point - simplex_points[i,:]]
    end

    A = [simplex_points[i,:] - simplex_points[1,:] for i in 2:length(simplex_points[1,:])]

    coefficients = A \ v  # Solve for coefficients to express v as a linear combination of simplex edges

    if all(0 .<= coefficients) && sum(coefficients) <= 1
        return true
    else
        return false
    end
end

end
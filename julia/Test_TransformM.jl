using Revise
using Plots

#includet("TransformM.jl")
#using .TransformM

include("TransformM.jl")
using .TransformM
#using Delaunay

#a_test = [[0,0],[1,0],[1,1],[0,1],[0,2],[1,2]]
#=
a_test = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0; 0.0 2.0; 1.0 2.0]
b_test = [2.0 2.0; 4.0 2.0; 3.0 3.0; 2.0 3.0; 2.0 4.0; 4.0 4.0]

tf = TransformM.Transform_Const(2,a_test,b_test)
po = TransformM.map_point(tf,[0.6,0.9])
println(po)
=#

a_test = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
b_test = [2.0 2.0; 3.0 2.0; 4.0 4.0; 2.0 3.0]
tf = TransformM.Transform_Const(2,a_test,b_test)



x_in = zeros(0)
y_in = zeros(0)
x_out = zeros(0)
y_out = zeros(0)
#c= Array{iter::RGBA{Float64}}(undef, 0)
#pyplot()
#p = Plots.scatter(; lab="")
#po = TransformM.map_point(tf,[1.0,1.0])
for y in 0:0.1:1
    for x in 0:0.1:1
        #println([x,y])
        #color1 = RGBA(0.1+0.8*x,0,0.1+0.8*y,1)
        #Plots.scatter!([x],[y])
        #append!(c,color1)
        append!(x_in,x)
        append!(y_in,y)
        po = TransformM.map_point(tf,[x,y])
        #Plots.scatter!([po[1]],[po[2]])
        append!(x_out,po[1])
        append!(y_out,po[2])
    end
end
Plots.plot(x_in, y_in, seriestype=:scatter)
Plots.plot!(x_out, y_out, seriestype=:scatter)

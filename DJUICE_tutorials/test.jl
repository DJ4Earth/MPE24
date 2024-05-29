### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ fb70c1ba-1af3-4ca3-aa6d-c0c928c79465
begin
	import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
	Pkg.status()
end

# ╔═╡ e61bf01a-1a0b-11ef-3ad8-619127c1205e
begin
using Enzyme
Enzyme.API.typeWarning!(false)
Enzyme.Compiler.RunAttributor[] = false

using DJUICE
using MAT

#Load model from MATLAB file
file = matopen(joinpath(@__DIR__, ".", "Data","PIG_Control_drag_dJUICE.mat"))

mat  = read(file, "md")
close(file)
md = model(mat)

#make model run faster
md.stressbalance.maxiter = 20

#Now call AD!
md.inversion.iscontrol = 1
md.inversion.independent = md.friction.coefficient
md.inversion.independent_string = "FrictionCoefficient"

md = solve(md, :grad)

# compute gradient by finite differences at each node
addJ = md.results["StressbalanceSolution"]["Gradient"]
end;

# ╔═╡ Cell order:
# ╠═fb70c1ba-1af3-4ca3-aa6d-c0c928c79465
# ╠═e61bf01a-1a0b-11ef-3ad8-619127c1205e

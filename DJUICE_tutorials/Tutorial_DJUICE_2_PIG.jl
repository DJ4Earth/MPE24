### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 4b6fdc20-b10a-4c3b-8bfc-b8f9f4ce033c
begin
	import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
end

# ╔═╡ 7ee597f0-fb53-4865-beee-3104b31c8ef3
begin
	using DJUICE
	using MAT
	
	file = matopen(joinpath(@__DIR__, ".", "Data","PIG_Control_B_dJUICE.mat"))  
	mat  = read(file, "md")
	close(file)
	md = model(mat)
end

# ╔═╡ 9ee2316a-0245-11ef-319d-63d9da8d5061
md"""
# Pine Island Glacier

Let's try to solve a real glacier on Antarctica, PIG.
"""

# ╔═╡ d4f79bc0-d944-4e3a-b2b9-592b1d40c0e4
begin
	md.inversion.iscontrol = false
	solve(md, :Stressbalance)
end

# ╔═╡ 2ac93dbd-f7d2-47f0-a68e-8ec9ff9438a1
plotmodel(md, md.materials.rheology_B)

# ╔═╡ be7e4edc-5f33-4414-87f0-253e40928e82
plotmodel(md, md.friction.coefficient)

# ╔═╡ 6fd77011-e5e0-4550-8e29-bf923e3255c3
plotmodel(md, sqrt.(md.initialization.vx.^2+md.initialization.vy.^2))

# ╔═╡ 452d1fbc-95ea-4e27-9477-4025b0987a14
plotmodel(md, md.results["StressbalanceSolution"]["Vel"])

# ╔═╡ d6e37a8a-0677-4c5a-b2ec-eae5c22a95f6
begin
	file_friction = matopen(joinpath(@__DIR__, ".", "Data","PIG_Control_drag_dJUICE.mat"))  
	mat_friction  = read(file_friction, "md")
	close(file)
	md_friction = model(mat_friction)
	md_friction.inversion.iscontrol = false
	md_friction=solve(md_friction, :Stressbalance)
end;

# ╔═╡ 8bf305ef-cce6-471e-81e0-0a78faaab5cc
plotmodel(md_friction, md_friction.results["StressbalanceSolution"]["Vel"])

# ╔═╡ 88adddc7-fe9c-4b33-92ce-0fe9ce14037c
plotmodel(md_friction, md_friction.friction.coefficient)

# ╔═╡ Cell order:
# ╟─9ee2316a-0245-11ef-319d-63d9da8d5061
# ╠═4b6fdc20-b10a-4c3b-8bfc-b8f9f4ce033c
# ╠═7ee597f0-fb53-4865-beee-3104b31c8ef3
# ╠═d4f79bc0-d944-4e3a-b2b9-592b1d40c0e4
# ╠═2ac93dbd-f7d2-47f0-a68e-8ec9ff9438a1
# ╠═be7e4edc-5f33-4414-87f0-253e40928e82
# ╠═6fd77011-e5e0-4550-8e29-bf923e3255c3
# ╠═452d1fbc-95ea-4e27-9477-4025b0987a14
# ╠═d6e37a8a-0677-4c5a-b2ec-eae5c22a95f6
# ╠═8bf305ef-cce6-471e-81e0-0a78faaab5cc
# ╠═88adddc7-fe9c-4b33-92ce-0fe9ce14037c

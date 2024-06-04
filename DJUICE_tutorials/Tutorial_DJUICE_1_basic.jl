### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 7808d04c-5b1a-4e45-aad4-d9d761597e03
begin
	import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
end

# ╔═╡ 47d2e93e-096d-4983-99ab-f138fd0057d8
using DJUICE#main

# ╔═╡ bd86b569-aabc-43da-9802-ac531d28a591
md"""
# Basics of DJUICE: A square ice shelf

This tutorial demonstrates how to compute the velocity field of a steady-state square ice shelf using the Differentiable Julia ICE (DJUICE) model. We follow the approach outlined in the Ice-Sheet and Sea-Level System Model (ISSM) tutorial, adapted from its original C++ implementation to Julia. The original ISSM tutorial can be found [here](https://issm.jpl.nasa.gov/documentation/tutorials/squareiceshelf/).
"""

# ╔═╡ deb25c09-55a6-4b95-a2ca-b90845d3806b
md"""
## Install the package

To follow this tutorial, you need to install the [`DJUICE`](https://github.com/DJ4Earth/DJUICE.jl) package. DJUICE is available on Julia package registry, and you can install it directly from the repository.


### Option 1: Using the Julia Package Manager
1. Open your Julia REPL.
2. Enter the package manager mode by pressing `]`.
3. Run the following command to add DJUICE from the main branch of the repository:
```julia
]add DJUICE#main
```

### Option 2: Using `Pkg` 
Use `Pkg` to instantiate the environment and install the dependencies in `Project.toml` from the mini-tutorial.
"""

# ╔═╡ 95641af7-1cbf-4a43-84ea-be40ca4a0acc
md"""
After installing, import the package:
"""

# ╔═╡ 2c3fc78b-595b-40d0-a16d-6058e76d5bb1
md"""
## Initialize the model


All the data belonging to a model (geometry, node coordinates, results, etc.) is held in the single object `Model`. This will create a new model named `md` whose `struct` is `DJUICE.Model`. The information contained in the model `md` is grouped by `struct`, that contain fields related to a particular aspect of the model (e.g. mesh, ice geometry, material properties, friction, stressbalance solution, results of the runs, etc.). When one creates a new model, all these fields are initialized, and ready to be used as a `DJUICE` model. 


To create a new model, use the following command. This command initializes the `md` object, setting up the model structure with all necessary fields.

"""

# ╔═╡ e19d512d-0a02-4e30-a06c-4ffd2576dd4b
begin
	model()
end

# ╔═╡ 246e4613-f7c5-4fc0-bee0-33fb521d189d
md"""## Generate Mesh

Use `md=triangle(md,domainname,resolution)` to generate an unstructured triangular mesh based on the domain outline defined in the file `domainname`, with a characteristic length `resolution` for the mesh.

We start with a square domain of $[0, 10^6]\times[0,10^6]$ with $5\times10^3~$ mesh resolution, all the units are in meters. Indeed, in `DJUICE` we use [SI](https://en.wikipedia.org/wiki/International_System_of_Units) unit system.

Then, we set the whole domain to be ice-covered, as an ice shelf.
"""

# ╔═╡ 1bf7fb45-de6e-4227-917c-7d8bd2738479
begin
	md = triangle(model(), issmdir()*"/test/Exp/Square.exp",50000.)
	# set ice mask 
	md = setmask(md,"all","")
	plotmodel(md, "mesh")
end

# ╔═╡ 416c8beb-b4b7-4a15-8584-a86fd8461aac
md"""## Set the geometry

We define the ice geometry on the given domain. 
### Ice thickness
Let's set the `thickness` of the ice shelf to be 
$$H(x,y)=h_{\max} + \frac{h_{\min}-h_{\max}}{L_y}y+0.1\frac{h_{\min}-h_{\max}}{L_x}x$$
where $h_\min=300$ and $h_\max=1000$"""

# ╔═╡ a6c3eb4a-fd60-4671-9bf7-a95df2b5ec5f
begin
	hmin=300.
	hmax=1000.
	ymin=minimum(md.mesh.y)
	ymax=maximum(md.mesh.y)
	xmin=minimum(md.mesh.x)
	xmax=maximum(md.mesh.x)
	
	md.geometry.thickness = hmax .+ (hmin-hmax)*(md.mesh.y .- ymin)./(ymax-ymin) .+ 
	                            0.1*(hmin-hmax)*(md.mesh.x .- xmin)./(xmax-xmin)
	# plot the ice thickness
	plotmodel(md, md.geometry.thickness)
end

# ╔═╡ dd19725d-d256-46a2-894c-13202a1b775d
md"""### Ice base
Because the ice shelf is floating, we can determine the `base` of the ice by the flotation criteria
$$b=-\frac{\rho_i}{\rho_w}*H,$$ where $\rho_i$ and $\rho_w$ are the density of ice and water. The physical constants are given by default in `md.materials` when initialize the model. """

# ╔═╡ a65e6922-2873-410e-8cb1-9f7a7a9763b2
begin
	md.geometry.base = -md.materials.rho_ice/md.materials.rho_water*md.geometry.thickness
	plotmodel(md, md.geometry.base)
end

# ╔═╡ 1d8e62dc-8cbf-4252-a00c-ba28372d8556
md"""### Ice surface

The `surface` of the ice is then calculated as $s=b+H$. """

# ╔═╡ 635a67ca-dae3-46ea-b509-bc8d04745040
begin
	md.geometry.surface   = md.geometry.base+md.geometry.thickness
	plotmodel(md, md.geometry.surface)
end

# ╔═╡ 22dc7566-ac5d-4f1c-bee5-cab4e52a4573
md"""### Bedrock
In this tutorial, we are going to work on an ice shelf. To gurantee the ice is completely floating, we set the sea `bed` (bedrock) elevation to be 10 meters deeper than the `base` of the ice:"""

# ╔═╡ 6120e046-c134-4f53-a92f-f25afe55ae86
begin
	md.geometry.bed       = md.geometry.base .-10
	plotmodel(md, md.geometry.bed)
end

# ╔═╡ 72fa1779-c2b6-4194-acc4-23ce24147bc6
md"""## Set intitial conditions

We set the initial velocity to zeros. """

# ╔═╡ cd0668fe-b4f2-45ee-a5b8-6db01210cda6
begin
	md.initialization.vx = zeros(md.mesh.numberofvertices)
	md.initialization.vy = zeros(md.mesh.numberofvertices)
end;

# ╔═╡ 5ded5721-77f7-41e3-a6d8-b51f2f738bf7
md"""
## Set physical parameters

Ice in large scale is a non-Neutownian, viscos fluid. Its governing dynamics are described by the Shelfy Stream Approximation (SSA), expressed as a system of PDEs: 

$$\nabla\cdot\boldsymbol{\sigma}+{\boldsymbol{\tau}}_b=\rho_i g H \nabla s$$
where $\boldsymbol{\tau}_b=(\tau_{bx}, \tau_{by})^T$ represents the basal shear stress, $\rho_i$ is the ice density, $g$ is the gravitational acceleration. $\boldsymbol{\sigma}$ is the stress tensor of the SSA model defined as

$$\boldsymbol{\sigma} = \mu H
    \begin{bmatrix}
         \displaystyle 4\frac{\partial u}{\partial x}+2\frac{\partial v}{\partial y} 
         & 
         \displaystyle \frac{\partial u}{\partial y}+\frac{\partial v}{\partial x}  \\
         \\
         \displaystyle \frac{\partial u}{\partial y}+\frac{\partial v}{\partial x} 
         &
         \displaystyle 2\frac{\partial u}{\partial x}+4\frac{\partial v}{\partial y}
    \end{bmatrix}.$$

The ice viscosity, $\mu$, is determined by Glen's flow-law, which in two dimensions reads: 

$$\mu =\frac{B}{2}\left( \left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial v}{\partial y}\right)^2 + \frac{1}{4}\left(\frac{\partial u}{\partial y} +\frac{\partial v}{\partial x}\right)^2 + \frac{\partial u}{\partial x}\frac{\partial v}{\partial y}\right)^{\frac{1-n}{2n}},$$

where $n = 3$ is the flow-law exponent, and $B$ is the pre-factor dependent on ice temperature, among other factors. 


The above PDEs is what `DJUICE` is going to solve. We will only need to set the pre-factor $B$ and exponent $n$ in this tutorial.

"""

# ╔═╡ 80ba77a2-2829-4f3e-8b45-3047dc7bab51
begin
	md.materials.rheology_B=1.815730284801701e+08*ones(md.mesh.numberofvertices)
	md.materials.rheology_n=3*ones(md.mesh.numberofelements);
end;

# ╔═╡ 131ed044-517b-4754-adf8-053482437689
md"""
The basal shear stress is related to the ice velocity by a friciton law, here we use the Budd friction law

$$\tau_b=C^2N^\frac{q}{p}|u_b|^{\frac{1}{q}-1}u_b$$

where the effective pressure $N$ is calculated by `dJUICE` at the base of the ice.

To start, we set the basal friction coefficient $C=20~\text{m}^{-1/2} \text{s}^{1/2}$, and the exponents $p=1$ and $q=1$.

"""

# ╔═╡ 2ebc291e-222f-4383-a608-f8558de177b1
begin
	md.friction.coefficient=20*ones(md.mesh.numberofvertices)
	md.friction.p=ones(md.mesh.numberofvertices)
	md.friction.q=ones(md.mesh.numberofvertices)
end;

# ╔═╡ c389a807-abe5-43be-9d0d-0a42e9613b97
md"""
## Boundary conditions

Let's set the left, bottom, right boundaries to be Dirichlet boundary, and the top boundary to be a calving front boundary. In `dJUICE`, the calving front boundary is automatically determined by the 0-levelset contour of `md.mask.ice_levelset`, which are the nodes inside the polygon defined in `./test/Exp/SquareFront.exp`.

The Dirichlet boundaries are the rest of the boundaries of the domain, by setting the values in `md.stressbalance.spcvx` and `md.stressbalance.spcvy`.
"""

# ╔═╡ 4cb62901-cf99-4e60-b778-658f7027b7c8
begin
	#Boundary conditions
	nodefront=ContourToNodes(md.mesh.x,md.mesh.y,issmdir()*"/test/Exp/SquareFront.exp",2.0) .& md.mesh.vertexonboundary
	md.stressbalance.spcvx = NaN*ones(md.mesh.numberofvertices)
	md.stressbalance.spcvy = NaN*ones(md.mesh.numberofvertices)
	pos = findall(md.mesh.vertexonboundary .& .~nodefront)
	md.mask.ice_levelset[findall(nodefront)] .= 0
	
	segmentsfront=md.mask.ice_levelset[md.mesh.segments[:,1:2]]==0
	segments = findall(vec(sum(Int64.(md.mask.ice_levelset[md.mesh.segments[:,1:2]].==0), dims=2)) .!=2)
	pos=md.mesh.segments[segments,1:2]
	md.stressbalance.spcvx[pos] .= 0.0
	md.stressbalance.spcvy[pos] .= 0.0
end;

# ╔═╡ ede33305-2bdc-4ce5-8fbf-215cff47229d
md"""
## Numerical tolerance

- `restol` is the mechanical equilibrium residual convergence criterion
- `reltol` is the velocity relative convergence criterion
- `abstol` is the velocity absolute convergence criterion

If the tolerance is set to `NaN`, that means it is not applied.
"""

# ╔═╡ c3110ce6-3839-4067-be0f-77453c4050c9
begin
	md.stressbalance.restol=0.05
	md.stressbalance.reltol=0.05
	md.stressbalance.abstol=NaN
end;

# ╔═╡ d67f0dd5-fce4-41fb-a0a7-b31e583f108a
md"""
## Solve

Now let's solve the nonlinar PDEs
"""

# ╔═╡ 0fc54de5-c0ac-4622-b681-745c90f2dec1
solve(md,:Stressbalance)

# ╔═╡ 4c3bd8b8-8b60-481f-9330-15955486ae6f
md"""
## plot solutions

The solutions are in `md.results["StressbalanceSolution"]`. We can plot the velocity magnitude by the following command.
"""

# ╔═╡ f864c816-1f4c-424e-839a-393e476a0557
plotmodel(md, md.results["StressbalanceSolution"]["Vel"])

# ╔═╡ Cell order:
# ╟─bd86b569-aabc-43da-9802-ac531d28a591
# ╟─deb25c09-55a6-4b95-a2ca-b90845d3806b
# ╠═7808d04c-5b1a-4e45-aad4-d9d761597e03
# ╟─95641af7-1cbf-4a43-84ea-be40ca4a0acc
# ╠═47d2e93e-096d-4983-99ab-f138fd0057d8
# ╟─2c3fc78b-595b-40d0-a16d-6058e76d5bb1
# ╠═e19d512d-0a02-4e30-a06c-4ffd2576dd4b
# ╟─246e4613-f7c5-4fc0-bee0-33fb521d189d
# ╠═1bf7fb45-de6e-4227-917c-7d8bd2738479
# ╟─416c8beb-b4b7-4a15-8584-a86fd8461aac
# ╠═a6c3eb4a-fd60-4671-9bf7-a95df2b5ec5f
# ╟─dd19725d-d256-46a2-894c-13202a1b775d
# ╠═a65e6922-2873-410e-8cb1-9f7a7a9763b2
# ╟─1d8e62dc-8cbf-4252-a00c-ba28372d8556
# ╠═635a67ca-dae3-46ea-b509-bc8d04745040
# ╟─22dc7566-ac5d-4f1c-bee5-cab4e52a4573
# ╠═6120e046-c134-4f53-a92f-f25afe55ae86
# ╟─72fa1779-c2b6-4194-acc4-23ce24147bc6
# ╠═cd0668fe-b4f2-45ee-a5b8-6db01210cda6
# ╟─5ded5721-77f7-41e3-a6d8-b51f2f738bf7
# ╠═80ba77a2-2829-4f3e-8b45-3047dc7bab51
# ╟─131ed044-517b-4754-adf8-053482437689
# ╠═2ebc291e-222f-4383-a608-f8558de177b1
# ╟─c389a807-abe5-43be-9d0d-0a42e9613b97
# ╠═4cb62901-cf99-4e60-b778-658f7027b7c8
# ╟─ede33305-2bdc-4ce5-8fbf-215cff47229d
# ╠═c3110ce6-3839-4067-be0f-77453c4050c9
# ╟─d67f0dd5-fce4-41fb-a0a7-b31e583f108a
# ╠═0fc54de5-c0ac-4622-b681-745c90f2dec1
# ╟─4c3bd8b8-8b60-481f-9330-15955486ae6f
# ╠═f864c816-1f4c-424e-839a-393e476a0557

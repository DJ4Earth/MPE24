# SIAM Conference on Mathematics of Planet Earth (MPE24): Material for minitutorials

## Requirements
For the tutorials we will use [Pluto.jl](https://plutojl.org/) notebooks in [Julia](https://julialang.org/).

### Installing Julia
Download and install Julia for your platform from [here](https://julialang.org/downloads/).

### Installing Pluto.jl
Start Julia 
```shell
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.10.4 (2024-06-04)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

```
and run the following commands to install Pluto.jl:
```julia
] add Pluto
```
Start Pluto.jl
```julia
import Pluto
Pluto.run()
```
Load the notebooks from this repository. Alternatively, you can view the static html version of the notebooks linked below.

## MT1: Differentiable Programming in Julia with Enzyme
Tuesday, June 11, 10:00 AM - 12:00 PM, Grand II - Ballroom Level

  - Valentin Churavy, Massachusetts Institute of Technology, U.S.
  - William Moses, University of Illinois Urbana-Champaign, U.S.
  - Michel Schanen, Argonne National Laboratory, U.S., "Differentiation of 2D Burgers using Enzyme", [source](Burgers_tutorial/burgers_tutorial.jl) [html](https://dj4earth.github.io/MPE24/Burgers_tutorial/burgers_tutorial.html)

We will demonstrate the differentiable programming paradigm in Julia. We will introduce the Julia programming language allowing users to write their first programs. This will be followed by an introduction to automatic differentiation (AD) concepts and a demonstration of the AD tool Enzyme. It is highly-efficient and its ability to perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools. We will provide notebooks for attendees to write Julia code and differentiate codes with Enzyme.

This session is co-authored by Sri HariKrishna Narayanan and Jan Hückelheim, Argonne National Laboratory, U.S.

## MT2: Differentiable Earth System Models in Julia
Tuesday, June 11, 2:45 PM - 4:45 PM, Grand II - Ballroom Level

  - Joseph L. Kump, University of Texas, U.S.
  - Sarah M. Williamson, University of Texas, U.S.
  - Gong Cheng, Dartmouth College, U.S.
    
We will demonstrate three Earth system models written in Julia and differentiated by the automatic differentiation tool Enzyme. We will present (1) an idealized ocean model showing circulation inside a closed domain, (2) dJUICE: an ice dynamics model that employs the finite element method, (3) ClimaSeaIce: a thermodynamic sea ice model interacting with the ocean, showing growth and melt over time. We will provide notebooks for attendees to differentiate the models for user-defined parameters and solve inverse problems to infer unknowns from sample observational data. Previous experience with Julia or Enzyme is not required.

This session is co-authored by Patrick Heimbach, University of Texas at Austin, U.S. and Mathieu Morlighem, Dartmouth College, U.S.

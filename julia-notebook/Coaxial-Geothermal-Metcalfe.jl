### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 5c7e7f90-f543-4350-91a0-cee3fd2311a3
md"""# Model of Coaxial Geothermal
Bob.Metcalfe@UTexas.edu TexasGEO.org"""

# ╔═╡ fbfcd4da-5b27-4c1e-9a13-d5be496374c8
md"""## Earth"""

# ╔═╡ 5bb01181-a51c-400a-b3b1-ffa272219541
const earth = (EarthSurfaceTemperature = 15.0, EarthTemperatureGradient = 0.025) # [C], [C/m]

# ╔═╡ 1569cfa5-95c8-4809-8661-050eb0b89e1c
md"""Ambient temperature in Celsius at depth in meters"""

# ╔═╡ bf15d5b7-1b83-406d-b331-8fd121dc00bb
function  AmbientTemperature(earth, depth)
	earth.EarthSurfaceTemperature + depth * earth.EarthTemperatureGradient
end

# ╔═╡ 93e87fed-0406-4968-a0d7-347a775a36b1
md"""Rock is Granite"""

# ╔═╡ d80f8b23-db44-4495-a4f0-0c89f4d11d50
md"""Fluid is Water"""

# ╔═╡ ccfb42b1-3413-4da9-b7dc-37a469debc58
begin
	const FluidDensity = WaterDensity = 997000.0 # [g/m3]
	const FluidSpecificHeat = WaterSpecificHeat = 4.186 # [J/gC]
	const FluidThermalConductivity = WaterThermalConductivity = 0.6 # [W/mC]
	nothing
end

# ╔═╡ c7bd11b4-4d73-459a-b659-7fd5a92ec231
md"""## Drilling"""

# ╔═╡ 3d2fd592-1e16-404a-8058-e39b0d6d968d
begin
	const RockDensity = GraniteDensity = 2750000 # [g/m3]
	const RockSpecificHeat = GraniteSpecificHeat = 0.790 # [J/gC]
	const RockThermalConductivity = GraniteThermalConductivity = 2.62 # [W/mC] range average
	nothing
end

# ╔═╡ 0f44987d-83dc-4852-8b20-64dca75dce37
md"""Volume of inner and outer pipes is equal for now"""

# ╔═╡ 90efbe32-482b-439b-8e83-1efc7b333a45
begin
	const NumberOfPipes = 15 # [n]
	const LengthOfPipe = 10.0 # [m]
	const DepthOfWell = NumberOfPipes * LengthOfPipe # [m]
	const InnerRadius = 0.1 # [m]
	const OuterRadius = InnerRadius * sqrt(2) # [m]
	println("Well depth ",DepthOfWell," meters, pipe radius ",InnerRadius," meters.")
	const InnerArea = 2 * InnerRadius * pi * LengthOfPipe # [m2]
	const OuterArea = 2 * OuterRadius * pi * LengthOfPipe # [m2]
	const InnerVolume = pi * InnerRadius^2 * LengthOfPipe # [m3]
	const OuterVolume = (pi * OuterRadius^2 * LengthOfPipe) - InnerVolume # [m3]
	nothing
end

# ╔═╡ 592b0843-4e70-481f-b590-c44b0df74f95
md"""Check to be sure inner and outer pipe volumes are equal (for now)"""

# ╔═╡ dfcb9a83-363d-47ce-8c89-caa76c0c34b4
if round(InnerVolume, digits=4) != round(OuterVolume, digits=4)
    println("Inner and outer volumnes are not equal.")
end

# ╔═╡ c38b783f-b1c7-4e0f-9d86-26a72a5e3d1a
md"""Let the drilling begin"""

# ╔═╡ a85a89f7-08d8-4dd5-b481-c59c377c710b
begin
	mutable struct Pipe{T,V}
	    innertemp::T
	    outertemp::V
	end
	Base.show(io::IO, p::Pipe) = print(io, "Pipe: innertemp = $(p.innertemp), outertemp = $(p.outertemp)")
	PipeString = [(ambienttemp = AmbientTemperature(earth, (ip-1)*LengthOfPipe); Pipe(ambienttemp, ambienttemp)) for ip = 1:NumberOfPipes]
end

# ╔═╡ 4c1881e8-5db7-4d8d-aaa5-f5efb975c81e
md"""## Completion of well"""

# ╔═╡ 0905fc13-c8e7-42fe-82a6-3d1d4a0cc376
begin
	const PumpSpeed = InnerVolume/5.0 # [m3/s]
	const PumpTime = InnerVolume/PumpSpeed # [s]
	nothing
end

# ╔═╡ 23717e41-6a97-454d-a4d1-68f1960a7c61
md"""## Operation"""

# ╔═╡ 7e6a5d90-f400-4801-bf79-0ec32cc17fd6
md"""Iteration on thermo updates"""

# ╔═╡ 62494b1e-718b-4c59-999d-73f5798d39b5
for run in 1:100
    for ip in 1:NumberOfPipes
        pipe = PipeString[ip]
        # outer pipe to inner pipe Joule flux
        O2Ic = FluidThermalConductivity
        O2Idt = pipe.outertemp - pipe.innertemp
        O2Ia = InnerArea
        O2Im = InnerVolume * FluidDensity
        O2Ipt = PumpTime
        
        O2Ijf = O2Ic * O2Idt * O2Ia * O2Im * O2Ipt
        # print("O2I ",run, O2Ijf,O2Ic,O2Idt,O2Ia,O2Im,O2Ipt)
        # ambient to outer Joule flux 
        A2Oc = RockThermalConductivity
        A2Odt = AmbientTemperature(earth, (ip-1)*LengthOfPipe) - pipe.outertemp
        A2Oa = OuterArea
        A2Om = OuterVolume * FluidDensity
        A2Opt = PumpTime
        
        A2Ojf = A2Oc * A2Odt * A2Oa * A2Om * A2Opt
        # print("A2O ",run,A2Ojf,A2Oc,A2Odt,A2Oa,A2Om,A2Opt)
        # Update temperatures
        pipe.innertemp += O2Ijf / (FluidSpecificHeat * O2Im)
        pipe.outertemp += -O2Ijf / (FluidSpecificHeat * A2Om)
        pipe.outertemp += A2Ojf / (RockSpecificHeat * A2Om)
    end
end

# ╔═╡ fc13daad-4381-4661-bd41-00bbd491f12f
md"""Pump - move fluid down and up coaxial pipes through rock"""

# ╔═╡ 03cabb18-68c8-11ef-0787-056e7f997aba
begin
	function printPipeString(message)
	    println(message)
	    for ip in 1:NumberOfPipes
	        println(ip," ", PipeString[ip])
	    end
	end
	# printPipeString("---------- start ----------")
	savedinnerpipe = PipeString[1].innertemp
	for ip in 1:NumberOfPipes-1
	    PipeString[ip].innertemp = PipeString[ip+1].innertemp # fluid moving up
	end
	savedouterpipe = PipeString[NumberOfPipes].outertemp
	for ip in NumberOfPipes:-1:2
	    PipeString[ip].outertemp = PipeString[ip-1].outertemp
	end
	PipeString[1].outertemp = PipeString[end].outertemp
	PipeString[1].outertemp = savedinnerpipe
	PipeString[NumberOfPipes].innertemp = savedouterpipe
	printPipeString("----------- stop -----------")
end

# ╔═╡ Cell order:
# ╟─5c7e7f90-f543-4350-91a0-cee3fd2311a3
# ╟─fbfcd4da-5b27-4c1e-9a13-d5be496374c8
# ╠═5bb01181-a51c-400a-b3b1-ffa272219541
# ╟─1569cfa5-95c8-4809-8661-050eb0b89e1c
# ╠═bf15d5b7-1b83-406d-b331-8fd121dc00bb
# ╟─93e87fed-0406-4968-a0d7-347a775a36b1
# ╟─d80f8b23-db44-4495-a4f0-0c89f4d11d50
# ╠═ccfb42b1-3413-4da9-b7dc-37a469debc58
# ╟─c7bd11b4-4d73-459a-b659-7fd5a92ec231
# ╠═3d2fd592-1e16-404a-8058-e39b0d6d968d
# ╟─0f44987d-83dc-4852-8b20-64dca75dce37
# ╠═90efbe32-482b-439b-8e83-1efc7b333a45
# ╟─592b0843-4e70-481f-b590-c44b0df74f95
# ╠═dfcb9a83-363d-47ce-8c89-caa76c0c34b4
# ╟─c38b783f-b1c7-4e0f-9d86-26a72a5e3d1a
# ╠═a85a89f7-08d8-4dd5-b481-c59c377c710b
# ╟─4c1881e8-5db7-4d8d-aaa5-f5efb975c81e
# ╠═0905fc13-c8e7-42fe-82a6-3d1d4a0cc376
# ╟─23717e41-6a97-454d-a4d1-68f1960a7c61
# ╟─7e6a5d90-f400-4801-bf79-0ec32cc17fd6
# ╠═62494b1e-718b-4c59-999d-73f5798d39b5
# ╟─fc13daad-4381-4661-bd41-00bbd491f12f
# ╠═03cabb18-68c8-11ef-0787-056e7f997aba

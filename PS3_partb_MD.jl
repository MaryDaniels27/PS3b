using StatFiles, DataFrames, Optim, Parameters, Plots, LinearAlgebra

car_characteristics = DataFrame(load("C:\\Users\\maryd\\OneDrive\\UW Madison\\Fall 2022\\Computational\\JF HW\\PS3_comp_b\\PS3\\Car_demand_characteristics_spec1.dta"))
# model_id is product j and market is year t 
groupby(car_characteristics, "Year")
# note we don't necessarily observe the same products in each market 
markets = unique(car_characteristics[:, 2])

instruments = DataFrame(load("C:\\Users\\maryd\\OneDrive\\UW Madison\\Fall 2022\\Computational\\JF HW\\PS3_comp_b\\PS3\\Car_demand_iv_spec1.dta"))
income = DataFrame(load("C:\\Users\\maryd\\OneDrive\\UW Madison\\Fall 2022\\Computational\\JF HW\\PS3_comp_b\\PS3\\Simulated_type_distribution.dta"))
income = Matrix{Float64}(income)

@with_kw struct Primitives
    r::Int64 = 100  #number of consumer types 
    λ_p::Float64 = 0.6  #the coefficient on consumer r's income for their random coefficient α_i
    markets::Array{Float64, 1} = unique(car_characteristics[:, 2])   #a vector with the list of markets 
end

prim = Primitives()


##### Problem 1 ######
#the year 1985 is the 31st element of my markets vector 
t = 31

function market_shares(t::Int64, δ_jt::Vector{Float64})  #calculating the marketshares for market t 
    @unpack r, λ_p, markets = prim
    t_ind = findall(x-> x == markets[t], car_characteristics[:, 2])
    # now we calculate demand for product j in market t by looping over consumer types r 
    σ_j = zeros(length(t_ind))
    for j = 1:length(t_ind)  #looping over all of the products in market t 
        for i = 1:r   #looping over individuals
            δ = δ_jt[j]   #mean utility for product j
            price = car_characteristics[:, 6][t_ind]
            μ = λ_p * income[i].* price[j]
            σ_j[j] += exp(δ + μ)/(1 + sum(exp.(δ_jt .+ (λ_p * income[i].*price))))
        end
    end

    σ_j = log.((1/r).*σ_j)  #averaging for each product. This will give us the new vector of log market shares/demand for products in market t.
    return σ_j
end 


t_ind = findall(x-> x == markets[31], car_characteristics[:, 2])
δ_j0 = Vector{Float64}(car_characteristics[:, 5])[t_ind]
market_shares(31, δ_j0)



function contraction_mapping(prim::Primitives, t::Int64)
    @unpack r, λ_p, markets = prim 
    t_ind = findall(x-> x == markets[t], car_characteristics[:, 2])  #finding all the products for market t 
    #Write a routine that inverts the demand fucntion for parameter value λ_p = 0.6
    # the initial guess of the mean utilities is the delta_iia column in the characteristics data set 
    δ_jt = Vector{Float64}(car_characteristics[:, 5])[t_ind]       
    # we will iterate over this in the future, updating δ_jt 

    #comparing the old and new market shares, we will iterate if the absolute difference is greater than tol = 1e-06
    σ_j = market_shares(t, δ_jt)  #computing new market shares 
    σ_0 = log.(Vector{Float64}(car_characteristics[:, 4][t_ind]))    #computing old market shares 
    δ_jt_1 = δ_jt + σ_0 - σ_j  #updating the mean utilities
    err = maximum(abs.(δ_jt_1 - δ_jt))  #the norm of the difference in mean uilities 
    err_shares = maximum(abs.(σ_0 - σ_j))  # the norm of the difference in log market shares 
    println("the norm of the difference in log shares is  ", err_shares)
    errors = zeros(200)  #to store the path of the norm difference in log market shares overtime 
    it = 0  #counter 
    errors[it+1] = err_shares
    tol = 1e-012
    while err > tol 
        it += 1
        println("on iteration", it)
        δ_jt = δ_jt_1  #update the old mean utilities 
        σ_j = market_shares(t, δ_jt)  #calculating the new market shares
        δ_jt_1 = δ_jt + σ_0 - σ_j  #updating the mean utilities
        err = maximum(abs.(δ_jt_1 - δ_jt))  #update error 
        err_shares = maximum(abs.(σ_0 - σ_j))  # the norm of the difference in log market shares 
        println("the norm of the difference in log shares is  ", err_shares)
        errors[it+1] = err_shares
    end   #end the while loop 

    return errors
end   #end the function

path = contraction_mapping(prim, t) 
#the path of the norm of the difference in the log market shares (elements 1 to 141)
Plots.plot(collect(1:1:141), path[1:141], title = "Path of the norm difference in log market shares")
Plots.savefig("difference_market_shares.png")

#Using Newton's method 

# first, we need to get the gradient which requires using to the the matrix σ of choice probabilities for product j from individual r 
@unpack r, λ_p, markets = prim 
t_ind = findall(x-> x == markets[31], car_characteristics[:, 2])  #finding all the products for market t = 1985
σ = zeros(length(t_ind), r)   #individual choice probabilities

for j = 1:length(t_ind)  #looping over all of the products in market t 
    for i = 1:r   #looping over individuals
        δ = δ_jt[j]   #mean utility for product j
        price = car_characteristics[:, 6][t_ind]
        μ = λ_p * income[i].* price[j]
        σ[j, i] = exp(δ + μ)/(1 + sum(exp.(δ_jt .+ (λ_p * income[i].*price))))
    end
end

Identity = diagm(0 => ones(length(t_ind)))  #identity matrix 
#Now we can calculate the Jacobian using the formula JF gave us 
Δ = (1/r) * Identity * (σ * (1 .- σ)') - (1/r) * (1 .- Identity) * (σ * σ')     #gradient 

#Using the Newton optimization function, I will find the optimal δ
#define a function that minimizes the difference between the new and old market shares 


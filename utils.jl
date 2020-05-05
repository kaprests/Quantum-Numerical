""" Comphys assignment 3 """

println("Importing libraries and stuff")
using LinearAlgebra
using SparseArrays
using PyPlot
println("Finnished importing libraries and stuff")


#############################################
### Physical and computational parameters ###
#############################################


# Physical parameters
const L = 1e-9 # Width of box
const NUM_POINTS = 1000
const DELTA_X = L/(NUM_POINTS - 1)

# Normalized/dimensionless parameters
const DELTA_X_NORM = DELTA_X / L


############################################
### Computation and analysis (functions) ###
############################################


function hamiltonian_matrix(;
                    N=NUM_POINTS, 
                    potential=fill(0, NUM_POINTS)
                    )
    """ Constructs the normalized hamiltonian matrix """
    diag::Array{Float64} = fill(2*N^2, N)
    diag = diag + potential
    off_diag::Array{Float64} = fill(-N^2, N-1)

    hamiltonian = SymTridiagonal(diag, off_diag)
    hamiltonian, diag, off_diag
end


function solve_eigenproblem(H, diag, off_diag)
    """ Solves eigenproblem """
    vals, vecs = eigen(H)
    #vals, vecs = LAPACK.stev!('V', diag, off_diag)
    vals, vecs
end


function analytic_eigenfunc(n, x)
    """ The analytic normalized eigenfunctions of the infinite well """
    sqrt(2)*sin(n*pi*x) * sqrt(DELTA_X_NORM)
end


function difference_norm(numerical, analytical)
    """ Compares numerical eigenfuncions with analytical. Returns error """
    difference = numerical - analytical
    sqrt(sum(difference .^2)) # Error
end


function eigenfunc_error_scaling()
    """ Investigates how the error of the numerical eigenfunctions scale with dx 
        Computes and plots
    """
    num_points_vec = collect(100:10:1000)
    del_x_vec = L ./ (num_points_vec .- 1)
    del_x_norm_vec = 1 ./ num_points_vec
    errors = []

    for num_points in num_points_vec
        delta_x_norm = 1/(num_points - 1)
        H, diag, off_diag = hamiltonian_matrix(
                                        N=num_points,
                                        potential=fill(0, num_points))
        vals, vecs = solve_eigenproblem(H, diag, off_diag)

        first_eigfunc_num = vecs[:,1]
        first_eigfunc_analy = [analytic_eigenfunc(1, x) for x in 0:delta_x_norm:1]
        error = difference_norm(first_eigfunc_num, first_eigfunc_analy)
        push!(errors, error)
    end

    plt.title("Error of computed groundstate")
	plt.xlabel(L"N")
	plt.ylabel("Error")
    plt.plot(num_points_vec, errors)
	plt.savefig("wf_error_scaling.pdf")
    plt.show()
end


function inner_product(func1, func2)
    """ Computes the inner product of two wave functions
        Completely uneeded as dot from LinearAlgebra already does this
    """
    dot(func1, func2)
end


function get_eigen_coeffs(eigenvecs, func)
    """ Computes the eigen coeffisients (eigenstate expansion) """
    transpose(conj(eigenvecs)) * func
end


function check_orthonormality(vecs)
    """ Checks orthonormality of numerical solutions, returns larges deviation """
    inner_prod_matrix = vecs * transpose(vecs)
    identity_matrix = Diagonal(ones(NUM_POINTS, NUM_POINTS))
    max_err = max((inner_prod_matrix - identity_matrix)...)
    println("Maximum deviation from orthonormality: ", max_err)
end


function Psi_evolve(t_norm; eigen_states, eigen_vals, Psi_initial)
    """ Numerical wave function """
    coeffs = get_eigen_coeffs(eigen_states, Psi_initial)
    Psi_exp = [exp(-im*lambda*t_norm) for lambda in eigen_vals]
    transpose(coeffs .* Psi_exp)*transpose(eigen_states) # Psi at t_norm
end


function dirac_delta(;N=NUM_POINTS, divider=2)
    """ INCOMPLETE Create normalized delta function at x' = 1/2 """
    Psi_delta = zeros(N)
    Psi_delta[div(N, divider)] = NUM_POINTS
    Psi_delta
end


function secant_method(x0, x1, tol; func=energy_eigenstate_below_v0)
    diff = tol + 1
    x2 = 0
    while diff > tol
        x2 = x1 - func(x1) * (x1-x0)/(func(x1)-func(x0))
        diff = abs(x2 - x1)
        x0, x1 = x1, x2
    end
    x2
end


function number_of_bound_states(vals, v0)
	""" check how many bound states for particle in a well with a barrier """
	bounded_vals = findall(x-> x<v0, vals)
	num_bound_states = length(bounded_vals)
	return num_bound_states
end


function forward_euler(H, N_time, dt, initial)
    Psi = initial

    for i in 0:N_time
        Psi = Psi - im*dt*H*Psi
    end

    Psi
end


function forward_euler_breakdown(H, dt, initial;N=NUM_POINTS)
    """ Finds how many steps before wf evolved by euler scheme, breaks down """
    Psi = initial
    max_iter = 1e5

    num_steps = 1
    while true
        num_steps += 1
        Psi = Psi - im*dt*H*Psi
        prob = dot(Psi, Psi)
        if abs(prob - 1) > 2
            return num_steps
        end
        if num_steps == max_iter
            return max_iter
        end
    end
end


function crank_nicolson(H, N_time, dt, initial)
    Psi = initial
    lhs_opr = Diagonal(ones(NUM_POINTS, NUM_POINTS)) + (im/2) * dt * H
    rhs_opr = Diagonal(ones(NUM_POINTS, NUM_POINTS)) - (im/2) * dt * H
    inv_lhs = inv(lhs_opr)

    for i in 0:N_time
        b = rhs_opr * Psi
        Psi = inv_lhs*b
    end

    Psi
end


##################
### Potentials ###
##################


function heaviside(x)
    0.5 * (sign(x) + 1)
end


function barrier(;v0)
    x_vec_norm = collect(0:DELTA_X_NORM:1)
    V_vec = zeros(NUM_POINTS)
    lower = 1/3
    upper = 2/3

    for (idx, x) in enumerate(x_vec_norm)
        V_vec[idx] = heaviside(x-lower) - heaviside(x-upper)
    end

    return V_vec *= v0
end


############################
### Plotting (functions) ###
############################


function plot_eigenstates_and_values(;vals, vecs, analytic_eigenfunc, nev=NUM_POINTS)
    # Eigenvalue plot (task 2.5)
    plt.title(string("N = ", NUM_POINTS))
    plt.plot([(pi*n)^2 for n in 1:nev], label="analytic")
    plt.plot(vals[1:nev], label="computed")
    plt.xlabel("n")
    plt.ylabel(L"\lambda_n = \frac{2mL^2}{\hbar^2}E_n")
    plt.legend()
	plt.savefig("eigenvalues_empty_box.pdf")
    plt.show()

	plt.title("Ratio of computed and analytical eigenvalues")
    plt.plot(vals[1:nev] ./ [(pi*n)^2 for n in 1:nev], label="analytic")
	plt.ylim(bottom=0.3, top=1.1)
    plt.xlabel("n")
    plt.ylabel(L"\lambda_{n_c} \left / \lambda_{n_a} \right.")
	plt.savefig("eigenval_ratio_empty_box.pdf")
	plt.show()

    # Eigenstate plot (task 2.6)
    for i in 1:3
        plt.plot([analytic_eigenfunc(i, x) for x in 0:DELTA_X_NORM:1],
                                                        label="analytical",
                                                        color="y")
        plt.plot(vecs[:,i], "--", label="numerical", color="k")
    end
	
	plt.title(string("N = ", NUM_POINTS))
    plt.xlabel("x'")
    plt.ylabel(L"\psi_n")
	plt.legend()
	plt.savefig("eigenstates_empty_box.pdf")
	plt.show()
end



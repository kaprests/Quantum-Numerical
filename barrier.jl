include("utils.jl")


#######################
### Utils functions ###
#######################


function energy_eigenstate_below_v0(val; v0=1e3)
	""" Roots of this function gives eigen functions with eigenvalues below v0 """
    k = sqrt(val)
    kappa = sqrt(v0 - val)

    (exp(kappa/3)*(kappa*sin(k/3)+ k*cos(k/3))^2
        - exp(-kappa/3)*(kappa*sin(k/3)-k*cos(k/3))^2)
end


function find_roots(;func=energy_eigenstate_below_v0)
    """
    # TODO: Define start values in a cooler more general way
    # i.e. not hardcode for particular barrier height

    function_vals = [func(val) for val in 0:0.0001:1e3]
    negs = findall(x -> x <= 0, [func(val) for val in 0:0.1:1e3])
    println("Megatives: ", negs)

    for idx in negs
        println(func(0.1*(idx-1)))
    end
    """

    root1 = secant_method(70, 71, 1e-12)
    root2 = secant_method(75, 76, 1e-12)

    root3 = secant_method(290, 291, 1e-12)
    root4 = secant_method(294, 295, 1e-12)

    root5 = secant_method(630, 631, 1e-12)
    root6 = secant_method(650, 651, 1e-12)

    println("Root1: ", root1)
    println("Root2: ", root2)
    println("Root3: ", root3)
    println("Root4: ", root4)
    println("Root5: ", root5)
    println("Root6: ", root6)

end


######################
### main functions ###
######################


function zero_barrier()
	# Define potential
	potential = barrier(v0=0)

    # Create normalized H matrix
    H, diag, off_diag = hamiltonian_matrix(potential=potential)

    # Solve ev problem
    vals, vecs = solve_eigenproblem(H, diag, off_diag) # energies and eigenstates

    # Eigenvals and -vecs plot
    plot_eigenstates_and_values(
                            vals=vals,
                            vecs=vecs,
                            analytic_eigenfunc=analytic_eigenfunc,
                            )

    # Error scaling with dx
    eigenfunc_error_scaling()

    """ Orthonormality test """
    check_orthonormality(vecs)
end


function high_barrier()
    ####################
    ### Eigen states ###
    ####################


    # Define potential
    v0 = 1e3
    potential = barrier(v0=v0)
    
    # Create normalized H matrix
    H, diag, off_diag = hamiltonian_matrix(potential=potential)

    # Solve eigenvalue problem
    vals, vecs = solve_eigenproblem(H, diag, off_diag)

    number_of_bound_eigenstates = number_of_bound_states(vals, v0)
    println("#bound eigenstates: ", number_of_bound_eigenstates)

    # print the eigenvals of the bound states
    for (i, val) in enumerate(vals[1:number_of_bound_eigenstates])
        println(string("eigenval", i,": "), val)
    end

    # Plot bound eigen states
    for i in 1:number_of_bound_eigenstates
        plt.plot(vecs[:, i], label=string(L"eigenstate #", i))
    end
    plt.xlabel("x'")
    plt.ylabel(L"\psi_n")
    plt.legend()
    plt.savefig("bound_eigenstates_barrier.pdf")
    plt.show()

    # Root finding
    find_roots() # Prints roots, corresponds to bound eigenvals

    # Plot f(lambda) (equation 3.4)
    plt.plot([energy_eigenstate_below_v0(val) for val in 0:1e3])
    plt.ylabel(L"f(\lambda)")
    plt.xlabel(L"\lambda")
    plt.savefig("f_of_lambda_roots.pdf")
    plt.show()


    ######################
    ### Time evolution ###
    ######################

    t_norm_final = pi/(vals[2] - vals[1]) # Final time

    # Define and plot initial state
    initial = (1/sqrt(2))*(vecs[:, 1] + vecs[:, 2])
    plt.xlabel("x'")
    plt.ylabel(L"\Psi")
    plt.plot(initial, label="initial")
    plt.plot(Psi_evolve(t_norm_final, eigen_states=vecs, eigen_vals=vals, Psi_initial=initial))
    plt.legend()
    plt.savefig("initial_and_final_state.pdf")
    plt.show()

    ### Eigen expansion ###

    num_timesteps = 1000
    dt = t_norm_final/num_timesteps
    prob_dist_matrix = zeros(num_timesteps, length(initial))

    t = 0
    println(t_norm_final)
    
    plt.plot(initial)
    for i in 1:num_timesteps
        t += dt
        Psi_evolved = Psi_evolve(t, eigen_states=vecs, eigen_vals=vals, Psi_initial=initial)
        if i%(num_timesteps/10) == 0
            plt.plot(Psi_evolved)
        end
        prob_dist_matrix[i, :] = abs.(Psi_evolved) .^2
    end
    
    plt.title("Time development")
    plt.xlabel("x'")
    plt.ylabel(L"\psi")
    plt.legend()
    plt.savefig("time_evolve_samples.pdf")
    plt.show()

    plt.title(L"\mid \Psi(x', t') \mid ^2")
    plt.ylabel("t'")
    plt.xlabel("x'")
    plt.imshow(prob_dist_matrix)
    plt.savefig("time_evolve_surface.pdf")
    plt.show()

    ### step by step ###

    t_end = t_norm_final
    dt = 0.1
    N_time = t_end/dt

    #Psi_initial_step = vecs[:, 1]
    #Psi_initial_step = initial
    Psi_initial_step = dirac_delta(divider=6)
    plt.plot(Psi_initial_step)
    plt.show()
    plt.legend()

    # forward Euler, steps before breakdown
    dt_vec = collect(1e-7:1e-7:1e-3)
    steps_vec = zeros(length(dt_vec))

    println(length(steps_vec), " ", length(dt_vec))

    for (i, dt) in enumerate(dt_vec)
        steps_vec[i] = forward_euler_breakdown(H, dt, Psi_initial_step)
    end

    plt.title(string(L"\Delta x' \approx ", round(DELTA_X_NORM, digits=3)))
    plt.plot(dt_vec .* (NUM_POINTS^2), steps_vec)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(L"\Delta t'\left / (\Delta x')^2 \right.")
    plt.ylabel("#steps")
    plt.savefig("steps_before_euler_collapse.pdf")
    plt.show()

    # Crank Nicolson
    Psi_evolved_cn = crank_nicolson(H, N_time, dt, Psi_initial_step)
    plt.plot(Psi_initial_step, label=L"\Psi_0 (initial)")
    plt.plot(Psi_evolved_cn, label=L"\Psi (final)")
    plt.xlabel("x'")
    plt.ylabel(L"\Psi")
    plt.savefig("crank_nicolson.pdf")
    plt.legend()
    plt.show()
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    #zero_barrier()
    high_barrier()
end



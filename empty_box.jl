include("utils.jl")


function main()
 	# Create normalized H matrix
    H, diag, off_diag = hamiltonian_matrix()

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

    # Time evolution

    # Initial wave functions
    initial1 = [analytic_eigenfunc(1, x) for x in 0:DELTA_X_NORM:1]
    initial2 = dirac_delta()
    
    t_norm_final1 = 0.05
    t_norm_final2 = 0.1

    Psi_evolved1 = Psi_evolve(t_norm_final1,
                            eigen_states=vecs,
                            eigen_vals=vals,
                            Psi_initial=initial1)

    Psi_evolved2 = Psi_evolve(t_norm_final2,
                            eigen_states=vecs,
                            eigen_vals=vals,
                            Psi_initial=initial2)

    N_time = 1000
    dt = t_norm_final2/N_time
    
    Psi_evolved2_cn = crank_nicolson(H, N_time, dt, initial2)

    plt.title(string(L"t'_{final} = ", t_norm_final1))
    plt.xlabel("x'")
    plt.ylabel(L"\Psi")
    plt.plot(initial1, label="initial")
    plt.plot(Psi_evolved1, label="evolved")
    plt.legend()
    plt.savefig("time_evolve_emptybox_1.pdf")
    plt.show()

    plt.title(string(L"t'_{final} = ", t_norm_final2))
    plt.xlabel("x'")
    plt.ylabel(L"\Psi")
    plt.plot(initial2, label="initial")
    plt.plot(Psi_evolved2, label="evolved")
    plt.legend()
    plt.savefig("time_evolve_emptybox_2.pdf")
    plt.show()

    plt.title(string(L"t'_{final} = ", t_norm_final2))
    plt.xlabel("x'")
    plt.ylabel(L"\Psi")
    plt.plot(initial2, label="initial")
    plt.plot(Psi_evolved2_cn, label="evolved")
    plt.legend()
    plt.savefig("time_evolve_emptybox_2_cn.pdf")
    plt.show()

    println("delta: ", dot(initial2, initial2))
    println("ev1: ", dot(Psi_evolved1, Psi_evolved1))
    println("ev2: ", dot(Psi_evolved2, Psi_evolved2))
    println("ev2 cn", dot(Psi_evolved2_cn, Psi_evolved2_cn))
end


if basename(PROGRAM_FILE) == basename(@__FILE__)
    main()
end



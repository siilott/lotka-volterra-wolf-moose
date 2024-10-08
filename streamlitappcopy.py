import pints
import pints.toy
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import scipy
import pints.plot
    
model = pints.toy.LotkaVolterraModel()

def load_lynx_hare_data(file_location):
    names = ["year", "hare", "lynx"]
    df = pd.read_csv(file_location, header=None, names=names)
    return df

df = load_lynx_hare_data("lynxhare.csv")
df['modified time'] = df['year'] 
mod_times = df['modified time'].values
observed_data = df[['hare', 'lynx']].values

plt.figure()
plt.xlabel('Years since 1845')
plt.ylabel('Population size')
plt.plot(mod_times,  observed_data)
plt.legend(['hare', 'lynx'])
plt.show()

# Ensure mod_times is not empty
if len(mod_times) == 0:
    st.error("No data available for the year range.")
else:
    # Set up the slider
    year_range = st.slider(
        'Select year range',
        min_value=int(mod_times.min()),
        max_value=int(mod_times.max()),
        value=[int(mod_times.min()), int(mod_times.max())]
    )

    # Create a mask to filter the data
    mask = (mod_times >= year_range[0]) & (mod_times <= year_range[1])
    filtered_years = mod_times[mask] - 1845
    values = observed_data[mask]
    filtered_hare_population = values[:, 0]
    filtered_lynx_population = values[:, 1]

    # Check if filtered data is available
    if len(filtered_years) == 0:
        st.error("No data available for the selected year range.")
    else:
        # Title for the plot
        st.title("Time Series Solution")

        # Plot the data
        fig1, ax1 = plt.subplots()
        ax1.plot(filtered_years, filtered_hare_population, label='Hare')
        ax1.plot(filtered_years, filtered_lynx_population, label='Lynx')
        ax1.set_xlabel('Years since 1845')
        ax1.set_ylabel('Population size in 1000s')
        ax1.legend()
        # Display the plot in the Streamlit app
        st.pyplot(fig1)

        # Title for the plot
        st.title("Phase Space Solution")

        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0, 150)
        ax2.set_ylim(0, 90)
        ax2.set_xlabel('Hare')
        ax2.set_ylabel('Lynx')
        scatter = ax2.scatter(filtered_hare_population, filtered_lynx_population, c=filtered_years)
        colorbar=fig2.colorbar(scatter, ax=ax2)
        # ax2.quiver(filtered_hare_population[1:], filtered_lynx_population[1:], np.diff(filtered_hare_population), np.diff(filtered_lynx_population)
        ax2.plot(filtered_hare_population, filtered_lynx_population)
    
        # Display the plot in the Streamlit app
        st.pyplot(fig2)   

        st.title("MCMC Fitting of Model to Data")
        # Create an object with links to the model and time series
        problem = pints.MultiOutputProblem(model, filtered_years, np.log(values))

        # Create a log posterior
        log_prior_theta = pints.UniformLogPrior(lower_or_boundaries=0, upper=2)
        log_prior_sigma = pints.GaussianLogPrior(mean=0, sd=3)
        log_prior = pints.ComposedLogPrior(log_prior_theta, log_prior_theta, log_prior_theta, log_prior_theta,
                                   log_prior_sigma, log_prior_sigma)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        # Run MCMC on the noisy data
        x0 = [[0.43, 0.16, 0.9, 0.27, 0.28, 0.26]] * 4
        mcmc = pints.MCMCController(log_posterior, 4, x0)
        mcmc.set_max_iterations(1000)

        chains = mcmc.run()   

        results = pints.MCMCSummary(
            chains=chains, 
            parameter_names=["a", "b", "c", "d", "sigma_1", "sigma_2"], 
            time=mcmc.time())

        # Select first chain
        chain1 = chains[0]

        # Remove warm up period
        chain1 = chain1[500:]

        # Create fine-grained list of times for plotting
        n_fine = 1000
        times_fine = np.linspace(min(filtered_years), max(filtered_years), n_fine)

        # Plot some predictions with these samples
        num_lines = 100
        hare = np.zeros((n_fine, num_lines))
        lynx = np.zeros((n_fine, num_lines))
        for i in range(num_lines):
            temp = np.exp(model.simulate(times=times_fine, parameters=chain1[i, :4]))
            hare[:, i] = temp[:, 0]
            lynx[:, i] = temp[:, 1]

        # Set parameters for optimization
        n_fine = 1000
        num_lines = 10
        times_fine = np.linspace(min(filtered_years), max(filtered_years), n_fine)

        # Arrays to hold simulated hare and lynx populations
        hare = np.zeros((n_fine, num_lines))
        lynx = np.zeros((n_fine, num_lines))


        # Collect results
        for i in range(num_lines):
            problem = pints.MultiOutputProblem(model, filtered_years, np.log(values))    
            error = pints.SumOfSquaresError(problem)

            # Optimization setup
            initial_parameters = [0.50, 0.10, 1.0, 0.50]
            bounds_lower = [0.01, 0.01, 0.01, 0.01]
            bounds_upper = [5, 5, 5, 5]

            transformation = pints.RectangularBoundariesTransformation(bounds_lower, bounds_upper)
            opt = pints.OptimisationController(
                error,
                initial_parameters,
                method=pints.CMAES,
                transformation=transformation
                )
            
            opt.set_log_interval(20)
            opt.set_max_evaluations(20000)
            
            optimized_parameters, _ = opt.run()
            optimized_simulation = np.exp(model.simulate(times = times_fine, parameters= optimized_parameters))
            hare[:, i] = optimized_simulation[:,0]
            lynx[:, i] = optimized_simulation[:,1]

        #Plotting
        fig3, ax3 = plt.subplots()

        # Plot the real data
        ax3.plot(filtered_years, filtered_hare_population, 'o-', label='Observed Hare')
        ax3.plot(filtered_years, filtered_lynx_population, 'o-', label='Observed Lynx')

        # Plot the hare and lynx populations with low opacity for visualization
        ax3.plot(times_fine, hare, color='blue', alpha=0.01)
        ax3.plot(times_fine, lynx, color='orange', alpha=0.01)

        # Set labels for the axes
        ax3.set_xlabel('Years since 1845')
        ax3.set_ylabel('Populations')

        ax3.legend()

        # Display the plot in the Streamlit app
        st.pyplot(fig3)
        
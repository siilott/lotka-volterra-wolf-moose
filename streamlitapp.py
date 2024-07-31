import pints
import pints.toy
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import scipy

def load_lynx_hare_data(file_location):
    names = ["year", "hare", "lynx"]
    df = pd.read_csv(file_location, header=None, names=names)
    return df

df = load_lynx_hare_data("/Users/sajai/Documents/lotka-volterra-wolf-moose/lynxhare.csv")
df['modified time'] = df['year'] - 1845
mod_times = df['modified time'].values
observed_data = df[['hare', 'lynx']].values

plt.figure()
plt.xlabel('Years since 1845')
plt.ylabel('Population size')
plt.plot(mod_times,  observed_data)
plt.legend(['hare', 'lynx'])
plt.show()

# Title for the plot
st.write("Hare and Lynx Population Over Time")

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
    filtered_years = mod_times[mask]
    values = observed_data[mask]
    filtered_hare_population = values[:, 0]
    filtered_lynx_population = values[:, 1]

    # Check if filtered data is available
    if len(filtered_years) == 0:
        st.error("No data available for the selected year range.")
    else:
        # Plot the data
        fig1, ax1 = plt.subplots()
        ax1.plot(filtered_years, filtered_hare_population, label='Hare')
        ax1.plot(filtered_years, filtered_lynx_population, label='Lynx')
        ax1.set_xlabel('Years since 1845')
        ax1.set_ylabel('Population size')
        ax1.legend()
        # Display the plot in the Streamlit app
        st.pyplot(fig1)

        st.title("Hare and Lynx Population Analysis")

        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0, 80)
        ax2.set_ylim(0, 80)
        ax2.set_xlabel('hare')
        ax2.set_ylabel('lynx')
        ax2.plot(filtered_hare_population, filtered_lynx_population)
        ax2.quiver(filtered_hare_population[1:], filtered_lynx_population[1:], np.diff(filtered_hare_population), np.diff(filtered_lynx_population))
        ax2.legend()
        # Display the plot in the Streamlit app
        st.pyplot(fig2)
    
        def dX_dt(X, t):
            a, b, c, d =  0.32, 0.13, 1.27, 0.38
            x, y = X
            dotx = x * (a - b * y)
            doty = y * (-c + d * x)
            return np.array([dotx, doty])

        def plot_lotka_volterra_phase_plane():
            a, b, c, d =  0.32, 0.13, 1.27, 0.38
            plt.figure(figsize=(10,10))

            init_x, init_y =  filtered_hare_population, filtered_lynx_population
    
            plt.plot(init_x, init_y, 'g*', markersize=30)

            for v in values:
                X0 = v                          # starting point
                X = scipy.integrate.odeint( dX_dt, X0, np.linspace(0,80,3000))
                plt.plot( X[:,0], X[:,1], lw=3, color='green')

            # plot nullclines
            x = np.linspace(0, 250, 24)
            y = np.linspace(0, 250, 24)

            plt.hlines(a/b, x.min(), x.max(), color='#F39200', lw=4, label='y-nullcline 1')
            plt.plot(x,(a/b)* np.ones_like(x), color='#0072bd', lw=4, label='x-nullcline 2')
            plt.vlines(c/d, x.min(), x.max(), color='#0072bd', lw=4, label='x-nullcline 1')
            plt.plot(x, (c/d)* np.ones_like(x), color='#F39200', lw=4, label='y-nullcline 2')

            # quiverplot - define a grid and compute direction at each point
            X, Y = np.meshgrid(x, y)  # create a grid
            DX = a * X - b * X * Y  # evaluate dx/dt
            DY = -c * Y + d * X * Y  # evaluate dy/dt
            M = (np.hypot(DX, DY))  # norm growth rate
            M[M == 0] = 1.  # avoid zero division errors

            plt.quiver(X, Y, DX / M, DY / M, M)
            plt.xlim(-0.05, 100)
            plt.ylim(-0.05, 250)
            plt.xlabel('Prey (Hare) Population')
            plt.ylabel('Predator (Lynx) Population')
            plt.title('Phase Plane of the Lotka-Volterra Model')
            plt.show()

        fig_phase_plane = plot_lotka_volterra_phase_plane()
        st.pyplot(fig_phase_plane)
        
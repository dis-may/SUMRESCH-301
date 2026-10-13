import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_errors(vehicle):
    with open(f"{vehicle}_errors_new.pkl", "rb") as file:
        errors = pickle.load(file)
        plt.plot(errors, '.', alpha=0.5, ms=0.5)

        # x = range(len(errors))
        # y = errors
        # # 2. Calculate the trendline (deg=1 means linear)
        # coefficients = np.polyfit(x, y, deg=1)
        # trend_function = np.poly1d(coefficients)

        # # 3. Plot the trendline
        # plt.plot(x, trend_function(x), color="red", linestyle="--", linewidth=2, label="Trendline")


        plt.title(f"{vehicle.capitalize()} Training Error")
        plt.yscale('log')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Batch')

        plt.savefig("./new_plots_with_rk4/" + f"{vehicle}_errors_new", dpi=300)
        plt.show()


plots = ["aggression", "fear", "love", "explorer"]
for vehicle in plots:
    plot_errors(vehicle)


# plot_errors("aggression")
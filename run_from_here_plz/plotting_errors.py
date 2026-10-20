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

        plt.savefig(f"{vehicle}_errors_new", dpi=300)
        plt.show()


def plot_testing_error(vehicle):
    with open(f"{vehicle}_epoch_error.pkl", "rb") as file:
        e = pickle.load(file)

    plt.figure(figsize=(7.5, 4))
    plt.plot(e)
    plt.xticks(range(len(e)))
    plt.xlabel("Epoch")
    plt.ylabel("MSE Error")
    plt.grid(True)
    plt.title(vehicle.capitalize() + " Testing Error")
    plt.savefig(vehicle + "_testing_epoch_error.png", dpi=300)
    plt.show()

def plot_all_testing_errors():
    vehicles = ["aggression", "fear", "love", "explorer"]
    errors = []
    for vehicle in vehicles:
        with open(f"{vehicle}_epoch_error.pkl", "rb") as file:
            errors.append(pickle.load(file))
    
    plt.figure(figsize=(7.5, 4))
    for i in range(len(errors)):
        e = errors[i]
        plt.plot(e, label=vehicles[i].capitalize())
    plt.xticks(range(len(e)))
    plt.xlabel("Epoch")
    plt.ylabel("MSE Error")
    plt.grid(True)
    plt.title("Testing Error")
    plt.legend()
    plt.savefig("all_testing_epoch_error.png", dpi=300)
    plt.show()

# plots = ["aggression", "fear", "love", "explorer"]
# for vehicle in plots:
#     plot_errors(vehicle)


# plot_errors("aggression")

# plot_testing_error("love")

plot_all_testing_errors()

import numpy as np
import timeit as it
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import gymnasium as gym


def vanilla_mse(observed, predicted):
    """Basic Python implementation of MSE."""
    if len(observed) != len(predicted):
        raise ValueError("Observed and predicted lists must have the same length.")
    squared_errors = 0.0
    for o, p in zip(observed, predicted):
        squared_errors += (o - p) ** 2
    return squared_errors / len(observed)

def mse_numpy(observed, predicted):
    """MSE using NumPy."""
    obs = np.array(observed)
    pred = np.array(predicted)
    return np.mean((obs - pred) ** 2)

def mse_sklearn(observed, predicted):
    """Calculate MSE using scikit-learn's built-in function."""
    return mean_squared_error(observed, predicted)

# Test data
observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

mse_functions = {
    'vanilla': vanilla_mse,
    'numpy': mse_numpy,
    'sklearn': mse_sklearn
}

for name, func in mse_functions.items():
    mse = func(observed, predicted)
    exec_time = it.timeit(lambda: func(observed, predicted), number=1000) / 1000
    print(f"MSE ({name}): {mse}, Time: {exec_time:.6f} seconds")

# Check if all MSE values are equal
if (vanilla_mse(observed, predicted) == mse_numpy(observed, predicted) == mse_sklearn(observed, predicted)):
    print("Test successful")
else:
    print("Test failed")


def generate_oscillatory_data(n_points, noise_scale=0.1):
    """Generate 1D oscillatory data with and without noise. """
    x = np.linspace(0, 10, n_points)
    y_true = np.cos(x)  # Oscillatory function 
    noise = np.random.normal(0, noise_scale, n_points)
    y_noisy = y_true + noise
    print(f"Data generated: {n_points} points, range=[0, 10], noise_scale={noise_scale}")
    return x, y_true, y_noisy

def plot_oscillatory_data(x, y_true, y_noisy):
    """Plot the true and noisy oscillatory data."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True Data", color="blue", linewidth=2)
    plt.scatter(x, y_noisy, label="Noisy Data", color="red", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("1D Oscillatory Data (Truth vs. Noisy)")
    plt.legend()
    plt.grid(True)
    plt.savefig("../data/oscillatory_data.png")
    plt.close()
    print("Oscillatory data plot saved to data/oscillatory_data.png.")


x,y,z = generate_oscillatory_data(343, noise_scale=0.4)
plot_oscillatory_data(x,y,z)

def plot_clustering_variance(x, y_noisy):
    """Perform clustering and plot variance vs. number of clusters.
    Parameters:
    -----------
    x : numpy.ndarray,
        The input values.
    y_noisy : numpy.ndarray,
        The noisy oscillatory data.
    """
    data = np.column_stack((x, y_noisy))    # Combine x and y_noisy into a 2D dataset
    variances = []                          # Initialize list to store variances
    cluster_range = range(1, 11)            # Test different numbers of clusters (from 1 to 10)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=26)
        kmeans.fit(data)
        variances.append(kmeans.inertia_)
    
    # Plot variance vs. number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, variances, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Variance (Inertia)")
    plt.title("Variance vs. Number of Clusters")
    plt.grid(True)
    plt.savefig("../data/clustering_variance.png")
    plt.close()
    print("Clustering completed. Variance plot saved to data/clustering_variance.png.")

x, y_true, y_noisy = generate_oscillatory_data(n_points=100, noise_scale=0.2)
plot_clustering_variance(x, y_noisy)


def run_linear_regression(x, y):
    """Linear regression using scikit-learn."""
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model.predict(x.reshape(-1, 1))

def run_neural_network(x, y, max_iter=1000):
    """Neural network regression."""
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=max_iter, random_state=42)
    model.fit(x.reshape(-1, 1), y)
    return model.predict(x.reshape(-1, 1))

def run_pinn(x, y, max_iter=1000):
    """Simplified PINN (same as NN for demonstration)."""
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=max_iter, random_state=42)
    model.fit(x.reshape(-1, 1), y)
    return model.predict(x.reshape(-1, 1))


# Train models
y_pred_lr = run_linear_regression(x, y_noisy)
y_pred_nn = run_neural_network(x, y_noisy)
y_pred_pinn = run_pinn(x, y_noisy)

print("Task completed: Linear Regression")
print("Task completed: Neural Network")
print("Task completed: PINN")


def monitor_progress(x, y, model, max_iter=100):
    """Monitor the progress of regression models."""
    losses = []
    for i in range(1, max_iter + 1):
        if hasattr(model, 'max_iter'):
            # For models like MLPRegressor, update max_iter
            model.max_iter = i
        else:
            # For models like LinearRegression, simply fit with the current data
            pass
        
        # Fit the model
        model.fit(x.reshape(-1, 1), y)
        
        # Predict and calculate loss
        y_pred = model.predict(x.reshape(-1, 1))
        losses.append(mean_squared_error(y, y_pred))
    return losses
    

# Track losses for all models
model_lr = LinearRegression()
losses_lr = monitor_progress(x, y_noisy, model_lr, 100)

model_nn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1, random_state=42)
losses_nn = monitor_progress(x, y_noisy, model_nn, 100)

model_pinn = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1, random_state=42)
losses_pinn = monitor_progress(x, y_noisy, model_pinn, 100)


def plot_combined_results(x, y_true, y_noisy, losses_lr, losses_nn, losses_pinn):
    """Combine plots for Tasks 5, 6, 7."""
    plt.figure(figsize=(10, 12))
    
    # Subplot 1: Regression results
    plt.subplot2grid((3,1), (0, 0))
    plt.scatter(x, y_noisy, label="Noisy Data", alpha=0.5)
    plt.plot(x, y_true, label="True Function", linewidth=2)
    plt.plot(x, y_pred_lr, "--", label="Linear Regression")
    plt.plot(x, y_pred_nn, "-.", label="Neural Network")
    plt.plot(x, y_pred_pinn, ":", label="PINN")
    plt.title("Regression Results")
    plt.grid()
    plt.legend()
    
    # Subplot 2: Error vs. Truth
    plt.subplot2grid((3,1), (1,0))
    plt.plot(x, y_true - y_pred_lr, "--", label="Linear Regression Error")
    plt.plot(x, y_true - y_pred_nn, "-.", label="Neural Network Error")
    plt.plot(x, y_true - y_pred_pinn, ":", label="PINN Error")
    plt.title("Error vs. Truth")
    plt.grid()
    plt.legend()
    
    # Subplot 3: Progress Monitoring (Loss vs. Iterations)
    plt.subplot2grid((3,1), (2,0))
    iterations = range(1, 101)
    plt.plot(iterations, losses_lr, "--", label="Linear Regression")
    plt.plot(iterations, losses_nn, "-.", label="Neural Network")
    plt.plot(iterations, losses_pinn, ":", label="PINN")
    plt.title("Loss vs. Iterations")
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("../data/combined_plots.png")
    plt.close()
    print("Combined plots saved to data/combined_plots.png")


plot_combined_results(x, y_true, y_noisy, losses_lr, losses_nn, losses_pinn)


def run_reinforcement_learning(episodes=10):
    """
    Run a simple reinforcement learning task using the gymnasium library.
    This script uses a random policy to interact with the CartPole environment.
    """
    # Create the CartPole environment
    env = gym.make("CartPole-v1")
    print("Running reinforcement learning on CartPole-v1...")
    
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()  # Reset the environment for a new episode
        total_reward = 0
        done = False
        
        # Interact with the environment until the episode ends
        while not done:
            action = env.action_space.sample()  # Take a random action
            next_state, reward, terminated, truncated, info = env.step(action)  # Execute the action
            done = terminated or truncated  # Check if the episode is done
            total_reward += reward  # Accumulate the reward
        
        # Record the total reward for this episode
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    
    # Close the environment
    env.close()
    
    # Print the average reward over all episodes
    print("Reinforcement learning completed.")
    print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f}")

# Run the RL script
run_reinforcement_learning(episodes=20)

import numpy as np
import matplotlib.pyplot as plt

class DatasetGenerator:
    def generate_random_2d(self, n_points):
        return np.random.rand(n_points, 2)

    def generate_noisy_function(self, n_points, func, noise_scale=0.1):
        """Generate dataset with noise around a function."""
        x = np.linspace(0, 1, n_points)
        y = func(x) + noise_scale * np.random.randn(n_points)
        return np.column_stack((x, y))

    def append_datasets(self, dataset1, dataset2):
        """Append two datasets."""
        return np.vstack((dataset1, dataset2))

    def save_dataset(self, dataset, filename):
        """Save dataset to a file."""
        np.savetxt(filename, dataset, delimiter=',')

    def plot_dataset(self, dataset, title, filename):
        """Plot and save the dataset."""
        plt.scatter(dataset[:, 0], dataset[:, 1], s=10)
        plt.title(title)
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    generator = DatasetGenerator()

    # Generate random dataset
    random_data = generator.generate_random_2d(100)
    generator.save_dataset(random_data, "../data/random_data.csv")
    generator.plot_dataset(random_data, "Random 2D Dataset", "../data/random_plot.png")

    # Generate noisy dataset around a function 
    func = lambda x: x**2
    noisy_data = generator.generate_noisy_function(100, func)
    generator.save_dataset(noisy_data, "../data/noisy_data.csv")
    generator.plot_dataset(noisy_data, "Noisy Dataset (y = x^2)", "../data/noisy_plot.png")

    # Append datasets
    combined_data = generator.append_datasets(random_data, noisy_data)
    generator.save_dataset(combined_data, "../data/combined_data.csv")
    generator.plot_dataset(combined_data, "Combined Dataset", "../data/combined_plot.png")
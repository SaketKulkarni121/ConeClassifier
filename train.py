import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import pickle
import os
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

PLOT_FLAG = False


class TrackConeSimulator:
    def __init__(self, num_points=20, noise_std=0.1):
        x = np.linspace(0, 10, num_points)
        y = np.cumsum(np.random.randn(num_points)) * 0.5

        tck, u = interpolate.splprep([x, y], s=1)

        self.spline_func = lambda t: interpolate.splev(t, tck)
        self.tck = tck

        self.x = x
        self.y = y

    def get_start_point_and_direction(self):
        # Get the first point (start point)
        start_x, start_y = self.spline_func(0)

        # Calculate the direction at t=0 (tangent vector)
        dx, dy = interpolate.splev(0, self.tck, der=1)

        # Normalize the direction vector
        direction = np.array([dx, dy])
        direction /= np.linalg.norm(direction)

        return np.array([start_x, start_y]), direction

    def generate_cones(self, cone_spacing=0.20, lateral_offset=0.15, lateral_noise=0.02, longitudinal_noise=0.02):
        t = np.linspace(0, 1, int(10 / cone_spacing))

        dx, dy = interpolate.splev(t, self.tck, der=1)

        normals = np.column_stack([-dy, dx])
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

        left_cones = []
        right_cones = []

        for i, (x, y, normal) in enumerate(zip(self.spline_func(t)[0], self.spline_func(t)[1], normals)):
            long_noise = np.random.normal(0, longitudinal_noise)

            lat_noise_left = np.random.normal(0, lateral_noise)
            left_cone = np.array([x, y]) + normal * (lateral_offset + lat_noise_left)
            left_cones.append(left_cone)

            lat_noise_right = np.random.normal(0, lateral_noise)
            right_cone = np.array([x, y]) - normal * (lateral_offset + lat_noise_right)
            right_cones.insert(0, right_cone)

        left_cones = np.array(left_cones)
        right_cones = np.array(right_cones)

        # Get start point and direction
        start_point, direction = self.get_start_point_and_direction()

        if PLOT_FLAG:
            plt.figure(figsize=(12, 6))
            t = np.linspace(0, 1, 200)
            x, y = self.spline_func(t)

            plt.plot(x, y, "g-", label="Track Centerline")
            plt.scatter(left_cones[:, 0], left_cones[:, 1], color="blue", label="Left Cones")
            plt.scatter(right_cones[:, 0], right_cones[:, 1], color="red", label="Right Cones")

            # Highlight initial cones
            plt.scatter(
                left_cones[:5, 0],
                left_cones[:5, 1],
                color="cyan",
                marker="x",
                s=100,
                label="Initial Left Cones",
            )
            plt.scatter(
                right_cones[-5:, 0],
                right_cones[-5:, 1],
                color="magenta",
                marker="x",
                s=100,
                label="Initial Right Cones",
                )

            # Plot the start point and direction
            plt.quiver(
                start_point[0],
                start_point[1],
                direction[0],
                direction[1],
                angles="xy",
                scale_units="xy",
                scale=0.5,
                color="purple",
                label="Start Direction",
            )
            plt.scatter(start_point[0], start_point[1], color="orange", label="Start Point")
            
            # print(right_cones)

            plt.title("Track Simulation with Cones")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.legend()
            plt.grid(True)
            plt.show()

        return left_cones, right_cones, start_point, direction

    def plot_track(self, left_cones, right_cones):
        if PLOT_FLAG:
            plt.figure(figsize=(12, 6))

            t = np.linspace(0, 1, 200)
            x, y = self.spline_func(t)

            plt.plot(x, y, "g-", label="Track Centerline")
            plt.scatter(left_cones[:, 0], left_cones[:, 1], color="blue", label="Left Cones")
            plt.scatter(right_cones[:, 0], right_cones[:, 1], color="red", label="Right Cones")

            plt.title("Track Simulation with Cones")

            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")

            plt.legend()
            plt.grid(True)
            plt.show()

    def generate_training_data(self, num_samples, num_splines):
        all_features = []
        all_labels = []
        num_initial_cones = 5  # Number of initial cones to include

        for spline_idx in range(num_splines):
            self.__init__(num_points=20)
            if spline_idx % 1000 == 0:
                print(f"  Generating spline {spline_idx + 1}/{num_splines}...")

            for sample_idx in range(num_samples):
                left_cones, right_cones, start_point, direction = self.generate_cones()
                cones = np.vstack([left_cones, right_cones])
                labels = np.array([0] * len(left_cones) + [1] * len(right_cones))

                # Extract initial cones
                first_left = left_cones[:num_initial_cones].flatten()
                first_right = right_cones[-num_initial_cones:].flatten()

                for cone in cones:
                    # Calculate distance from start point to cone
                    distance = np.linalg.norm(cone - start_point)

                    # Enhanced feature vector
                    features = [
                        *cone,
                        distance,
                        *start_point,
                        *direction,
                        *first_left,
                        *first_right,
                    ]
                    all_features.append(features)

                all_labels.extend(labels)
                
            if sample_idx % 1000 == 0:
                save_training_data_incrementally((np.array(all_features), np.array(all_labels)))
                
                all_features.clear()
                all_labels.clear()

        print("Training data generation complete.")

        return True

    def train_cone_classifier(self, X, y, test_size, max_iter):
        print("Training XGBoost classifier...")

        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Hyperparameter tuning
        param_dist = {
            "max_depth": [6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "n_estimators": [500, 1000, 2000],
            "lambda": [1, 5, 10],  # L2 regularization
            "alpha": [0, 1, 5],  # L1 regularization
        }

        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        search = RandomizedSearchCV(xgb_model, param_dist, n_iter=20, cv=3, scoring="accuracy", verbose=2, n_jobs=-1)
        search.fit(X_train_scaled, y_train)

        # Train the best model
        best_params = search.best_params_
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train_scaled, y_train, early_stopping_rounds=50, eval_set=[(X_test_scaled, y_test)], verbose=50)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Final Accuracy: {accuracy * 100:.2f}%")

        # Save model and scaler
        model.save_model("best_model.bin")
        joblib.dump(scaler, "scaler.bin")

        return model, accuracy, scaler


    def test_classifier_on_random_splines(self, classifier, scaler, num_splines, num_samples):
        print(f"Testing classifier on {num_splines} random splines...")

        accuracies = []
        num_initial_cones = 5

        for spline_idx in range(num_splines):
            test_simulator = TrackConeSimulator()

            all_features = []
            all_labels = []

            for _ in range(num_samples):
                left_cones, right_cones, start_point, direction = test_simulator.generate_cones()
                cones = np.vstack([left_cones, right_cones])
                labels = np.array([0] * len(left_cones) + [1] * len(right_cones))

                # Extract initial cones
                first_left = left_cones[:num_initial_cones].flatten()
                first_right = right_cones[-num_initial_cones:].flatten()

                for cone in cones:
                    # Calculate distance from start point to cone
                    distance = np.linalg.norm(cone - start_point)

                    # Enhanced feature vector
                    features = [
                        *cone,
                        distance,
                        *start_point,
                        *direction,
                        *first_left,
                        *first_right,
                    ]
                    all_features.append(features)

                all_labels.extend(labels)

            X_test = np.array(all_features)
            y_test = np.array(all_labels)

            X_test_scaled = scaler.transform(X_test)
            dtest = xgb.DMatrix(X_test_scaled)

            # Predict using the classifier
            y_pred = classifier.predict(dtest)
            y_pred = (y_pred > 0.5).astype(int)  # Convert to binary classification

            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        average_accuracy = np.mean(accuracies)
        print(f"Average accuracy over {num_splines} splines: {average_accuracy:.4f}")
        return average_accuracy


def save_training_data_incrementally(data, filename="training_data.pkl"):
    """Save training data to a pickle file incrementally."""
    mode = 'ab' if os.path.exists(filename) else 'wb'
    
    with open(filename, mode) as f:
        pickle.dump(data, f)



def load_training_data(filename="training_data.pkl"):
    """Load training data from a pickle file if it exists"""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None


if __name__ == "__main__":
    print("Initializing TrackConeSimulator...")
    simulator = TrackConeSimulator()

    print("Generating training data...")

    training_data = load_training_data()
    if training_data is None:
        completed= simulator.generate_training_data(num_samples=1, num_splines=5000000)
        if completed:
            print("Training data generation complete.")
            X, y = load_training_data()
    else:
        print("Loading training data from file...")
        X, y = training_data

    # Try to load pre-trained model and scaler if they exist
    if os.path.exists("model.bin") and os.path.exists("scaler.bin"):
        print("Loading pre-trained model and scaler...")
        classifier = xgb.Booster()
        classifier.load_model("model.bin")
        scaler = joblib.load("scaler.bin")
    else:
        print("No pre-trained model found, training new model...")
        classifier, accuracy, scaler = simulator.train_cone_classifier(X, y, test_size=0.2, max_iter=1000)

    print("Testing classifier on random splines...")
    average_accuracy = simulator.test_classifier_on_random_splines(classifier, scaler, num_splines=10, num_samples=1000)
    print(f"Average Test Accuracy: {average_accuracy:.4f}")
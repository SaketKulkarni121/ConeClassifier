import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

class TrackConeSimulator:
    def __init__(self, num_points=20, noise_std=0.1):
        x = np.linspace(0, 10, num_points)
        y = np.cumsum(np.random.randn(num_points)) * 0.5
        
        tck, u = interpolate.splprep([x, y], s=1)
        
        self.spline_func = lambda t: interpolate.splev(t, tck)
        self.tck = tck
        
        self.x = x
        self.y = y
    
    def generate_cones(self, cone_spacing=.30, lateral_offset=.15, lateral_noise=0.02, longitudinal_noise=0.02):
        t = np.linspace(0, 1, int(10/cone_spacing))
        
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
            right_cones.append(right_cone)
            
        return np.array(left_cones), np.array(right_cones)

    def plot_track(self, left_cones, right_cones):
        plt.figure(figsize=(12, 6))
        
        t = np.linspace(0, 1, 200)
        x, y = self.spline_func(t)
        
        plt.plot(x, y, 'g-', label='Track Centerline')
        plt.scatter(left_cones[:, 0], left_cones[:, 1], color='blue', label='Left Cones')
        plt.scatter(right_cones[:, 0], right_cones[:, 1], color='red', label='Right Cones')
        
        plt.title('Track Simulation with Cones')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def generate_training_data(self, num_samples=1000, num_splines=200):
        all_features = []
        all_labels = []
        
        for spline_idx in range(num_splines):
            self.__init__(num_points=20)
            
            for sample_idx in range(num_samples):
                left_cones, right_cones = self.generate_cones()
                
                cones = np.vstack([left_cones, right_cones])
                labels = np.array([0]*len(left_cones) + [1]*len(right_cones))
                
                for cone in cones:
                    track_x, track_y = self.spline_func(np.linspace(0, 1, 100))
                    
                    distances = np.linalg.norm(np.column_stack((track_x, track_y)) - cone, axis=1)
                    distance = np.min(distances)
                    
                    angle = np.arctan2(cone[1] - track_y[np.argmin(distances)], cone[0] - track_x[np.argmin(distances)])
                    angles = np.arctan2(np.diff(track_y), np.diff(track_x))
                    
                    curvature = np.mean(np.diff(angles))
                    
                    features = [*cone, distance, angle, curvature]
                    
                    all_features.append(features)
                    
                all_labels.extend(labels)
                
        return np.array(all_features), np.array(all_labels)

    def train_cone_classifier(self, X, y, test_size=0.2, max_iter=1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            max_iter=max_iter,
            learning_rate_init=0.005,
            alpha=0.01,
            random_state=42
        )
        
        classifier.fit(X_train_scaled, y_train)
        
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        joblib.dump(classifier, 'cone_classifier.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        
        return classifier, accuracy

def test_classifier_on_random_splines(classifier, simulator, scaler, num_splines=30, num_samples=100):
    accuracies = []
    
    for spline_idx in range(num_splines):
        test_simulator = TrackConeSimulator()
        
        all_features = []
        all_labels = []
        
        for _ in range(num_samples):
            left_cones, right_cones = test_simulator.generate_cones()
            cones = np.vstack([left_cones, right_cones])
            labels = np.array([0]*len(left_cones) + [1]*len(right_cones))
            
            for cone in cones:
                track_x, track_y = test_simulator.spline_func(np.linspace(0, 1, 100))
                
                distances = np.linalg.norm(np.column_stack((track_x, track_y)) - cone, axis=1)
                distance = np.min(distances)
                
                angle = np.arctan2(cone[1] - track_y[np.argmin(distances)], cone[0] - track_x[np.argmin(distances)])
                angles = np.arctan2(np.diff(track_y), np.diff(track_x))
                
                curvature = np.mean(np.diff(angles))
                
                features = [*cone, distance, angle, curvature]
                
                all_features.append(features)
                
            all_labels.extend(labels)
            
        X_test = np.array(all_features)
        y_test = np.array(all_labels)
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
    average_accuracy = np.mean(accuracies)
    return average_accuracy

if __name__ == "__main__":
    simulator = TrackConeSimulator()
    X, y = simulator.generate_training_data(num_samples=1000, num_splines=10)
    classifier, accuracy = simulator.train_cone_classifier(X, y)
    print(f"Final Training Accuracy: {accuracy:.4f}")
    average_accuracy = test_classifier_on_random_splines(classifier, simulator, joblib.load('scaler.joblib'))

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import joblib
import time
import pickle

class TrackValidator:
    def __init__(self, classifier_path='cone_classifier.joblib', scaler_path='scaler.joblib'):
        self.classifier = joblib.load(classifier_path)
        self.scaler = joblib.load(scaler_path)
        self.total = 0
        self.incorrect = 0
    
    def generate_random_track(self, num_points=30):
        x = np.linspace(0, 10, num_points)
        y = np.cumsum(np.random.randn(num_points)) * 0.5
        
        tck, u = interpolate.splprep([x, y], s=1)
        
        spline_func = lambda t: interpolate.splev(t, tck)
        
        return spline_func, tck
    
    def generate_cones(self, spline_func, tck, cone_spacing=0.3, lateral_offset=0.15, lateral_noise=0.04, longitudinal_noise=0.02):
        t = np.linspace(0, 1, int(10/cone_spacing))
        dx, dy = interpolate.splev(t, tck, der=1)
        
        normals = np.column_stack([-dy, dx])
        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
        
        left_cones = []
        right_cones = []
        for i, (x, y, normal) in enumerate(zip(spline_func(t)[0], spline_func(t)[1], normals)):
            lat_noise_left = np.random.normal(0, lateral_noise)
            left_cone = np.array([x, y]) + normal * (lateral_offset + lat_noise_left)
            left_cones.append(left_cone)
            
            lat_noise_right = np.random.normal(0, lateral_noise)
            right_cone = np.array([x, y]) - normal * (lateral_offset + lat_noise_right)
            right_cones.append(right_cone)
            
        return np.array(left_cones), np.array(right_cones)
    
    def prepare_cone_features(self, spline_func, cones):
        features = []
        
        for cone in cones:
            track_x, track_y = spline_func(np.linspace(0, 1, 100))
            
            distances = np.linalg.norm( np.column_stack((track_x, track_y)) - cone, axis=1)
            distance = np.min(distances)
            
            features.append([*cone, distance])
        return np.array(features)
    
    def classify_cones(self, features):
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)
    
    def visualize_results(self, spline_func, left_cones, right_cones, predicted_labels):
        plt.figure(figsize=(12, 6))
        
        t = np.linspace(0, 1, 200)
        x, y = spline_func(t)
        
        plt.plot(x, y, 'g-', label='Track Centerline')
        
        all_cones = np.vstack([left_cones, right_cones])
        misclassified_mask = np.zeros(len(all_cones), dtype=bool)
        misclassified_mask[:len(left_cones)] = (predicted_labels[:len(left_cones)] != 0)
        misclassified_mask[len(left_cones):] = (predicted_labels[len(left_cones):] != 1)
        
        cone_colors = np.array(['blue'] * len(left_cones) + ['red'] * len(right_cones))
        cone_colors[misclassified_mask] = 'y'
        
        self.total += len(all_cones)
        self.incorrect += np.sum(misclassified_mask)
        
        plt.scatter(
            all_cones[:, 0], 
            all_cones[:, 1],
            color=cone_colors,
            label='Cones'
        )
        
        plt.title('Cone Classification Results')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        plt.legend()
        plt.grid(True)
        plt.pause(1)
        plt.clf()
        
    def getAccuracy(self):
        return (self.total, self.incorrect, (self.total - self.incorrect) / self.total)
        
def load_training_data(filename='training_data.pkl'):
    """Load training data from a pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    validator = TrackValidator()
    
    total_time = 0
    
    training_data = load_training_data()
    
    for i in range(30):
        spline_func, tck = validator.generate_random_track()
        left_cones, right_cones = validator.generate_cones(spline_func, tck)
        all_cones = np.vstack([left_cones, right_cones])
        features = validator.prepare_cone_features(spline_func, all_cones)
        
        start_time = time.time()
        predicted_labels = validator.classify_cones(features)
        
        end_time = time.time()
        
        total_time += (end_time - start_time)
        
        validator.visualize_results(spline_func, left_cones, right_cones, predicted_labels)
    
    total, incorrect, accuracy = validator.getAccuracy()
    print(f"Total cones: {total}")
    print(f"Incorrectly classified cones: {incorrect}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average time per iteration: {total_time / 30:.4f} seconds")

if __name__ == "__main__":
    main()

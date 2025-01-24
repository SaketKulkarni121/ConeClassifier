import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import random

class TrackSimulator:
    def __init__(self):
        self.track_width = 3
        self.cone_spacing_straight = 4  # Cone spacing in straight segments (meters)
        self.cone_spacing_turn_min = 0.5  # Minimum cone spacing in turns (meters)
        self.cone_spacing_turn_max = 2  # Maximum cone spacing in turns (meters)
        self.side_cone_distance = 1.5  # Distance of cones on the left and right side (meters)
        self.error = 0.2  # Random error for cone placement
        self.num_cones_max = 35  # Maximum number of cones per track
        self.max_track_length = 20  # Max track length in meters
        self.max_turns = 2  # Maximum of 2 turns per track
    
    def generate_random_spline(self):
        # Generate random control points for the spline
        x_points = [0]  # Start at x = 0
        y_points = [0]  # Start at y = 0

        total_length = 0
        turns_placed = 0  # Track number of turns
        
        # Create the straight portion before turning
        while total_length < self.max_track_length and turns_placed < self.max_turns:
            # Randomly decide whether to make a turn or stay straight
            turn_prob = random.uniform(0, 1)
            
            if turn_prob > 0.5 and turns_placed < self.max_turns:  # 50% chance to turn
                # Add a turn (left or right)
                segment_length = np.random.uniform(5, 7)  # Turn segment length
                curvature = np.random.uniform(0.1, 0.3)  # Moderate curvature for turns
                
                # Randomly choose the direction of the turn: left or right
                direction = random.choice([-1, 1])  # -1 for left turn, 1 for right turn
                angle_change = np.radians(np.random.uniform(10, 30)) * direction  # Random angle change
                
                dx = segment_length * np.cos(angle_change)
                dy = segment_length * np.sin(angle_change)
                
                # Append the new point based on current direction
                x_points.append(x_points[-1] + dx)
                y_points.append(y_points[-1] + dy)
                
                turns_placed += 1  # Increment the turn counter
            else:
                # Add a straight segment
                segment_length = np.random.uniform(4, 8)  # Straight segment length
                x_points.append(x_points[-1])
                y_points.append(y_points[-1] + segment_length)  # Move straight upwards
            
            total_length += segment_length
        
        # Ensure the track is smooth by interpolating the points into a spline
        try:
            spline = interpolate.splprep([x_points, y_points], s=0.1, per=False)  # Small smoothing factor
        except Exception as e:
            print(f"Error interpolating points: {e}")
            return None
        
        # Start point and direction vector
        start_point = (x_points[0], y_points[0])
        direction_vector = (x_points[1] - x_points[0], y_points[1] - y_points[0])
        
        print(f"Start point: {start_point}")
        print(f"Direction vector: {direction_vector}")
        
        return spline, x_points, y_points, start_point, direction_vector


    def generate_cones(self, spline):
        # Interpolate along the spline
        t = np.linspace(0, 1, 200)
        x, y = interpolate.splev(t, spline[0])

        cones_left_x = []
        cones_left_y = []
        cones_right_x = []
        cones_right_y = []
        
        total_length = 0

        # Debugging: Check if the track is being generated
        print(f"Track generated with {len(x)} points")

        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Determine cone spacing based on track curvature
            if np.abs(dy/dx) > 0.05:  # Significant change in slope (turn)
                cone_spacing = np.random.uniform(self.cone_spacing_turn_min, self.cone_spacing_turn_max)
            else:  # Straight segment
                cone_spacing = self.cone_spacing_straight
            
            # Debugging: Show spacing and segment details
            print(f"Segment {i}: Distance = {distance:.2f}, Spacing = {cone_spacing:.2f}")

            num_cones = int(distance // cone_spacing)
            print(f"Placing {num_cones} cones in this segment")
            
            for j in range(num_cones):
                t_cone = j * cone_spacing / distance
                cone_x = (1 - t_cone) * x[i-1] + t_cone * x[i]
                cone_y = (1 - t_cone) * y[i-1] + t_cone * y[i]
                
                # Calculate direction of track segment (for placing cones on left/right)
                angle = np.arctan2(dy, dx)  # Angle of the track direction
                
                # Add cones to the left and right sides
                left_cone_x = cone_x - self.side_cone_distance * np.sin(angle)
                left_cone_y = cone_y + self.side_cone_distance * np.cos(angle)
                
                right_cone_x = cone_x + self.side_cone_distance * np.sin(angle)
                right_cone_y = cone_y - self.side_cone_distance * np.cos(angle)
                
                # Add random error to cone positions
                left_cone_x += np.random.uniform(-self.error, self.error)
                left_cone_y += np.random.uniform(-self.error, self.error)
                
                right_cone_x += np.random.uniform(-self.error, self.error)
                right_cone_y += np.random.uniform(-self.error, self.error)
                
                cones_left_x.append(left_cone_x)
                cones_left_y.append(left_cone_y)
                
                cones_right_x.append(right_cone_x)
                cones_right_y.append(right_cone_y)
                
                if len(cones_left_x) + len(cones_right_x) >= self.num_cones_max:
                    print(f"Max cones reached: {self.num_cones_max}")
                    break
            if len(cones_left_x) + len(cones_right_x) >= self.num_cones_max:
                break
        
        print(f"Generated {len(cones_left_x)} left cones and {len(cones_right_x)} right cones.")
        return cones_left_x, cones_left_y, cones_right_x, cones_right_y

    def plot_track(self, spline, cones_left_x, cones_left_y, cones_right_x, cones_right_y, start_point, direction_vector):
        # Interpolate the spline to get smooth track line
        t = np.linspace(0, 1, 200)
        x, y = interpolate.splev(t, spline[0])

        # Plot the track and cones
        plt.figure(figsize=(6, 12))  # Set the graph to 20 meters high and 10 meters wide
        
        # Plot start point and direction vector
        plt.scatter(start_point[0], start_point[1], c='yellow', s=100, label='Start Point')
        plt.quiver(start_point[0], start_point[1], direction_vector[0], direction_vector[1], 
                  color='yellow', scale=1.0, width=0.01, label='Initial Direction')
        
        plt.plot(x, y, 'g-', label='Track Centerline')
        plt.scatter(cones_left_x, cones_left_y, c='r', label='Left Side Cones', s=10)
        plt.scatter(cones_right_x, cones_right_y, c='b', label='Right Side Cones', s=10)
        plt.title('Random Track with Cones')
        plt.xlabel('X coordinate (meters)')
        plt.ylabel('Y coordinate (meters)')
        plt.xlim(-10, 10)  # Ensure the car stays within the 10m range left to right
        plt.ylim(0, 20)  # Y goes from bottom to top (0 to 20 meters)
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def generate_track(self):
        # Generate a random spline track with cones
        temp = self.generate_random_spline()
        if temp is None:
            return
        spline, x_points, y_points, start_point, direction_vector = temp
        print(f"Start point: {start_point}")
        print(f"Direction vector: {direction_vector}")
        
        cones_left_x, cones_left_y, cones_right_x, cones_right_y = self.generate_cones(spline)
        
        self.plot_track(spline, cones_left_x, cones_left_y, cones_right_x, cones_right_y, start_point, direction_vector)


def main():
    simulator = TrackSimulator()
    simulator.generate_track()

if __name__ == "__main__":
    main()

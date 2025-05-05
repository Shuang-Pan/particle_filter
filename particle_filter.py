import json
import numpy as np
import matplotlib.pyplot as plt

# Function to compute rotation error in degrees using axis-angle method
def rotation_error_degrees(rot1, rot2):
    delta_rotation = np.dot(np.linalg.inv(rot1), rot2)  # Relative rotation matrix
    trace = np.clip(np.trace(delta_rotation), -1.0, 3.0)  # Ensure valid range
    angle_rad = np.arccos((trace - 1) / 2)
    return np.degrees(angle_rad)

# Particle Filter Implementation
class ParticleFilter:
    def __init__(self, num_particles, noise_std_rotation, noise_std_translation):
        self.num_particles = num_particles
        self.noise_std_rotation = noise_std_rotation
        self.noise_std_translation = noise_std_translation
        self.particles = []

    def initialize(self, initial_transform):
        rotation = np.array(initial_transform['rotation_matrix'])
        translation = np.array(initial_transform['translation_vector'])
        self.particles = [
            {
                "rotation_matrix": rotation + np.random.normal(0.1, self.noise_std_rotation, (3, 3)),
                "translation_vector": translation + np.random.normal(0, self.noise_std_translation, 3),
            }
            for _ in range(self.num_particles)
        ]

    def predict(self):
        for particle in self.particles:
            particle["rotation_matrix"] += np.random.normal(2, self.noise_std_rotation, (3, 3))
            particle["translation_vector"] += np.random.normal(-3, self.noise_std_translation, 3)

    def update(self, observation):
        weights = []
        observation_rotation = np.array(observation['rotation_matrix'])
        observation_translation = np.array(observation['translation_vector'])

        for particle in self.particles:
            particle_rotation = np.array(particle['rotation_matrix'])
            particle_translation = np.array(particle['translation_vector'])

            rot_error = rotation_error_degrees(particle_rotation, observation_rotation)
            trans_error = np.linalg.norm(particle_translation - observation_translation)

            weight = np.exp(-rot_error / 10) * np.exp(-trans_error / 5)
            weights.append(weight)

        weights = np.array(weights)
        weights /= np.sum(weights)

        # Resample particles
        resampled_indices = np.random.choice(len(self.particles), size=self.num_particles, p=weights)
        self.particles = [self.particles[i] for i in resampled_indices]

    def estimate(self):
        avg_rotation = np.mean([p["rotation_matrix"] for p in self.particles], axis=0)
        avg_translation = np.mean([p["translation_vector"] for p in self.particles], axis=0)
        return {
            "rotation_matrix": avg_rotation.tolist(),
            "translation_vector": avg_translation.tolist(),
        }

# Load JSON files
with open('true_transforms_98.json', 'r') as f:
    true_transforms = json.load(f)

with open('errored_transforms_2_R_only.json', 'r') as f:
    errored_transforms = json.load(f)

with open('estimated_transforms_98.json', 'r') as f:
    estimated_transforms = json.load(f)

# Particle filter parameters
num_particles = 100
noise_std_rotation = 0.05  # Small noise for rotation matrix
noise_std_translation = 1  # Small noise for translation vector

filtered_transforms = {}
rotation_errors_estimated = []
translation_errors_estimated = []
rotation_errors_errored = []
translation_errors_errored = []
rotation_errors_filtered = []
translation_errors_filtered = []

# Apply particle filter
for key in errored_transforms:
    # Calculate errors for estimated transforms
    estimated_transform = estimated_transforms[key]
    true_transform = true_transforms[key]
    rotation_errors_estimated.append(rotation_error_degrees(
        np.array(estimated_transform['rotation_matrix']),
        np.array(true_transform['rotation_matrix'])
    ))
    translation_errors_estimated.append(np.linalg.norm(
        np.array(estimated_transform['translation_vector']) -
        np.array(true_transform['translation_vector'])
    ))

    # Calculate errors for errored transforms
    errored_transform = errored_transforms[key]
    rotation_errors_errored.append(rotation_error_degrees(
        np.array(errored_transform['rotation_matrix']),
        np.array(true_transform['rotation_matrix'])
    ))
    translation_errors_errored.append(np.linalg.norm(
        np.array(errored_transform['translation_vector']) -
        np.array(true_transform['translation_vector'])
    ))

    # Particle filter
    pf = ParticleFilter(num_particles, noise_std_rotation, noise_std_translation)
    pf.initialize(errored_transform)
    pf.predict()
    pf.update(estimated_transform)
    filtered_transform = pf.estimate()
    filtered_transforms[key] = filtered_transform

    # Calculate errors for filtered transforms
    rotation_errors_filtered.append(rotation_error_degrees(
        np.array(filtered_transform['rotation_matrix']),
        np.array(true_transform['rotation_matrix'])
    ))
    translation_errors_filtered.append(np.linalg.norm(
        np.array(filtered_transform['translation_vector']) -
        np.array(true_transform['translation_vector'])
    ))

# Save filtered transforms
with open('filtered_transforms.json', 'w') as f:
    json.dump(filtered_transforms, f, indent=4)

# # Plot rotation and translation errors
# plt.figure(figsize=(12, 6))

# # Plot rotation errors
# plt.subplot(1, 2, 1)
# plt.plot(rotation_errors_estimated, label='Estimated Transforms', marker='o')
# plt.plot(rotation_errors_errored, label='Errored Transforms', marker='x')
# plt.plot(rotation_errors_filtered, label='Filtered Transforms', marker='^')
# plt.title('Rotation Errors (Degrees)')
# plt.xlabel('Frame Index')
# plt.ylabel('Rotation Error (Degrees)')
# plt.legend()

# # Plot translation errors
# plt.subplot(1, 2, 2)
# plt.plot(translation_errors_estimated, label='Estimated Transforms', marker='o')
# plt.plot(translation_errors_errored, label='Errored Transforms', marker='x')
# plt.plot(translation_errors_filtered, label='Filtered Transforms', marker='^')
# plt.title('Translation Errors')
# plt.xlabel('Frame Index')
# plt.ylabel('Translation Error')
# plt.legend()

# plt.tight_layout()
# plt.show()
# Generate the time array from frame index
frame_indices = np.arange(len(rotation_errors_estimated))
time_seconds = frame_indices / 2.0  # Time in seconds is half the frame index

# Plot rotation and translation errors
plt.figure(figsize=(12, 6))

# Plot rotation errors
plt.subplot(1, 2, 1)
plt.plot(time_seconds, rotation_errors_estimated, label='Transforms from computer vision', marker='o')
plt.plot(time_seconds, rotation_errors_errored, label='Transforms from forward kinematics', marker='x')
plt.plot(time_seconds, rotation_errors_filtered, label='Transforms from particle filter', marker='^')
plt.title('Rotation Errors (Degrees)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Rotation Error (Degrees)')
plt.legend()

# Plot translation errors
plt.subplot(1, 2, 2)
plt.plot(time_seconds, translation_errors_estimated, label='Transforms from computer vision', marker='o')
plt.plot(time_seconds, translation_errors_errored, label='Transforms from forward kinematics', marker='x')
plt.plot(time_seconds, translation_errors_filtered, label='Transforms from particle filter', marker='^')
plt.title('Translation Errors')
plt.xlabel('Time (Seconds)')
plt.ylabel('Translation Error (mm)')
plt.legend()

plt.tight_layout()
plt.show()

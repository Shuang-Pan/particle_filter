import json
import numpy as np
import random

def add_rotation_error(rotation_matrix, error_degrees):
    # Convert rotation matrix to axis-angle representation
    axis = np.array([random.uniform(-1, 1) for _ in range(3)])
    axis /= np.linalg.norm(axis)  # Normalize the axis
    angle_rad = np.radians(error_degrees)
    
    # Compute the skew-symmetric matrix for the axis
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    # Compute the rotation error matrix using Rodrigues' formula
    rotation_error = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * np.dot(K, K)
    
    # Apply the rotation error
    errored_rotation = np.dot(rotation_matrix, rotation_error)
    return errored_rotation

def add_translation_error(translation_vector, error_magnitude):
    # Add random translation error
    direction = np.array([1, 1, 1])/np.linalg.norm([1, 1, 1])
    error_vector = direction * error_magnitude
    errored_translation = translation_vector + error_vector
    return errored_translation

# Load the true transforms
with open('true_transforms_98.json', 'r') as f:
    true_transforms = json.load(f)

# Add errors to create new errored transforms
errored_transforms = {}
rotation_error_degrees = 15 
translation_error_magnitude = 0 # 7

for key, transform in true_transforms.items():
    true_rotation = np.array(transform['rotation_matrix'])
    true_translation = np.array(transform['translation_vector'])
    
    # Add errors
    errored_rotation = add_rotation_error(true_rotation, rotation_error_degrees+np.random.uniform(5))
    errored_translation = add_translation_error(true_translation, 0) #3
    
    # Store the errored transform
    errored_transforms[key] = {
        'rotation_matrix': errored_rotation.tolist(),
        'translation_vector': errored_translation.tolist()
    }

# Save the errored transforms to a new JSON file
with open('errored_transforms_2_R_only.json', 'w') as f:
    json.dump(errored_transforms, f, indent=4)

print("Errored transforms file created as 'errored_transforms.json'")

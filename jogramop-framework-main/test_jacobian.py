import pybullet as p
import numpy as np

def get_jacobian(robot_id, end_effector_link_id, arm_joint_ids):
    num_dof = len(arm_joint_ids)  # Should be 7 for the arm

    # Get the current joint positions, velocities, and accelerations
    joint_states = p.getJointStates(robot_id, arm_joint_ids)
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [0.0] * num_dof  # Assuming zero velocities
    joint_accelerations = [0.0] * num_dof  # Assuming zero accelerations

    # Ensure local position is a 3-element array (usually [0, 0, 0] if calculating at the end effector)
    local_position = [0.0, 0.0, 0.0]

    # Check all input sizes
    assert len(joint_positions) == num_dof, "Joint positions length does not match numDof"
    assert len(joint_velocities) == num_dof, "Joint velocities length does not match numDof"
    assert len(joint_accelerations) == num_dof, "Joint accelerations length does not match numDof"
    assert len(local_position) == 3, "Local position must be a 3-element vector"

    # Calculate the Jacobian
    jac_t, jac_r = p.calculateJacobian(
        robot_id,
        end_effector_link_id,
        local_position,
        joint_positions,
        joint_velocities,
        joint_accelerations
    )

    return np.array(jac_t), np.array(jac_r)

# Example usage
robot_id = 1  # Assuming you have already connected and loaded a robot model
end_effector_link_id = 11  # Example link ID for end effector
arm_joint_ids = [0, 1, 2, 3, 4, 5, 6]  # Replace with your actual arm joint IDs

jac_t, jac_r = get_jacobian(robot_id, end_effector_link_id, arm_joint_ids)
print("Translational Jacobian:", jac_t)
print("Rotational Jacobian:", jac_r)
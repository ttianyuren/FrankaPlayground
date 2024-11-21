import pybullet as p
import pybullet_data
import numpy as np
import math
import random
import time

# Object parameter lists
sphere_params = []
cylinder_params = []
cuboid_params = []
plane_params = []  # For planes


# Function to perform IK and return residue
def calculate_inverse_kinematics(
    robot_id,
    end_effector_link_index,
    target_pos,
    target_ori=None,
    threshold=0.01,
    max_iters=100,
):
    # If orientation is not provided, use the current orientation
    if target_ori is None:
        link_state = p.getLinkState(robot_id, end_effector_link_index)
        target_ori = link_state[1]  # Extract current orientation (quaternion)

    # Calculate IK solution (joint positions)
    joint_angles = p.calculateInverseKinematics(
        robot_id, end_effector_link_index, target_pos, target_ori
    )

    # Set the robot joints to the computed IK solution

    p.resetJointState(robot_id, [0, 1, 2, 3, 4, 5, 6], joint_angles, targetVelocity=0)

    # Get the actual end effector position after applying IK
    actual_state = p.getLinkState(robot_id, end_effector_link_index)
    actual_pos = actual_state[4]  # Position of the end effector

    # Calculate residue (distance between target and actual positions)
    residue = np.linalg.norm(np.array(target_pos) - np.array(actual_pos))

    # Check if the solution is valid (residue below threshold)
    solution_found = residue <= threshold

    return joint_angles, residue, solution_found


# Generate a random pose with position and orientation
def random_pose(x_range=[-0.8, 0.8], y_range=[-0.8, 0.8], z_range=[0.0, 1.8]):
    position = [
        random.uniform(x_range[0], x_range[1]),
        random.uniform(y_range[0], y_range[1]),
        random.uniform(z_range[0], z_range[1]),
    ]
    orientation = p.getQuaternionFromEuler(
        [
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
        ]
    )
    return position, orientation


# Generate a random size within a range
def random_size(min_size, max_size):
    return random.uniform(min_size, max_size)


# Apply damping to an object
def apply_damping(object_id, linear_damping=0.9, angular_damping=0.9):
    p.changeDynamics(
        object_id, -1, linearDamping=linear_damping, angularDamping=angular_damping
    )


# Create and return a random sphere object
def create_random_sphere(min_size=0.05, max_size=0.5):
    radius = random_size(min_size, max_size)
    position, orientation = random_pose()
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual_shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1]
    )
    sphere_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )
    apply_damping(sphere_id)
    sphere_params.append({"id": sphere_id, "radius": radius})


# Create and return a random cuboid object
def create_random_cuboid(min_size=0.05, max_size=0.5):
    half_extents = [
        random_size(min_size, max_size) / 2,
        random_size(min_size, max_size) / 2,
        random_size(min_size, max_size) / 2,
    ]
    position, orientation = random_pose()
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0, 1, 0, 1]
    )
    cuboid_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )
    apply_damping(cuboid_id)
    cuboid_params.append({"id": cuboid_id, "dimensions": half_extents})


# Create and return a random cylinder object
def create_random_cylinder(min_size=0.05, max_size=0.5):
    radius = random_size(min_size, max_size) / 2
    height = random_size(min_size * 2, max_size * 3)
    position, orientation = random_pose()
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height
    )
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0, 0, 1, 1]
    )
    cylinder_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation,
    )
    apply_damping(cylinder_id)
    cylinder_params.append({"id": cylinder_id, "radius": radius, "height": height})


# Create the simulation environment
def create_scene():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Create a plane (z=0)
    plane_id = p.loadURDF("plane.urdf")
    plane_params.append({"plane_eq": [0, 0, 1, 0]})
    print(f"Plane: Equation={plane_params[0]['plane_eq']}")

    # Load a robot model (Franka Panda)
    urdf_file = "franka_panda/panda.urdf"
    base_position = (-0.4, 0, -0.0)
    robot_id = p.loadURDF(urdf_file, basePosition=base_position, useFixedBase=True)

    # Set initial joint positions for the robot
    initial_joint_positions = [
        -0.41,
        0.71,
        -0.00,
        -1.12,
        1.95,
        1.33,
        -1.33,
        0.0,
        0.0,
        0.01,
        0.01,
        0.0,
    ]
    for i in range(len(initial_joint_positions)):
        p.resetJointState(robot_id, i, initial_joint_positions[i])

    # Create random objects in the environment
    for _ in range(3):
        create_random_sphere(0.05, 0.3)

    for _ in range(3):
        create_random_cuboid(0.05, 0.5)

    for _ in range(3):
        create_random_cylinder(0.05, 0.5)

    # Start the simulation loop
    start_time = time.time()
    time_limit = 200
    enable_state_printing = 1
    enable_robot_motion = 0

    A = 0.4  # Amplitude
    omega = 0.8  # Frequency

    # Initialize previous parameters to compare
    prev_sphere_params = {}
    prev_cuboid_params = {}
    prev_cylinder_params = {}
    prev_joint_positions = ()

    while True:
        p.stepSimulation()

        if enable_robot_motion:
            for joint_index in range(len(initial_joint_positions)):
                # Compute the sine wave target for the joint
                target_position = initial_joint_positions[joint_index] + A * np.sin(
                    omega * (time.time() - start_time)
                )

                # Apply position control to the joint
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_position,
                    force=500,
                )

        if enable_state_printing:
            scene_change = 0
            # Update and print spheres if parameters change
            for sphere in sphere_params:
                position, _ = p.getBasePositionAndOrientation(sphere["id"])
                position = [round(coord, 3) for coord in position]

                if (
                    sphere["id"] not in prev_sphere_params
                    or prev_sphere_params[sphere["id"]] != position
                ):
                    print(
                        f"Sphere (ID={sphere['id']}): Center={position}, Radius={round(sphere['radius'], 3)}"
                    )
                    prev_sphere_params[sphere["id"]] = position
                    scene_change = 1

            # Update and print cuboids if parameters change
            for cuboid in cuboid_params:
                position, orientation = p.getBasePositionAndOrientation(cuboid["id"])
                position = [round(coord, 3) for coord in position]
                orientation = [round(ori, 3) for ori in orientation]
                dimensions = [round(i, 3) for i in cuboid["dimensions"]]

                if cuboid["id"] not in prev_cuboid_params or prev_cuboid_params[
                    cuboid["id"]
                ] != (position, orientation):
                    print(
                        f"Cuboid (ID={cuboid['id']}): Dimensions={dimensions}, Position={position}, Orientation={orientation}"
                    )
                    prev_cuboid_params[cuboid["id"]] = (position, orientation)
                    scene_change = 1

            # Update and print cylinders if parameters change
            for cylinder in cylinder_params:
                position, orientation = p.getBasePositionAndOrientation(cylinder["id"])
                position = [round(coord, 3) for coord in position]
                orientation = [round(ori, 3) for ori in orientation]

                if cylinder["id"] not in prev_cylinder_params or prev_cylinder_params[
                    cylinder["id"]
                ] != (position, orientation):
                    print(
                        f"Cylinder (ID={cylinder['id']}): Radius={round(cylinder['radius'], 3)}, Height={round(cylinder['height'], 3)}, Position={position}, Orientation={orientation}"
                    )
                    prev_cylinder_params[cylinder["id"]] = (position, orientation)
                    scene_change = 1

            if not scene_change:
                # Collect robot joint positions (in degrees) as a tuple
                num_joints = p.getNumJoints(robot_id)
                joint_positions_deg = tuple(
                    round(math.degrees(p.getJointState(robot_id, joint_index)[0]), 1)
                    for joint_index in range(num_joints)
                )

                # Print all joint positions if they have changed
                if joint_positions_deg != prev_joint_positions:
                    print(f"Robot Joints: {joint_positions_deg}")
                    prev_joint_positions = joint_positions_deg

        # Check simulation time limit
        if time.time() - start_time > time_limit:
            break

    p.disconnect()


# Example usage
if __name__ == "__main__":
    create_scene()

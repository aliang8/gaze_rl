import ss
import gymnasium
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for headless rendering
import matplotlib.pyplot as plt
import os

# Create the environment with render_mode="none" instead of "headless"
# "none" will still create the offscreen renderer but not the viewer
env = gymnasium.make("ss/FrankaLiftEnv-v0", render_mode="none")

# Reset the environment
observation, info = env.reset(seed=42)

# Lists to store trajectory data
eef_positions = []
joint_positions = []
gripper_positions = []
camera_views = {
    'front_view': [],
    'top_view': [],
    'left_view': [],
    'right_view': []
}

# Run a simple loop (no control)
num_steps = 100
for step in range(num_steps):
    action = (np.zeros(6), 0)  # (position control, gripper control)
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Store trajectory data
    eef_positions.append(observation["eef_pos"].copy())
    joint_positions.append(observation["joint_pos"].copy())
    gripper_positions.append(observation["gripper_pos"].copy())
    
    # Store camera views
    for view_name in camera_views.keys():
        if view_name in observation:
            camera_views[view_name].append(observation[view_name].copy())
    
    # Optional: print some information
    if step % 10 == 0:
        print(f"Step {step}/{num_steps}")
        print(f"End effector position: {observation['eef_pos']}")

# Close the environment
env.close()

# Convert lists to numpy arrays for easier manipulation
eef_positions = np.array(eef_positions)
joint_positions = np.array(joint_positions)
gripper_positions = np.array(gripper_positions)

# Create directory for saving frames
os.makedirs("trajectory_frames", exist_ok=True)

# Create a single plot with the whole trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(eef_positions[:, 0], eef_positions[:, 1], eef_positions[:, 2], 'b-', linewidth=2)
ax.scatter(eef_positions[0, 0], eef_positions[0, 1], eef_positions[0, 2], c='g', s=100, label='Start')
ax.scatter(eef_positions[-1, 0], eef_positions[-1, 1], eef_positions[-1, 2], c='r', s=100, label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Franka End Effector Complete Trajectory')
ax.legend()
plt.savefig("trajectory_full.png", dpi=150)
plt.close(fig)

print("Full trajectory plot saved to 'trajectory_full.png'")

# Check if camera views were captured
have_camera_views = any(len(views) > 0 for views in camera_views.values())

# Debug information about camera views
print(f"Camera views available: {have_camera_views}")
for name, views in camera_views.items():
    if views:
        print(f"  {name}: {len(views)} frames, shape: {views[0].shape}")

# Save frames with both trajectory and camera views
print("Saving trajectory frames with camera views...")

# Create frames for every few steps to keep the number of frames manageable
frame_step = max(1, num_steps // 50)  # About 50 frames total
for i in range(0, num_steps, frame_step):
    # Create a figure with multiple subplots
    if have_camera_views:
        fig = plt.figure(figsize=(18, 10))
        # First subplot for 3D trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    else:
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory up to the current point
    ax1.plot(eef_positions[:i+1, 0], eef_positions[:i+1, 1], eef_positions[:i+1, 2], 'b-', linewidth=2)
    
    # Plot the current end effector position
    ax1.scatter(eef_positions[i, 0], eef_positions[i, 1], eef_positions[i, 2], c='r', s=100)
    
    # Add start position
    ax1.scatter(eef_positions[0, 0], eef_positions[0, 1], eef_positions[0, 2], c='g', s=100)
    
    # Set labels and title
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'End Effector Trajectory (Step {i+1}/{num_steps})')
    
    # Set consistent view limits
    ax1.set_xlim([min(eef_positions[:, 0]) - 0.1, max(eef_positions[:, 0]) + 0.1])
    ax1.set_ylim([min(eef_positions[:, 1]) - 0.1, max(eef_positions[:, 1]) + 0.1])
    ax1.set_zlim([min(eef_positions[:, 2]) - 0.1, max(eef_positions[:, 2]) + 0.1])
    
    # Add camera views if available
    if have_camera_views:
        view_positions = {
            'front_view': (2, 3, 2),
            'top_view': (2, 3, 3),
            'left_view': (2, 3, 4),
            'right_view': (2, 3, 5)
        }
        
        for view_name, subplot_pos in view_positions.items():
            if view_name in camera_views and len(camera_views[view_name]) > i:
                ax = fig.add_subplot(*subplot_pos)
                ax.imshow(camera_views[view_name][i])
                ax.set_title(f'{view_name.replace("_", " ").title()}')
                ax.axis('off')  # Hide axes
    
    # Add joint positions as a subplot
    if have_camera_views:
        ax_joints = fig.add_subplot(2, 3, 6)
    else:
        # If no camera views, add below the trajectory
        ax_joints = fig.add_subplot(212)
    
    # Plot joint positions
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
    ax_joints.bar(joint_names, joint_positions[i])
    ax_joints.set_title(f'Joint Positions (Step {i+1})')
    ax_joints.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"trajectory_frames/frame_{i:04d}.png", dpi=100)
    plt.close(fig)
    
    if i % 10 == 0:
        print(f"Rendered frame {i+1}/{num_steps}")

print("\nTip: You can create a video from the frames using ffmpeg with this command:")
print("ffmpeg -framerate 10 -pattern_type glob -i 'trajectory_frames/frame_*.png' -c:v libx264 -pix_fmt yuv420p franka_trajectory.mp4")

# Create a simple plot of the end effector position over time
plt.figure(figsize=(12, 8))

# Plot each coordinate over time
plt.subplot(3, 1, 1)
plt.plot(range(num_steps), eef_positions[:, 0], 'r-')
plt.ylabel('X Position')
plt.title('End Effector Position Over Time')

plt.subplot(3, 1, 2)
plt.plot(range(num_steps), eef_positions[:, 1], 'g-')
plt.ylabel('Y Position')

plt.subplot(3, 1, 3)
plt.plot(range(num_steps), eef_positions[:, 2], 'b-')
plt.ylabel('Z Position')
plt.xlabel('Time Step')

plt.tight_layout()
plt.savefig("position_over_time.png", dpi=150)
plt.close()

print("Position over time plot saved to 'position_over_time.png'")
print("Individual trajectory frames with views saved in 'trajectory_frames/' directory")
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import plot_basis
from pytransform3d.transformations import plot_transform
from scipy.spatial.transform import Rotation as R

def visualize_robot_policy_sequence(translations, quaternions, save_gif=False,
                                    gif_filename='robot_policy_visualization.gif'):
    """
    Visualize a sequence of translations and quaternions in 3D space.

    Args:
    - translations: A sequence (list or array) of 3D positions (Nx3 array).
    - quaternions: A sequence (list or array) of quaternions (Nx4 array).
    - save_gif: Boolean to save the visualization as a GIF (default: False).
    - gif_filename: Filename for the saved GIF (default: 'robot_policy_visualization.gif').

    Returns:
    - None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(translations)):
        translation = translations[i]
        quaternion = quaternions[i]

        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Clear the previous plot
        ax.cla()

        # Visualize the translation (position)
        ax.scatter(translation[0], translation[1], translation[2], color='r', label='Position')

        # Visualize the quaternion (orientation)
        plot_transform(ax=ax, A2B=np.eye(4), s=0.5)
        plot_basis(ax=ax, R=rotation_matrix, p=translation, s=0.5, alpha=0.6)

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.draw()
        plt.pause(0.5)  # Pause for visualization

    # Optionally save the sequence as a GIF
    if save_gif:
        import imageio
        images = []
        for i in range(len(translations)):
            translation = translations[i]
            quaternion = quaternions[i]
            rotation_matrix = R.from_quat(quaternion).as_matrix()

            # Clear the previous plot
            ax.cla()

            # Visualize the translation (position)
            ax.scatter(translation[0], translation[1], translation[2], color='r', label='Position')

            # Visualize the quaternion (orientation)
            plot_transform(ax=ax, A2B=np.eye(4), s=0.5)
            plot_basis(ax=ax, R=rotation_matrix, p=translation, s=0.5, alpha=0.6)

            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            # Save each frame
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)

        imageio.mimsave(gif_filename, images, fps=2)

    plt.show()

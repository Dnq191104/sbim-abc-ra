import torch
import numpy as np

# Try importing specific functions
try:
    from sbibm.tasks import get_task, get_available_tasks
    from sbibm.metrics.c2st import c2st
    print("Successfully imported sbibm functions")
except ImportError as e:
    print(f"Import error: {e}")
    # Try alternative imports
    try:
        import sbibm
        get_task = sbibm.tasks.get_task
        get_available_tasks = sbibm.tasks.get_available_tasks
        c2st = sbibm.metrics.c2st.c2st
        print("Alternative import successful")
    except Exception as e2:
        print(f"Alternative import failed: {e2}")
        exit(1)


def probe_task(task_name: str, num_observation: int = 1):
    """
    Probe a task to understand its structure, shapes, and data types.
    This helps us design the ABC RA implementation correctly.
    """
    print(f"\n=== Probing task: {task_name} ===")

    task = get_task(task_name)

    # Task metadata
    print("Task metadata:")
    print(f"  dim_data: {task.dim_data}")
    print(f"  dim_parameters: {task.dim_parameters}")
    print(f"  num_observations: {task.num_observations}")
    print(f"  name: {task.name}")

    # Prior
    print("\nPrior:")
    prior = task.get_prior()
    prior_dist = task.get_prior_dist()

    # Sample from prior
    theta_samples = prior(num_samples=5)
    print(f"  theta shape: {theta_samples.shape}")
    print(f"  theta dtype: {theta_samples.dtype}")
    print(f"  theta sample[0]: {theta_samples[0]}")

    # Simulator
    print("\nSimulator:")
    simulator = task.get_simulator()
    x_samples = simulator(theta_samples)

    print(f"  x shape: {x_samples.shape}")
    print(f"  x dtype: {x_samples.dtype}")

    # Check if x is already flattened or structured
    if len(x_samples.shape) == 2:
        print(f"  x appears to be: [batch_size, feature_dim] = {x_samples.shape}")
        print(f"  x[0]: {x_samples[0]}")
    elif len(x_samples.shape) == 1:
        print(f"  x appears to be: 1D vector, length = {x_samples.shape[0]}")
        print(f"  x[0]: {x_samples[0]}")
    else:
        print(f"  x has shape {x_samples.shape} - structured data!")
        print(f"  x[0]: {x_samples[0]}")

    # Test flattening
    x_flat = x_samples[0].flatten()
    print(f"  x[0] flattened: shape {x_flat.shape}, values {x_flat}")

    # Observation
    print("\nObservation:")
    x_obs = task.get_observation(num_observation)
    print(f"  x_obs shape: {x_obs.shape}")
    print(f"  x_obs dtype: {x_obs.dtype}")
    print(f"  x_obs: {x_obs}")

    # Reference posterior
    print("\nReference posterior:")
    theta_ref = task.get_reference_posterior_samples(num_observation)
    print(f"  theta_ref shape: {theta_ref.shape}")
    print(f"  theta_ref dtype: {theta_ref.dtype}")
    print(f"  theta_ref mean: {theta_ref.mean(dim=0)}")
    print(f"  theta_ref std: {theta_ref.std(dim=0)}")

    # Transforms
    print("\nTransforms:")
    transforms = task._get_transforms(automatic_transforms_enabled=False)["parameters"]
    print(f"  transforms type: {type(transforms)}")
    if hasattr(transforms, 'inv'):
        print("  transforms has inv method")

    # Test transforms
    theta_test = theta_samples[0:1]
    print(f"  original theta: {theta_test}")
    if transforms is not None:
        theta_transformed = transforms(theta_test)
        print(f"  transformed theta: {theta_transformed}")
        theta_inv = transforms.inv(theta_transformed)
        print(f"  inverse transformed: {theta_inv}")

    return {
        'task': task,
        'theta_shape': theta_samples.shape[1:],
        'x_shape': x_samples.shape[1:],
        'x_obs_shape': x_obs.shape,
        'transforms': transforms
    }


def test_multiple_tasks():
    """Test probe on different task types"""
    tasks_to_test = [
        "gaussian_linear",  # Simple
        "two_moons",        # Multimodal
        "slcp",            # Complex posterior
    ]

    results = {}
    for task_name in tasks_to_test:
        try:
            results[task_name] = probe_task(task_name, num_observation=1)
        except Exception as e:
            print(f"Error probing {task_name}: {e}")

    return results


if __name__ == "__main__":
    # Test one task in detail
    probe_task("gaussian_linear", 1)

    # Test multiple tasks
    print("\n" + "="*50)
    print("TESTING MULTIPLE TASKS")
    test_multiple_tasks()

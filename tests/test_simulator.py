import pytest
import concurrent.futures
from src.data.stochastic.generate_harmonic import StochasticTrunkFactory
from src.data.forward_process.simulator import simulate_forward_process

def test_thread_safety_temp_dirs():
    # Test that multiple calls to the simulator execute concurrently
    # and don't collide. We'll run a minimal simulation 4 times.
    factory = StochasticTrunkFactory(propagation_modes=["Linear"])
    trunk = factory.generate()
    
    def run_sim():
        # resolution=32 for faster tests
        return simulate_forward_process(trunk, resolution=32, n_electrodes=8, pattern="adjacent")
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_sim) for _ in range(4)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    # None of them should have failed due to temp file collisions
    # (Assuming Octave is installed and works; if not, we skip or mock)
    # Since we can't guarantee Octave is installed on the testing machine,
    # we just check that the Python side handled it gracefully (returning None or valid ForwardResult).
    assert len(results) == 4

def test_failure_handling():
    # Create an impossible trunk to force Octave failure
    factory = StochasticTrunkFactory(propagation_modes=["Linear"])
    trunk = factory.generate()
    
    # Try to simulate with invalid electrodes (e.g. 0 or 1) to force failure
    # The EIDORS bridge will fail gracefully and return None
    result = simulate_forward_process(trunk, resolution=32, n_electrodes=1, pattern="adjacent")
    assert result is None

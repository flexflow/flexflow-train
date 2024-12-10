import numpy as np

def generate_poisson_arrivals(n, arrival_rate):
    """
    Generate arrival times for n requests following a Poisson process.
    
    Parameters:
    n (int): Number of requests to generate
    arrival_rate (float): Average arrival rate (requests per second)
    
    Returns:
    numpy.ndarray: Array of arrival times in seconds
    """
    # Generate n exponentially distributed inter-arrival times
    # For a Poisson process, inter-arrival times follow exponential distribution
    inter_arrival_times = np.random.exponential(scale=1/arrival_rate, size=n)
    
    # Calculate cumulative sum to get arrival times
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Round to 6 decimal places for practical purposes (microsecond precision)
    arrival_times = np.round(arrival_times, decimals=6)
    
    return arrival_times

# Example usage
if __name__ == "__main__":
    # Example: Generate 10 arrivals with rate of 2 requests per second
    n_requests = 250
    rate = 0.1  # requests per second
    
    arrivals = generate_poisson_arrivals(n_requests, rate)
    
    print("Arrival times (seconds):")
    for i, time in enumerate(arrivals, 1):
        print(f"Request {i}: {time:.6f} s")
    
    # Calculate average inter-arrival time to verify
    avg_inter_arrival = np.diff(arrivals).mean()
    print(f"\nAverage inter-arrival time: {avg_inter_arrival:.6f} s")
    print(f"Expected inter-arrival time: {1/rate:.6f} s")
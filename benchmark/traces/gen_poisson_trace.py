import numpy as np
import random

random.seed(0)
np.random.seed(0)

def generate_poisson_events(rate, time_duration):
    # calculates the total number of events using a Poisson distribution
    num_events = np.random.poisson(rate * time_duration)
    # generates inter-arrival times between events using an exponential distribution with a mean of 1.0 / rate
    inter_arrival_times = np.random.exponential(1.0 / rate, num_events)
    # cumulatively summed to obtain the event times, resulting in a sequence of event times that follows a Poisson process
    event_times = np.cumsum(inter_arrival_times)
    return num_events, event_times, inter_arrival_times

duration = 30 # seconds
rate = 7 # requests per second

num_events, event_times, inter_arrival_times = generate_poisson_events(rate, duration)
print(num_events)

# for item in event_times:
#     print("{:.3f}".format(item))
# print(inter_arrival_times)
with open("poisson_trace_duration{}_rate{}.txt".format(duration, rate), "w") as f:
    f.write("Number of Events: {}, Duration: {}, Rate: {}\n".format(num_events, duration, rate))
    f.write("==========\n")
    for item in event_times:
        f.write("{:.3f}\n".format(item))
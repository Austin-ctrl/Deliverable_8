# training will generate
#(state, action, reward, next_state, done)

# REPLAY BUFFER
from collections import deque
import random

class replay_buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self,
            state,
            action,
            reward,
            next_state,
            done):

        # append
        self.buffer.append((state,
                            action,
                            reward,
                            next_state,
                            done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):

        return len(self.buffer)
    
    
if __name__ == "__main__":

    buffer = replay_buffer(1000)

    buffer.add(
        "state1",
        3,
        1.0,
        "state2",
        False
    )

    print(len(buffer))

    batch = buffer.sample(1)

    print(batch)
    
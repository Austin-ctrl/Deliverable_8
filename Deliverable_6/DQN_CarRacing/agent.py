from model import DQN
import torch
import random
import torch.optim as optim

class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
            
        # dqn paramters
        self.n_actions = 5

        # exploration
        self.epsilon = 1.0  # we are at 100 % rando

        self.gamma = 0.99  # discount factor for bellman

        # networks
        self.online_net = DQN(self.n_actions).to(self.device)
        self.target_net = DQN(self.n_actions).to(self.device)

        # copy the target weights into the target netork
        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )
        
        
        # we need an optimizer to update the network. basically, if a loss is computed, we back propagote, then the optimizer will update the weigtts.
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=1e-4
        )
        self.loss_fn = torch.nn.MSELoss()  # mean squared error loss function
        
    # Things to note are that online_net learns every target_net stays frozen.
    # Every 1000 steps, we'll copy the online net to the target net and refresht eh target
    def update_target_net(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

    def select_action(self, state):
        # exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # exploitation, using the learned q values
        with torch.no_grad():
            q_values = self.online_net(state.to(self.device))
            action = torch.argmax(q_values, dim=1).item()  # best actions among the 5 actions
            return action
        
    def decay_epsilon(self):
        # decay epsilon
        self.epsilon = max(0.1, self.epsilon - 0.00009)  # linear decay
    def train_step(self,
                replay_buffer,
                batch_size):

        batch = replay_buffer.sample(
            batch_size
        )
        # unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        current_q = self.online_net(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        )
        # future q values
        with torch.no_grad():

            next_q = self.target_net(next_states)

            next_q = next_q.max(dim=1)[0]

        # This the bellman
        targets = rewards + (
            self.gamma *
            next_q *
            (1 - dones)
        )
        loss = self.loss_fn(
            current_q.squeeze(),
            targets
        )
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":

    agent = Agent()

    state = torch.randn(
        1,
        4,
        84,
        84
    )

    action = agent.select_action(state)

    print("Selected Action:", action)

# Action mappings
# 0 = do nothing
# 1 = steer right
# 2 = steer left
# 3 = gas
# 4 = brake









        
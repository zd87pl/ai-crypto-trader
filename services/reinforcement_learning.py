"""
Reinforcement Learning for Trading Strategy Optimization

This module implements a reinforcement learning framework for trading strategy optimization.
It uses Q-learning with function approximation to learn optimal trading actions based on
market conditions, technical indicators, and social sentiment data.
"""

import os
import json
import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from datetime import datetime
from collections import deque

# Configure logging
logger = logging.getLogger("reinforcement_learning")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/reinforcement_learning.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [RL] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TradingRLAgent:
    """
    Reinforcement Learning agent for trading strategy optimization.
    Uses Q-learning with function approximation to learn optimal trading policies.
    """
    
    def __init__(self, 
                 state_features: List[str],
                 actions: List[str] = ["BUY", "HOLD", "SELL"],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 update_target_every: int = 100,
                 random_seed: Optional[int] = None):
        """
        Initialize the RL agent for trading.
        
        Args:
            state_features: List of feature names that define the state
            actions: List of possible trading actions
            learning_rate: Learning rate for neural network updates
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate at which epsilon decays over time
            memory_size: Size of replay memory
            batch_size: Number of samples to use for learning in each update
            update_target_every: Number of steps between target network updates
            random_seed: Random seed for reproducibility
        """
        self.state_features = state_features
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize step counter
        self.step_counter = 0
        
        # Initialize training metrics
        self.training_history = []
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"Reinforcement Learning Agent initialized with:")
        logger.info(f"- State features: {state_features}")
        logger.info(f"- Actions: {actions}")
        logger.info(f"- Learning rate: {learning_rate}")
        logger.info(f"- Gamma: {gamma}")
        logger.info(f"- Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
        logger.info(f"- Memory size: {memory_size}")
        logger.info(f"- Batch size: {batch_size}")
        logger.info(f"- Update target every: {update_target_every} steps")
    
    def _initialize_models(self) -> None:
        """
        Initialize the neural network models for Q-learning.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam
            
            # Hide TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            # Create main Q-network
            self.model = Sequential()
            self.model.add(Dense(24, input_dim=len(self.state_features), activation='relu'))
            self.model.add(Dense(24, activation='relu'))
            self.model.add(Dense(len(self.actions), activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            # Create target Q-network
            self.target_model = Sequential()
            self.target_model.add(Dense(24, input_dim=len(self.state_features), activation='relu'))
            self.target_model.add(Dense(24, activation='relu'))
            self.target_model.add(Dense(len(self.actions), activation='linear'))
            self.target_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            # Initialize target model with same weights as main model
            self.target_model.set_weights(self.model.get_weights())
            
            self.tf_available = True
            logger.info("TensorFlow models initialized successfully.")
            
        except ImportError:
            logger.warning("TensorFlow not available. Using NumPy-based models instead.")
            self.tf_available = False
            
            # Number of neurons in each layer
            input_size = len(self.state_features)
            hidden_size = 24
            output_size = len(self.actions)
            
            # Initialize weights and biases with random values
            self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
            self.bias1 = np.zeros(hidden_size)
            self.weights2 = np.random.randn(hidden_size, hidden_size) * 0.1
            self.bias2 = np.zeros(hidden_size)
            self.weights3 = np.random.randn(hidden_size, output_size) * 0.1
            self.bias3 = np.zeros(output_size)
            
            # Target network weights and biases
            self.target_weights1 = self.weights1.copy()
            self.target_bias1 = self.bias1.copy()
            self.target_weights2 = self.weights2.copy()
            self.target_bias2 = self.bias2.copy()
            self.target_weights3 = self.weights3.copy()
            self.target_bias3 = self.bias3.copy()
    
    def _numpy_predict(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """
        Predict Q-values using NumPy-based model.
        
        Args:
            state: The state vector
            use_target: Whether to use the target network
            
        Returns:
            Q-values for each action
        """
        if use_target:
            # Use target network
            weights1, bias1 = self.target_weights1, self.target_bias1
            weights2, bias2 = self.target_weights2, self.target_bias2
            weights3, bias3 = self.target_weights3, self.target_bias3
        else:
            # Use main network
            weights1, bias1 = self.weights1, self.bias1
            weights2, bias2 = self.weights2, self.bias2
            weights3, bias3 = self.weights3, self.bias3
        
        # Forward pass through the network
        hidden1 = np.maximum(0, np.dot(state, weights1) + bias1)  # ReLU
        hidden2 = np.maximum(0, np.dot(hidden1, weights2) + bias2)  # ReLU
        output = np.dot(hidden2, weights3) + bias3  # Linear
        
        return output
    
    def _numpy_train(self, states: np.ndarray, targets: np.ndarray) -> float:
        """
        Train the NumPy-based model on a batch of data.
        
        Args:
            states: Batch of state vectors
            targets: Target Q-values
            
        Returns:
            Loss value
        """
        # Learning rate
        alpha = self.learning_rate
        
        # Store original outputs for loss calculation
        original_outputs = np.array([self._numpy_predict(state) for state in states])
        
        # Train on each sample
        for i, state in enumerate(states):
            # Forward pass
            hidden1 = np.maximum(0, np.dot(state, self.weights1) + self.bias1)
            hidden2 = np.maximum(0, np.dot(hidden1, self.weights2) + self.bias2)
            output = np.dot(hidden2, self.weights3) + self.bias3
            
            # Compute gradients
            d_output = output - targets[i]
            
            # Backpropagation for output layer
            d_weights3 = np.outer(hidden2, d_output)
            d_bias3 = d_output
            
            # Backpropagation for second hidden layer
            d_hidden2 = np.dot(d_output, self.weights3.T)
            d_hidden2[hidden2 <= 0] = 0  # ReLU derivative
            d_weights2 = np.outer(hidden1, d_hidden2)
            d_bias2 = d_hidden2
            
            # Backpropagation for first hidden layer
            d_hidden1 = np.dot(d_hidden2, self.weights2.T)
            d_hidden1[hidden1 <= 0] = 0  # ReLU derivative
            d_weights1 = np.outer(state, d_hidden1)
            d_bias1 = d_hidden1
            
            # Update weights and biases
            self.weights3 -= alpha * d_weights3
            self.bias3 -= alpha * d_bias3
            self.weights2 -= alpha * d_weights2
            self.bias2 -= alpha * d_bias2
            self.weights1 -= alpha * d_weights1
            self.bias1 -= alpha * d_bias1
        
        # Calculate mean squared error loss
        new_outputs = np.array([self._numpy_predict(state) for state in states])
        loss = np.mean((new_outputs - targets) ** 2)
        
        return loss
    
    def _update_target_network(self) -> None:
        """
        Update the target network with weights from the main network.
        """
        if self.tf_available:
            self.target_model.set_weights(self.model.get_weights())
        else:
            # Copy weights and biases from main to target network
            self.target_weights1 = self.weights1.copy()
            self.target_bias1 = self.bias1.copy()
            self.target_weights2 = self.weights2.copy()
            self.target_bias2 = self.bias2.copy()
            self.target_weights3 = self.weights3.copy()
            self.target_bias3 = self.bias3.copy()
    
    def preprocess_state(self, state_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert a state dictionary to a numpy array.
        
        Args:
            state_dict: Dictionary of state features and values
            
        Returns:
            Numpy array representing the state
        """
        # Extract features in the correct order
        state_array = np.array([state_dict.get(feature, 0.0) for feature in self.state_features])
        
        # Ensure correct shape for single state
        return state_array.reshape(1, -1)
    
    def preprocess_batch_states(self, state_dicts: List[Dict[str, float]]) -> np.ndarray:
        """
        Convert a list of state dictionaries to a numpy array.
        
        Args:
            state_dicts: List of state feature dictionaries
            
        Returns:
            Numpy array representing the batch of states
        """
        batch_states = np.zeros((len(state_dicts), len(self.state_features)))
        
        for i, state_dict in enumerate(state_dicts):
            for j, feature in enumerate(self.state_features):
                batch_states[i, j] = state_dict.get(feature, 0.0)
        
        return batch_states
    
    def act(self, state_dict: Dict[str, float], training: bool = True) -> str:
        """
        Choose an action based on the current state.
        
        Args:
            state_dict: Dictionary of state features and values
            training: Whether the agent is in training mode (use epsilon-greedy)
            
        Returns:
            Selected action as a string
        """
        # Preprocess state
        state = self.preprocess_state(state_dict)
        
        # Epsilon-greedy policy during training
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(len(self.actions))
            return self.actions[action_idx]
        
        # Choose action with highest Q-value
        if self.tf_available:
            q_values = self.model.predict(state, verbose=0)[0]
        else:
            q_values = self._numpy_predict(state[0])
        
        action_idx = np.argmax(q_values)
        return self.actions[action_idx]
    
    def remember(self, state: Dict[str, float], action: str, reward: float, 
                 next_state: Dict[str, float], done: bool) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        action_idx = self.actions.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self) -> Dict[str, float]:
        """
        Train the agent on a batch of experiences from memory.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < self.batch_size:
            return {"loss": None, "mean_q": None, "epsilon": self.epsilon}
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract batch components
        states = [experience[0] for experience in minibatch]
        actions = [experience[1] for experience in minibatch]
        rewards = [experience[2] for experience in minibatch]
        next_states = [experience[3] for experience in minibatch]
        dones = [experience[4] for experience in minibatch]
        
        # Preprocess states
        batch_states = self.preprocess_batch_states(states)
        batch_next_states = self.preprocess_batch_states(next_states)
        
        if self.tf_available:
            # Get current Q values from main network
            current_q = self.model.predict(batch_states, verbose=0)
            
            # Get next Q values from target network
            next_q = self.target_model.predict(batch_next_states, verbose=0)
            
            # Update target Q values
            target_q = current_q.copy()
            for i in range(self.batch_size):
                if dones[i]:
                    target_q[i, actions[i]] = rewards[i]
                else:
                    target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
            
            # Train the model
            history = self.model.fit(batch_states, target_q, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            
        else:
            # NumPy implementation
            # Get current Q values
            current_q = np.array([self._numpy_predict(state) for state in batch_states])
            
            # Get next Q values from target network
            next_q = np.array([self._numpy_predict(state, use_target=True) for state in batch_next_states])
            
            # Update target Q values
            target_q = current_q.copy()
            for i in range(self.batch_size):
                if dones[i]:
                    target_q[i, actions[i]] = rewards[i]
                else:
                    target_q[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
            
            # Train the model
            loss = self._numpy_train(batch_states, target_q)
        
        # Update counter and target network if needed
        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self._update_target_network()
            logger.info(f"Updated target network at step {self.step_counter}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Calculate mean Q value for metrics
        mean_q = np.mean([np.max(q) for q in current_q])
        
        # Record training metrics
        metrics = {
            "loss": loss,
            "mean_q": float(mean_q),
            "epsilon": self.epsilon,
            "step": self.step_counter
        }
        self.training_history.append(metrics)
        
        return metrics
    
    def train(self, env, episodes: int = 1000, max_steps: int = 500, 
              render_every: int = 100, save_path: Optional[str] = None) -> Dict:
        """
        Train the agent on an environment.
        
        Args:
            env: Training environment with reset(), step() and render() methods
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            render_every: How often to render the environment
            save_path: Path to save the model
            
        Returns:
            Dictionary of training results
        """
        rewards_history = []
        
        for episode in range(1, episodes + 1):
            # Reset environment
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            # Track episode metrics
            episode_actions = {action: 0 for action in self.actions}
            
            while not done and steps < max_steps:
                # Choose action
                action = self.act(state)
                episode_actions[action] += 1
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Remember experience
                self.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.memory) >= self.batch_size:
                    metrics = self.replay()
                    
                    # Log training progress periodically
                    if steps % 100 == 0:
                        logger.info(f"Episode {episode}/{episodes}, Step {steps}: "
                                   f"Loss = {metrics['loss']:.4f}, Mean Q = {metrics['mean_q']:.4f}, "
                                   f"Epsilon = {metrics['epsilon']:.4f}")
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Calculate action distribution
            action_distribution = {action: count/steps for action, count in episode_actions.items()}
            
            # Record episode results
            rewards_history.append(total_reward)
            logger.info(f"Episode {episode}/{episodes} finished after {steps} steps. "
                       f"Total reward: {total_reward:.2f}, Average reward: {np.mean(rewards_history[-100:]):.2f}")
            logger.info(f"Action distribution: {action_distribution}")
            
            # Render environment periodically
            if render_every > 0 and episode % render_every == 0:
                if hasattr(env, 'render'):
                    env.render()
            
            # Save model periodically
            if save_path and episode % 100 == 0:
                self.save(f"{save_path}_episode_{episode}")
        
        # Final save
        if save_path:
            self.save(save_path)
        
        # Return training results
        results = {
            "rewards_history": rewards_history,
            "average_reward": float(np.mean(rewards_history[-100:])),
            "training_history": self.training_history,
            "final_epsilon": self.epsilon
        }
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save the agent's models and parameters.
        
        Args:
            path: Path to save the model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save parameters
        params = {
            "state_features": self.state_features,
            "actions": self.actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "update_target_every": self.update_target_every,
            "step_counter": self.step_counter,
            "training_timestamp": datetime.now().isoformat()
        }
        
        with open(f"{path}_params.json", 'w') as f:
            json.dump(params, f, indent=2)
        
        # Save models
        if self.tf_available:
            self.model.save(f"{path}_model")
            logger.info(f"Model saved to {path}_model")
        else:
            # Save NumPy weights
            np.savez(f"{path}_weights.npz", 
                    weights1=self.weights1, bias1=self.bias1,
                    weights2=self.weights2, bias2=self.bias2,
                    weights3=self.weights3, bias3=self.bias3,
                    target_weights1=self.target_weights1, target_bias1=self.target_bias1,
                    target_weights2=self.target_weights2, target_bias2=self.target_bias2,
                    target_weights3=self.target_weights3, target_bias3=self.target_bias3)
            logger.info(f"Weights saved to {path}_weights.npz")
    
    def load(self, path: str) -> None:
        """
        Load the agent's models and parameters.
        
        Args:
            path: Path to load the model from
        """
        # Load parameters
        try:
            with open(f"{path}_params.json", 'r') as f:
                params = json.load(f)
            
            self.state_features = params["state_features"]
            self.actions = params["actions"]
            self.learning_rate = params["learning_rate"]
            self.gamma = params["gamma"]
            self.epsilon = params["epsilon"]
            self.epsilon_end = params["epsilon_end"]
            self.epsilon_decay = params["epsilon_decay"]
            self.memory_size = params["memory_size"]
            self.batch_size = params["batch_size"]
            self.update_target_every = params["update_target_every"]
            self.step_counter = params["step_counter"]
            
            logger.info(f"Loaded parameters from {path}_params.json")
            
            # Load models
            if self.tf_available:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(f"{path}_model")
                self.target_model = tf.keras.models.load_model(f"{path}_model")
                logger.info(f"Loaded TensorFlow model from {path}_model")
            else:
                # Load NumPy weights
                weights = np.load(f"{path}_weights.npz")
                self.weights1 = weights['weights1']
                self.bias1 = weights['bias1']
                self.weights2 = weights['weights2']
                self.bias2 = weights['bias2']
                self.weights3 = weights['weights3']
                self.bias3 = weights['bias3']
                self.target_weights1 = weights['target_weights1']
                self.target_bias1 = weights['target_bias1']
                self.target_weights2 = weights['target_weights2']
                self.target_bias2 = weights['target_bias2']
                self.target_weights3 = weights['target_weights3']
                self.target_bias3 = weights['target_bias3']
                logger.info(f"Loaded NumPy weights from {path}_weights.npz")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_q_values(self, state_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Get Q-values for all actions for a given state.
        
        Args:
            state_dict: Dictionary of state features and values
            
        Returns:
            Dictionary mapping actions to Q-values
        """
        # Preprocess state
        state = self.preprocess_state(state_dict)
        
        # Get Q-values
        if self.tf_available:
            q_values = self.model.predict(state, verbose=0)[0]
        else:
            q_values = self._numpy_predict(state[0])
        
        # Map to action names
        return {action: float(q_values[i]) for i, action in enumerate(self.actions)}
    
    def get_training_history(self) -> List[Dict]:
        """
        Get the agent's training history.
        
        Returns:
            List of training metrics dictionaries
        """
        return self.training_history
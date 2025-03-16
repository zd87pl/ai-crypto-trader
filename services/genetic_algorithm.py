"""
Genetic Algorithm for Trading Strategy Optimization

This module implements a genetic algorithm framework for optimizing trading strategies.
It evolves a population of strategy parameter sets, using mutation, crossover,
and selection operations to improve trading performance over generations.
"""

import os
import json
import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
from datetime import datetime
from copy import deepcopy

# Configure logging
logger = logging.getLogger("genetic_algorithm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/genetic_algorithm.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [GeneticAlgorithm] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class GeneticAlgorithm:
    """
    A genetic algorithm implementation for optimizing trading strategies.
    """
    
    def __init__(self, param_ranges: Dict[str, Tuple], 
                 fitness_function: Callable, 
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 elitism_pct: float = 0.1,
                 tournament_size: int = 3,
                 random_seed: Optional[int] = None):
        """
        Initialize the genetic algorithm.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) tuples
            fitness_function: Function that evaluates a parameter set and returns a fitness score
            population_size: Number of individuals in the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover between individuals
            elitism_pct: Percentage of top individuals to preserve unchanged
            tournament_size: Number of individuals in tournament selection
            random_seed: Seed for random number generator
        """
        self.param_ranges = param_ranges
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_pct = elitism_pct
        self.tournament_size = tournament_size
        
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation_history = []
        
        logger.info(f"Genetic Algorithm initialized with:")
        logger.info(f"- Population size: {population_size}")
        logger.info(f"- Generations: {generations}")
        logger.info(f"- Mutation rate: {mutation_rate}")
        logger.info(f"- Crossover rate: {crossover_rate}")
        logger.info(f"- Elitism percentage: {elitism_pct}")
        logger.info(f"- Parameters: {list(param_ranges.keys())}")
    
    def initialize_population(self, seeded_individuals: List[Dict] = None) -> None:
        """
        Initialize the population with random individuals and optionally include seeded individuals.
        
        Args:
            seeded_individuals: List of predefined parameter sets to include in the initial population
        """
        self.population = []
        
        # Add seeded individuals if provided
        if seeded_individuals:
            for individual in seeded_individuals:
                # Ensure all parameters are within bounds
                bounded_individual = {}
                for param, value in individual.items():
                    if param in self.param_ranges:
                        min_val, max_val = self.param_ranges[param]
                        bounded_individual[param] = max(min_val, min(max_val, value))
                    else:
                        bounded_individual[param] = value
                self.population.append(bounded_individual)
        
        # Generate random individuals for the rest of the population
        remaining = self.population_size - len(self.population)
        for _ in range(remaining):
            individual = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    individual[param] = random.randint(min_val, max_val)
                else:
                    individual[param] = random.uniform(min_val, max_val)
            self.population.append(individual)
        
        logger.info(f"Population initialized with {len(self.population)} individuals")
        logger.info(f"Seeded individuals: {len(seeded_individuals) if seeded_individuals else 0}")
    
    def evaluate_population(self) -> None:
        """
        Evaluate the fitness of all individuals in the population.
        """
        self.fitness_scores = []
        for individual in self.population:
            fitness = self.fitness_function(individual)
            self.fitness_scores.append(fitness)
            
            # Track best individual
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = deepcopy(individual)
        
        logger.info(f"Population evaluated. Best fitness: {self.best_fitness}")
    
    def selection(self) -> List[Dict]:
        """
        Select individuals for reproduction using tournament selection.
        
        Returns:
            List of selected individuals
        """
        selected = []
        
        # Elitism: Preserve top individuals
        elites_count = max(1, int(self.elitism_pct * self.population_size))
        indices = sorted(range(len(self.fitness_scores)), 
                         key=lambda i: self.fitness_scores[i], 
                         reverse=True)
        
        for i in range(elites_count):
            selected.append(deepcopy(self.population[indices[i]]))
        
        # Tournament selection for the rest
        while len(selected) < self.population_size:
            tournament_indices = random.sample(range(len(self.population)), 
                                             self.tournament_size)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(deepcopy(self.population[winner_idx]))
        
        return selected
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Perform crossover between two parents to create two children.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two child individuals
        """
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1 = {}
        child2 = {}
        
        # Choose random crossover points for each parameter
        for param in self.param_ranges:
            if random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def mutation(self, individual: Dict) -> Dict:
        """
        Mutate an individual by randomly changing parameters.
        
        Args:
            individual: The individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = deepcopy(individual)
        
        for param, (min_val, max_val) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                # Apply different mutation strategies based on parameter type
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter: adjust by small discrete steps
                    step = max(1, int((max_val - min_val) * 0.1))  # 10% of range
                    delta = random.choice([-step, step])
                    mutated[param] = max(min_val, min(max_val, mutated[param] + delta))
                else:
                    # Float parameter: adjust by percentage or small random change
                    if random.random() < 0.5:
                        # Percentage adjustment (±20%)
                        factor = random.uniform(0.8, 1.2)
                        mutated[param] = max(min_val, min(max_val, mutated[param] * factor))
                    else:
                        # Random value from parameter range
                        range_size = max_val - min_val
                        delta = random.uniform(-0.1 * range_size, 0.1 * range_size)
                        mutated[param] = max(min_val, min(max_val, mutated[param] + delta))
        
        return mutated
    
    def evolve_generation(self) -> None:
        """
        Evolve the population by one generation through selection, crossover, and mutation.
        """
        # Select individuals for reproduction
        selected = self.selection()
        
        # Create new generation through crossover and mutation
        new_population = []
        
        # Preserve elite individuals
        elites_count = max(1, int(self.elitism_pct * self.population_size))
        new_population.extend(selected[:elites_count])
        
        # Crossover and mutation for the rest
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
    
    def run(self, seeded_individuals: List[Dict] = None) -> Dict:
        """
        Run the genetic algorithm for the specified number of generations.
        
        Args:
            seeded_individuals: List of predefined parameter sets to include in the initial population
            
        Returns:
            The best individual found
        """
        # Initialize and evaluate initial population
        self.initialize_population(seeded_individuals)
        self.evaluate_population()
        
        # Store initial generation data
        self.record_generation(0)
        
        # Evolve for specified number of generations
        for generation in range(1, self.generations + 1):
            logger.info(f"Evolving generation {generation}/{self.generations}")
            
            # Evolve population
            self.evolve_generation()
            
            # Evaluate new population
            self.evaluate_population()
            
            # Record generation data
            self.record_generation(generation)
            
            # Log progress
            avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
            logger.info(f"Generation {generation}: Best fitness = {self.best_fitness}, Avg fitness = {avg_fitness:.4f}")
        
        logger.info(f"Genetic algorithm completed. Best individual: {self.best_individual}")
        logger.info(f"Best fitness: {self.best_fitness}")
        
        return self.best_individual
    
    def record_generation(self, generation: int) -> None:
        """
        Record data for the current generation.
        
        Args:
            generation: The current generation number
        """
        # Calculate statistics
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        min_fitness = min(self.fitness_scores)
        max_fitness = max(self.fitness_scores)
        
        # Find current best individual
        best_idx = self.fitness_scores.index(max_fitness)
        current_best = deepcopy(self.population[best_idx])
        
        # Store generation data
        generation_data = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'best_fitness': max_fitness,
            'avg_fitness': avg_fitness,
            'min_fitness': min_fitness,
            'best_individual': current_best,
            'diversity': self.calculate_diversity()
        }
        
        self.generation_history.append(generation_data)
    
    def calculate_diversity(self) -> float:
        """
        Calculate the diversity of the current population.
        
        Returns:
            A score between 0-1 representing population diversity
        """
        if not self.population or len(self.population) < 2:
            return 0.0
        
        # Calculate normalized parameter distances
        param_distances = []
        
        for param, (min_val, max_val) in self.param_ranges.items():
            param_range = max_val - min_val
            if param_range == 0:
                continue
                
            values = [ind[param] for ind in self.population]
            normalized_values = [(v - min_val) / param_range for v in values]
            
            # Calculate variance of the parameter
            variance = np.var(normalized_values)
            param_distances.append(variance)
        
        # Average variance across all parameters
        return float(np.mean(param_distances))
    
    def get_generation_history(self) -> List[Dict]:
        """
        Get the history of all generations.
        
        Returns:
            List of generation data dictionaries
        """
        return self.generation_history
    
    def get_best_individual(self) -> Dict:
        """
        Get the best individual found.
        
        Returns:
            Dictionary of the best parameter set
        """
        return deepcopy(self.best_individual)
    
    def get_population_diversity(self) -> Dict:
        """
        Get diversity metrics for the current population.
        
        Returns:
            Dictionary of diversity metrics
        """
        diversity = {}
        
        # Overall diversity score
        diversity['overall'] = self.calculate_diversity()
        
        # Parameter-specific diversity
        diversity['parameters'] = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            param_range = max_val - min_val
            if param_range == 0:
                diversity['parameters'][param] = 0.0
                continue
                
            values = [ind[param] for ind in self.population]
            normalized_values = [(v - min_val) / param_range for v in values]
            variance = float(np.var(normalized_values))
            diversity['parameters'][param] = variance
        
        return diversity
import numpy as np
import matplotlib.pyplot as plt

# Параметри алгоритму
POPULATION_SIZE = 1000  # Розмір популяції
SELECTION_RATE = 0.2   # Коефіцієнт вибору
CROSSOVER_RATE = 0.7   # Коефіцієнт кросоверу
MUTATION_RATE = 0.1    # Коефіцієнт мутації
GENERATIONS = 40       # Кількість поколінь
SEARCH_SPACE = [-512, 512]  # Область пошуку
# SEARCH_SPACE = [-32.768, 32.768]  # Область пошуку
EXPERIMENTS = 2    # Кількість експериментів

# Функція Еклі (Ackley function)
# def ackley_function(x, y):
#     a, b, c = 20, 0.2, 2 * np.pi
#     part1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
#     part2 = -np.exp(0.5 * (np.cos(c*x) + np.cos(c*y)))
#     return part1 + part2 + a + np.exp(1)

def ackley_function(x, y):
    """
    Функція Egg Holder, яку потрібно мінімізувати.
    """
    return -(y + 47) * np.sin(np.sqrt(abs(y + (x / 2) + 47))) - x * np.sin(np.sqrt(abs(x - (y + 47))))

# Ініціалізація популяції (рівномірний розподіл)
def initialize_population(size):
    grid_size = int(np.sqrt(size))
    x_vals = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], grid_size)
    y_vals = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], grid_size)
    xv, yv = np.meshgrid(x_vals, y_vals)
    population = np.column_stack([xv.ravel(), yv.ravel()])
    
    return population[:size]

# Оцінка пристосованості (фітнес-функція)
def fitness(population):
    return 1 / (1 + np.apply_along_axis(lambda ind: ackley_function(ind[0], ind[1]), 1, population))

# Відбір найкращих
def selection(population, fitness_values):
    num_selected = int(SELECTION_RATE * len(population))
    indices = np.argsort(-fitness_values)[:num_selected]
    return population[indices]

# Кросовер (рівномірний)
def crossover(parents):
    num_offspring = POPULATION_SIZE - len(parents)
    offspring = []
    for _ in range(num_offspring):
        p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
        mask = np.random.rand(2) < 0.5
        child = np.where(mask, p1, p2)
        offspring.append(child)
    return np.array(offspring)

# Мутація (додавання випадкового вектора довжини 1)
def mutation(population):
    mutation_mask = np.random.rand(len(population)) < MUTATION_RATE
    mutation_direction = np.random.uniform(-1, 1, (len(population), 2))
    mutation_direction /= np.linalg.norm(mutation_direction, axis=1, keepdims=True)
    population[mutation_mask] += mutation_direction[mutation_mask]
    return np.clip(population, SEARCH_SPACE[0], SEARCH_SPACE[1])

# Основний алгоритм
def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    for generation in range(GENERATIONS):
        fit_values = fitness(population)
        parents = selection(population, fit_values)
        offspring = crossover(parents)
        population = np.vstack((parents, offspring))
        population = mutation(population)
    best_index = np.argmin([ackley_function(ind[0], ind[1]) for ind in population])
    return population[best_index], ackley_function(population[best_index][0], population[best_index][1])

# Запуск експериментів
best_solutions = []
for _ in range(EXPERIMENTS):
    best_solution, best_value = genetic_algorithm()
    best_solutions.append(best_value)

# Вивід результатів
best_found = min(best_solutions)
print(f"Найменше знайдене значення: {best_found}")

import numpy as np
import matplotlib.pyplot as plt

# Параметри алгоритму
POPULATION_SIZE = 1000   # Розмір популяції
SELECTION_RATE = 0.2   # Коефіцієнт вибору
CROSSOVER_RATE = 0.7   # Коефіцієнт кросоверу
MUTATION_RATE = 0.1    # Коефіцієнт мутації
GENERATIONS = 40       # Кількість поколінь
SEARCH_SPACE = [-512, 512]  # Область пошуку
EXPERIMENTS = 1   # Кількість експериментів

# Функція Egg Holder
def ackley_function(x, y):
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
    values = np.apply_along_axis(lambda ind: ackley_function(ind[0], ind[1]), 1, population)
    # для мінімізації: зміщуємо значення так, щоб вони були позитивні
    shift = abs(values.min()) + 1e-6
    return 1 / (values + shift)

# Відбір найкращих
def selection(population, fitness_values):
    num_selected = int(SELECTION_RATE * len(population))
    indices = np.argsort(-fitness_values)[:num_selected]
    return population[indices]

# Кросовер (BLX-α)
def blend_crossover(parent1, parent2, alpha=0.5):
    child1, child2 = [], []
    for x1, x2 in zip(parent1, parent2):
        d = abs(x2 - x1)
        low = min(x1, x2) - alpha * d
        high = max(x1, x2) + alpha * d
        child1.append(np.random.uniform(low, high))
        child2.append(np.random.uniform(low, high))
    return np.array(child1), np.array(child2)

# Обрізання особин у межах пошуку
def clip_individual(ind, bounds):
    return np.array([np.clip(val, low, high) for val, (low, high) in zip(ind, bounds)])

# Мутація (випадковий вектор довжини 1)
def mutation(population):
    mutation_mask = np.random.rand(len(population)) < MUTATION_RATE
    mutation_direction = np.random.uniform(-1, 1, (len(population), 2))
    mutation_direction /= np.linalg.norm(mutation_direction, axis=1, keepdims=True)
    population[mutation_mask] += mutation_direction[mutation_mask]
    return np.clip(population, SEARCH_SPACE[0], SEARCH_SPACE[1])

# Основний алгоритм з візуалізацією
def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    plt.figure(figsize=(8, 6))
    for generation in range(GENERATIONS):
        plt.clf()
        plt.title(f'Покоління {generation + 1}')
        plt.xlim(SEARCH_SPACE[0], SEARCH_SPACE[1])
        plt.ylim(SEARCH_SPACE[0], SEARCH_SPACE[1])
        plt.scatter(population[:,0], population[:,1], s=5, color='blue', alpha=0.4)
        plt.pause(0.1)

        fit_values = fitness(population)
        parents = selection(population, fit_values)
        offspring = []
        for _ in range((POPULATION_SIZE - len(parents)) // 2):
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
            child1, child2 = blend_crossover(p1, p2)
            offspring.append(child1)
            offspring.append(child2)
        population = np.vstack((parents, offspring))
        population = mutation(population)

    plt.ioff()
    plt.show()

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

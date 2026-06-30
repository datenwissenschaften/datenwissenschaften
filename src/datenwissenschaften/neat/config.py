from pathlib import Path


def write_neat_config(
    path: Path,
    *,
    num_inputs: int,
    num_outputs: int,
    pop_size: int,
) -> None:
    if num_inputs < 1:
        raise ValueError("num_inputs must be positive.")
    if num_outputs < 1:
        raise ValueError("num_outputs must be positive.")
    if pop_size < 128:
        raise ValueError("pop_size must be at least 128.")

    effective_pop_size = max(128, min(pop_size, 1024))

    if num_inputs <= 256:
        initial_connection = "full_direct"
        num_hidden = 8
    elif num_inputs <= 512:
        initial_connection = "partial_direct 0.25"
        num_hidden = 8
    else:
        initial_connection = "partial_direct 0.05"
        num_hidden = 0

    elitism = max(2, min(5, effective_pop_size // 60))
    species_elitism = max(1, min(2, elitism // 2))
    survival_threshold = 0.2

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        f"""[NEAT]
fitness_criterion      = max
fitness_threshold      = 10000
pop_size               = {effective_pop_size}
reset_on_extinction    = False
no_fitness_termination = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.1
bias_mutate_rate        = 0.5
bias_replace_rate       = 0.05

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.10
conn_delete_prob        = 0.005

enabled_default         = True
enabled_mutate_rate     = 0.005

feed_forward            = True
initial_connection      = {initial_connection}

node_add_prob           = 0.02
node_delete_prob        = 0.0

num_hidden              = {num_hidden}
num_inputs              = {num_inputs}
num_outputs             = {num_outputs}

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.0
weight_min_value        = -30.0
weight_mutate_power     = 0.1
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.05

[DefaultSpeciesSet]
compatibility_threshold = 2.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 75
species_elitism      = {species_elitism}

[DefaultReproduction]
elitism            = {elitism}
survival_threshold = {survival_threshold}
""",
        encoding="utf-8",
    )

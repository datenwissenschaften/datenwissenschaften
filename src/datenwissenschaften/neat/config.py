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
    if pop_size < 1:
        raise ValueError("pop_size must be positive.")

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        f"""[NEAT]
fitness_criterion      = max
fitness_threshold      = 10000
pop_size               = {min(pop_size, 1024)}
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
bias_mutate_power       = 0.2
bias_mutate_rate        = 0.5
bias_replace_rate       = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.15
conn_delete_prob        = 0.02

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

node_add_prob           = 0.03
node_delete_prob        = 0.01

num_hidden              = 16
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
weight_mutate_power     = 0.2
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 2.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 50
species_elitism      = 2

[DefaultReproduction]
elitism            = 10
survival_threshold = 0.3
""",
        encoding="utf-8",
    )

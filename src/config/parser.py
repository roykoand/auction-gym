import json
import numpy as np
from copy import deepcopy


def parse_config(path):
    with open(path) as f:
        config = json.load(f)

    # Set up Random Number Generator
    rng = np.random.default_rng(config["random_seed"])
    np.random.seed(config["random_seed"])

    # Number of runs
    num_runs = config.get("num_runs", 1)

    # Max number of slots in every auction round
    # Multi-slot is currently not fully supported.
    max_slots = 1

    # Technical parameters for distribution of latent embeddings
    embedding_size = config["embedding_size"]
    embedding_variance = config["embedding_variance"]
    obs_embedding_size = config["obs_embedding_size"]

    # Expand agent-config if there are multiple copies
    agent_configs = []
    num_agents = 0
    for agent_config in config["agents"]:
        if "num_copies" in agent_config:
            for _ in range(1, agent_config["num_copies"] + 1):
                agent_config_copy = deepcopy(agent_config)
                agent_config_copy["name"] += f" {num_agents + 1}"
                agent_configs.append(agent_config_copy)
                num_agents += 1
        else:
            agent_configs.append(agent_config)
            num_agents += 1

    # First sample item catalog (so it is consistent over different configs with the same seed)
    # Agent : (item_embedding, item_value)
    agents2items = {
        agent_config["name"]: rng.normal(
            0.0, embedding_variance, size=(agent_config["num_items"], embedding_size)
        )
        for agent_config in agent_configs
    }

    agents2item_values = {
        agent_config["name"]: rng.lognormal(0.1, 0.2, agent_config["num_items"])
        for agent_config in agent_configs
    }

    # Add intercepts to embeddings (Uniformly in [-4.5, -1.5], this gives nicer distributions for P(click))
    for agent, items in agents2items.items():
        agents2items[agent] = np.hstack(
            (items, -3.0 - 1.0 * rng.random((items.shape[0], 1)))
        )
    return (
        rng,
        config,
        agent_configs,
        agents2items,
        agents2item_values,
        num_runs,
        max_slots,
        embedding_size,
        embedding_variance,
        obs_embedding_size,
    )

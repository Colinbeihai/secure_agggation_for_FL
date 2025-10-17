def fed_avg(updates):
    total_samples = sum([u['num_samples'] for u in updates])
    avg_weights = {}
    for k in updates[0]['weights'].keys():
        avg_weights[k] = sum(u['weights'][k] * u['num_samples'] / total_samples for u in updates)
    return avg_weights

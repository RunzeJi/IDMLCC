import flwr as fl
import csv, cfg

NUM_ROUNDS = cfg.SERVER_ROUNDS
NUM_WORKERS = 1
FILENAME = cfg.ACC_FILE_NAME

aggregated_acc_list = []

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / sum(examples)
    aggregated_acc_list.append(aggregated_accuracy)
    return {"accuracy": aggregated_accuracy}

fl.server.start_server(server_address='0.0.0.0:8080',
                       config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
                       strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
                                                          fraction_fit=1.0,
                                                          min_fit_clients=NUM_WORKERS,
                                                          min_evaluate_clients=NUM_WORKERS,
                                                          min_available_clients=NUM_WORKERS,
                                                          ),)

with open(FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['iteration', 'accuracy'])

    for i, number in enumerate(aggregated_acc_list, start=1):
        writer.writerow([i, number])

print(aggregated_acc_list)

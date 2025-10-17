import torch
import random
import time
import os
import csv
import concurrent.futures
from server.server import Server
from client.client_manager import ClientManager
from utils.config_parser import load_config
from utils.logger import Logger

if __name__ == "__main__":
    # -------------------- 1. 初始化配置 --------------------
    config = load_config("config.yaml")
    logger = Logger("results/logs/train.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    server = Server(config, logger, device)
    clients = ClientManager(config, logger, device)

    drop_prob = config.get("dropout_prob", 0.1)               # 每轮客户端掉线概率
    min_clients = config.get("min_clients_to_aggregate", 7)   # 最少客户端聚合数
    max_workers = config.get("max_parallel_clients", len(clients.clients))  # 并行线程数

    # 协议与基准配置
    secure_mode = config.get("secure_aggregation", "none").lower()
    bench_cfg = config.get("benchmark", {"enable": False, "csv_path": "results/logs/bench.csv"})
    bench_enable = bool(bench_cfg.get("enable", False))
    bench_csv = bench_cfg.get("csv_path", "results/logs/bench.csv")

    def write_bench_row(row):
        if not bench_enable:
            return
        os.makedirs(os.path.dirname(bench_csv), exist_ok=True)
        file_exists = os.path.exists(bench_csv)
        write_header = True
        if file_exists:
            try:
                write_header = os.path.getsize(bench_csv) == 0
            except OSError:
                write_header = True
        headers = [
            "round",
            "protocol",
            "device",
            "num_online_clients",
            "train_time_s",
            "agg_time_s",
            "accuracy",
            "aggregated",
        ]
        with open(bench_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                w.writeheader()
            w.writerow(row)

    logger.log("Starting Federated Learning...")

    # -------------------- 2. 训练轮次 --------------------
    for round in range(config["num_rounds"]):
        logger.log(f"\n=== Round {round + 1}/{config['num_rounds']} ===")

        # 分发全局模型
        global_model = server.get_global_model()
        clients.distribute_model(global_model)

        updates = []

        # -------------------- 3. 并行客户端训练 --------------------
        train_start = time.time()
        def train_client(client):
            """单个客户端训练任务"""
            if random.random() > drop_prob:
                logger.log(f"Client {client.id} training locally...")
                return client.train_local()
            else:
                logger.log(f"Client {client.id} dropped out this round.")
                return None

        # 使用线程池并行执行客户端训练
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(train_client, client) for client in clients.clients]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    # 附加 client_id，便于安全聚合识别
                    # 注意：此处的 client_id 来自于并发上下文，故在返回时不丢失
                    # 我们在分发任务时已绑定每个 client 实例
                    # 这里无法直接从 future 获取 client 对象，改为在 client.train_local 返回后由外层补充
                    # 为保证 client_id，修改 train_client 以返回 (id, payload)
                    pass
        # 由于上面的并发闭包中不便直接注入 id，这里重写并发逻辑以携带 id
        updates = []
        def run_and_tag(client):
            res = train_client(client)
            if res is None:
                return None
            res["client_id"] = client.id
            return res
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_and_tag, client) for client in clients.clients]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    updates.append(res)
        train_time = time.time() - train_start

        # -------------------- 4. 聚合判断 --------------------
        agg_time = 0.0
        aggregated = False
        if len(updates) >= min_clients:
            agg_start = time.time()
            if secure_mode == "secagg_plus":
                from protocols.secagg_plus import aggregate_secure as secure_aggregate
                online_ids = [u["client_id"] for u in updates]
                global_template = server.get_global_model()
                new_weights = secure_aggregate(round, global_template, updates, online_ids)
                server.global_model.load_state_dict({k: v.to(server.device) for k, v in new_weights.items()})
                aggregated = True
            elif secure_mode == "fastsecagg":
                from protocols.fastsecagg import aggregate_secure as secure_aggregate
                online_ids = [u["client_id"] for u in updates]
                global_template = server.get_global_model()
                new_weights = secure_aggregate(round, global_template, updates, online_ids)
                server.global_model.load_state_dict({k: v.to(server.device) for k, v in new_weights.items()})
                aggregated = True
            else:
                server.aggregate(updates)
                aggregated = True
            agg_time = time.time() - agg_start

            acc = server.evaluate()
            logger.log(f"Round {round + 1} Accuracy: {acc:.4f}")
            logger.log(f"Aggregated from {len(updates)} clients (mode={secure_mode})")

            write_bench_row({
                "round": round + 1,
                "protocol": secure_mode,
                "device": str(device),
                "num_online_clients": len(updates),
                "train_time_s": __builtins__.round(train_time, 6),
                "agg_time_s": __builtins__.round(agg_time, 6),
                "accuracy": __builtins__.round(acc, 6),
                "aggregated": True,
            })
        else:
            logger.log(f"Round {round + 1} skipped aggregation: only {len(updates)} clients available.")
            write_bench_row({
                "round": round + 1,
                "protocol": secure_mode,
                "device": str(device),
                "num_online_clients": len(updates),
                "train_time_s": __builtins__.round(train_time, 6),
                "agg_time_s": 0.0,
                "accuracy": "",
                "aggregated": False,
            })

    logger.log("Training completed.")

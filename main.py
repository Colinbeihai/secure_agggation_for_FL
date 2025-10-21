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
            "mask_sum_time_s",
            "unmask_time_s",
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
    es_cfg = config.get("early_stop", {"enable": False})
    es_enable = bool(es_cfg.get("enable", False))
    target_acc = float(es_cfg.get("target_accuracy", 1.0))
    patience = int(es_cfg.get("patience", 0))
    min_delta = float(es_cfg.get("min_delta", 0.0))

    best_acc = -1.0
    no_improve = 0
    total_train_time = 0.0
    total_agg_time = 0.0
    stop_reason = "max_rounds_reached"

    # open-ended training until early-stop or max_num_rounds
    num_rounds_cfg = config.get("num_rounds", None)
    max_rounds = int(config.get("max_num_rounds", num_rounds_cfg if num_rounds_cfg is not None else 100))
    round = 0
    while round < max_rounds:
        logger.log(f"\n=== Round {round + 1} ===")

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

        # 并发执行并为每个结果附带 client_id
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
        total_train_time += train_time

        # -------------------- 4. 聚合判断 --------------------
        agg_time = 0.0
        aggregated = False
        if len(updates) >= min_clients:
            agg_start = time.time()
            mask_sum_time = 0.0
            unmask_time = 0.0
            if secure_mode == "secagg_plus":
                from protocols.secagg_plus import aggregate_secure as secure_aggregate
                online_ids = [u["client_id"] for u in updates]
                global_template = server.get_global_model()
                new_weights, stats = secure_aggregate(round, global_template, updates, online_ids)
                mask_sum_time = float(stats.get("mask_sum_time_s", 0.0))
                unmask_time = float(stats.get("unmask_time_s", 0.0))
                server.global_model.load_state_dict({k: v.to(server.device) for k, v in new_weights.items()})
                aggregated = True
            elif secure_mode == "fastsecagg":
                from protocols.fastsecagg import aggregate_secure as secure_aggregate
                online_ids = [u["client_id"] for u in updates]
                global_template = server.get_global_model()
                new_weights, stats = secure_aggregate(round, global_template, updates, online_ids)
                mask_sum_time = float(stats.get("mask_sum_time_s", 0.0))
                unmask_time = float(stats.get("unmask_time_s", 0.0))
                server.global_model.load_state_dict({k: v.to(server.device) for k, v in new_weights.items()})
                aggregated = True
            elif secure_mode == "scsecagg":
                from protocols.scsecagg import aggregate_secure as secure_aggregate, ScSecAgg
                online_ids = [u["client_id"] for u in updates]
                global_template = server.get_global_model()
                # read config for scsecagg
                sc_cfg = config.get("scsecagg", {})
                num_servers = int(sc_cfg.get("num_servers", 5))
                read_threshold = int(sc_cfg.get("read_threshold", 3))
                storage_factor = int(sc_cfg.get("storage_factor", 2))
                instance = ScSecAgg(num_servers=num_servers, read_threshold=read_threshold, storage_factor=storage_factor)
                new_weights, stats = secure_aggregate(round, global_template, updates, online_ids, instance=instance)
                mask_sum_time = float(stats.get("mask_sum_time_s", 0.0))
                unmask_time = float(stats.get("unmask_time_s", 0.0))
                server.global_model.load_state_dict({k: v.to(server.device) for k, v in new_weights.items()})
                aggregated = True
            else:
                server.aggregate(updates)
                aggregated = True
            agg_time = time.time() - agg_start
            total_agg_time += agg_time

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
                "mask_sum_time_s": __builtins__.round(mask_sum_time, 6),
                "unmask_time_s": __builtins__.round(unmask_time, 6),
                "accuracy": __builtins__.round(acc, 6),
                "aggregated": True,
            })
            # Early stopping checks
            if es_enable:
                improved = (acc >= best_acc + min_delta)
                if improved:
                    best_acc = acc
                    no_improve = 0
                else:
                    no_improve += 1
                if acc >= target_acc:
                    logger.log(f"Target accuracy reached: {acc:.4f} >= {target_acc:.4f}. Stopping.")
                    stop_reason = "target_accuracy_reached"
                    break
                if patience > 0 and no_improve >= patience:
                    logger.log(f"Early stop: no improvement for {patience} rounds (best={best_acc:.4f}).")
                    stop_reason = "no_improvement_patience"
                    break
        else:
            logger.log(f"Round {round + 1} skipped aggregation: only {len(updates)} clients available.")
            write_bench_row({
                "round": round + 1,
                "protocol": secure_mode,
                "device": str(device),
                "num_online_clients": len(updates),
                "train_time_s": __builtins__.round(train_time, 6),
                "agg_time_s": 0.0,
                "mask_sum_time_s": 0.0,
                "unmask_time_s": 0.0,
                "accuracy": "",
                "aggregated": False,
            })
        round += 1

    logger.log(f"Training completed. reason={stop_reason}, rounds={round}, best_acc={best_acc:.4f}, total_train_time_s={total_train_time:.3f}, total_agg_time_s={total_agg_time:.3f}")

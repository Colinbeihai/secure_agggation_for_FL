# secure_aggregation_sc.py
import torch
import numpy as np
from typing import List, Optional
import time

class SecureAggregationSC:
    """
    基于阶梯码的安全聚合协议
    参考论文: "How to Read and Update Coded Distributed Storage Robustly and Optimally"
    """
    
    def __init__(self, num_servers: int = 6, recovery_threshold: int = 4, 
                 storage_factor: int = 2, model_dim: int = 1000, 
                 field_size: int = 2**31 - 1, seed: int = 42):
        """
        初始化安全聚合协议
        
        Args:
            num_servers: 服务器数量 (N)
            recovery_threshold: 恢复阈值，任意R_r个服务器可恢复消息 (R_r)
            storage_factor: 存储因子，控制存储开销 (K_c)
            model_dim: 模型维度 (L)
            field_size: 有限域大小
            seed: 随机种子
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.N = num_servers
        self.R_r = recovery_threshold
        self.K_c = storage_factor
        self.L = model_dim
        self.q = field_size
        
        # 验证参数有效性
        if not (0 < self.K_c <= self.R_r <= self.N):
            raise ValueError("参数必须满足: 0 < K_c ≤ R_r ≤ N")
        
        # 计算阶梯结构参数
        self.G = self.N - self.R_r + 1
        self.alpha = [self.N - self.R_r + self.K_c + 1 - i for i in range(1, self.G + 1)]
        self.beta = [self.N + 1 - i for i in range(1, self.G + 1)]
        
        print(f"系统参数: N={self.N}, R_r={self.R_r}, K_c={self.K_c}, L={self.L}")
        print(f"阶梯参数: G={self.G}, alpha={self.alpha}, beta={self.beta}")
        
        # 生成Cauchy矩阵
        self.C = self._generate_cauchy_matrix()
        print("Cauchy矩阵生成完成")
    
    def _generate_cauchy_matrix(self) -> torch.Tensor:
        """生成Cauchy编码矩阵"""
        # 选择不同的x和f值以确保矩阵可逆
        x_values = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15][:self.N], dtype=torch.float32)
        f_values = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16][:self.N], dtype=torch.float32)
        
        C = torch.zeros(self.N, self.N)
        for i in range(self.N):
            for j in range(self.N):
                C[i, j] = 1.0 / (x_values[i] - f_values[j])
        
        return C
    
    def _sc_generate_simple(self, vector: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """
        简化的阶梯结构生成
        实际应用中可以按照论文的精确递归结构实现
        
        Args:
            vector: 输入向量
            noise_std: 噪声标准差
        """
        M = torch.zeros(self.N, self.L)
        
        # 将向量分成G个块
        block_size = self.L // self.G
        remainder = self.L % self.G
        
        current_pos = 0
        for i in range(self.G):
            # 当前块的大小
            current_block_size = block_size + (1 if i < remainder else 0)
            
            if current_block_size == 0:
                continue
                
            # 提取当前块
            block_data = vector[current_pos:current_pos + current_block_size]
            
            # 扩展到alpha_i行 (简化处理)
            rows_to_use = min(self.alpha[i], self.N)
            
            # 在阶梯结构中放置数据
            for row in range(rows_to_use):
                end_pos = current_pos + min(current_block_size, self.L - current_pos)
                if current_pos < end_pos:
                    M[row, current_pos:end_pos] = block_data[:end_pos - current_pos]
            
            # 添加噪声到剩余行
            for row in range(rows_to_use, min(self.beta[i], self.N)):
                noise = torch.normal(0, noise_std, (current_block_size,))
                end_pos = current_pos + min(current_block_size, self.L - current_pos)
                if current_pos < end_pos:
                    M[row, current_pos:end_pos] = noise[:end_pos - current_pos]
            
            current_pos += current_block_size
        
        return M
    
    def client_prepare_update(self, model_update: torch.Tensor, 
                            client_id: int = 0, 
                            security_level: int = 1) -> torch.Tensor:
        """
        客户端准备安全更新
        
        Args:
            model_update: 模型更新梯度
            client_id: 客户端ID（用于随机种子）
            security_level: 安全级别
            
        Returns:
            encoded_update: 编码后的安全更新
        """
        if len(model_update) != self.L:
            raise ValueError(f"模型更新维度应为 {self.L}, 但得到 {len(model_update)}")
        
        # 设置客户端特定的随机种子
        torch.manual_seed(client_id + 42)
        
        # 生成阶梯结构矩阵
        M_delta = self._sc_generate_simple(model_update, noise_std=0.01)
        
        # Cauchy编码
        encoded_update = torch.matmul(self.C, M_delta)
        
        # 添加额外的客户端特定噪声以增强安全性
        client_noise = torch.normal(0, 0.001, encoded_update.shape)
        encoded_update += client_noise
        
        return encoded_update
    
    def server_aggregate_updates(self, encoded_updates: List[torch.Tensor],
                               dropout_servers: Optional[List[int]] = None) -> torch.Tensor:
        """
        服务器聚合客户端更新
        
        Args:
            encoded_updates: 客户端编码更新列表
            dropout_servers: 掉线服务器索引
            
        Returns:
            aggregated: 聚合后的编码数据
        """
        if dropout_servers is None:
            dropout_servers = []
        
        if not encoded_updates:
            raise ValueError("没有可聚合的更新")
        
        # 初始化聚合结果
        aggregated = torch.zeros_like(encoded_updates[0])
        
        # 聚合所有客户端的更新
        for i, update in enumerate(encoded_updates):
            aggregated += update
        
        print(f"聚合了 {len(encoded_updates)} 个客户端的更新")
        print(f"掉线服务器: {dropout_servers}")
        
        return aggregated
    
    def recover_global_update(self, aggregated_encoded: torch.Tensor,
                            available_servers: Optional[List[int]] = None) -> torch.Tensor:
        """
        从聚合的编码数据中恢复全局模型更新
        
        Args:
            aggregated_encoded: 聚合的编码数据
            available_servers: 可用服务器索引
            
        Returns:
            global_update: 恢复的全局模型更新
        """
        if available_servers is None:
            available_servers = list(range(self.N))
        
        if len(available_servers) < self.R_r:
            raise ValueError(f"需要至少 {self.R_r} 个服务器进行恢复，当前只有 {len(available_servers)} 个")
        
        # 选择可用的服务器数据
        selected_servers = available_servers[:self.R_r]
        available_data = aggregated_encoded[selected_servers]
        available_C = self.C[selected_servers]
        
        print(f"使用服务器 {selected_servers} 进行恢复")
        
        try:
            # 使用伪逆求解线性系统
            C_pinv = torch.linalg.pinv(available_C)
            recovered_M = torch.matmul(C_pinv, available_data)
            
            # 从恢复的矩阵中提取模型更新（取第一行作为简化）
            global_update = recovered_M[0, :self.L]
            
            return global_update
            
        except Exception as e:
            print(f"恢复过程中出错: {e}")
            # 备用方案：直接平均
            print("使用备用恢复方案...")
            return torch.mean(aggregated_encoded[:, :self.L], dim=0)
    
    def evaluate_recovery_accuracy(self, original_updates: List[torch.Tensor],
                                recovered_update: torch.Tensor) -> dict:
        """
        评估恢复准确性
        
        Args:
            original_updates: 原始客户端更新列表
            recovered_update: 恢复的全局更新
            
        Returns:
            metrics: 评估指标字典
        """
        # 计算真实的全局更新（直接平均）
        true_global = torch.stack(original_updates).mean(dim=0)
        
        # 计算误差
        mse = torch.mean((recovered_update - true_global) ** 2).item()
        relative_error = torch.norm(recovered_update - true_global) / torch.norm(true_global)
        cosine_sim = torch.nn.functional.cosine_similarity(
            recovered_update.unsqueeze(0), true_global.unsqueeze(0)
        ).item()
        
        metrics = {
            'mse': mse,
            'relative_error': relative_error.item(),
            'cosine_similarity': cosine_sim,
            'true_norm': torch.norm(true_global).item(),
            'recovered_norm': torch.norm(recovered_update).item()
        }
        
        return metrics


def demo_secure_aggregation():
    """演示完整的安全聚合流程"""
    print("=" * 60)
    print("基于阶梯码的安全聚合演示")
    print("=" * 60)
    
    # 参数设置
    N = 6           # 服务器数量
    R_r = 4         # 恢复阈值
    K_c = 2         # 存储因子
    MODEL_DIM = 500  # 模型维度
    NUM_CLIENTS = 5  # 客户端数量
    
    # 初始化安全聚合协议
    print("\n1. 初始化安全聚合系统...")
    sec_agg = SecureAggregationSC(
        num_servers=N,
        recovery_threshold=R_r,
        storage_factor=K_c,
        model_dim=MODEL_DIM
    )
    
    # 生成模拟客户端更新
    print(f"\n2. 生成 {NUM_CLIENTS} 个客户端的模拟更新...")
    client_updates = []
    for i in range(NUM_CLIENTS):
        # 模拟模型更新（正态分布）
        update = torch.normal(0, 0.1, (MODEL_DIM,))
        client_updates.append(update)
        print(f"   客户端 {i}: 更新范数 = {torch.norm(update):.4f}")
    
    # 客户端编码
    print("\n3. 客户端进行安全编码...")
    encoded_updates = []
    for i, update in enumerate(client_updates):
        encoded = sec_agg.client_prepare_update(update, client_id=i)
        encoded_updates.append(encoded)
        print(f"   客户端 {i}: 编码完成, 形状: {encoded.shape}")
    
    # 模拟服务器掉线
    dropout_servers = [4, 5]  # 服务器4和5掉线
    available_servers = [i for i in range(N) if i not in dropout_servers]
    
    # 服务器聚合
    print(f"\n4. 服务器聚合更新 (掉线服务器: {dropout_servers})...")
    start_time = time.time()
    aggregated = sec_agg.server_aggregate_updates(encoded_updates, dropout_servers)
    aggregation_time = time.time() - start_time
    print(f"   聚合时间: {aggregation_time:.4f} 秒")
    
    # 恢复全局更新
    print(f"\n5. 从可用服务器 {available_servers} 恢复全局更新...")
    start_time = time.time()
    recovered_update = sec_agg.recover_global_update(aggregated, available_servers)
    recovery_time = time.time() - start_time
    print(f"   恢复时间: {recovery_time:.4f} 秒")
    
    # 评估恢复准确性
    print("\n6. 评估恢复准确性...")
    metrics = sec_agg.evaluate_recovery_accuracy(client_updates, recovered_update)
    
    print("\n恢复结果评估:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   相对误差: {metrics['relative_error']:.6f}")
    print(f"   余弦相似度: {metrics['cosine_similarity']:.6f}")
    print(f"   真实更新范数: {metrics['true_norm']:.4f}")
    print(f"   恢复更新范数: {metrics['recovered_norm']:.4f}")
    
    # 性能统计
    total_time = aggregation_time + recovery_time
    print(f"\n性能统计:")
    print(f"   总时间: {total_time:.4f} 秒")
    print(f"   客户端数量: {NUM_CLIENTS}")
    print(f"   模型维度: {MODEL_DIM}")
    print(f"   掉线容错: {len(dropout_servers)}/{N} 服务器")
    
    return sec_agg, metrics


def test_robustness():
    """测试系统的鲁棒性"""
    print("\n" + "=" * 60)
    print("鲁棒性测试: 不同掉线情况下的恢复能力")
    print("=" * 60)
    
    # 初始化
    sec_agg = SecureAggregationSC(num_servers=6, recovery_threshold=4, 
                                 storage_factor=2, model_dim=200)
    
    # 生成测试数据
    client_updates = [torch.normal(0, 0.1, (200,)) for _ in range(3)]
    encoded_updates = [sec_agg.client_prepare_update(update, i) for i, update in enumerate(client_updates)]
    
    # 测试不同的掉线情况
    dropout_scenarios = [
        [],           # 无掉线
        [5],          # 1个服务器掉线
        [1, 2],       # 2个服务器掉线
        [3, 4, 5],    # 3个服务器掉线（达到极限）
    ]
    
    print("\n掉线场景测试:")
    for i, dropouts in enumerate(dropout_scenarios):
        available = [j for j in range(6) if j not in dropouts]
        
        if len(available) < sec_agg.R_r:
            print(f"  场景 {i+1}: 服务器 {dropouts} 掉线 - 无法恢复 (只有 {len(available)} 个服务器)")
            continue
        
        try:
            aggregated = sec_agg.server_aggregate_updates(encoded_updates, dropouts)
            recovered = sec_agg.recover_global_update(aggregated, available)
            metrics = sec_agg.evaluate_recovery_accuracy(client_updates, recovered)
            
            print(f"  场景 {i+1}: 服务器 {dropouts} 掉线 - 恢复成功, 相对误差: {metrics['relative_error']:.6f}")
            
        except Exception as e:
            print(f"  场景 {i+1}: 服务器 {dropouts} 掉线 - 恢复失败: {e}")


if __name__ == "__main__":
    # 运行主演示
    sec_agg, metrics = demo_secure_aggregation()
    
    # 运行鲁棒性测试
    test_robustness()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
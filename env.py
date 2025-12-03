import numpy as np

class CuttingStockEnv:
    """
    环境实现：
    - patterns: n x m 矩阵（每个模式下产生每个 item 的数量）
    - trim_losses: 长度 n    pattern 的修剪损失（cm）
    - item_lengths: 长度 m    每种商品的长度（cm）
    - demand_prob: 一个概率向量，表示每种物品被需求的相对可能性。用于生成随机需求
    - dmin, dmax: 每个周期总需求的上下限，需求总量将在此范围内随机生成
    - s_max, x_max:每种物品的库存上限, 每个周期可切割的原材料最大数量（模拟生产资源的限制）
    - cost params: h_plus = h_plus_factor * item_length    库存持有成本
    - h_minus = h_minus_factor * item_length    缺货/失销成本
    - g_j = g_factor * trim_loss_j    切损成本
    """
    def __init__(self, patterns, trim_losses, item_lengths, demand_prob,
                 dmin=40, dmax=50, s_max=70, x_max=30,
                 h_plus_factor=0.01, h_minus_factor=1.0, g_factor=0.1, seed=None):
        self.A = np.array(patterns)
        self.A = self.A.astype(int)
        self.n, self.m = self.A.shape
        self.trim_losses = np.array(trim_losses)
        self.item_lengths = np.array(item_lengths)
        self.demand_prob = np.array(demand_prob) / np.sum(demand_prob)
        self.dmin = int(dmin); self.dmax = int(dmax)
        self.s_max = int(s_max)
        self.x_max = int(x_max)
        self.h_plus = h_plus_factor * self.item_lengths
        self.h_minus = h_minus_factor * self.item_lengths
        self.g = g_factor * self.trim_losses
        self.rng = np.random.default_rng(seed)

    def reset(self, init_inventory=None):
        '''
        复位函数
        全零库存\给定库存
        '''
        if init_inventory is None:
            self.s = np.zeros(self.m, dtype=int)
        else:
            self.s = np.array(init_inventory, dtype=int)
        return self.s.copy()

    def sample_demand(self):
        '''
        随机生成需求
        '''
        d_total = int(self.rng.integers(self.dmin, self.dmax+1))
        d = self.rng.multinomial(d_total, self.demand_prob)
        return d

    def is_feasible(self, x):
        '''
        可行性检查
        '''
        x = np.array(x, dtype=int)
        produced = self.A.T.dot(x)
        if np.any(self.s + produced > self.s_max):
            return False
        if np.sum(x) > self.x_max:
            return False
        if np.any(x < 0):
            return False
        return True

    def step(self, x):
        x = np.array(x, dtype=int)
        d = self.sample_demand()
        produced = self.A.T.dot(x)
        s_next = self.s + produced - d
        holding = np.maximum(s_next, 0)
        lost = np.maximum(-s_next, 0)
        cost_trim = np.dot(self.g, x)
        cost_hold = np.dot(self.h_plus, holding)
        cost_lost = np.dot(self.h_minus, lost)
        cost = cost_trim + cost_hold + cost_lost
        s_next = np.maximum(self.s + produced - d, 0).astype(int)
        self.s = s_next.copy()
        return s_next, float(cost), d

    def post_decision_state(self, s, x):
        '''
        事后状态
        '''
        return s + self.A.T.dot(x)
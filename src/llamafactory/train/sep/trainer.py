from .tree_search import MCTSTreeSearch
from trl import PPOTrainer
from .trainer import CustomDPOTrainer

class EXITPPOTrainer:
    def __init__(self, model, ref_model, finetuning_args, tokenizer, **kwargs):
        # 初始化 DPOTrainer
        self.dpo_trainer = CustomDPOTrainer(model=model, ref_model=ref_model, finetuning_args=finetuning_args, **kwargs)
        
        # 初始化树搜索模块
        self.tree_search = MCTSTreeSearch(model, tokenizer=tokenizer)

        # 初始化 PPO Trainer
        self.ppo_trainer = PPOTrainer(model=model, args=finetuning_args, tokenizer=tokenizer, **kwargs)

    def train_step(self, inputs):
        # 执行 EXIT 中的树搜索，生成专家策略
        search_paths = self.tree_search.search(inputs["text"], current_state={})
        
        # 使用生成的专家策略进行模仿学习
        self.dpo_trainer.update_strategy(search_paths)

        # 使用 PPO 进行策略优化
        self.ppo_trainer.train_step(inputs)

    def evaluate(self, inputs):
        # 在评估阶段，可以选择单独使用树搜索策略评估，也可以使用 DPO 或 PPO 评估
        return self.dpo_trainer.evaluate(inputs)

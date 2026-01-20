import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CVTrainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", type=str, required=True, help="训练配置文件")
    train_parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--config", type=str, required=True, help="评估配置文件")
    eval_parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from .core import load_yaml_file
        from .builder import build_trainer, build_loggers
        
        config_path = Path(args.config)
        os.chdir(config_path.parent)
        
        config = load_yaml_file(str(config_path))
        stages_config = config.get("stages", [])
        
        for stage_config in stages_config:
            loggers = build_loggers(stage_config.get("logger", []))
            trainer = build_trainer(stage_config, loggers)
            
            if args.resume:
                import torch
                trainer.load_state_dict(torch.load(args.resume))
            
            trainer.train(epochs=stage_config.get("epochs", 100))
            
            for logger in loggers:
                logger.close()
    
    elif args.command == "eval":
        from .core import load_yaml_file
        from .builder import build_trainer
        import torch
        
        config_path = Path(args.config)
        os.chdir(config_path.parent)
        
        config = load_yaml_file(str(config_path))
        
        checkpoint_path = args.checkpoint or config.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError("checkpoint path must be specified")
        
        trainer = build_trainer(config)
        trainer.load_state_dict(torch.load(checkpoint_path))
        
        eval_split = config.get("eval_split", "val")
        trainer.eval(eval_split)
        
        print(f"Evaluation on {eval_split}:")
        for name, meter in trainer.meters.items():
            print(f"  {name}: {meter.get_value()}")


if __name__ == "__main__":
    main()

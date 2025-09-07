import argparse
import sys
from pathlib import Path

from src.infrastructure.config.config_loader import ConfigLoader
from src.infrastructure.di.container import Container
from src.infrastructure.cli.checkpoint_menu import select_resume_checkpoint


def _cmd_train(args) -> int:
    # Load configs
    loader = ConfigLoader(Path(args.config))
    model_cfg = loader.load_model_config()
    training_cfg = loader.load_training_config()
    hardware_cfg = loader.load_hardware_config()
    data_cfg = loader.load_data_config()

    # Wire dependencies
    container = Container()
    container.register_configs(model_cfg, training_cfg, hardware_cfg, data_cfg)
    training_service = container.get_training_service()

    # Delegate interactive/non-interactive resume selection to infrastructure helper
    ckpt_repo = container.get_checkpoint_repository()
    resume_checkpoint = select_resume_checkpoint(ckpt_repo)

    # Set resume checkpoint on the training service, if any
    if hasattr(training_service, 'set_resume_checkpoint'):
        training_service.set_resume_checkpoint(resume_checkpoint)

    # Validate and run
    training_service.validate_configuration(model_cfg, training_cfg)
    result = training_service.train_model(model_cfg, training_cfg)

    # Minimal reporting
    print("Training complete.")
    print("Final checkpoint:", result.final_checkpoint_path)
    print("Best val loss:", result.best_val_loss)
    print("Total steps:", result.total_steps)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refactored training entrypoint")
    parser.add_argument("command", choices=["train"], help="Action to perform")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")

    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if args.command == "train":
        return _cmd_train(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from configs import get_config
from trainer import Trainer

if __name__ =="__main__":
    config = get_config().config
    trainer = Trainer(config)
    trainer.run()
    
import importlib

from ctu.trainers.base_trainer import BaseTrainer

def get_trainer(opt):
  trainer_filename = "ctu.trainers." + opt.model + "_trainer"
  trainer_lib = importlib.import_module(trainer_filename)

  trainer = None
  target_trainer_name = opt.model + 'trainer'
  for name, cls in trainer_lib.__dict__.items():
    if name.lower() == target_trainer_name.lower() and issubclass(cls, BaseTrainer):
      trainer = cls

  if trainer is None:
    raise ValueError("In {}.py, there should be a subclass of BaseTrainer "
                     "with class name that matches {} in lowercase.".format(
                       trainer_filename, target_trainer_name))

  return trainer

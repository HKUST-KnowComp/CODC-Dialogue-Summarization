"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel
from onmt.models.bin_class_model import BinClassModel
from onmt.models.input_attention import InputAttention

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel", "check_sru_requirement", 
           "InputAttention"]
